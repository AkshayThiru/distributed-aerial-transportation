import os

import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from meshcat import Visualizer
from scipy import constants
from scipy.linalg import polar

from utils.geometry_utils import faces_from_vertex_rep
from utils.math_utils import rotation_matrix_a_to_b

_INTEGRATION_STEPS_PER_ROTATION_PROJECTION = 20

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
_QUADROTOR_OBJ_PATH = _DIR_PATH + "/../objs/quadrotor.obj"
_OBJ_MATERIAL = gm.MeshLambertMaterial(
    color=0xFFFFFF, wireframe=False, opacity=1, reflectivity=0.5
)
_MESH_MATERIAL = gm.MeshBasicMaterial(
    color=0xFF22DD, wireframe=True, linewidth=2, opacity=1
)
_FORCE_MATERIAL = gm.MeshLambertMaterial(
    color=0x47A7B8, wireframe=False, opacity=0.75, reflectivity=0.5
)

_QUADROTOR_RADIUS = 0.3  # [m].
_DRAW_FORCE_ARROWS = False
_FORCE_TAIL_RADIUS = 0.01  # [m].
_FORCE_HEAD_BASE_RADIUS = 0.03  # [m].
_FORCE_HEAD_LENGTH = 0.1  # [m].
_FORCE_SCALING = 0.2  # [m/N].
_FORCE_MIN_LENGTH = 0.05  # [m].
_FORCE_OFFSET = np.array([0, 0, 0.0])  # [m].


class RQPParameters:
    def __init__(
        self, m: np.ndarray, J: np.ndarray, ml: float, Jl: np.ndarray, r: np.ndarray
    ) -> None:
        """Inputs:
        m:  quadrotor masses, [kg], [m1, ..., mn],
        J:  quadrotor inertias, [kg-m], J[:, :, i] is the inertia of i-th quadrotor.
        ml: payload mass, [kg],
        Jl: payload inertia in body frame, [kg-m^2],
        r:  rigid link attachment position in body frame, [m], [r1, ..., rn].
        """
        self.n = r.shape[1]
        assert m.shape == (self.n,)
        assert J.shape == (3, 3, self.n)
        assert Jl.shape == (3, 3)
        assert r.shape == (3, self.n)
        self.m = m
        self.J = J
        self.ml = ml
        self.Jl = Jl
        self.r = r
        # total mass.
        self.mT = np.sum(m) + ml
        # Relative center of mass position (in body frame).
        self.x_com = np.sum(r * m, axis=1) / self.mT
        # Rigid link attachment position relative to CoM.
        self.r_com = (self.r.T - self.x_com).T
        # Total body inertia.
        self.JT = Jl - ml * pin.skewSquare(self.x_com, self.x_com)
        for i in range(self.n):
            self.JT = self.JT - m[i] * pin.skewSquare(
                self.r_com[:, i], self.r_com[:, i]
            )
        self.JT_inv = np.linalg.inv(self.JT)
        self.J_inv = np.empty((3, 3, self.n))
        for i in range(self.n):
            self.J_inv[:, :, i] = np.linalg.inv(J[:, :, i])


class RQPState:
    def __init__(
        self,
        R: np.ndarray,
        w: np.ndarray,
        xl: np.ndarray,
        vl: np.ndarray,
        Rl: np.ndarray,
        wl: np.ndarray,
    ) -> None:
        """Inputs:
        R:  quadrotor rotations in SO(3), R[:, :, i] for the i-th quadrotor,
        w:  w[:, i] is the body angular velocity of the i-th quadrotor,
        xl: payload CoM position in R^3,
        vl: payload CoM velocity,
        Rl: payload rotation in SO(3),
        wl: payload body angular velocity, dRl = Rl * hat(wl).
        """
        self.n = w.shape[1]
        assert R.shape == (3, 3, self.n)
        assert w.shape == (3, self.n)
        assert xl.shape == (3,)
        assert vl.shape == (3,)
        assert Rl.shape == (3, 3)
        assert wl.shape == (3,)
        self.R = R
        self.w = w
        self.xl = xl
        self.vl = vl
        self.Rl = Rl
        self.wl = wl
        self.project_R()
        self._counter = 0

    def project_R(self) -> None:
        """Project R[:, :, i] and Rl onto SO(3).
        See Algorithm 6.4.1 in G.H. Golub, C.F. Van Loan, "Matrix Computations, Fourth Edition", 2013.
        """
        self.Rl, _ = polar(self.Rl)
        for i in range(self.n):
            self.R[:, :, i], _ = polar(self.R[:, :, i])

    def integrate(
        self, dw: np.ndarray, dvl: np.ndarray, dwl: np.ndarray, dt: float
    ) -> None:
        """Integrate state along acceleration."""
        assert dw.shape == self.w.shape
        assert dvl.shape == (3,)
        assert dwl.shape == (3,)
        for i in range(self.n):
            self.R[:, :, i] = self.R[:, :, i] @ pin.exp3(
                (self.w[:, i] + dw[:, i] * dt / 2) * dt
            )
            self.w[:, i] = self.w[:, i] + dw[:, i] * dt
        self.xl = self.xl + self.vl * dt + dvl * dt**2 / 2
        self.vl = self.vl + dvl * dt
        self.Rl = self.Rl @ pin.exp3((self.wl + dwl * dt / 2) * dt)
        self.wl = self.wl + dwl * dt
        self._counter = self._counter + 1
        if self._counter >= _INTEGRATION_STEPS_PER_ROTATION_PROJECTION:
            self.project_R()
            self._counter = 0


class RQPDynamics:
    """Dynamics of a rigid payload carried by rigidly attached quadrotors.
    state:  state of the system,
    xi:     position of i-th quadrotor CoM, = xl + Rl ri (internal variable),
    fi:     thrust by i-th quadrotor (input) (in ground frame),
    Mi:     moment by i-th quadrotor (input) (in i-th quadrotor body frame),
    Ti:     force on payload by i-th quadrotor (internal variable) (in ground frame).
    Dynamics:
    mi xi'' = fi Ri e3 - mi g (0,0,1) - Ti,
    Ji wi' + hat(wi) Ji wi = Mi,
    ml dxl' = sum_i Ti - ml g (0,0,1),
    Jl wl' + hat(wl) Jl wl = sum_i hat(ri) Rl.T Ti,
    """

    def __init__(self, params: RQPParameters, state: RQPState, dt: float) -> None:
        assert params.n == state.n
        self.num_quadrotors = params.n
        self.params = params
        self.state = state
        self.gravity = -constants.g * np.array([0, 0, 1])
        self.dt = dt

    def forward_dynamics(
        self, w: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the acceleration given the quadrotor inputs.
        Inputs:
        w:  wrench, = (f, M),
        f:  force (in ground frame), = [f1, ..., fn],
        M:  moment (in i-th quadrotor body frame), = [M1, ..., Mn].
        Outputs:
        (dw, dvl, dwl); with
        dw:     derivative of state.w,
        dvl:    acceleration of state.xl,
        dwl:    derivative of state.wl.
        """
        f, M = w
        assert f.shape == (self.num_quadrotors,)
        assert M.shape == (3, self.num_quadrotors)
        # Quadrotor angular accelarations.
        dw = np.empty((3, self.num_quadrotors))
        for i in range(self.num_quadrotors):
            dw[:, i] = self.params.J_inv[:, :, i] @ (
                M[:, i]
                - pin.skew(self.state.w[:, i])
                @ self.params.J[:, :, i]
                @ self.state.w[:, i]
            )
        # Center of mass accelerations.
        quad_force = self.state.R[:, 2, :] * f
        dv_com = np.sum(quad_force, axis=1) / self.params.mT + self.gravity
        net_moment_com = np.sum(
            np.cross(
                self.params.r_com,
                self.state.Rl.T @ quad_force,
                axisa=0,
                axisb=0,
                axisc=0,
            ),
            axis=1,
        )
        dwl = self.params.JT_inv @ (
            net_moment_com - pin.skew(self.state.wl) @ self.params.JT @ self.state.wl
        )
        # Payload acceleration.
        dvl = (
            dv_com
            - self.state.Rl
            @ (pin.skewSquare(self.state.wl, self.state.wl) + pin.skew(dwl))
            @ self.params.x_com
        )
        return (dw, dvl, dwl)

    @staticmethod
    def inverse_dynamics_error(
        s: RQPState,
        p: RQPParameters,
        w: tuple[np.ndarray, np.ndarray],
        acc: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Computes error in dynamics equations.
        Inputs:
        s:      state of the system,
        p:      system parameters,
        w:      wrench, = (f, M),
        f:      force applied by the actuators (in ground frame),
        M:      moment applied by quadrotors (in quadrotor body frame),
        acc:    acceleration, = (dw, dvl, dwl).
        Outputs:
        err:    Norm of dynamics error.
        """
        f, M = w
        dw, dvl, dwl = acc
        gravity_vec = -constants.g * np.array([0, 0, 1])
        dv_quad = (
            np.vstack(dvl) + s.Rl @ (pin.skewSquare(s.wl, s.wl) + pin.skew(dwl)) @ p.r
        )
        internal_force = s.R[:, 2, :] * f + np.vstack(gravity_vec) * p.m - p.m * dv_quad
        com_acc_err = np.linalg.norm(
            p.ml * dvl - p.ml * gravity_vec - np.sum(internal_force, axis=1)
        )
        load_moment = np.sum(
            np.cross(p.r, s.Rl.T @ internal_force, axisa=0, axisb=0, axisc=0), axis=1
        )
        com_ang_err = np.linalg.norm(
            p.Jl @ dwl + pin.skew(s.wl) @ p.Jl @ s.wl - load_moment
        )
        quad_ang_err_sq = 0.0
        for i in range(p.n):
            quad_ang_err_sq += (
                np.linalg.norm(
                    p.J[:, :, i] @ dw[:, i]
                    + pin.skew(s.w[:, i]) @ p.J[:, :, i] @ s.w[:, i]
                    - M[:, i]
                )
                ** 2
            )
        dyn_err = np.sqrt(com_acc_err**2 + com_ang_err**2 + quad_ang_err_sq)
        return dyn_err

    def integrate(self, w: tuple[np.ndarray, np.ndarray]) -> None:
        f, M = w
        assert f.shape == (self.num_quadrotors,)
        assert M.shape == (3, self.num_quadrotors)
        acc = self.forward_dynamics(w)
        self.state.integrate(*acc, self.dt)


class RQPCollision:
    """Class for storing the collision objects of the RQP system."""

    def __init__(
        self, payload_vertices: np.ndarray, payload_mesh_vertices: np.ndarray
    ) -> None:
        """
        Inputs:
        payload_vertices:       vertices of the payload at I_SE(3) configuration, (n, 3) ndarray.
        payload_mesh_vertices:  vertices of the payload mesh (used for collision).
        """
        assert payload_vertices.shape[1] == 3
        assert payload_mesh_vertices.shape[1] == 3
        # Payload object.
        payload_faces = faces_from_vertex_rep(payload_vertices)
        self.payload_obj = gm.TriangularMeshGeometry(payload_vertices, payload_faces)
        # Payload mesh (and collision) object.
        self.payload_mesh_vertices = payload_mesh_vertices
        payload_mesh_faces = faces_from_vertex_rep(payload_mesh_vertices)
        self.payload_mesh = gm.TriangularMeshGeometry(
            payload_mesh_vertices, payload_mesh_faces
        )
        # Quadrotor object.
        self.quad_obj = gm.ObjMeshGeometry.from_file(_QUADROTOR_OBJ_PATH)


class RQPVisualizer:
    """Class for visualizing the RQP system."""

    def __init__(
        self, param: RQPParameters, col: RQPCollision, vis: Visualizer
    ) -> None:
        self.param = param
        self.col = col

        # Display system in zero configuration.
        # Set payload object.
        vis["rqp_payload"].set_object(self.col.payload_obj, _OBJ_MATERIAL)
        # Set payload meshes.
        vis["rqp_payload_mesh"].set_object(self.col.payload_mesh, _MESH_MATERIAL)
        # Set quadrotor objects and meshes.
        n = self.param.n
        for i in range(n):
            quadrotor = "rqp_quadrotor_" + str(i)
            quadrotor_mesh = "rqp_quadrotor_mesh_" + str(i)
            vis[quadrotor].set_object(self.col.quad_obj, _OBJ_MATERIAL)
            vis[quadrotor_mesh].set_object(gm.Sphere(_QUADROTOR_RADIUS), _MESH_MATERIAL)
            T = tf.translation_matrix(self.param.r[:, i])
            vis[quadrotor].set_transform(T)
            vis[quadrotor_mesh].set_transform(T)
        # Set force arrows.
        if _DRAW_FORCE_ARROWS:
            for i in range(n):
                force_tail = "rqp_force_tail_" + str(i)
                force_head = "rqp_force_head_" + str(i)
                vis[force_tail].set_object(
                    gm.Cylinder(height=_FORCE_MIN_LENGTH, radius=_FORCE_TAIL_RADIUS),
                    _FORCE_MATERIAL,
                )
                vis[force_head].set_object(
                    gm.Cylinder(
                        height=_FORCE_HEAD_LENGTH,
                        radiusBottom=_FORCE_HEAD_BASE_RADIUS,
                        radiusTop=0.0,
                    ),
                    _FORCE_MATERIAL,
                )
                T = tf.translation_matrix(
                    np.array([0, 0, _FORCE_MIN_LENGTH / 2])
                    + self.param.r[:, i]
                    + _FORCE_OFFSET
                )
                T[:3, :3] = rotation_matrix_a_to_b(
                    np.array([0, 1, 0]), np.array([0, 0, 1])
                )
                vis[force_tail].set_transform(T)
                T[2, 3] += _FORCE_MIN_LENGTH / 2 + _FORCE_HEAD_LENGTH / 2
                vis[force_head].set_transform(T)

    def update(self, state: RQPState, f: np.ndarray, vis: Visualizer) -> None:
        xl = state.xl
        Rl = state.Rl
        Rq = state.R
        r = self.param.r
        n = self.param.n
        assert f.shape == (n,)
        # Update payload object and mesh.
        T = tf.translation_matrix(xl)
        T[:3, :3] = Rl
        vis["rqp_payload"].set_transform(T)
        vis["rqp_payload_mesh"].set_transform(T)
        # Update quadrotor objects and meshes.
        for i in range(n):
            quadrotor = "rqp_quadrotor_" + str(i)
            quadrotor_mesh = "rqp_quadrotor_mesh_" + str(i)
            T = tf.translation_matrix(xl + Rl @ r[:, i])
            T[:3, :3] = Rq[:, :, i]
            vis[quadrotor].set_transform(T)
            vis[quadrotor_mesh].set_transform(T)
        # Update force arrows.
        if _DRAW_FORCE_ARROWS:
            for i in range(n):
                force_tail = "rqp_force_tail_" + str(i)
                force_head = "rqp_force_head_" + str(i)
                force_dir = Rq[:, 2, i]
                force_length = np.max([f[i] * _FORCE_SCALING, _FORCE_MIN_LENGTH])
                T = tf.translation_matrix(
                    xl + Rl @ (r[:, i] + _FORCE_OFFSET) + force_length / 2 * force_dir
                )
                T[:3, :3] = rotation_matrix_a_to_b(np.array([0, 1, 0]), force_dir)
                # Create new force tail.
                vis[force_tail].delete()
                vis[force_tail].set_object(
                    gm.Cylinder(height=force_length, radius=_FORCE_TAIL_RADIUS),
                    _FORCE_MATERIAL,
                )
                vis[force_tail].set_transform(T)
                # Update force heads.
                T[:3, 3] += (force_length + _FORCE_HEAD_LENGTH) / 2 * force_dir
                vis[force_head].set_transform(T)
