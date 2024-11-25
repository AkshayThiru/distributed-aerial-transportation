import os

import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from meshcat import Visualizer
from scipy import constants
from scipy.linalg import cholesky, polar, solve

from utils.geometry_utils import faces_from_vertex_rep
from utils.math_utils import rotation_matrix_a_to_b

_INTEGRATION_STEPS_PER_ROTATION_PROJECTION = 20

_OBJ_MATERIAL = gm.MeshLambertMaterial(
    color=0xFFFFFF, wireframe=False, opacity=1, reflectivity=0.5
)
_MESH_MATERIAL = gm.MeshBasicMaterial(
    color=0xFF22DD, wireframe=True, linewidth=2, opacity=1
)
_FORCE_MATERIAL = gm.MeshLambertMaterial(
    color=0x47A7B8, wireframe=False, opacity=0.75, reflectivity=0.5
)

_ACTUATOR_RADIUS = 0.05  # [m].
_ACTUATOR_MESH_RADIUS = 0.15 # [m].
_LINK_RADIUS = 0.01  # [m].
_DRAW_FORCE_ARROWS = False
_FORCE_TAIL_RADIUS = 0.01  # [m].
_FORCE_HEAD_BASE_RADIUS = 0.03  # [m].
_FORCE_HEAD_LENGTH = 0.1  # [m].
_FORCE_SCALING = 0.2  # [m/N].
_FORCE_MIN_LENGTH = 0.05  # [m].


class PMRLParameters:
    def __init__(
        self, m: np.ndarray, ml: float, Jl: np.ndarray, r: np.ndarray, L: np.ndarray
    ) -> None:
        """Inputs:
        m:  quadrotor masses, [kg], [m1, ..., mn],
        ml: payload mass, [kg],
        Jl: payload inertia in body frame, [kg-m^2],
        r:  rigid link attachment position in body frame, [m], [r1, ..., rn],
        L:  rigid link lengths, [m], [L1, ..., Ln].
        """
        self.n = m.shape[0]
        assert m.shape == (self.n,)
        assert Jl.shape == (3, 3)
        assert r.shape == (3, self.n)
        assert L.shape == (self.n,)
        self.m = m
        self.ml = ml
        self.Jl = Jl
        self.r = r
        self.L = L
        self.Jl_inv = np.linalg.inv(Jl)
        self.Jl_inv_chol = cholesky(self.Jl_inv)
        self.diag_m_inv = np.diag(1 / m)
        self.m_times_r = r * m
        self.m_times_L = L * m
        self.ml_times_L = L * ml
        self.r_divided_L = r / L


class PMRLState:
    def __init__(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        xl: np.ndarray,
        vl: np.ndarray,
        Rl: np.ndarray,
        wl: np.ndarray,
    ) -> None:
        """Inputs:
        q:  rigid link unit vectors in world frame in (S^2)^n, [q1, ..., qn],
        dq: tangent vector for q, [dq1, ..., dqn], <qi, dqi> = 0,
        xl: payload CoM position in R^3,
        vl: payload CoM velocity,
        Rl: payload rotation in SO(3),
        wl: payload body angular velocity, dRl = Rl * hat(wl).
        """
        assert q.shape == dq.shape
        assert q.shape[0] == 3
        assert xl.shape == (3,)
        assert vl.shape == (3,)
        assert Rl.shape == (3, 3)
        assert wl.shape == (3,)
        self.q = q
        self.dq = dq
        self.project_q()
        self.xl = xl
        self.vl = vl
        self.Rl = Rl
        self.wl = wl
        self.project_R()
        self._counter = 0

    def project_q(self) -> None:
        """Project q onto (S^2)^n and dq onto T_q (S^2)^n."""
        assert all(np.linalg.norm(self.q, axis=0) > 0)
        self.q = self.q / np.linalg.norm(self.q, axis=0)
        self.dq = self.dq - self.q * (self.q * self.dq).sum(axis=0)

    def project_R(self) -> None:
        """Project Rl onto SO(3).
        See Algorithm 6.4.1 in G.H. Golub, C.F. Van Loan, "Matrix Computations, Fourth Edition", 2013.
        """
        self.Rl, _ = polar(self.Rl)

    def integrate(
        self, ddq: np.ndarray, dvl: np.ndarray, dwl: np.ndarray, dt: float
    ) -> None:
        """Integrate state along acceleration.
        Note: ddq should satisfy q[:,i].T @ ddq[:,i] + dq[:,i].T @ dq[:,i] = 0.
        """
        assert ddq.shape == self.q.shape
        assert dvl.shape == (3,)
        assert dwl.shape == (3,)
        self.q = self.q + self.dq * dt + ddq * dt**2 / 2
        self.dq = self.dq + ddq * dt
        self.project_q()
        self.xl = self.xl + self.vl * dt + dvl * dt**2 / 2
        self.vl = self.vl + dvl * dt
        self.Rl = self.Rl @ pin.exp3((self.wl + dwl * dt / 2) * dt)
        self.wl = self.wl + dwl * dt
        self._counter = self._counter + 1
        if self._counter >= _INTEGRATION_STEPS_PER_ROTATION_PROJECTION:
            self.project_R()
            self._counter = 0


class PMRLDynamics:
    """Dynamics of a rigid payload carried by point mass robots with rigid links.
    state:  state of the system,
    xi:     position of i-th point mass, = xl + Li qi + Rl ri (internal variable),
    fi:     thrust on i-th point mass (input) (in ground frame),
    Ti:     tension (or compression) in i-th link (internal variable).
    Dynamics:
    mi xi'' = fi - mi g (0,0,1) - Ti qi,
    ml dxl' = sum_i Ti qi - ml g (0,0,1),
    Jl wl' + hat(wl) Jl wl = sum_i hat(ri) Ti Rl.T qi,
    qi.T dqi' = - norm(dqi)**2.
    """

    def __init__(self, params: PMRLParameters, state: PMRLState, dt: float) -> None:
        assert params.m.shape[0] == state.q.shape[1]
        self.num_quadrotors = params.m.shape[0]
        self.params = params
        self.state = state
        self.gravity = -constants.g * np.array([0, 0, 1])
        self.dt = dt

    def forward_dynamics(
        self, f: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Computes the acceleration given the force.
        Inputs:
        f:  force (in ground frame), = [f1, ..., fn].
        Outputs:
        (ddq, dvl, dwl), T; with
        ddq:    acceleration of state.q,
        dvl:    acceleration of state.xl,
        dwl:    derivative of state.wl,
        T:      tension (or compression) force in the rigid links, = [T1, ..., Tn].
        """
        assert f.shape == (3, self.num_quadrotors)
        wl_hat_sq = pin.skewSquare(self.state.wl, self.state.wl)
        cor_acc = self.params.Jl_inv @ np.cross(
            self.state.wl, self.params.Jl @ self.state.wl
        )
        cor_mat = self.state.Rl @ (wl_hat_sq - pin.skew(cor_acc))
        add_force = f - cor_mat @ self.params.m_times_r
        tension_rhs = (add_force * self.state.q).sum(
            axis=0
        ) + self.params.m_times_L * np.linalg.norm(self.state.dq, axis=0) ** 2
        tension_rhs = tension_rhs / self.params.m
        r_cross_q_mat = np.cross(
            self.params.r, self.state.Rl.T @ self.state.q, axisa=0, axisb=0, axisc=0
        )
        temp_mat_ = self.params.Jl_inv_chol @ r_cross_q_mat
        # Matrix times its transpose is implicitly fast.
        tension_lhs = (
            self.params.diag_m_inv
            + 1 / self.params.ml * (self.state.q.T @ self.state.q)
            + (temp_mat_.T @ temp_mat_)
        )

        T = solve(
            tension_lhs,
            tension_rhs,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
            assume_a="pos",
        )
        q_times_T = self.state.q @ T
        r_cross_q_mat_times_T = self.params.Jl_inv @ r_cross_q_mat @ T
        ddq = (
            (add_force - self.state.q * T) / self.params.m_times_L
            - np.vstack(q_times_T) / self.params.ml_times_L
            - self.state.Rl @ pin.skew(r_cross_q_mat_times_T) @ self.params.r_divided_L
        )
        dvl = q_times_T / self.params.ml + self.gravity
        dwl = r_cross_q_mat_times_T - cor_acc
        return (ddq, dvl, dwl), T

    @staticmethod
    def inverse_dynamics_error(
        s: PMRLState,
        p: PMRLParameters,
        f: np.ndarray,
        T: np.ndarray,
        acc: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Computes error in dynamics equations.
        Inputs:
        s:      state of the system,
        p:      system parameters,
        f:      force applied by the actuators,
        T:      tension in the links, = [T1, ..., Tn],
        acc:    acceleration, = (ddq, dvl, dwl).
        Outputs:
        err:    Norm of dynamics error.
        """
        ddq, dvl, dwl = acc
        gravity_vec = constants.g * np.array([[0], [0], [1]])
        dv_quad = (
            np.vstack(dvl)
            + ddq * p.L
            + s.Rl @ (pin.skewSquare(s.wl, s.wl) + pin.skew(dwl)) @ p.r
        )
        quad_acc_err = (
            np.linalg.norm(dv_quad * p.m - f + gravity_vec * p.m + s.q * T) ** 2
        )
        load_acc_err = np.linalg.norm(p.ml * dvl - s.q @ T + p.ml * gravity_vec.T) ** 2
        r_cross_q = np.cross(p.r, s.Rl.T @ s.q, axisa=0, axisb=0, axisc=0)
        load_ang_err = (
            np.linalg.norm(p.Jl @ dwl + pin.skew(s.wl) @ p.Jl @ s.wl - r_cross_q @ T)
            ** 2
        )
        ddq_residual_err = (
            np.linalg.norm((s.q * ddq).sum(axis=0) + np.linalg.norm(s.dq, axis=0) ** 2)
            ** 2
        )
        dyn_err = np.sqrt(quad_acc_err + load_acc_err + load_ang_err + ddq_residual_err)
        return dyn_err

    def integrate(self, f: np.ndarray) -> None:
        assert f.shape == (3, self.num_quadrotors)
        acc, _ = self.forward_dynamics(f)
        self.state.integrate(*acc, self.dt)


class PMRLCollision:
    """Class for storing the collision objects of the system."""

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


class PMRLVisualizer:
    """Class for visualizing the PMRL system."""

    def __init__(
        self, params: PMRLParameters, col: PMRLCollision, vis: Visualizer
    ) -> None:
        self.params = params
        self.col = col

        # Display system in zero configuration.
        # Set payload.
        vis["pmrl_payload"].set_object(self.col.payload_obj, _OBJ_MATERIAL)
        # Set payload meshes.
        vis["pmrl_payload_mesh"].set_object(self.col.payload_mesh, _MESH_MATERIAL)
        # Set rigid links, acutator objects, and actuator meshes.
        n = self.params.n
        for i in range(n):
            link = "pmrl_link_" + str(i)
            vis[link].set_object(
                gm.Cylinder(self.params.L[i], _LINK_RADIUS)
            )
            T = tf.translation_matrix(np.array([0, 0, self.params.L[i] / 2]) + self.params.r[:, i])
            T[:3, :3] = rotation_matrix_a_to_b(np.array([0, 1, 0]), np.array([0, 0, 1]))
            vis[link].set_transform(T)
            # Set actuators and meshes.
            actuator = "pmrl_actuator_" + str(i)
            actuator_mesh = "pmrl_actuator_mesh_" + str(i)
            vis[actuator].set_object(
                gm.Sphere(_ACTUATOR_RADIUS), _OBJ_MATERIAL
            )
            vis[actuator_mesh].set_object(
                gm.Sphere(_ACTUATOR_MESH_RADIUS), _MESH_MATERIAL
            )
            T = tf.translation_matrix(np.array([0, 0, self.params.L[i]]) + self.params.r[:, i])
            vis[actuator].set_transform(T)
            vis[actuator_mesh].set_transform(T)
        # Set force arrows.
        if _DRAW_FORCE_ARROWS:
            for i in range(n):
                force_tail = "pmrl_force_tail_" + str(i)
                force_head = "pmrl_force_head_" + str(i)
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
                    np.array([0, 0, _FORCE_MIN_LENGTH / 2 + self.params.L[i]]) + self.params.r[:, i]
                )
                T[:3, :3] = rotation_matrix_a_to_b(np.array([0, 1, 0]), np.array([0, 0, 1]))
                vis[force_tail].set_transform(T)
                T[2, 3] += _FORCE_MIN_LENGTH / 2 + _FORCE_HEAD_LENGTH / 2
                vis[force_head].set_transform(T)

    def update(self, state: PMRLState, f: np.ndarray, vis: Visualizer) -> None:
        xl = state.xl
        Rl = state.Rl
        r = self.params.r
        q = state.q
        L = self.params.L
        n = self.params.n
        assert f.shape == (3, n)

        # Update payload object and mesh.
        T = tf.translation_matrix(xl)
        T[:3, :3] = Rl
        vis["pmrl_payload"].set_transform(T)
        vis["pmrl_payload_mesh"].set_transform(T)
        # Update rigid links, acutator objects, and actuator meshes.
        for i in range(n):
            link = "pmrl_link_" + str(i)
            T = tf.translation_matrix(xl + Rl @ r[:, i] + q[:, i] * L[i] / 2)
            T[:3, :3] = rotation_matrix_a_to_b(np.array([0, 1, 0]), q[:, i])
            vis[link].set_transform(T)
            # Set actuators and meshes.
            actuator = "pmrl_actuator_" + str(i)
            actuator_mesh = "pmrl_actuator_mesh_" + str(i)
            T[:3, 3] += q[:, i] * L[i] / 2
            vis[actuator].set_transform(T)
            vis[actuator_mesh].set_transform(T)
        # Update force arrows.
        if _DRAW_FORCE_ARROWS:
            for i in range(n):
                force_tail = "pmrl_force_tail_" + str(i)
                force_head = "pmrl_force_head_" + str(i)
                force_length = np.linalg.norm(f[:, i])
                if force_length == 0:
                    force_dir = np.array([0, 0, 1])
                else:
                    force_dir = f[:, i] / force_length
                force_length = np.max([force_length * _FORCE_SCALING, _FORCE_MIN_LENGTH])
                T = tf.translation_matrix(xl + Rl @ r[:, i] + q[:, i] * L[i] + force_length / 2 * force_dir)
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
