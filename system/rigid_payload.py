import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from meshcat import Visualizer
from scipy import constants
from scipy.linalg import cholesky, polar

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
_FORCE_TAIL_RADIUS = 0.01  # [m].
_FORCE_HEAD_BASE_RADIUS = 0.03  # [m].
_FORCE_HEAD_LENGTH = 0.1  # [m].
_FORCE_SCALING = 1.0  # [m/N].
_FORCE_MIN_LENGTH = 0.05  # [m].


class RPParameters:
    def __init__(self, ml: float, Jl: np.ndarray, r: np.ndarray) -> None:
        """Inputs:
        ml: payload mass, [kg],
        Jl: payload inertia in body frame, [kg-m^2],
        r:  rigid link attachment position in body frame, [m], [r1, ..., rn].
        """
        self.n = r.shape[1]
        assert Jl.shape == (3, 3)
        assert r.shape == (3, self.n)
        self.ml = ml
        self.Jl = Jl
        self.r = r
        self.Jl_inv = np.linalg.inv(Jl)
        self.Jl_inv_chol = cholesky(self.Jl_inv)


class RPState:
    def __init__(
        self, xl: np.ndarray, vl: np.ndarray, Rl: np.ndarray, wl: np.ndarray
    ) -> None:
        """Inputs:
        xl: payload CoM position in R^3,
        vl: payload CoM velocity,
        Rl: payload rotation in SO(3),
        wl: payload body angular velocity, dRl = Rl * hat(wl).
        """
        assert xl.shape == (3,)
        assert vl.shape == (3,)
        assert Rl.shape == (3, 3)
        assert wl.shape == (3,)
        self.xl = xl
        self.vl = vl
        self.Rl = Rl
        self.wl = wl
        self.project_R()
        self._counter = 0

    def project_R(self) -> None:
        """Project Rl onto SO(3).
        See Algorithm 6.4.1 in G.H. Golub, C.F. Van Loan, "Matrix Computations, Fourth Edition", 2013.
        """
        self.Rl, _ = polar(self.Rl)

    def integrate(self, dvl: np.ndarray, dwl: np.ndarray, dt: float) -> None:
        """Integrate state along acceleration."""
        assert dvl.shape == (3,)
        assert dwl.shape == (3,)
        self.xl = self.xl + self.vl * dt + dvl * dt**2 / 2
        self.vl = self.vl + dvl * dt
        self.Rl = self.Rl @ pin.exp3((self.wl + dwl * dt / 2) * dt)
        self.wl = self.wl + dwl * dt
        self._counter = self._counter + 1
        if self._counter >= _INTEGRATION_STEPS_PER_ROTATION_PROJECTION:
            self.project_R()
            self._counter = 0


class RPDynamics:
    """Dynamics of a rigid payload carried by force actuators.
    state:  state of the system,
    fi:     force by the i-th point actuator at position ri (input) (in ground frame),
    Dynamics:
    ml dxl' = sum_i fi - ml g (0,0,1),
    Jl wl' + hat(wl) Jl wl = sum_i hat(ri) Rl.T fi.
    """

    def __init__(self, params: RPParameters, state: RPState, dt: float) -> None:
        self.num_actuators = params.n
        self.params = params
        self.state = state
        self.gravity = -constants.g * np.array([0, 0, 1])
        self.dt = dt

    def forward_dynamics(
        self, f: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the acceleration given the force.
        Inputs:
        f:  force (in ground frame), = [f1, ..., fn].
        Outputs:
        (dvl, dwl); with
        dvl:    acceleration of state.xl,
        dwl:    derivative of state.wl.
        """
        assert f.shape == (3, self.num_actuators)
        net_force = np.sum(f, axis=1)
        net_moment = np.sum(
            np.cross(self.params.r, self.state.Rl.T @ f, axisa=0, axisb=0, axisc=0),
            axis=1,
        )

        dvl = net_force / self.params.ml + self.gravity
        dwl = self.params.Jl_inv @ (
            net_moment - np.cross(self.state.wl, self.params.Jl @ self.state.wl)
        )

        return (dvl, dwl)

    @staticmethod
    def inverse_dynamics_error(
        s: RPState, p: RPParameters, f: np.ndarray, acc: tuple[np.ndarray, np.ndarray]
    ) -> float:
        """Computes error in dynamics equations.
        Inputs:
        s:      state of the system,
        p:      rigid payload parameters,
        f:      force applied by the actuators,
        acc:    acceleration, = (dvl, dwl).
        Outputs:
        err:    Norm of dynamics error.
        """
        dvl, dwl = acc
        gravity_vec = -constants.g * np.array([0, 0, 1])
        net_force = np.sum(f, axis=1) + p.ml * gravity_vec.T
        net_moment = np.sum(
            np.cross(p.r, s.Rl.T @ f, axisa=0, axisb=0, axisc=0), axis=1
        )
        load_acc_err = np.linalg.norm(p.ml * dvl - net_force) ** 2
        load_ang_err = (
            np.linalg.norm(p.Jl @ dwl + pin.skew(s.wl) @ p.Jl @ s.wl - net_moment) ** 2
        )
        dyn_err = np.sqrt(load_acc_err + load_ang_err)
        return dyn_err

    def integrate(self, f: np.ndarray) -> None:
        assert f.shape == (3, self.num_actuators)
        acc = self.forward_dynamics(f)
        self.state.integrate(*acc, self.dt)


class RPCollision:
    """Class for storing the collision objects of the RP system."""

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


class RPVisualizer:
    """Class for visualizing the RP system."""

    def __init__(self, param: RPParameters, col: RPCollision, vis: Visualizer) -> None:
        self.param = param
        self.col = col

        # Display system in zero configuration.
        # Set payload object.
        vis["rp_payload"].set_object(self.col.payload_obj, _OBJ_MATERIAL)
        # Set payload meshes.
        vis["rp_payload_mesh"].set_object(self.col.payload_mesh, _MESH_MATERIAL)
        # Set actuator objects.
        n = self.param.n
        for i in range(n):
            actuator = "rp_actuator_" + str(i)
            vis[actuator].set_object(gm.Sphere(_ACTUATOR_RADIUS), _OBJ_MATERIAL)
            vis[actuator].set_transform(tf.translation_matrix(self.param.r[:, i]))
        # Set force arrows.
        for i in range(n):
            force_tail = "rp_force_tail_" + str(i)
            force_head = "rp_force_head_" + str(i)
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
                np.array([0, 0, _FORCE_MIN_LENGTH / 2]) + self.param.r[:, i]
            )
            T[:3, :3] = rotation_matrix_a_to_b(np.array([0, 1, 0]), np.array([0, 0, 1]))
            vis[force_tail].set_transform(T)
            T[2, 3] += _FORCE_MIN_LENGTH / 2 + _FORCE_HEAD_LENGTH / 2
            vis[force_head].set_transform(T)

    def update(self, state: RPState, f: np.ndarray, vis: Visualizer) -> None:
        xl = state.xl
        Rl = state.Rl
        r = self.param.r
        n = self.param.n
        assert f.shape == (3, n)
        # Update payload object and mesh.
        T = tf.translation_matrix(xl)
        T[:3, :3] = Rl
        vis["rp_payload"].set_transform(T)
        vis["rp_payload_mesh"].set_transform(T)
        # Update actuator objects.
        for i in range(n):
            actuator = "rp_actuator_" + str(i)
            vis[actuator].set_transform(tf.translation_matrix(xl + Rl @ r[:, i]))
        # Update force arrows.
        for i in range(n):
            force_tail = "rp_force_tail_" + str(i)
            force_head = "rp_force_head_" + str(i)
            force_length = np.linalg.norm(f[:, i])
            if force_length == 0:
                force_dir = np.array([0, 0, 1])
            else:
                force_dir = f[:, i] / force_length
            force_length = np.max([force_length * _FORCE_SCALING, _FORCE_MIN_LENGTH])
            T = tf.translation_matrix(xl + Rl @ r[:, i] + force_length / 2 * force_dir)
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
