import numpy as np
import pinocchio as pin
from scipy import constants
from scipy.linalg import cholesky, polar

_INTEGRATION_STEPS_PER_ROTATION_PROJECTION = 20


class RPParameters:
    def __init__(
        self, ml: float, Jl: np.ndarray, r: np.ndarray
    ) -> None:
        """Inputs:
        ml: payload mass, [kg],
        Jl: payload inertia in body frame, [kg-m^2],
        r:  rigid link attachment position in body frame, [m], [r1, ..., rn],
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
        self,
        xl: np.ndarray,
        vl: np.ndarray,
        Rl: np.ndarray,
        wl: np.ndarray,
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

    def integrate(
        self, dvl: np.ndarray, dwl: np.ndarray, dt: float
    ) -> None:
        """Integrate state along acceleration.
        """
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
    Jl wl' + hat(wl) Jl wl = sum_i hat(ri) Rl.T fi,
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
        dwl:    derivative of state.wl,
        """
        assert f.shape == (3, self.num_actuators)
        net_force = np.sum(f, axis=1)
        net_moment = np.cross(
            self.params.r, self.state.Rl.T @ f, axisa=0, axisb=0, axisc=0
        ) @ np.ones((self.num_actuators,))

        dvl = net_force / self.params.ml + self.gravity
        dwl = self.params.Jl_inv @ (net_moment - np.cross(
            self.state.wl, self.params.Jl @ self.state.wl
        ))
        
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
        acc:    acceleration, = (dvl, dwl),
        Outputs:
        err:    Norm of dynamics error.
        """
        dvl, dwl = acc
        gravity_vec = -constants.g * np.array([0, 0, 1])
        net_force = np.sum(f, axis=1) + p.ml * gravity_vec.T
        net_moment = np.cross(p.r, s.Rl.T @ f, axisa=0, axisb=0, axisc=0) @ np.ones((p.n,))
        load_acc_err = np.linalg.norm(p.ml * dvl - net_force) ** 2
        load_ang_err = np.linalg.norm(p.Jl @ dwl + pin.skew(s.wl) @ p.Jl @ s.wl - net_moment) ** 2
        dyn_err = np.sqrt(load_acc_err + load_ang_err)
        return dyn_err

    def integrate(self, f: np.ndarray) -> None:
        assert f.shape == (3, self.num_actuators)
        acc = self.forward_dynamics(f)
        self.state.integrate(*acc, self.dt)
