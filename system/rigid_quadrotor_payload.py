import numpy as np
import pinocchio as pin
from scipy import constants
from scipy.linalg import polar

_INTEGRATION_STEPS_PER_ROTATION_PROJECTION = 20


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
        """Integrate state along acceleration.
        """
        assert dw.shape == self.w.shape
        assert dvl.shape == (3,)
        assert dwl.shape == (3,)
        for i in range(self.n):
            self.R[:, :, i] = self.R[:, :, i] @ pin.exp3(
                (self.w[:, i] + dw[:, i] * dt / 2) * dt
            )
            self.w[:, i] = self.w[:, i] + dw[:, i] * dt
        self.xl = self.xl + self.vl * dt + dvl * dt ** 2 / 2
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
    mi xi'' = fi - mi g (0,0,1) - Ti,
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
        self, f: np.ndarray, M: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the acceleration given the quadrotor inputs.
        Inputs:
        f:  force (in ground frame), = [f1, ..., fn].
        M:  moment (in i-th quadrotor body frame), = [M1, ..., Mn].
        Outputs:
        (dw, dvl, dwl); with
        dw:     derivative of state.w,
        dvl:    acceleration of state.xl,
        dwl:    derivative of state.wl.
        """
        assert f.shape == (3, self.num_quadrotors)
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
        dv_com = np.sum(f, axis=1) / self.params.mT + self.gravity
        net_moment_com = np.sum(
            np.cross(self.params.r_com, self.state.Rl.T @ f, axisa=0, axisb=0, axisc=0),
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
        f: np.ndarray,
        M: np.ndarray,
        acc: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        """Computes error in dynamics equations.
        Inputs:
        s:      state of the system,
        p:      system parameters,
        f:      force applied by the actuators (in ground frame),
        M:      moment applied by quadrotors (in quadrotor body frame),
        acc:    acceleration, = (dw, dvl, dwl).
        Outputs:
        err:    Norm of dynamics error.
        """
        dw, dvl, dwl = acc
        gravity_vec = -constants.g * np.array([0, 0, 1])
        dv_quad = (
            np.vstack(dvl) + s.Rl @ (pin.skewSquare(s.wl, s.wl) + pin.skew(dwl)) @ p.r
        )
        internal_force = f + np.vstack(gravity_vec) * p.m - p.m * dv_quad
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
        dyn_err = np.sqrt(com_acc_err ** 2 + com_ang_err ** 2 + quad_ang_err_sq)
        return dyn_err

    def integrate(self, f: np.ndarray, M: np.ndarray) -> None:
        assert f.shape == (3, self.num_quadrotors)
        assert M.shape == (3, self.num_quadrotors)
        acc = self.forward_dynamics(f, M)
        self.state.integrate(*acc, self.dt)
