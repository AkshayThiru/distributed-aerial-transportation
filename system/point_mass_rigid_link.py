import numpy as np
import pinocchio as pin
from scipy import constants
from scipy.linalg import cholesky, polar, solve

_INTEGRATION_STEPS_PER_ROTATION_PROJECTION = 20


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
        self.q = self.q + self.dq * dt + ddq * dt ** 2 / 2
        self.dq = self.dq + ddq * dt
        self.project_q()
        self.xl = self.xl + self.vl * dt + dvl * dt ** 2 / 2
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
