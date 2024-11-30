from dataclasses import dataclass
from time import perf_counter
from typing import List

import cvxpy as cv
import numpy as np
import pinocchio as pin
from scipy import constants
from scipy.linalg import cho_factor, cho_solve

from control.rqp_centralized import SolverStatistics
from system.rigid_quadrotor_payload import (RQPCollision, RQPParameters,
                                            RQPState)


@dataclass
class Index:
    i: int
    is_leader: bool


class RQPPrimalSolver:
    """Solver for the primal step of the dual decomposition RQP controller.

    min_{dv_com, dvl, dwl, fi, Fi, Mi}
        (k_f/n) ||(fi + Fi) - mT g e3||^2 + (k_m/n) ||Mi + r_comi x Rl.T fi||^2
        + k_feq ||fi - fi_eq||^2
        + k_dvl 1{i == leader} ||dvl - dvl_des||^2
        + k_dwl 1{i == leader} ||dwl - dwl_des||^2
        + k_smooth (||fi||^2 - <fi, qi'>^2)
        + c.T (fi, Fi, Mi),
    s.t.    mT dv_com = (fi + Fi) - mT g e3,
            JT dwl + wl x JT wl = r_comi x Rl.T fi + Mi,
            dvl = dv_com - Rl hat(wl)^2 x_com + Rl hat(x_com) dwl,
            fi_z >= min_fz,
            ||fi||_2 <= sec(max_f_ang) fi_z,
            ||fi||_2 <= max_f,
            e3.T Rl e3 >= cos(max_p_ang) -> second-order CBF,
            ||wl||^2 <= max_wl ** 2 -> CBF,
            ||vl||^2 <= max_vl ** 2 -> CBF.

    Consensus equations:
    Fi + fi = sum_j fj,
    Mi + r_comi x Rl.T fi = sum_j r_comj x Rl.T fj.
    """

    def __init__(
        self,
        params: RQPParameters,
        col: RQPCollision,
        idx: Index,
        state: RQPState,
        dt: float,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # Set system constants.
        self._set_system_constants(params, col, idx, dt)
        # Set controller constants.
        self._set_controller_constants()
        # Set CVX variables: (dv_com, dvl, dwl, fi, Fi, Mi).
        self._set_cvx_variables()
        # Set CVX parameters.
        self._set_cvx_parameters()

        self.cons = []
        # Dynamics constraint.
        #   mT dv_com = (fi + Fi) - mT g e3,
        #   JT dwl + wl x JT wl = r_comi x Rl.T fi + Mi.
        self._set_dynamics_constraint()
        # Kinematics constraint.
        #   dvl = dv_com - Rl hat(wl)^2 x_com + Rl hat(x_com) dwl.
        self._set_kinematics_constraint()
        # z-force lower bound constraint.
        #   fi_z >= min_fz.
        self._set_zforce_lower_bound_constraint()
        # Force cone angle bound constraint.
        #   ||fi||_2 <= sec(max_f_ang) fi_z.
        self._set_force_cone_angle_bound_constraint()
        # Force norm bound constraint.
        #   ||f_i||_2 <= max_f.
        self._set_force_norm_bound_constraint()
        # Payload angle CBF constraint.
        #   h = e3.T Rl e3 - cos(max_p_ang) >= 0 -> second-order CBF,
        #   dh = e3.T Rl hat(wl) e3,
        #   d2h = e3.T Rl hat(wl)^2 e3 - e3.T Rl hat(e3) dwl,
        #   d2h + (alpha1 + alpha2) dh + alpha2 alpha2 h >= 0.
        self._set_payload_angle_cbf_constraint()
        # Angular velocity norm CBF constraint.
        #   h = max_wl ** 2 - ||wl||^2 >= 0 -> CBF,
        #   dh = -2 wl.T dwl,
        #   dh + alpha h >= 0.
        self._set_angular_velocity_norm_cbf_constraint()
        # Velocity norm CBF constraint.
        #   h = max_vl ** 2 - ||vl||^2 >= 0 -> CBF,
        #   dh = -2 vl.T dvl,
        #   dh + alpha h >= 0.
        self._set_velocity_norm_cbf_constraint()
        # TODO: Environment collision constraints.

        self.cost = 0
        # Total force cost.
        #   (k_f/n) ||(fi + Fi) - mT g e3||^2.
        self._set_total_force_cost()
        # Total moment cost.
        #   (k_m/n) ||Mi + r_comi x Rl.T fi||^2.
        self._set_total_moment_cost()
        # Equilibrium force cost.
        #   k_feq ||fi - fi_eq||^2.
        self._set_equilibrium_force_cost()
        # Desired acceleration cost.
        #   k_dvl 1{i == leader} ||dvl - dvl_des||^2
        #   ~= k_dvl 1{i == leader} (||dvl||^2 - 2 dvl_des.T dvl).
        self._set_desired_acceleration_cost()
        # Desired angular acceleration cost.
        #   k_dwl 1{i == leader} ||dwl - dwl_des||^2
        #   ~= k_dwl 1{i == leader} (||dwl||^2 - 2 dwl_des.T dwl).
        self._set_desired_angular_acceleration_cost()
        # Force smoothing cost.
        #   qi'_j = Ri exp(hat(wi) * dt) ej,
        #   k_smooth fi.T (I_3 - qi'_3 qi'_3.T) fi,
        #   = k_smooth (qi'_1.T fi)^2 + (qi'_2.T fi)^2.
        self._set_force_smoothing_cost()
        # Dual decomposition cost.
        #   c.T (fi, Fi, Mi).
        self._set_dual_decomposition_cost()

        dvl_des = np.zeros((3,))
        dwl_des = np.zeros((3,))
        acc_des = (dvl_des, dwl_des)
        self._update_cvx_parameters(state, acc_des)

        self._set_warm_start()
        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if self.verbose:
            print("Optimization problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)

    def _set_system_constants(
        self,
        params: RQPParameters,
        col: RQPCollision,
        idx: Index,
        dt: float,
    ) -> None:
        assert params.n >= 3
        self.n = params.n
        self.params = params
        self.col = col
        self.idx = idx

        self.dt = dt
        self.g = constants.g
        self.mT = params.mT
        self.JT = params.JT
        self.x_com = params.x_com
        self.r_com = params.r_com
        self.payload_vertices = col.payload_mesh_vertices

        # Set steady state forces.
        # Steady state forces point upwards,
        # and result in zero net acceleration and angular acceleration.
        self.fi_eq = np.zeros((3,))
        self.f_eq = np.zeros((3, self.n))
        wrench_mat = np.empty((3, self.n))
        wrench_mat[0, :] = np.ones((self.n,))
        rhs_vec = np.array([self.mT * self.g, 0.0, 0.0])
        for j in range(self.n):
            wrench_mat[1:, j] = pin.skew(self.r_com[:, j])[:2, 2]
        self.f_eq[2, :] = np.linalg.lstsq(wrench_mat, rhs_vec, rcond=None)[0]
        self.fi_eq[2] = self.f_eq[2, self.idx.i]
        # Check steady-state acceleration and angular acceleration.
        if self.verbose:
            net_acc = np.sum(self.f_eq, axis=1) - self.mT * self.g * np.array(
                [0.0, 0.0, 1.0]
            )
            net_ang = np.sum(
                np.cross(self.r_com, self.f_eq, axisa=0, axisb=0, axisc=0), axis=1
            )
            print(f"Equilibrium acceleration norm: {np.linalg.norm(net_acc)}")
            print(f"Equilibrium angular acceleration norm: {np.linalg.norm(net_ang)}")

        # Dependent constants.
        self.JT_inv = params.JT_inv
        self.J_inv_ri_hat = self.JT_inv @ pin.skew(self.r_com[:, self.idx.i])

    def _set_controller_constants(self) -> None:
        # Constraints:
        # z-force lower bound.
        self.min_fz = self.mT * self.g / (self.n * 10.0)
        # Force cone angle bound.
        self.max_f_ang = np.pi / 6.0
        self.sec_max_f_ang = 1.0 / np.cos(self.max_f_ang)
        # Force norm bound.
        self.max_f = (2.0 / self.n) * self.mT * self.g
        # Payload angle CBF constants.
        max_p_ang = np.pi / 12.0
        self.cos_max_p_ang = np.cos(max_p_ang)
        self.alpha1_max_p_ang_cbf = 1.0
        self.alpha2_max_p_ang_cbf = 1.0
        # Angular velocity norm CBF constants.
        max_wl = np.pi / 6.0
        self.max_wl_sq = max_wl**2
        self.alpha_max_wl_cbf = 1.0
        # Velocity norm CBF constants.
        max_vl = 1.0
        self.max_vl_sq = max_vl**2
        self.alpha_max_vl_cbf = 1.0

        # Costs:
        # Total force constant.
        self.k_f = 0.1 / self.n
        # Total moment constant.
        self.k_m = 0.1 / self.n
        # Equilibrium force constant.
        self.k_feq = 0.1
        # Desired acceleration constant.
        self.k_dvl = 1.0 * self.idx.is_leader
        # Desired angular acceleration constant.
        self.k_dwl = 1.0 * self.idx.is_leader
        # Force smooth constant.
        # NOTE: Controller is more stable without smoothing.
        self.k_smooth = 0 / self.dt**2

    def _set_cvx_variables(self) -> None:
        self.dv_com = cv.Variable(3)
        self.dvl = cv.Variable(3)
        self.dwl = cv.Variable(3)
        self.fi = cv.Variable(3)
        self.Fi = cv.Variable(3)
        self.Mi = cv.Variable(3)

    def _set_cvx_parameters(self) -> None:
        # State parameters.
        self.xl = cv.Parameter(3)
        self.vl = cv.Parameter(3)
        self.Rl = cv.Parameter((3, 3))
        self.wl = cv.Parameter(3)
        # Desired acceleration parameters.
        self.dvl_des = cv.Parameter(3)
        self.dwl_des = cv.Parameter(3)
        # Dependent parameters.
        self.v_norm2 = cv.Parameter(nonneg=True)
        self.w_norm2 = cv.Parameter(nonneg=True)
        self.R_w_hat = cv.Parameter((3, 3))
        self.R_w_hat_sq = cv.Parameter((3, 3))
        self.J_inv_w_cross_Jw = cv.Parameter(3)
        self.Rqi_orth = cv.Parameter((3, 2))
        # Dual decomposition parameters.
        self.c_fi = cv.Parameter(3)
        self.c_Fi = cv.Parameter(3)
        self.c_Mi = cv.Parameter(3)

    def _update_cvx_parameters(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
        c_fi: np.ndarray = np.zeros((3,)),
        c_Fi: np.ndarray = np.zeros((3,)),
        c_Mi: np.ndarray = np.zeros((3,)),
    ) -> None:
        # State parameters.
        self.xl.value = state.xl
        self.vl.value = state.vl
        self.Rl.value = state.Rl
        self.wl.value = state.wl
        # Desired acceleration parameters.
        self.dvl_des.value = acc_des[0]
        self.dwl_des.value = acc_des[1]
        # Dependent parameters.
        self.v_norm2.value = np.dot(state.vl, state.vl)
        self.w_norm2.value = np.dot(state.wl, state.wl)
        self.R_w_hat.value = state.Rl @ pin.skew(state.wl)
        self.R_w_hat_sq.value = state.Rl @ pin.skewSquare(state.wl, state.wl)
        self.J_inv_w_cross_Jw.value = self.JT_inv @ np.cross(
            state.wl, self.JT @ state.wl
        )
        self.Rqi_orth.value = (
            state.R[:, :, self.idx.i]
            @ pin.exp3(state.w[:, self.idx.i] * self.dt)[:, :2]
        )
        # Dual decomposition parameters.
        self.c_fi.value = c_fi
        self.c_Fi.value = c_Fi
        self.c_Mi.value = c_Mi

    # Constraints.
    def _set_dynamics_constraint(self) -> None:
        gravity_vector = -self.g * np.array([0.0, 0.0, 1.0])
        moment = self.J_inv_ri_hat @ self.Rl.T @ self.fi + self.JT_inv @ self.Mi
        self.cons += [
            self.mT * self.dv_com == (self.fi + self.Fi) + self.mT * gravity_vector,
            self.dwl + self.J_inv_w_cross_Jw == moment,
        ]

    def _set_kinematics_constraint(self) -> None:
        self.cons += [
            self.dvl
            == self.dv_com
            - self.R_w_hat_sq @ self.x_com
            + self.Rl @ pin.skew(self.x_com) @ self.dwl
        ]

    def _set_zforce_lower_bound_constraint(self) -> None:
        self.cons += [self.fi[2] >= self.min_fz]

    def _set_force_cone_angle_bound_constraint(self) -> None:
        self.cons += [cv.norm2(self.fi) <= self.sec_max_f_ang * self.fi[2]]

    def _set_force_norm_bound_constraint(self) -> None:
        self.cons += [cv.norm2(self.fi) <= self.max_f]

    def _set_payload_angle_cbf_constraint(self) -> None:
        e3 = np.array([0.0, 0.0, 1.0])
        self.cons += [
            self.R_w_hat_sq[2, 2]
            - e3 @ self.Rl @ pin.skew(e3) @ self.dwl
            + (self.alpha1_max_p_ang_cbf + self.alpha2_max_p_ang_cbf)
            * self.R_w_hat[2, 2]
            + (self.alpha1_max_p_ang_cbf * self.alpha2_max_p_ang_cbf)
            * (self.Rl[2, 2] - self.cos_max_p_ang)
            >= 0
        ]

    def _set_angular_velocity_norm_cbf_constraint(self) -> None:
        self.cons += [
            -2 * self.wl @ self.dwl
            + self.alpha_max_wl_cbf * (self.max_wl_sq - self.w_norm2)
            >= 0
        ]

    def _set_velocity_norm_cbf_constraint(self) -> None:
        self.cons += [
            -2 * self.vl @ self.dvl
            + self.alpha_max_vl_cbf * (self.max_vl_sq - self.v_norm2)
            >= 0
        ]

    # Costs.
    def _set_total_force_cost(self) -> None:
        f_err = (self.fi + self.Fi) - self.mT * self.g * np.array([0.0, 0.0, 1.0])
        self.cost += self.k_f * cv.sum_squares(f_err)

    def _set_total_moment_cost(self) -> None:
        moment = pin.skew(self.r_com[:, self.idx.i]) @ self.Rl.T @ self.fi + self.Mi
        self.cost += self.k_m * cv.sum_squares(moment)

    def _set_equilibrium_force_cost(self) -> None:
        feq_err = self.fi - self.fi_eq
        self.cost += self.k_feq * cv.sum_squares(feq_err)

    def _set_desired_acceleration_cost(self) -> None:
        self.cost += self.k_dvl * (
            cv.sum_squares(self.dvl) - 2 * self.dvl_des @ self.dvl
        )

    def _set_desired_angular_acceleration_cost(self) -> None:
        self.cost += self.k_dwl * (
            cv.sum_squares(self.dwl) - 2 * self.dwl_des @ self.dwl
        )

    def _set_force_smoothing_cost(self) -> None:
        self.cost += self.k_smooth * cv.sum_squares(self.Rqi_orth.T @ self.fi)

    def _set_dual_decomposition_cost(self) -> None:
        self.cost += self.c_fi @ self.fi + self.c_Fi @ self.Fi + self.c_Mi @ self.Mi

    # Warm start.
    def _set_warm_start(self) -> None:
        self.prev_fi = self.fi_eq
        self.prev_Fi = np.sum(self.f_eq, axis=1) - self.prev_fi
        self.prev_Mi = -self.J_inv_ri_hat @ self.prev_fi

        self.dv_com.value = np.zeros((3,))
        self.dvl.value = np.zeros((3,))
        self.dwl.value = np.zeros((3,))
        self.fi.value = self.prev_fi
        self.Fi.value = self.prev_Fi
        self.Mi.value = self.prev_Mi

    def solve(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
        c_fi: np.ndarray = np.zeros((3,)),
        c_Fi: np.ndarray = np.zeros((3,)),
        c_Mi: np.ndarray = np.zeros((3,)),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        self._update_cvx_parameters(state, acc_des, c_fi, c_Fi, c_Mi)
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)
        if self.prob.status == cv.OPTIMAL:
            self.prev_fi = self.fi.value
            self.prev_Fi = self.Fi.value
            self.prev_Mi = self.Mi.value
        elif self.verbose:
            print(f"Problem not solved to optimality, status: {self.prob.status}")
        solve_time = self.prob.solver_stats.solve_time
        return self.prev_fi, self.prev_Fi, self.prev_Mi, solve_time

    def set_leader(self) -> None:
        self.idx.is_leader = True

    def unset_leader(self) -> None:
        self.idx.is_leader = False

    def strong_convexity_matrix(self, state: RQPState) -> np.ndarray:
        eps = 1e-6
        mat = eps * np.eye(9)

        temp = np.zeros((3, 9))
        # k_smooth term.
        q_12 = (
            state.R[:, :, self.idx.i]
            @ pin.exp3(state.w[:, self.idx.i] * self.dt)[:, :2]
        )
        temp[:2, :3] = q_12.T
        mat += 2 * self.k_smooth * (temp.T @ temp)
        # k_feq term.
        temp[:, :3] = np.eye(3)
        mat += 2 * self.k_feq * (temp.T @ temp)
        # k_f term.
        temp[:, 3:6] = np.eye(3)
        mat += 2 * self.k_f * (temp.T @ temp)
        # k_m term.
        coeff_m_f = pin.skew(self.r_com[:, self.idx.i]) @ state.Rl.T
        temp[:, :3] = coeff_m_f
        temp[:, 3:6] = np.zeros((3, 3))
        temp[:, 6:] = np.eye(3)
        mat += 2 * self.k_m * (temp.T @ temp)
        # k_dwl term.
        coeff_dwl_f = self.JT_inv @ pin.skew(self.r_com[:, self.idx.i]) @ state.Rl.T
        coeff_dwl_M = self.JT_inv
        temp[:, :3] = coeff_dwl_f
        temp[:, 3:6] = np.zeros((3, 3))
        temp[:, 6:] = coeff_dwl_M
        mat += 2 * self.k_dwl * (temp.T @ temp)
        # k_dvl term.
        coeff_dvl_f = (
            np.eye(3) / self.mT + state.Rl @ pin.skew(self.x_com) @ coeff_dwl_f
        )
        coeff_dvl_F = np.eye(3) / self.mT
        coeff_dvl_M = state.Rl @ pin.skew(self.x_com) @ coeff_dwl_M
        temp[:, :3] = coeff_dvl_f
        temp[:, 3:6] = coeff_dvl_F
        temp[:, 6:] = coeff_dvl_M
        mat += 2 * self.k_dvl * (temp.T @ temp)

        return mat


class RQPDDController:
    """Dual decomposition-based distributed solver for the RQP controller."""

    def __init__(
        self,
        params: RQPParameters,
        col: RQPCollision,
        state: RQPState,
        dt: float,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # Set system constants.
        self._set_system_constants(params, col, dt)
        # Set controller constants.
        self._set_controller_constants()
        # Set variables: (f, F, M, lambda_F, lambda_M).
        self._set_variables()

        # Initialize primal solvers.
        self.primal_solvers: List[RQPPrimalSolver] = []
        for i in range(self.n):
            idx = Index(i, (i == self.leader_idx))
            solver = RQPPrimalSolver(params, col, idx, state, dt, verbose=False)
            self.primal_solvers.append(solver)

        # Set warm start solutions.
        self._set_warm_start()

    def _set_system_constants(
        self,
        params: RQPParameters,
        col: RQPCollision,
        dt: float,
    ) -> None:
        assert params.n >= 3
        self.n = params.n
        self.params = params
        self.col = col
        self.dt = dt

        self.r_com = params.r_com

    def _set_controller_constants(self) -> None:
        # Leader index.
        self.leader_idx = 0

        # Convergence tolerance (inf-norm).
        self.prim_inf_tol = 1e-2  # [N].
        self.max_iter = 100
        # Dual ascent regularization term.
        self.beta = 0
        # Constraint matrix.
        self.A = np.empty((6 * self.n, 9 * self.n))
        # Quasi-Newton matrix Cholesky decomposition object.
        self.qn_mat_chol = cho_factor(np.eye(6 * self.n))

    def _set_variables(self) -> None:
        # Primal variables.
        self.f = np.zeros((3, self.n))
        self.F = np.zeros((3, self.n))
        self.M = np.zeros((3, self.n))
        # Dual variables.
        self.lambda_F = np.zeros((3, self.n))
        self.lambda_M = np.zeros((3, self.n))

    # Warm start.
    def _set_warm_start(self) -> None:
        self.f = self.primal_solvers[0].f_eq
        for i in range(self.n):
            self.F[:, i] = self.primal_solvers[0].prev_Fi
            self.M[:, i] = self.primal_solvers[0].prev_Mi

    def _qn_mat_cholesky(self, state: RQPState) -> None:
        # Compute strong convexity inverse matrix.
        Q_inv = np.zeros((9 * self.n, 9 * self.n))
        for i in range(self.n):
            Qi_inv = np.linalg.inv(
                self.primal_solvers[i].strong_convexity_matrix(state)
            )
            Q_inv[9 * i : 9 * (i + 1), 9 * i : 9 * (i + 1)] = 0.5 * (Qi_inv + Qi_inv.T)
        # Compute constraint matrix.
        A = np.zeros((6 * self.n, 9 * self.n))
        for i in range(self.n):
            A[6 * i : 6 * i + 3, 9 * i + 3 : 9 * i + 6] = np.eye(3)
            A[6 * i + 3 : 6 * (i + 1), 9 * i + 6 : 9 * (i + 1)] = np.eye(3)
            for j in range(self.n):
                if j == i:
                    continue
                A[6 * i : 6 * i + 3, 9 * j : 9 * j + 3] = -np.eye(3)
                A[6 * i + 3 : 6 * (i + 1), 9 * j : 9 * j + 3] = (
                    -pin.skew(self.r_com[:, j]) @ state.Rl.T
                )
        # Quasi-Newton matrix.
        qn_mat = A @ Q_inv @ A.T + self.beta * np.eye(6 * self.n)
        self.A = A
        self.qn_mat_chol = cho_factor(qn_mat)

    def _get_primal_inf_err(self, state: RQPState) -> float:
        primal_inf_err_F = np.empty((3, self.n))
        primal_inf_err_M = np.empty((3, self.n))
        moments_ = np.cross(self.r_com, state.Rl.T @ self.f, axisa=0, axisb=0, axisc=0)
        for i in range(self.n):
            primal_inf_err_F[:, i] = self.F[:, i] - (
                np.sum(self.f, axis=1) - self.f[:, i]
            )
            primal_inf_err_M[:, i] = self.M[:, i] - (
                np.sum(moments_, axis=1) - moments_[:, i]
            )
        primal_inf_err = np.max(
            (
                np.linalg.norm(primal_inf_err_F, np.inf),
                np.linalg.norm(primal_inf_err_M, np.inf),
            )
        )
        return primal_inf_err

    def _dual_ascent_step(self, state: RQPState) -> None:
        # Primal optimal value.
        prim_opt = np.empty((9, self.n))
        prim_opt[:3, :] = self.f
        prim_opt[3:6, :] = self.F
        prim_opt[6:, :] = self.M
        # Dual gradient.
        dual_grad = self.A @ prim_opt.reshape((9 * self.n,), order="F")

        # Dual step.
        dual_step = cho_solve(self.qn_mat_chol, dual_grad).reshape(
            (6, self.n), order="F"
        )
        # Dual update.
        self.lambda_F += dual_step[:3, :]
        self.lambda_M += dual_step[3:, :]

    def control(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, SolverStatistics]:
        """Consensus equations:
        Fi + fi = sum_j fj,
        Mi + r_comi x Rl.T fi = sum_j r_comj x Rl.T fj.
        """
        total_solve_time = 0.0
        start_time = perf_counter()
        self._qn_mat_cholesky(state)
        stop_time = perf_counter()
        total_solve_time += stop_time - start_time

        iter = 0
        err_seq = []
        while True:
            primal_solve_time = 0.0
            for i in range(self.n):
                # Update cost parameters.
                c_Fi = self.lambda_F[:, i]
                c_Mi = self.lambda_M[:, i]
                c_fi = -(np.sum(self.lambda_F, axis=1) - c_Fi) + state.Rl @ pin.skew(
                    self.r_com[:, i]
                ) @ (np.sum(self.lambda_M, axis=1) - c_Mi)
                # Solve primal problems.
                self.f[:, i], self.F[:, i], self.M[:, i], time_ = self.primal_solvers[
                    i
                ].solve(state, acc_des, c_fi, c_Fi, c_Mi)
                primal_solve_time = np.max((primal_solve_time, time_))
            total_solve_time += primal_solve_time
            iter += 1

            start_time = perf_counter()
            # Get primal infeasibility error.
            primal_inf_err = self._get_primal_inf_err(state)
            # Check if dual gradient satisfies tolerance.
            if (primal_inf_err < self.prim_inf_tol) or (iter > self.max_iter):
                break
            err_seq.append(primal_inf_err)
            # Update dual variables.
            self._dual_ascent_step(state)
            stop_time = perf_counter()
            total_solve_time += stop_time - start_time
        stats = SolverStatistics(iter, total_solve_time, err_seq)
        return self.f, stats

    def get_force_cone_angle_bound(self) -> float:
        return self.primal_solvers[0].max_f_ang

    def set_force_err_tolerance(self, tol: float):
        self.prim_inf_tol = tol

    def set_max_iter(self, max_iter: int) -> None:
        self.max_iter = max_iter
