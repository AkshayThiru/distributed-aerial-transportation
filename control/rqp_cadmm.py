from dataclasses import dataclass
from time import perf_counter
from typing import List

import cvxpy as cv
import numpy as np
import pinocchio as pin
from scipy import constants

from control.rqp_centralized import SolverStatistics
from system.rigid_quadrotor_payload import (RQPCollision, RQPParameters,
                                            RQPState)


@dataclass
class Index:
    i: int
    is_leader: bool


class RQPPrimalSolver:
    """Solver for the primal step of the consensus-ADMM RQP controller.

    min_{dv_com, dvl, dwl, f}
        (k_f/n) ||sum_j f - mT g e3||^2 + (k_m/n) ||sum_j r_comi x Rl.T fi||^2
        + k_feq ||fi - fi_eq||^2
        + k_dvl 1{i == leader} ||dvl - dvl_des||^2
        + k_dwl 1{i == leader} ||dwl - dwl_des||^2
        + k_smooth (||fi||^2 - <fi, qi'>^2)
        + <lambda_f, f> + (rho/2) ||f - f_mean||^2,
    s.t.    mT dv_com = sum_j fi - mT g e3,
            JT dwl + wl x JT wl = sum_j r_comi x Rl.T fi,
            dvl = dv_com - Rl hat(wl)^2 x_com + Rl hat(x_com) dwl,
            fi_z >= min_fz,
            ||fi||_2 <= sec(max_f_ang) fi_z,
            ||fi||_2 <= max_f,
            e3.T Rl e3 >= cos(max_p_ang) -> second-order CBF,
            ||wl||^2 <= max_wl ** 2 -> CBF,
            ||vl||^2 <= max_vl ** 2 -> CBF.
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
        # Set CVX variables: (dv_com, dvl, dwl, f).
        self._set_cvx_variables()
        # Set CVX parameters.
        self._set_cvx_parameters()

        self.cons = []
        # Dynamics constraint.
        #   mT dv_com = sum_j fi - mT g e3,
        #   JT dwl + wl x JT wl = sum_j r_comi x Rl.T fi.
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
        #   (k_f/n) ||sum_j f - mT g e3||^2.
        self._set_total_force_cost()
        # Total moment cost.
        #   (k_m/n) ||sum_j r_comi x Rl.T fi||^2.
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
        # Consensus-ADMM cost.
        #   <lambda_f, f> + (rho/2) (||f||^2 - 2 <f_mean, f>).
        self._set_cadmm_cost()

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
        self.f_eq = np.zeros((3, self.n))
        wrench_mat = np.empty((3, self.n))
        wrench_mat[0, :] = np.ones((self.n,))
        rhs_vec = np.array([self.mT * self.g, 0.0, 0.0])
        for i in range(self.n):
            wrench_mat[1:, i] = pin.skew(self.r_com[:, i])[:2, 2]
        self.f_eq[2, :] = np.linalg.lstsq(wrench_mat, rhs_vec, rcond=None)[0]
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
        self.J_inv_r_hat = np.empty((3, 3, self.n))
        for i in range(self.n):
            self.J_inv_r_hat[:, :, i] = self.JT_inv @ pin.skew(self.r_com[:, i])

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
        self.f = cv.Variable((3, self.n))

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
        # Consensus-ADMM parameters.
        self.lambda_f = cv.Parameter((3, self.n))
        self.cadmm_rho = cv.Parameter(nonneg=True)
        self.f_mean_rho = cv.Parameter((3, self.n))

    def _update_cvx_parameters(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
        lambda_f: np.ndarray = None,
        cadmm_rho: float = 0,
        f_mean: np.ndarray = None,
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
        # Consensus-ADMM parameters.
        if lambda_f is None:
            lambda_f = np.zeros((3, self.n))
        if f_mean is None:
            f_mean = np.zeros((3, self.n))
        self.lambda_f.value = lambda_f
        self.cadmm_rho.value = cadmm_rho
        self.f_mean_rho.value = cadmm_rho * f_mean

    # Constraints.
    def _set_dynamics_constraint(self) -> None:
        gravity_vector = -self.g * np.array([0.0, 0.0, 1.0])
        moment = 0
        for i in range(self.n):
            moment += self.J_inv_r_hat[:, :, i] @ self.Rl.T @ self.f[:, i]
        self.cons += [
            self.mT * self.dv_com == cv.sum(self.f, axis=1) + self.mT * gravity_vector,
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
        self.cons += [self.f[2, self.idx.i] >= self.min_fz]

    def _set_force_cone_angle_bound_constraint(self) -> None:
        self.cons += [
            cv.norm2(self.f[:, self.idx.i])
            <= self.sec_max_f_ang * self.f[2, self.idx.i]
        ]

    def _set_force_norm_bound_constraint(self) -> None:
        self.cons += [cv.norm2(self.f[:, self.idx.i]) <= self.max_f]

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
        f_err = cv.sum(self.f, axis=1) - self.mT * self.g * np.array([0.0, 0.0, 1.0])
        self.cost += self.k_f * cv.sum_squares(f_err)

    def _set_total_moment_cost(self) -> None:
        moment = 0
        for i in range(self.n):
            moment += pin.skew(self.r_com[:, i]) @ self.Rl.T @ self.f[:, i]
        self.cost += self.k_m * cv.sum_squares(moment)

    def _set_equilibrium_force_cost(self) -> None:
        feq_err = (self.f - self.f_eq)[:, self.idx.i]
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
        self.cost += self.k_smooth * cv.sum_squares(
            self.Rqi_orth.T @ self.f[:, self.idx.i]
        )

    def _set_cadmm_cost(self) -> None:
        # <lambda_f, f> + (rho/2) (||f||^2 - 2 <f_mean, f>)
        self.cost += (
            cv.sum(cv.multiply(self.lambda_f, self.f))
            + (self.cadmm_rho / 2) * cv.sum_squares(self.f)
            - cv.sum(cv.multiply(self.f_mean_rho, self.f))
        )

    # Warm start.
    def _set_warm_start(self) -> None:
        self.prev_f = self.f_eq

        self.dv_com.value = np.zeros((3,))
        self.dvl.value = np.zeros((3,))
        self.dwl.value = np.zeros((3,))
        self.f.value = self.f_eq

    def solve(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
        lambda_f: np.ndarray = None,
        cadmm_rho: float = 0,
        f_mean: np.ndarray = None,
    ) -> tuple[np.ndarray, float]:
        self._update_cvx_parameters(state, acc_des, lambda_f, cadmm_rho, f_mean)
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)
        if self.prob.status == cv.OPTIMAL:
            self.prev_f = self.f.value
        elif self.verbose:
            print(f"Problem not solved to optimality, status: {self.prob.status}")
        solve_time = self.prob.solver_stats.solve_time
        return self.prev_f, solve_time

    def set_leader(self) -> None:
        self.idx.is_leader = True

    def unset_leader(self) -> None:
        self.idx.is_leader = False


class RQPCADMMController:
    """Consensus ADMM-based distributed solver for the RQP controller."""

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

        # Convergence tolerance (inf-norm) for residuals.
        self.res_tol = 1e-2  # [N].
        self.use_total_res = True
        self.max_iter = 100
        # Rho constants.
        self.rho0 = 1.0
        self.tau_incr = 1.0
        self.rho_max = 2.0

    def _set_variables(self) -> None:
        # Primal variables.
        self.f = np.zeros((3, self.n, self.n))
        self.f_mean = np.zeros((3, self.n))
        # Dual variables.
        self.lambda_f = np.zeros((3, self.n, self.n))

    # Warm start.
    def _set_warm_start(self) -> None:
        for i in range(self.n):
            self.f[:, :, i] = self.primal_solvers[0].f_eq
        self.f_mean = self.primal_solvers[0].f_eq

    def _get_mean_and_residual(
        self, state: RQPState
    ) -> tuple[np.ndarray, float, float]:
        # Compute primal mean and aggregate variables.
        f_app, F, M = (
            np.empty((3, self.n)),
            np.empty((3, self.n)),
            np.empty((3, self.n)),
        )
        f_mean = np.zeros((3, self.n))
        for i in range(self.n):
            f_app[:, i] = self.f[:, i, i]
            F[:, i] = np.sum(self.f[:, :, i], axis=1) - self.f[:, i, i]
            moments_ = np.cross(
                self.r_com, state.Rl.T @ self.f[:, :, i], axisa=0, axisb=0, axisc=0
            )
            M[:, i] = np.sum(moments_, axis=1) - moments_[:, i]
            f_mean += self.f[:, :, i]
        f_mean = f_mean / self.n
        # Compute (total and aggregate) residuals.
        moments_app_ = np.cross(
            self.r_com, state.Rl.T @ f_app, axisa=0, axisb=0, axisc=0
        )
        total_res = 0.0
        agg_err_F = np.empty((3, self.n))
        agg_err_M = np.empty((3, self.n))
        for i in range(self.n):
            total_res = np.max(
                (total_res, np.linalg.norm(self.f[:, :, i] - f_mean, np.inf))
            )
            agg_err_F[:, i] = F[:, i] - (np.sum(f_app, axis=1) - f_app[:, i])
            agg_err_M[:, i] = M[:, i] - (
                np.sum(moments_app_, axis=1) - moments_app_[:, i]
            )
        agg_res = np.max(
            (
                np.linalg.norm(agg_err_F, np.inf),
                np.linalg.norm(agg_err_M, np.inf),
            )
        )
        res = agg_res
        if self.use_total_res:
            res = total_res
        return f_mean, res

    def _dual_update(self, rho: float) -> None:
        for i in range(self.n):
            self.lambda_f[:, :, i] += rho * (self.f[:, :, i] - self.f_mean)

    def control(
        self,
        state: RQPState,
        acc_des: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, SolverStatistics]:
        iter = 0
        rho = self.rho0
        total_solve_time = 0.0
        err_seq = []
        while True:
            primal_solve_time = 0.0
            for i in range(self.n):
                # Solve primal problems.
                self.f[:, :, i], time_ = self.primal_solvers[i].solve(
                    state, acc_des, self.lambda_f[:, :, i], rho, self.f_mean
                )
                primal_solve_time = np.max((primal_solve_time, time_))
            total_solve_time += primal_solve_time
            iter += 1

            start_time = perf_counter()
            # Rho update.
            rho = np.min((rho * self.tau_incr, self.rho_max))
            # Get mean and residual.
            self.f_mean, res = self._get_mean_and_residual(state)
            # Check if residual satisfies tolerance.
            if (res < self.res_tol) or (iter > self.max_iter):
                break
            err_seq.append(res)
            # Update dual variables.
            self._dual_update(rho)
            stop_time = perf_counter()
            total_solve_time += stop_time - start_time
        # Return applied forces.
        f_app = np.zeros((3, self.n))
        for i in range(self.n):
            f_app[:, i] = self.f[:, i, i]
        stats = SolverStatistics(iter, total_solve_time, err_seq)
        return f_app, stats

    def get_force_cone_angle_bound(self) -> float:
        return self.primal_solvers[0].max_f_ang

    def set_force_err_tolerance(self, tol: float, use_total_res: bool = True) -> None:
        self.res_tol = tol
        self.use_total_res = use_total_res

    def set_max_iter(self, max_iter: int) -> None:
        self.max_iter = max_iter
