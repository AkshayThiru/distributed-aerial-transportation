from dataclasses import dataclass
from typing import Any, List

import cvxpy as cv
import hppfcl
import numpy as np
import pinocchio as pin
from scipy import constants

from system.rigid_quadrotor_payload import (RQPCollision, RQPParameters,
                                            RQPState)
from utils.math_utils import rotation_matrix_a_to_b
from utils.so3_tracking_controllers import (So3PDParameters, So3SMParameters,
                                            so3_pd_tracking_control,
                                            so3_sm_tracking_control)


@dataclass
class SolverStatistics:
    iter: int
    solve_time: float
    collision: bool
    min_env_dist: float
    err_seq: List[float] | None = None


class RQPCentralizedController:
    """Centralized controller for the rigid-quadrotor-payload system.

    min_{dv_com, dvl, dwl, f} k_f ||sum_i f - mT g e3||^2 + k_m ||sum_i r_comi x Rl.T fi||^2
                              + k_feq sum_i ||fi - fi_eq||^2
                              + k_dvl ||dvl - dvl_des||^2 + k_dwl ||dwl - dwl_des||^2
                              + k_smooth sum_i (||fi||^2 - <fi, qi'>^2),
    s.t.    mT dv_com = sum_i fi - mT g e3,
            JT dwl + wl x JT wl = sum_i r_comi x Rl.T fi,
            dvl = dv_com - Rl hat(wl)^2 x_com + Rl hat(x_com) dwl,
            fi_z >= min_fz, for i = 1, ..., n,
            ||fi||_2 <= sec(max_f_ang) fi_z, for i = 1, ..., n,
            ||fi||_2 <= max_f, for i = 1, ..., n,
            e3.T Rl e3 >= cos(max_p_ang) -> second-order CBF,
            ||wl||^2 <= max_wl ** 2 -> CBF,
            ||vl||^2 <= max_vl ** 2 -> CBF,
            dist(obstacle) >= dist_eps, for obstacle in env -> CBF.
    """

    def __init__(
        self,
        params: RQPParameters,
        col: RQPCollision,
        state: RQPState,
        dt: float,
        env: Any = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # Set system constants.
        self._set_system_constants(params, col, dt, env)
        # Set controller constants.
        self._set_controller_constants()
        # Set CVX variables: (dv_com, dvl, dwl, f).
        self._set_cvx_variables()
        # Set CVX parameters.
        self._set_cvx_parameters()

        self.cons = []
        # Dynamics constraint.
        #   mT dv_com = sum_i fi - mT g e3,
        #   JT dwl + wl x JT wl = sum_i r_comi x Rl.T fi.
        self._set_dynamics_constraint()
        # Kinematics constraint.
        #   dvl = dv_com - Rl hat(wl)^2 x_com + Rl hat(x_com) dwl.
        self._set_kinematics_constraint()
        # z-force lower bound constraint: for each i = 1, ..., n,
        #   fi_z >= min_fz.
        self._set_zforce_lower_bound_constraint()
        # Force cone angle bound constraint: for each i = 1, ..., n,
        #   ||fi||_2 <= sec(max_f_ang) fi_z.
        self._set_force_cone_angle_bound_constraint()
        # Force norm bound constraint: for each i = 1, ..., n,
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
        # Environment collision avoidance CBF constraint.
        self._set_env_collision_avoidance_cbf_constraint()

        self.cost = 0
        # Total force cost.
        #   k_f ||sum_i f - mT g e3||^2.
        self._set_total_force_cost()
        # Total moment cost.
        #   k_m ||sum_i r_comi x Rl.T fi||^2.
        self._set_total_moment_cost()
        # Equilibrium force cost.
        #   k_feq sum_i ||fi - fi_eq||^2.
        self._set_equilibrium_force_cost()
        # Desired acceleration cost.
        #   k_dvl ||dvl - dvl_des||^2 ~= k_dvl ||dvl||^2 - 2 k_dvl (dvl_des.T dvl).
        self._set_desired_acceleration_cost()
        # Desired angular acceleration cost.
        #   k_dwl ||dwl - dwl_des||^2 ~= k_dwl ||dwl||^2 - 2 k_dwl (dwl_des.T dwl).
        self._set_desired_angular_acceleration_cost()
        # Force smoothing cost.
        #   qi'_j = Ri exp(hat(wi) * dt) ej,
        #   k_smooth sum_i fi.T (I_3 - qi'_3 qi'_3.T) fi,
        #   = k_smooth sum_i (qi'_1.T fi)^2 + (qi'_2.T fi)^2.
        self._set_force_smoothing_cost()

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
        dt: float,
        env: Any,
    ) -> None:
        assert params.n >= 3
        self.n = params.n
        self.params = params
        self.col = col
        self.env = env

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
        # Environment collision avoidance CBF constants.
        self.dist_eps = 0.1  # [m].
        self.vision_radius = self.col.collision_radius + 5.0  # [m].
        self.nenv_cbfs = 10
        #   Backup CBF constants.
        self.max_deceleration = self.col.max_deceleration
        self.alpha_env_cbf = 2.0

        # Costs:
        # Total force constant.
        self.k_f = 0.1
        # Total moment constant.
        self.k_m = 0.1
        # Equilibrium force constant.
        self.k_feq = 0.1
        # Desired acceleration constant.
        self.k_dvl = 1.0
        # Desired angular acceleration constant.
        self.k_dwl = 1.0
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
        self.Rq_orth = cv.Parameter((6, self.n))
        # Environment collision avoidance CBF.
        self.env_cbf_lhs = cv.Parameter((self.nenv_cbfs, 3))
        self.env_cbf_rhs = cv.Parameter(self.nenv_cbfs)

    def _update_cvx_parameters(
        self, state: RQPState, acc_des: tuple[np.ndarray, np.ndarray]
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
        Rq_orth_ = np.empty((6, self.n))
        for i in range(self.n):
            Rq_ = state.R[:, :, i] @ pin.exp3(state.w[:, i] * self.dt)
            Rq_orth_[:, i] = Rq_[:, :2].reshape((6,))
        self.Rq_orth.value = Rq_orth_
        # Environment collision CBF.
        self._set_collision_avoidance_cbf_parameters(state)

    def _set_collision_avoidance_cbf_parameters(self, state: RQPState) -> None:
        self.env_cbf_lhs.value = np.zeros((self.nenv_cbfs, 3))
        self.collision = False
        self.min_env_dist = self.vision_radius
        self.env_cbf_rhs.value = (
            -self.alpha_env_cbf
            * (self.min_env_dist - self.dist_eps)
            * np.ones((self.nenv_cbfs,))
        )
        if self.env is None:
            return

        # Set system collision object and transformation.
        capsule_radius = self.col.collision_radius
        capsule_height = 0.5 * np.dot(state.vl, state.vl) / self.max_deceleration
        obj = hppfcl.Capsule(capsule_radius, capsule_height)
        tf1 = hppfcl.Transform3f()
        speed = np.linalg.norm(state.vl)
        if speed == 0:
            tf1.setTranslation(state.xl)
        else:
            capsule_dir = state.vl / speed
            tf1.setTranslation(state.xl + capsule_height / 2.0 * capsule_dir)
            tf1.setRotation(
                rotation_matrix_a_to_b(np.array([0.0, 0.0, 1.0]), capsule_dir)
            )
        # Get collision data.
        data = self.env.centralized_distance(obj, tf1, self.vision_radius)
        self.collision, min_dists, nearest_pts1, nearest_pts2 = data
        k = np.min([min_dists.shape[0], self.nenv_cbfs])
        if (k > 0) and (speed > 0):
            self.min_env_dist = np.min(min_dists)
            if k < min_dists.shape[0]:
                idx = np.argpartition(min_dists, k)[:k]
            else:
                idx = np.arange(k)
            # Compute CBF constants.
            lhs = np.zeros((self.nenv_cbfs, 3))
            rhs = np.zeros((self.nenv_cbfs,))
            for i in range(k):
                idxi = idx[i]
                di = min_dists[idxi]
                if di <= 1e-4:
                    continue
                projection = np.dot(nearest_pts1[idxi, :] - state.xl, capsule_dir)
                projection = np.max([0.0, np.min([capsule_height, projection])])
                min_time = np.sqrt(
                    2.0 * (capsule_height - projection) / self.max_deceleration
                )
                min_time = np.max([0.0, speed / self.max_deceleration - min_time])
                normal = nearest_pts1[idxi, :] - nearest_pts2[idxi, :]
                normal = normal / np.linalg.norm(normal)
                lhs[i] = normal * min_time
                rhs[i] = -self.alpha_env_cbf * (di - self.dist_eps) - np.dot(
                    normal, state.vl
                )
            self.env_cbf_lhs.value = lhs
            self.env_cbf_rhs.value = rhs

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
        self.cons += [self.f[2, :] >= self.min_fz]

    def _set_force_cone_angle_bound_constraint(self) -> None:
        self.cons += [cv.norm2(self.f, axis=0) <= self.sec_max_f_ang * self.f[2, :]]

    def _set_force_norm_bound_constraint(self) -> None:
        self.cons += [cv.norm2(self.f, axis=0) <= self.max_f]

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

    def _set_env_collision_avoidance_cbf_constraint(self) -> None:
        self.cons += [self.env_cbf_lhs @ self.dvl >= self.env_cbf_rhs]

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
        feq_err = self.f - self.f_eq
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
        for i in range(self.n):
            self.cost += self.k_smooth * cv.sum_squares(
                self.Rq_orth[:, i].reshape((3, 2)).T @ self.f[:, i]
            )

    # Warm start.
    def _set_warm_start(self) -> None:
        self.prev_f = self.f_eq

        self.dv_com.value = np.zeros((3,))
        self.dvl.value = np.zeros((3,))
        self.dwl.value = np.zeros((3,))
        self.f.value = self.f_eq

    def control(
        self, state: RQPState, acc_des: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, SolverStatistics]:
        self._update_cvx_parameters(state, acc_des)
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)
        if self.prob.status == cv.OPTIMAL:
            self.prev_f = self.f.value
        elif self.verbose:
            print(f"Problem not solved to optimality, status: {self.prob.status}")
        stats = SolverStatistics(
            -1, self.prob.solver_stats.solve_time, self.collision, self.min_env_dist
        )
        return self.prev_f, stats

    def get_force_cone_angle_bound(self) -> float:
        return self.max_f_ang

    def get_dist_eps(self) -> float:
        return self.dist_eps


class RQPLowLevelController:
    """Low-level (moment + thrust) controller for the quadrotor system."""

    def __init__(
        self,
        so3_controller_type: str,
        params: RQPParameters,
        max_f_ang: float,
    ) -> None:
        self._set_constants(params, max_f_ang)

        dwd = np.zeros((3,))
        if so3_controller_type == "pd":
            self.ll_params = self.pd_params
            self.controller = lambda R, Rd, w, wd, i: so3_pd_tracking_control(
                R, Rd, w, wd, dwd, self.J[:, :, i], self.ll_params
            )
        elif so3_controller_type == "sm":
            self.ll_params = self.sm_params
            self.controller = lambda R, Rd, w, wd, i: so3_sm_tracking_control(
                R, Rd, w, wd, dwd, self.J[:, :, i], self.ll_params
            )
        else:
            raise NotImplementedError

    def _set_constants(self, params: RQPParameters, max_f_ang: float) -> None:
        # System constants.
        self.J = params.J

        # Controller constants.
        #   PD tracking constants.
        k_R = 0.25
        k_Omega = 0.075
        self.pd_params = So3PDParameters(k_R, k_Omega)
        #   SM tracking constants.
        r = 0.5
        k_R = 1.415
        l_R = 0.707
        k_s = 0.113
        l_s = 0.057
        self.sm_params = So3SMParameters(r, k_R, l_R, k_s, l_s)
        #   Cone angle bound CBF constants.
        self.cos_max_f_ang = np.cos(max_f_ang)
        self.alpha1_cbf = 5.0
        self.alpha2_cbf = 5.0

    def _rotation_from_unit_vector(self, q: np.ndarray) -> np.ndarray:
        R = np.empty((3, 3))
        sin_x = -q[1]
        cos_x = np.sqrt(q[0] ** 2 + q[2] ** 2)
        sin_y = q[0] / cos_x
        cos_y = q[2] / cos_x
        R[0, 0] = cos_y
        R[1, 0] = 0.0
        R[2, 0] = -sin_y
        R[0, 1] = sin_x * sin_y
        R[1, 1] = cos_x
        R[2, 1] = cos_y * sin_x
        R[:, 2] = q
        return R

    def control(
        self, state: RQPState, f_des: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert f_des.shape == (3, state.n)
        assert all(f_des[2, :] > 0.0)
        f = np.zeros((state.n,))
        M = np.zeros((3, state.n))
        for i in range(state.n):
            # Set thrust forces.
            f[i] = np.dot(f_des[:, i], state.R[:, 2, i])
            # Set moments.
            qd = f_des[:, i] / np.linalg.norm(f_des[:, i])
            Rd = self._rotation_from_unit_vector(qd)
            # wd = state.w[:, i] # NOTE: Causes instability.
            wd = np.zeros((3,))
            M[:, i] = self.controller(state.R[:, :, i], Rd, state.w[:, i], wd, i)

        return f, M
