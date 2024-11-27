import cvxpy as cv
import numpy as np
import pinocchio as pin
from scipy import constants

from system.rigid_payload import RPCollision, RPParameters, RPState


class RPCentralizedController:
    """Centralized controller for the rigid payload system.

    min_{dvl, dwl, f} k_f ||sum_i f - ml g e3||^2 + k_feq sum_i ||fi - fi_eq||^2
                      + k_dvl ||dvl - dvl_des||^2 + k_dwl ||dwl - dwl_des||^2,
    s.t.    ml dvl = sum_i fi - ml g e3,
            Jl dwl + wl x Jl wl = sum_i ri x Rl.T fi,
            fi_z >= min_fz, for i = 1, ..., n,
            ||fi||_2 <= tan(max_f_ang) fi_z, for i = 1, ..., n,
            ||fi||_2 <= f_max, for i = 1, ..., n,
            e3.T Rl e3 >= cos(max_p_ang) -> second-order CBF,
            ||wl||^2 <= max_wl ** 2 -> CBF,
            ||vl||^2 <= max_vl ** 2 -> CBF,
    """

    def __init__(
        self,
        params: RPParameters,
        col: RPCollision,
        state: RPState,
        dt: float,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # Set system constants.
        self._set_system_constants(params, col, dt)
        # Set controller constants.
        self._set_controller_constants()
        # Set CVX variables: (dvl, dwl, f).
        self._set_cvx_variables()
        # Set CVX parameters.
        self._set_cvx_parameters()

        self.cons = []
        # Dynamics constraint:
        #   ml dvl = sum_i fi - ml g e3,
        #   Jl dwl + wl x Jl wl = sum_i ri x Rl.T fi.
        self._set_dynamics_constraint()
        # z-force lower bound constraint: for each i = 1, ..., n,
        #   fi_z >= min_fz.
        self._set_zforce_lower_bound_constraint()
        # Force cone angle bound constraint: for each i = 1, ..., n,
        #   ||fi||_2 <= tan(max_f_ang) fi_z.
        self._set_force_cone_angle_bound_constraint()
        # Force norm bound constraint: for each i = 1, ..., n,
        #   ||f_i||_2 <= max_f.
        self._set_force_norm_bound_constraint()
        # Payload angle CBF constraint.
        #   h = e3.T Rl e3 - cos(max_p_ang) >= 0 -> second-order CBF,
        #   dh = e3.T Rl hat(wl) e3,
        #   d2h = e3.T Rl hat(wl)^2 e3 - e3.T Rl hat(e3) dwl,
        #   ddh + (alpha1 + alpha2) dh + alpha2 alpha2 h >= 0.
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
        #   k_f ||sum_i f - ml g e3||^2.
        self._set_total_force_cost()
        # Equilibrium force cost.
        #   k_feq sum_i ||fi - fi_eq||^2.
        self._set_equilibrium_force_cost()
        # Desired acceleration cost.
        #   k_dvl ||dvl - dvl_des||^2 ~= k_dvl ||dvl||^2 - 2 k_dvl (dvl_des.T dvl).
        self._set_desired_acceleration_cost()
        # Desired angular acceleration cost.
        #   k_dwl ||dwl - dwl_des||^2 ~= k_dwl ||dwl||^2 - 2 k_dwl (dwl_des.T dwl).
        self._set_desired_angular_acceleration_cost()

        dvl_des = np.zeros((3,))
        dwl_des = np.zeros((3,))
        acc_des = (dvl_des, dwl_des)
        self._update_cvx_parameters(state, acc_des)

        self.prev_f = self.f_eq
        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if self.verbose:
            print("Optimization problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)

    def _set_system_constants(
        self,
        params: RPParameters,
        col: RPCollision,
        dt: float,
    ) -> None:
        assert params.n >= 3
        self.n = params.n
        self.params = params
        self.col = col

        self.dt = dt
        self.g = constants.g
        self.ml = params.ml
        self.Jl = params.Jl
        self.r = params.r
        self.payload_vertices = col.payload_mesh_vertices

        # Set steady state forces.
        # Steady state forces point upwards,
        # and result in zero net acceleration and angular acceleration.
        self.f_eq = np.zeros((3, self.n))
        wrench_mat = np.empty((3, self.n))
        wrench_mat[0, :] = np.ones((self.n,))
        rhs_vec = np.array([self.ml * self.g, 0.0, 0.0])
        for i in range(self.n):
            wrench_mat[1:, i] = pin.skew(self.r[:, i])[:2, 2]
        self.f_eq[2, :] = np.linalg.lstsq(wrench_mat, rhs_vec, rcond=None)[0]
        # Check steady-state acceleration and angular acceleration.
        if self.verbose:
            net_acc = np.sum(self.f_eq, axis=1) - self.ml * self.g * np.array(
                [0.0, 0.0, 1.0]
            )
            net_ang = np.sum(
                np.cross(self.r, self.f_eq, axisa=0, axisb=0, axisc=0), axis=1
            )
            print(f"Equilibrium acceleration norm: {np.linalg.norm(net_acc)}")
            print(f"Equilibrium angular acceleration norm: {np.linalg.norm(net_ang)}")

        # Dependent constants.
        self.Jl_inv = params.Jl_inv
        self.J_inv_r_hat = np.empty((3, 3, self.n))
        for i in range(self.n):
            self.J_inv_r_hat[:, :, i] = self.Jl_inv @ pin.skew(self.r[:, i])

    def _set_controller_constants(self) -> None:
        # Constraints:
        # z-force lower bound.
        self.min_fz = self.ml * self.g / (self.n * 10.0)
        # Force cone angle bound.
        max_f_ang = np.pi / 6.0
        self.sec_max_f_ang = 1.0 / np.cos(max_f_ang)
        # Force norm bound.
        self.max_f = (2.0 / self.n) * self.ml * self.g
        # Payload angle CBF constants.
        max_p_ang = np.pi / 6.0
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
        self.k_f = 0.1
        # Equilibrium force constant.
        self.k_feq = 0.1
        # Desired acceleration constant.
        self.k_dvl = 1.0
        # Desired angular acceleration constant.
        self.k_dwl = 1.0

    def _set_cvx_variables(self) -> None:
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
        self.v_norm2 = cv.Parameter()
        self.w_norm2 = cv.Parameter()
        self.R_w_hat = cv.Parameter((3, 3))
        self.R_w_hat_sq = cv.Parameter((3, 3))
        self.J_inv_w_cross_Jw = cv.Parameter(3)

    def _update_cvx_parameters(
        self, state: RPState, acc_des: tuple[np.ndarray, np.ndarray]
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
        self.J_inv_w_cross_Jw.value = self.Jl_inv @ np.cross(
            state.wl, self.Jl @ state.wl
        )

    # Constraints.
    def _set_dynamics_constraint(self) -> None:
        gravity_vector = -self.g * np.array([0.0, 0.0, 1.0])
        moment = 0
        for i in range(self.n):
            moment += self.J_inv_r_hat[:, :, i] @ self.Rl.T @ self.f[:, i]
        self.cons += [
            self.ml * self.dvl == cv.sum(self.f, axis=1) + self.ml * gravity_vector,
            self.dwl + self.J_inv_w_cross_Jw == moment,
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

    # Costs.
    def _set_total_force_cost(self) -> None:
        f_err = cv.sum(self.f, axis=1) - self.ml * self.g * np.array([0.0, 0.0, 1.0])
        self.cost += self.k_f * cv.sum_squares(f_err)

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

    def control(
        self, state: np.ndarray, acc_des: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        self._update_cvx_parameters(state, acc_des)
        self.prob.solve(solver=cv.CLARABEL, warm_start=True)
        if self.prob.status == cv.OPTIMAL:
            self.prev_f = self.f.value
            return self.f.value
        else:
            if self.verbose:
                print(f"Problem not solved to optimality, status: {self.prob.status}")
            return self.prev_f
