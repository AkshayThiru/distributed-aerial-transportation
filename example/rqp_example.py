import pickle
from dataclasses import dataclass
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from control.rqp_cadmm import RQPCADMMController
from control.rqp_centralized import (RQPCentralizedController,
                                     RQPLowLevelController)
from control.rqp_dd import RQPDDController
from example.env_forest import Forest
from example.setup import rqp_setup
from system.rigid_quadrotor_payload import RQPDynamics, RQPState
from utils.math_utils import compute_aggregate_statistics

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"


@dataclass
class RQPStateData:
    R: np.ndarray
    w: np.ndarray
    xl: np.ndarray
    vl: np.ndarray
    Rl: np.ndarray
    wl: np.ndarray


def _desired_acceleration_forest(
    s: RQPState, t: float, env: Forest
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.zeros((3,))
    x_ref[0] = s.xl[0] + 1.5  # [m].
    norm = np.linalg.norm(s.xl[:2] - env.mountain_center)
    if norm >= env.mountain_radius:
        x_ref[2] = 1.5  # [m].
    else:
        x_ref[2] = (
            np.sqrt(env.mountain_sphere_radius**2 - norm**2)
            - env.mountain_center_depth
            + 1.5
        )  # [m].
    v_ref = np.array([0.5, 0.0, 0.0])

    k_p = 1.0
    k_v = 1.0
    dvl_des = -k_v * (s.vl - v_ref) - k_p * (s.xl - x_ref)
    norm = np.linalg.norm(dvl_des)
    if norm > 0:
        dvl_des = dvl_des / norm * np.min((norm, 1.0))

    dwl_des = np.array([0.0, 0.0, 0.0])

    acc_des = (dvl_des, dwl_des)
    return acc_des, x_ref, v_ref


def _print_stats(iter: List[int], solve_time: List[float]) -> None:
    if len(iter) > 0:
        iter_stats = compute_aggregate_statistics(np.array(iter))
        print(
            f"Solver iterations: "
            + f"min: {iter_stats[0]:5.2f}, "
            + f"max: {iter_stats[1]:5.2f}, "
            + f"avg: {iter_stats[2]:5.2f}, "
            + f"std: {iter_stats[3]:5.2f}"
        )
    if len(solve_time) > 0:
        solver_time_stats = compute_aggregate_statistics(np.array(solve_time))
        print(
            f"Solver solve time (ms): "
            + f"min: {solver_time_stats[0] * 1e3:7.3f}, "
            + f"max: {solver_time_stats[1] * 1e3:7.3f}, "
            + f"avg: {solver_time_stats[2] * 1e3:7.3f}, "
            + f"std: {solver_time_stats[3] * 1e3:7.3f}"
        )


def main() -> None:
    n = 3
    dt = 1e-3
    hl_rel_freq = 10
    log_freq = hl_rel_freq
    # controller_type = "centralized"
    controller_type = "dual-decomposition"
    # controller_type = "consensus-admm"

    T = 100.0  # [s].
    env = Forest()
    _desired_acceleration = lambda s, t: _desired_acceleration_forest(s, t, env)

    params, col, s0 = rqp_setup(n)
    dyn = RQPDynamics(params, s0, dt)
    if controller_type == "centralized":
        hl_controller = RQPCentralizedController(
            params, col, s0, dt, env, verbose=False
        )
    elif controller_type == "dual-decomposition":
        hl_controller = RQPDDController(params, col, s0, dt, env, verbose=False)
    elif controller_type == "consensus-admm":
        hl_controller = RQPCADMMController(params, col, s0, dt, env, verbose=False)
    else:
        raise NotImplementedError
    max_f_ang = hl_controller.get_force_cone_angle_bound()
    ll_controller = RQPLowLevelController("pd", params, max_f_ang)

    t_seq = np.arange(0, T, dt)
    state_seq = []
    x_err_seq = []
    v_err_seq = []
    f_des_seq = []
    iter_seq = []
    solve_time_seq = []
    min_env_dist_seq = []
    w_seq = []
    for i in tqdm(range(len(t_seq))):
        if i % hl_rel_freq == 0:
            acc_des, x_ref, v_ref = _desired_acceleration(dyn.state, t_seq[i])
            f_des, stats = hl_controller.control(dyn.state, acc_des)
            # Logging.
            f_des_seq.append(f_des)
            if not (stats.iter == -1):
                iter_seq.append(stats.iter)
            solve_time_seq.append(stats.solve_time)
            min_env_dist_seq.append(stats.min_env_dist)
        w = ll_controller.control(dyn.state, f_des)
        dyn.integrate(w)
        if i % log_freq == 0:
            w_seq.append(w)
            x_err_seq.append(np.linalg.norm(x_ref - dyn.state.xl))
            v_err_seq.append(np.linalg.norm(v_ref - dyn.state.vl))
            s_ = dyn.state
            sd_ = RQPStateData(s_.R, s_.w, s_.xl, s_.vl, s_.Rl, s_.wl)
            state_seq.append(sd_)
    _print_stats(iter_seq, solve_time_seq)

    logs = dict()
    logs["n"] = n
    logs["dt"] = dt
    logs["T"] = T
    logs["hl_rel_freq"] = hl_rel_freq
    logs["log_freq"] = log_freq

    logs["num_trees"] = env.num_trees
    logs["tree_pos"] = env.tree_pos

    logs["controller_type"] = controller_type

    logs["state_seq"] = state_seq
    logs["x_err_seq"] = x_err_seq
    logs["v_err_seq"] = v_err_seq
    logs["f_des_seq"] = f_des_seq
    logs["iter_seq"] = iter_seq
    logs["solve_time_seq"] = solve_time_seq
    logs["min_env_dist_seq"] = min_env_dist_seq
    logs["w_seq"] = w_seq

    # Save log file.
    file_name = "logs/rqp_forest_" + controller_type + ".pkl"
    with open(file_name, "wb") as file:
        pickle.dump(logs, file)

    fig_width, fig_height = 3.54, 3.54  # [in].
    _, ax = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    t_seq = np.linspace(0.0, T, len(x_err_seq))
    ax[0].plot(t_seq, np.array(x_err_seq), "-b", lw=1)
    t_seq = np.linspace(0.0, T, len(v_err_seq))
    ax[1].plot(t_seq, np.array(v_err_seq), "-b", lw=1)

    plt.show()

    fig_width, fig_height = 3.54, 3.54  # [in].
    _, ax = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    solve_time_seq = np.array(solve_time_seq) * 1e3  # [ms].
    t_seq = np.linspace(0.0, T, len(solve_time_seq))
    ax[0].plot(t_seq, solve_time_seq, "-b", lw=1)
    min_env_dist_seq = np.array(min_env_dist_seq) + 1e-6
    t_seq = np.linspace(0.0, T, len(min_env_dist_seq))
    ax[1].plot(t_seq, min_env_dist_seq, "-b", lw=1)
    ax[1].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    main()
