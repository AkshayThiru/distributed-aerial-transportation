from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import meshcat
import numpy as np
from tqdm import tqdm

from control.rqp_cadmm import RQPCADMMController
from control.rqp_centralized import (RQPCentralizedController,
                                     RQPLowLevelController)
from control.rqp_dd import RQPDDController
from example.env_forest import Forest
from example.setup import rqp_setup
from system.rigid_quadrotor_payload import RQPDynamics, RQPState, RQPVisualizer
from utils.math_utils import compute_aggregate_statistics

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"

_VISUALIZE = True


def _desired_acceleration_noenv(
    s: RQPState, t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = 1.0
    height = 1.0
    freq = 1 / 2.0
    x_ref = np.array([radius * np.cos(freq * t), radius * np.sin(freq * t), height])
    v_ref = np.array(
        [-radius * freq * np.sin(freq * t), radius * freq * np.cos(freq * t), 0.0]
    )
    a_ref = np.array(
        [
            -radius * freq**2 * np.cos(freq * t),
            -radius * freq**2 * np.sin(freq * t),
            0.0,
        ]
    )
    k_p = 1.0
    k_v = 1.0
    dvl_des = a_ref - k_v * (s.vl - v_ref) - k_p * (s.xl - x_ref)

    dwl_des = np.array([np.sin(t), np.cos(t), np.pi / 12])

    acc_des = (dvl_des, dwl_des)
    return acc_des, x_ref, v_ref


def _desired_acceleration_forest(
    s: RQPState, t: float, env: Forest
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.zeros((3,))
    x_ref[0] = s.xl[0] + 1.0  # [m].
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


def _plot_convergence_rate(dyn: RQPDynamics, args: tuple) -> None:
    # Define high-level controllers.
    hlc_dd = RQPDDController(*args)
    hlc_cadmm = RQPCADMMController(*args)
    # Set tolerances.
    hlc_dd.set_force_err_tolerance(0.0)
    hlc_cadmm.set_force_err_tolerance(0.0, False)
    max_iter = 25
    hlc_dd.set_max_iter(max_iter)
    hlc_cadmm.set_max_iter(max_iter)

    nsample = 100
    err_seq_dd = []
    err_seq_cadmm = []
    for _ in tqdm(range(nsample)):
        # Set random desired accelerations.
        dvl_des = (np.random.random((3,)) - 0.5) * 10.0
        dwl_des = (np.random.random((3,)) - 0.5) * 10.0
        acc_des = (dvl_des, dwl_des)
        # Get error lists.
        _, stats = hlc_dd.control(dyn.state, acc_des)
        err_seq_dd.append(stats.err_seq)
        _, stats = hlc_cadmm.control(dyn.state, acc_des)
        err_seq_cadmm.append(stats.err_seq)
    err_seq_dd = np.array(err_seq_dd)
    err_seq_cadmm = np.array(err_seq_cadmm)

    fig_width, fig_height = 3.54, 2.0  # [in].
    _, ax = plt.subplots(
        1,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    err_stats_dd = compute_aggregate_statistics(err_seq_dd)
    err_stats_cadmm = compute_aggregate_statistics(err_seq_cadmm)
    iters = np.arange(max_iter)
    ax.plot(iters, err_stats_dd[2], "--b", lw=1, label=r"DD")
    ax.plot(iters, err_stats_cadmm[2], "--r", lw=1, label=r"C-ADMM")
    ax.fill_between(
        iters, err_stats_dd[0] + 1e-10, err_stats_dd[1], alpha=0.1, color="b", lw=1
    )
    ax.fill_between(
        iters,
        err_stats_cadmm[0] + 1e-10,
        err_stats_cadmm[1],
        alpha=0.1,
        color="r",
        lw=1,
    )
    ax.legend(loc="upper right")
    ax.set_yscale("log")

    plt.show()


def main() -> None:
    n = 3
    dt = 1e-3
    hl_rel_freq = 10
    vis_rel_freq = 10
    # controller_type = "centralized"
    controller_type = "dual-decomposition"
    # controller_type = "consensus-admm"

    if _VISUALIZE:
        vis = meshcat.Visualizer()
        vis.open()

    T = 75.0  # [s].
    env = Forest()
    if _VISUALIZE:
        env.visualize_env(vis)
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

    # args = (params, col, s0, dt, False)
    # _plot_convergence_rate(dyn, args)

    if _VISUALIZE:
        visualizer = RQPVisualizer(params, col, vis)

    t_seq = np.arange(0, T, dt)
    x_err = np.empty(t_seq.shape)
    v_err = np.empty(t_seq.shape)
    iter = []
    solve_time = []
    min_env_dist = []
    for i in tqdm(range(len(t_seq))):
        if i % hl_rel_freq == 0:
            acc_des, x_ref, v_ref = _desired_acceleration(dyn.state, t_seq[i])
            f_des, stats = hl_controller.control(dyn.state, acc_des)
            if not (stats.iter == -1):
                iter.append(stats.iter)
            solve_time.append(stats.solve_time)
            min_env_dist.append(stats.min_env_dist)
        w = ll_controller.control(dyn.state, f_des)
        dyn.integrate(w)
        if _VISUALIZE and (i % vis_rel_freq == 0):
            visualizer.update(dyn.state, w[0], vis)
        x_err[i] = np.linalg.norm(x_ref - dyn.state.xl)
        v_err[i] = np.linalg.norm(v_ref - dyn.state.vl)
    _print_stats(iter, solve_time)

    fig_width, fig_height = 3.54, 3.54  # [in].
    _, ax = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    ax[0].plot(t_seq, x_err, "-b", lw=1)
    ax[1].plot(t_seq, v_err, "-b", lw=1)

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
    solve_time = np.array(solve_time)
    t_seq = np.linspace(0.0, T, len(solve_time))
    ax[0].plot(t_seq, solve_time, "-b", lw=1)
    min_env_dist = np.array(min_env_dist)
    t_seq = np.linspace(0.0, T, len(min_env_dist))
    ax[1].plot(t_seq, min_env_dist, "-b", lw=1)

    plt.show()


if __name__ == "__main__":
    main()
