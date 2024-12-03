import os
import pickle
from time import perf_counter, sleep
from typing import Any, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import meshcat
import numpy as np
from matplotlib import patches
from matplotlib.colors import to_rgb
from scipy.signal import savgol_filter
from tqdm import tqdm

from control.rqp_cadmm import RQPCADMMController
from control.rqp_centralized import RQPCentralizedController
from control.rqp_dd import RQPDDController
from example.env_forest import Forest
from example.rqp_example import RQPStateData
from example.setup import rqp_setup
from system.rigid_quadrotor_payload import (RQPCollision, RQPParameters,
                                            RQPState, RQPVisualizer)
from utils.math_utils import compute_aggregate_statistics

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"
_HALF_COL_WIDTH = 3.54  # [in].
_FULL_COL_WIDTH = 5.0  # 7.16  # [in].
_FIG_DPI = 200
_SAVE_DPI = 1000  # >= 600.
_SAVE_FIG = True


_GRASS_COLOR = to_rgb("#70AB94")
_BARK_COLOR = to_rgb("#694B37")
_MESH_COLOR = to_rgb("#FF22DD")
_QUADROTOR_COLOR = to_rgb("#1590A0")
_PAYLOAD_COLOR = to_rgb("#D70E36")
_VISIONCONE_COLOR = to_rgb("#A8AEAC")


def _visualization(controller_type: str, logs: dict, env: Forest) -> None:
    min_fps: int = 24
    fast_forwardx: float = 5.0

    if controller_type == "centralized":
        replay_fraction = 0.72
    elif controller_type == "dual-decomposition":
        replay_fraction = 0.75
    elif controller_type == "consensus-admm":
        replay_fraction = 0.85

    vis = meshcat.Visualizer()
    vis.open()

    env.visualize_env(vis)

    n: int = logs["n"]
    T: float = logs["T"]
    dt: float = logs["dt"]
    hl_rel_freq: int = logs["hl_rel_freq"]
    log_freq: int = logs["log_freq"]
    assert hl_rel_freq == log_freq
    t_seq = np.arange(0.0, T, log_freq * dt)
    state_seq: List[RQPStateData] = logs["state_seq"]
    assert len(t_seq) == len(state_seq)

    params, col, _ = rqp_setup(n)
    visualizer = RQPVisualizer(params, col, vis)

    camera_offset = np.array([0.0, -5.0, 9.0])  # [m], z_offset > 8.0 m.
    xl = []
    for i in range((len(state_seq))):
        xl.append(state_seq[i].xl)
    xl = np.array(xl).T
    xl_filtered = savgol_filter(xl, 5000, 3, axis=1)
    camera_pos = xl_filtered[:, 0] + camera_offset
    vis.set_cam_pos(camera_pos)
    vis.set_cam_target(np.zeros((3,)))

    sleep(2)

    vis_rel_freq = int(np.max([1.0, np.floor(1.0 / (min_fps * dt * log_freq))]))
    spf = dt * log_freq * vis_rel_freq / fast_forwardx
    start_time = perf_counter()
    for i in tqdm(range(len(t_seq))):
        # Dynamics code (not used).
        # Camera pose and target update.
        camera_pos = xl_filtered[:, i] + camera_offset
        target_pos = xl_filtered[:, i]
        vis.set_cam_pos(camera_pos)
        vis.set_cam_target(target_pos)
        if i % vis_rel_freq == 0:
            if t_seq[i] > T * replay_fraction:
                break
            # Visualization update.
            f = np.zeros((n,))
            s = state_seq[i]
            state = RQPState(s.R, s.w, s.xl, s.vl, s.Rl, s.wl)
            visualizer.update(state, f, vis)

            # Sleep for appropriate time.
            stop_time = perf_counter()
            elapsed_time = stop_time - start_time
            if elapsed_time < spf:
                sleep(spf - elapsed_time)
            start_time = perf_counter()


def _draw_capsule(
    ax: Any, c1: np.ndarray, c2: np.ndarray, radius: float, **kwargs
) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)

    height = np.linalg.norm(c1 - c2)
    if height == 0:
        ax.plot(radius * x_circ + c1[0], radius * y_circ + c1[1], **kwargs)
    dir = (c2 - c1) / height
    orth = np.array([-dir[1], dir[0]])
    ang = np.arctan2(orth[1], orth[0])
    theta1 = np.linspace(ang, ang + np.pi, 50)
    theta2 = np.linspace(ang + np.pi, ang + 2 * np.pi, 50)
    x = np.empty((100, 2))
    x[:50, 0] = c1[0] + radius * np.cos(theta1)
    x[:50, 1] = c1[1] + radius * np.sin(theta1)
    x[50:, 0] = c2[0] + radius * np.cos(theta2)
    x[50:, 1] = c2[1] + radius * np.sin(theta2)
    ax.plot(x[:, 0], x[:, 1], **kwargs)


def _plot_xy_trajectory(
    controller_type: str,
    logs: dict,
    env: Forest,
    params: RQPParameters,
    col: RQPCollision,
    hl_controller=Any,
) -> None:
    # Figure properties.
    font_size = 8
    font_dict = {
        "fontsize": font_size,  # [pt]
        "fontstyle": "normal",
        "fontweight": "normal",
    }
    axis_margins = 0.05

    fig_height = 2.0  # 2.5  # [inch].
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(_FULL_COL_WIDTH, fig_height),
        dpi=_FIG_DPI,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    # Set grass color.
    # ax.set_facecolor(_GRASS_COLOR)
    # Draw forest and mountain.
    theta = np.linspace(0, 2 * np.pi, 100)
    r = env.mountain_radius
    c = env.mountain_center
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)
    # hill = patches.Circle(c, r, fc=_GRASS_COLOR, ec='black', ls="--", lw=1.0, label=r"hill")
    # ax.add_patch(hill)
    ax.plot(
        r * x_circ + c[0],
        r * y_circ + c[1],
        ls="--",
        lw=1,
        color=_GRASS_COLOR,
        # label=r"hill",
    )
    r = env.bark_radius
    for i in range(env.num_trees):
        c = env.tree_pos[i, :2]
        if i == 0:
            tree = patches.Circle(
                c, r, fc=_BARK_COLOR, ec="black", lw=1.0, label=r"trees"
            )
            # ax.plot(r * x_circ + c[0], r * y_circ + c[1], ls="-", lw=1, color=_BARK_COLOR, label=r"trees")
        else:
            tree = patches.Circle(c, r, fc=_BARK_COLOR, ec="black", lw=1.0)
            # ax.plot(r * x_circ + c[0], r * y_circ + c[1], ls="-", lw=1, color=_BARK_COLOR)
        ax.add_patch(tree)
    # Draw system trajectory.
    state_seq: List[RQPStateData] = logs["state_seq"]
    xl = []
    for i in range((len(state_seq))):
        xl.append(state_seq[i].xl)
    xl = np.array(xl).T
    ax.plot(xl[0, :], xl[1, :], ls="--", lw=1, color="black", label=r"$x_l$")
    # Draw mesh, vision cones, payload, and quadrotor at key frames.
    T: int = logs["T"]
    dt: float = logs["dt"]
    log_freq: int = logs["log_freq"]

    if controller_type == "centralized":
        key_frames = np.array([0.5])  # as fractions.
    elif controller_type == "dual-decomposition":
        key_frames = np.array([0.16, 0.55])  # as fractions.
    elif controller_type == "consensus-admm":
        key_frames = np.array([0.19, 0.51, 0.72])  # as fractions.
    key_idx = (key_frames * T / (dt * log_freq)).astype(int)
    for k in range(len(key_idx)):
        i: int = key_idx[k]
        # Payload.
        si = state_seq[i]
        xq = (si.xl.reshape((3, 1)) + si.Rl @ params.r)[:2, :]
        if k == 0:
            payload = patches.Polygon(
                xq.T,
                closed=True,
                fc=_PAYLOAD_COLOR,
                ec="black",
                lw=0.5,
                label=r"payload",
            )
        else:
            payload = patches.Polygon(
                xq.T, closed=True, fc=_PAYLOAD_COLOR, ec="black", lw=0.5
            )
        ax.add_patch(payload)
        # Quadrotors.
        r = col.quadrotor_radius
        for j in range(params.n):
            if k == 0 and j == 0:
                quadrotor = patches.Circle(
                    xq[:, j],
                    r,
                    fc=_QUADROTOR_COLOR,
                    ec="black",
                    lw=0.5,
                    alpha=0.75,
                    label=r"quadrotor",
                )
            else:
                quadrotor = patches.Circle(
                    xq[:, j], r, fc=_QUADROTOR_COLOR, ec="black", lw=0.5, alpha=0.75
                )
            ax.add_patch(quadrotor)
        # Collision mesh.
        r = col.collision_radius
        dec = col.max_deceleration
        c1 = xl[:, i]
        c2 = xl[:, i] + 0.5 * np.linalg.norm(si.vl) / dec * si.vl
        if k == 0:
            _draw_capsule(
                ax,
                c1[:2],
                c2[:2],
                r,
                ls="--",
                lw=1,
                color=_MESH_COLOR,
                label=r"collision capsule",
            )
            # ax.plot(r * x_circ + c1[0], r * y_circ + c[1], ls="--", lw=1, color=_MESH_COLOR, label=r"collision capsule")
        else:
            _draw_capsule(ax, c1[:2], c2[:2], r, ls="--", lw=1, color=_MESH_COLOR)
            # ax.plot(r * x_circ + c1[0], r * y_circ + c[1], ls="--", lw=1, color=_MESH_COLOR)
        # Vision cones.
        if controller_type == "centralized":
            hlc: RQPCentralizedController = hl_controller
            r = hlc.vision_radius
            if k == 0:
                vc = patches.Circle(
                    c1,
                    r,
                    fc=_VISIONCONE_COLOR,
                    ec="none",
                    lw=1.0,
                    alpha=0.25,
                    label=r"vision cone",
                )
            else:
                vc = patches.Circle(
                    c1, r, fc=_VISIONCONE_COLOR, ec="none", lw=1.0, alpha=0.25
                )
            ax.add_patch(vc)
        if controller_type in ["dual-decomposition", "consensus-admm"]:
            hlc: RQPDDController | RQPCADMMController = hl_controller
            r = hlc.primal_solvers[0].vision_radius
            ang = hlc.primal_solvers[0].vision_cone_ang
            for j in range(params.n):
                dir = (si.Rl @ params.r)[:2, j]
                dir_ang = np.arctan2(dir[1], dir[0])
                theta1 = (dir_ang - ang) * 180 / np.pi
                theta2 = (dir_ang + ang) * 180 / np.pi
                if k == 0 and j == 0:
                    vc = patches.Wedge(
                        xq[:, j],
                        r,
                        theta1,
                        theta2,
                        fc=_VISIONCONE_COLOR,
                        ec="none",
                        alpha=0.25,
                        label=r"vision cone",
                    )
                else:
                    vc = patches.Wedge(
                        xq[:, j],
                        r,
                        theta1,
                        theta2,
                        fc=_VISIONCONE_COLOR,
                        ec="none",
                        alpha=0.25,
                    )
                ax.add_patch(vc)
    leg = ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=1.0,
        ncol=2,
        fancybox=False,
        edgecolor="black",
        labelspacing=0.15,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        labelsize=font_size,
    )
    ax.margins(axis_margins, axis_margins)
    ax.axis("equal")
    ax.set_xlim([5.0, 55.0])  # [m].
    ax.set_ylim([-7.0, 7.0])  # [m].

    plt.show()

    if _SAVE_FIG:
        fig.savefig(
            _DIR_PATH + "/../plots/rqp_forest_xy_" + controller_type + ".png",
            dpi=_SAVE_DPI,
        )  # , bbox_inches='tight')


def _plot_min_dist(controller_type: str, logs: dict, dist_eps: float) -> None:
    T: float = logs["T"]
    if controller_type == "centralized":
        Tf = 0.72 * T
    elif controller_type == "dual-decomposition":
        Tf = 0.75 * T
    elif controller_type == "consensus-admm":
        Tf = 0.85 * T

    # Figure properties.
    font_size = 8
    font_dict = {
        "fontsize": font_size,  # [pt]
        "fontstyle": "normal",
        "fontweight": "normal",
    }
    axis_margins = 0.05

    fig_height = 2.0  # [inch].
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(_HALF_COL_WIDTH, fig_height),
        dpi=_FIG_DPI,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # Plot minimum distance.
    min_env_dist_seq: List[float] = logs["min_env_dist_seq"]
    min_env_dist_seq = np.array(min_env_dist_seq)
    t_seq = np.linspace(0.0, T, len(min_env_dist_seq))
    ax.plot(
        t_seq,
        min_env_dist_seq,
        "-b",
        lw=1,
        label=r"$\text{min}_i \ d(x_l(t), \mathcal{O}_i)$",
    )
    ax.plot(
        t_seq,
        dist_eps * np.ones_like(t_seq),
        "--k",
        lw=1,
        label=r"$\epsilon_{\text{margin}}$",
    )
    ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
        labelspacing=0.15,
    )
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=font_size,
    )
    ax.set_yscale("log")
    # ax.set_xticks([0, 12, 24, 36, 47])
    ax.set_xlim([0.0, Tf])
    ax.set_xlabel(r"time $\ (\text{s})$", **font_dict)
    ax.set_ylabel(r"minimum distance $\ (\text{m})$", **font_dict)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if _SAVE_FIG:
        fig.savefig(
            _DIR_PATH + "/../plots/rqp_forest_min_dist_" + controller_type + ".png",
            dpi=_SAVE_DPI,
        )  # , bbox_inches='tight')

def _print_stats(logs: dict) -> None:
    iter_seq: List[int] = logs["iter_seq"]
    iter_seq = np.array(iter_seq)
    # iter_seq = iter_seq[iter_seq <= 20]
    if len(iter_seq) > 0:
        iter_stats = compute_aggregate_statistics(iter_seq)
        print(
            f"Solver iterations: "
            + f"min: {iter_stats[0]:5.2f}, "
            + f"max: {iter_stats[1]:5.2f}, "
            + f"avg: {iter_stats[2]:5.2f}, "
            + f"std: {iter_stats[3]:5.2f}"
        )
    solve_time_seq = logs["solve_time_seq"]
    if len(solve_time_seq) > 0:
        solver_time_stats = compute_aggregate_statistics(np.array(solve_time_seq))
        print(
            f"Solver solve time (ms): "
            + f"min: {solver_time_stats[0] * 1e3:7.3f}, "
            + f"max: {solver_time_stats[1] * 1e3:7.3f}, "
            + f"avg: {solver_time_stats[2] * 1e3:7.3f}, "
            + f"std: {solver_time_stats[3] * 1e3:7.3f}"
        )


def main() -> None:
    # controller_type = "centralized"
    controller_type = "dual-decomposition"
    # controller_type = "consensus-admm"

    file_name = _DIR_PATH + "/../logs/rqp_forest_" + controller_type + ".pkl"
    with open(file_name, "rb") as file:
        logs = pickle.load(file)

    env = Forest()
    env.num_trees = logs["num_trees"]
    env.tree_pos = logs["tree_pos"]

    n: int = logs["n"]
    dt: float = logs["dt"]
    params, col, s0 = rqp_setup(n)
    if controller_type == "centralized":
        hl_controller = RQPCentralizedController(
            params, col, s0, dt, env, verbose=False
        )
    elif controller_type == "dual-decomposition":
        hl_controller = RQPDDController(params, col, s0, dt, env, verbose=False)
    elif controller_type == "consensus-admm":
        hl_controller = RQPCADMMController(params, col, s0, dt, env, verbose=False)

    _visualization(controller_type, logs, env)
    _plot_xy_trajectory(controller_type, logs, env, params, col, hl_controller)
    _plot_min_dist(controller_type, logs, hl_controller.get_dist_eps())
    _print_stats(logs)


if __name__ == "__main__":
    main()
