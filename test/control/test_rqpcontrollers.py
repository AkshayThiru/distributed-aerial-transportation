import matplotlib as mpl
import matplotlib.pyplot as plt
import meshcat
import numpy as np

from control.rqp_centralized import (RQPCentralizedController,
                                     RQPLowLevelController)
from control.rqp_distributed import RQPDistributedController
from example.setup import rqp_setup
from system.rigid_quadrotor_payload import RQPDynamics, RQPState, RQPVisualizer

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"


def _desired_acceleration(
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


def main() -> None:
    n = 3
    dt = 1e-3
    hl_rel_freq = 10
    # controller_type = "centralized"
    controller_type = "dual-decomposition"

    vis = meshcat.Visualizer()
    vis.open()

    params, col, s0 = rqp_setup(n)
    if controller_type == "centralized":
        hl_controller = RQPCentralizedController(params, col, s0, dt, verbose=True)
    elif controller_type == "dual-decomposition":
        hl_controller = RQPDistributedController(params, col, s0, dt, verbose=True)
    else:
        raise NotImplementedError
    ll_controller = RQPLowLevelController("pd", params, hl_controller.max_f_ang)
    visualizer = RQPVisualizer(params, col, vis)

    t_seq = np.arange(0, 20, dt)
    dyn = RQPDynamics(params, s0, dt)
    x_err = np.empty(t_seq.shape)
    v_err = np.empty(t_seq.shape)
    for i in range(len(t_seq)):
        if i % hl_rel_freq == 0:
            acc_des, x_ref, v_ref = _desired_acceleration(dyn.state, t_seq[i])
            f_des = hl_controller.control(dyn.state, acc_des)
        w = ll_controller.control(dyn.state, f_des)
        dyn.integrate(w)
        if i % hl_rel_freq == 0:
            visualizer.update(dyn.state, w[0], vis)
        x_err[i] = np.linalg.norm(x_ref - dyn.state.xl)
        v_err[i] = np.linalg.norm(v_ref - dyn.state.vl)

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


if __name__ == "__main__":
    main()
