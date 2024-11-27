from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from scipy.linalg import polar

from utils.so3_tracking_controllers import (so3_pd_tracking_control,
                                            so3_sm_tracking_control)

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"


def _get_so3_tracking_errors(
    control: Callable[[np.ndarray, np.ndarray], np.ndarray],
    R: np.ndarray,
    Rd: np.ndarray,
    w: np.ndarray,
    wd: np.ndarray,
    J: np.ndarray,
    dt: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert R.shape == (3, 3) and Rd.shape == (3, 3)
    assert w.shape == (3,) and wd.shape == (3,)
    J_inv = np.linalg.inv(J)

    t_seq = np.arange(0, T, dt)
    e_R = np.empty(t_seq.shape)
    e_Omega = np.empty(t_seq.shape)
    M_norm = np.empty(t_seq.shape)
    for i in range(len(t_seq)):
        M = control(R, w)
        e_R[i] = 1 / 2 * np.trace(np.eye(3) - Rd.T @ R)
        e_Omega[i] = np.linalg.norm(w - R.T @ Rd @ wd)
        M_norm[i] = np.linalg.norm(M)

        dw = J_inv @ (M - np.cross(w, J @ w))
        R = R @ pin.exp3((w + dw * dt / 2) * dt)
        R, _ = polar(R)
        w = w + dw * dt

    return e_R, e_Omega, M_norm


def _test_so3_tracking_control() -> None:
    T = 5.0
    dt = 1e-3

    R = np.eye(3)
    Rd = tf.random_rotation_matrix()[:3, :3]
    w = np.zeros((3,))
    wd = np.zeros((3,))
    dwd = np.zeros((3,))

    J = np.diag([2.32, 2.32, 4]) * 1e-3

    scale = 1 / 0.0820 * J[0, 0]
    k_R = 8.81 * scale
    k_Omega = 2.54 * scale
    pd_params = (k_R, k_Omega)
    pd_control = lambda R, w: so3_pd_tracking_control(R, Rd, w, wd, dwd, J, pd_params)
    pd_errs = _get_so3_tracking_errors(pd_control, R, Rd, w, wd, J, dt, T)

    r = 0.5
    k_R = 50.0 * scale
    l_R = 25.0 * scale
    k_s = 4.0 * scale
    l_s = 2.0 * scale
    sm_params = (r, k_R, l_R, k_s, l_s)
    sm_control = lambda R, w: so3_sm_tracking_control(R, Rd, w, wd, dwd, J, sm_params)
    sm_errs = _get_so3_tracking_errors(sm_control, R, Rd, w, wd, J, dt, T)

    fig_width, fig_height = 3.54, 5  # [in].
    _, ax = plt.subplots(
        3,
        1,
        figsize=(fig_width, fig_height),
        dpi=200,
        sharex=True,
        layout="constrained",
    )
    t_seq = np.arange(0, T, dt)
    ax[0].plot(t_seq, pd_errs[0], "-b", lw=1, label=r"PD")
    ax[0].plot(t_seq, sm_errs[0], "-r", lw=1, label=r"SMC")
    ax[0].set_ylabel(r"$e_R(R, R_d)$")
    ax[0].legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    ax[1].plot(t_seq, pd_errs[1], "-b", lw=1, label=r"PD")
    ax[1].plot(t_seq, sm_errs[1], "-r", lw=1, label=r"SMC")
    ax[1].set_ylabel(r"$e_\Omega(\omega, \omega_d)$")
    ax[1].legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    ax[2].plot(t_seq, pd_errs[1], "-b", lw=1, label=r"PD")
    ax[2].plot(t_seq, sm_errs[1], "-r", lw=1, label=r"SMC")
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$\Vert M\Vert_2$")
    ax[2].legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )

    plt.show()


if __name__ == "__main__":
    _test_so3_tracking_control()
