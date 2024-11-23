import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from pinocchio.utils import rotate

from system.rigid_quadrotor_payload import RQPState


def _rqp_nrobot_state_trajectory(t: float, n: int) -> tuple[RQPState, tuple]:
    # xl trajectory.
    k1 = np.pi / 2  # |
    k2 = 2 / 3 * np.pi  # | - constants.
    a = k1 * t  # |
    b = k2 * t  # | - angles.
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    da, dda = k1, 0
    db, ddb = k2, 0
    xl = np.array([ca, sa, sb])
    vl = np.array([-sa * da, ca * da, cb * db])
    dvl = np.array(
        [-ca * da**2 - sa * dda, -sa * da**2 + ca * dda, -sb * db**2 + cb * ddb]
    )

    # R trajectories.
    if t <= 5:
        Rl = rotate("z", (2 * np.pi) * np.sin(np.pi / 2 * t))
        wl = np.array([0, 0, np.pi**2 * np.cos(np.pi / 2 * t)])
        dwl = np.array([0, 0, -np.pi**3 / 2 * np.sin(np.pi / 2 * t)])
    else:
        Rl = rotate("x", (2 * np.pi) * np.sin(np.pi / 2 * t))
        wl = np.array([np.pi**2 * np.cos(np.pi / 2 * t), 0, 0])
        dwl = np.array([-np.pi**3 / 2 * np.sin(np.pi / 2 * t), 0, 0])
    R = np.empty((3, 3, n))
    w = np.empty((3, n))
    dw = np.empty((3, n))
    for i in range(n):
        R[:, :, i] = Rl
        w[:, i] = wl
        dw[:, i] = dwl

    # Full trajectory.
    state = RQPState(R, w, xl, vl, Rl, wl)
    acc = (dw, dvl, dwl)
    return state, acc


def main() -> None:
    n = 4
    dt = 1e-3
    t_seq = np.arange(0, 10, dt)
    R_err_seq, w_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    xl_err_seq, vl_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    Rl_err_seq, wl_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    state_t, acc_t = _rqp_nrobot_state_trajectory(0, n)
    R_err_seq[0], xl_err_seq[0], Rl_err_seq[0] = 0, 0, 0
    for i in range(1, len(t_seq)):
        state_t.integrate(*acc_t, dt)
        s_, acc_t = _rqp_nrobot_state_trajectory(t_seq[i], n)
        R_err_seq[i] = norm(state_t.R - s_.R)
        w_err_seq[i] = norm(state_t.w - s_.w)
        xl_err_seq[i] = norm(state_t.xl - s_.xl)
        vl_err_seq[i] = norm(state_t.vl - s_.vl)
        Rl_err_seq[i] = norm(state_t.Rl - s_.Rl)
        wl_err_seq[i] = norm(state_t.wl - s_.wl)

    plt.plot(t_seq, R_err_seq, label="R")
    plt.plot(t_seq, w_err_seq, label="w")
    plt.plot(t_seq, xl_err_seq, label="xl")
    plt.plot(t_seq, vl_err_seq, label="vl")
    plt.plot(t_seq, Rl_err_seq, label="Rl")
    plt.plot(t_seq, wl_err_seq, label="wl")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
