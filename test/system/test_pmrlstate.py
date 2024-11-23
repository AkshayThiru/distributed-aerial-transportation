import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from pinocchio.utils import rotate

from system.point_mass_rigid_link import PMRLState


def _pmrl_3robot_state_trajectory(t: float) -> tuple[PMRLState, tuple]:
    # q trajectory.
    k1 = np.pi / 2  # |
    k2 = 2 / 3 * np.pi  # | - constants.
    k3 = np.pi / 5  # |
    a = k1 * t  # |
    b = k3 * np.sin(k2 * t)  # | - angles.
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    da, dda = k1, 0
    db, ddb = k3 * k2 * np.cos(k2 * t), -k3 * k2**2 * np.sin(k2 * t)
    q_ = np.array([ca * sb, sa * sb, cb])
    dq_ = np.array(
        [-sa * sb * da + ca * cb * db, ca * sb * da + sa * cb * db, -sb * db]
    )
    ddq_ = np.array(
        [
            -ca * sb * da**2
            - 2 * sa * cb * da * db
            - sa * sb * dda
            - ca * sb * db**2
            + ca * cb * ddb,
            -sa * sb * da**2
            + 2 * ca * cb * da * db
            + ca * sb * dda
            - sa * sb * db**2
            + sa * cb * ddb,
            -cb * db**2 - sb * ddb,
        ]
    )
    q = np.vstack([q_] * 3).T
    dq = np.vstack([dq_] * 3).T
    ddq = np.vstack([ddq_] * 3).T

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

    # Rl trajectory.
    if t <= 5:
        Rl = rotate("z", (2 * np.pi) * np.sin(np.pi / 2 * t))
        wl = np.array([0, 0, np.pi**2 * np.cos(np.pi / 2 * t)])
        dwl = np.array([0, 0, -np.pi**3 / 2 * np.sin(np.pi / 2 * t)])
    else:
        Rl = rotate("x", (2 * np.pi) * np.sin(np.pi / 2 * t))
        wl = np.array([np.pi**2 * np.cos(np.pi / 2 * t), 0, 0])
        dwl = np.array([-np.pi**3 / 2 * np.sin(np.pi / 2 * t), 0, 0])

    # Full trajectory.
    state = PMRLState(q, dq, xl, vl, Rl, wl)
    acc = (ddq, dvl, dwl)
    return state, acc


def main() -> None:
    dt = 1e-3
    t_seq = np.arange(0, 10, dt)
    q_err_seq, dq_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    xl_err_seq, vl_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    Rl_err_seq, wl_err_seq = np.empty(t_seq.shape), np.empty(t_seq.shape)
    state_t, acc_t = _pmrl_3robot_state_trajectory(0)
    q_err_seq[0], xl_err_seq[0], Rl_err_seq[0] = 0, 0, 0
    for i in range(1, len(t_seq)):
        state_t.integrate(*acc_t, dt)
        s_, acc_t = _pmrl_3robot_state_trajectory(t_seq[i])
        q_err_seq[i] = norm(state_t.q - s_.q)
        dq_err_seq[i] = norm(state_t.dq - s_.dq)
        xl_err_seq[i] = norm(state_t.xl - s_.xl)
        vl_err_seq[i] = norm(state_t.vl - s_.vl)
        Rl_err_seq[i] = norm(state_t.Rl - s_.Rl)
        wl_err_seq[i] = norm(state_t.wl - s_.wl)

    plt.plot(t_seq, q_err_seq, label="q")
    plt.plot(t_seq, dq_err_seq, label="dq")
    plt.plot(t_seq, xl_err_seq, label="xl")
    plt.plot(t_seq, vl_err_seq, label="vl")
    plt.plot(t_seq, Rl_err_seq, label="Rl")
    plt.plot(t_seq, wl_err_seq, label="wl")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
