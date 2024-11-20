import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from system.rigid_quadrotor_payload import RQPDynamics, RQPParameters, RQPState


def _rqp_3robot_parameters() -> RQPParameters:
    m = np.array([0.5] * 3)
    ml = 0.225
    Jq = np.diag([2.32, 2.32, 4]) * 1e-3
    J = np.empty((3, 3, 3))
    for i in range(3):
        J[:, :, i] = Jq
    Jl = np.diag([2.1, 1.87, 3.97]) * 1e-2
    r = np.vstack(
        [
            np.array([-0.42, -0.27, 0]),
            np.array([0.48, -0.27, 0]),
            np.array([-0.06, 0.55, 0]),
        ]
    ).T
    return RQPParameters(m, J, ml, Jl, r)


def _rqp_3robots_init_state() -> RQPState:
    R = np.empty((3, 3, 3))
    for i in range(3):
        R[:, :, i] = np.eye(3)
    w = np.zeros((3, 3))
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return RQPState(R, w, xl, vl, Rl, wl)


def _rqp_3robot_wrench_trajectory(
    p: RQPParameters, t: float
) -> tuple[np.ndarray, np.ndarray]:
    f_ = np.array([0, 0, 1]) * p.mT * constants.g / 3
    k1_d, k2_d = p.mT * constants.g / 5, p.mT * constants.g / 5
    f1_d, f2_d = np.pi / 2, np.pi
    f_ = f_ + np.array(
        [k1_d * np.cos(f1_d * t), k1_d * np.sin(f1_d * t), k2_d * np.sin(f2_d * t)]
    )
    M = np.random.random((3, 3)) - 0.5
    return np.vstack([f_] * 3).T, M


def main() -> None:
    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn_err_seq = np.empty(t_seq.shape)
    s0 = _rqp_3robots_init_state()
    p = _rqp_3robot_parameters()
    dyn = RQPDynamics(p, s0, dt)
    for i in range(len(t_seq)):
        f, M = _rqp_3robot_wrench_trajectory(p, t_seq[i])
        acc = dyn.forward_dynamics(f, M)
        dyn_err_seq[i] = RQPDynamics.inverse_dynamics_error(dyn.state, p, f, M, acc)
        dyn.integrate(f, M)

    plt.plot(t_seq, dyn_err_seq)
    plt.show()


if __name__ == "__main__":
    main()
