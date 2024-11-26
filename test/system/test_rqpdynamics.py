import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from system.rigid_quadrotor_payload import RQPDynamics, RQPParameters, RQPState


def _rqp_nrobot_parameters(n: int) -> RQPParameters:
    m = np.array([0.5] * n)
    ml = 0.225
    Jq = np.diag([2.32, 2.32, 4]) * 1e-3
    J = np.empty((3, 3, n))
    for i in range(n):
        J[:, :, i] = Jq
    Jl = np.diag([2.1, 1.87, 3.97]) * 1e-2
    if n == 3:
        r = np.vstack(
            [
                np.array([-0.42, -0.27, 0]),
                np.array([0.48, -0.27, 0]),
                np.array([-0.06, 0.55, 0]),
            ]
        ).T
    else:
        r = (np.random.random((3, n)) - 0.5) * 2.0
    return RQPParameters(m, J, ml, Jl, r)


def _rqp_nrobots_init_state(n: int) -> RQPState:
    R = np.empty((3, 3, n))
    for i in range(n):
        R[:, :, i] = np.eye(3)
    w = np.zeros((3, n))
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return RQPState(R, w, xl, vl, Rl, wl)


def _rqp_nrobot_wrench_trajectory(
    p: RQPParameters, t: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    f = np.random.random((n,)) * p.mT * constants.g / n
    M = np.random.random((3, n)) - 0.5
    return (f, M)


def main() -> None:
    n = 4
    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn_err_seq = np.empty(t_seq.shape)
    s0 = _rqp_nrobots_init_state(n)
    p = _rqp_nrobot_parameters(n)
    dyn = RQPDynamics(p, s0, dt)
    for i in range(len(t_seq)):
        w = _rqp_nrobot_wrench_trajectory(p, t_seq[i], n)
        acc = dyn.forward_dynamics(w)
        dyn_err_seq[i] = RQPDynamics.inverse_dynamics_error(dyn.state, p, w, acc)
        dyn.integrate(w)

    plt.plot(t_seq, dyn_err_seq)
    plt.show()


if __name__ == "__main__":
    main()
