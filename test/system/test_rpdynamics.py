import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from system.rigid_payload import RPDynamics, RPParameters, RPState


def _rp_nactuators_parameters(n: int) -> RPParameters:
    ml = 0.225
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
    return RPParameters(ml, Jl, r)


def _rp_nactuators_init_state(n: int) -> RPState:
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return RPState(xl, vl, Rl, wl)


def _rp_nactuators_force_trajectory(p: RPParameters, t: float, n: int) -> np.ndarray:
    if n == 3:
        f_ = np.array([0, 0, 1]) * p.ml * constants.g / 3
        k1_d, k2_d = p.ml * constants.g / 5, p.ml * constants.g / 5
        f1_d, f2_d = np.pi / 2, np.pi
        f_ = f_ + np.array(
            [k1_d * np.cos(f1_d * t), k1_d * np.sin(f1_d * t), k2_d * np.sin(f2_d * t)]
        )
        f = np.vstack([f_] * 3).T
    else:
        f = np.random.random((3, n)) * p.ml * constants.g / n
    return f


def main() -> None:
    n = 4
    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn_err_seq = np.empty(t_seq.shape)
    s0 = _rp_nactuators_init_state(n)
    p = _rp_nactuators_parameters(n)
    dyn = RPDynamics(p, s0, dt)
    for i in range(len(t_seq)):
        f = _rp_nactuators_force_trajectory(p, t_seq[i], n)
        acc = dyn.forward_dynamics(f)
        dyn_err_seq[i] = RPDynamics.inverse_dynamics_error(dyn.state, p, f, acc)
        dyn.integrate(f)

    plt.plot(t_seq, dyn_err_seq)
    plt.show()


if __name__ == "__main__":
    main()
