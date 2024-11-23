import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from system.point_mass_rigid_link import (PMRLDynamics, PMRLParameters,
                                          PMRLState)


def _pmrl_nrobot_parameters(n: int) -> PMRLParameters:
    m = np.array([0.5] * n)
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
        r = np.random.random((3, n))
    L = np.array([1.0] * n)
    return PMRLParameters(m, ml, Jl, r, L)


def _pmrl_nrobots_init_state(n: int) -> PMRLState:
    q = np.vstack([np.array([0.0, 0.0, 1.0])] * n).T
    dq = np.vstack([np.array([0.0, 0.0, 0.0])] * n).T
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return PMRLState(q, dq, xl, vl, Rl, wl)


def _pmrl_nrobot_force_trajectory(p: PMRLParameters, t: float, n: int) -> np.ndarray:
    total_mass = p.m.sum() + p.ml
    if n == 3:
        f_ = np.array([0, 0, 1]) * total_mass * constants.g / 3
        k1_d, k2_d = total_mass * constants.g / 5, total_mass * constants.g / 5
        f1_d, f2_d = np.pi / 2, np.pi
        f_ = f_ + np.array(
            [k1_d * np.cos(f1_d * t), k1_d * np.sin(f1_d * t), k2_d * np.sin(f2_d * t)]
        )
        f = np.vstack([f_] * 3).T
    else:
        f = np.random.random((3, n)) * total_mass * constants.g / n
    return f


def main() -> None:
    n = 4
    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn_err_seq = np.empty(t_seq.shape)
    s0 = _pmrl_nrobots_init_state(n)
    p = _pmrl_nrobot_parameters(n)
    dyn = PMRLDynamics(p, s0, dt)
    for i in range(len(t_seq)):
        f = _pmrl_nrobot_force_trajectory(p, t_seq[i], n)
        acc, T = dyn.forward_dynamics(f)
        dyn_err_seq[i] = PMRLDynamics.inverse_dynamics_error(dyn.state, p, f, T, acc)
        dyn.integrate(f)

    plt.plot(t_seq, dyn_err_seq)
    plt.show()


if __name__ == "__main__":
    main()
