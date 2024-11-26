import meshcat
import numpy as np
from scipy import constants

from example.setup import pmrl_setup
from system.point_mass_rigid_link import (PMRLDynamics, PMRLParameters,
                                          PMRLState, PMRLVisualizer)


def _pmrl_nactuators_force_trajectory(
    p: PMRLParameters, s: PMRLState, t: float, n: int
) -> np.ndarray:
    total_mass = p.m.sum() + p.ml
    f_ = np.array(
        [
            np.sin(np.pi * t) * total_mass * constants.g / (15 * n),
            np.cos(np.pi * t) * total_mass * constants.g / (9 * n),
            total_mass * constants.g * (1.0 / n + np.sin(np.pi * t) / (9 * n)),
        ]
    )
    return np.vstack([f_] * n).T


def main() -> None:
    n = 3

    vis = meshcat.Visualizer()
    vis.open()

    params, col, s0 = pmrl_setup(n)
    visualizer = PMRLVisualizer(params, col, vis)

    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn = PMRLDynamics(params, s0, dt)
    for i in range(len(t_seq)):
        f = _pmrl_nactuators_force_trajectory(params, dyn.state, t_seq[i], n)
        dyn.integrate(f)
        visualizer.update(dyn.state, f, vis)


if __name__ == "__main__":
    main()
