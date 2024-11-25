import meshcat
import numpy as np
from scipy import constants

from example.setup import rp_setup
from system.rigid_payload import (RPDynamics, RPParameters, RPState,
                                  RPVisualizer)


def _rp_nactuators_force_trajectory(
    p: RPParameters, s: RPState, t: float, n: int
) -> np.ndarray:
    f_ = np.array(
        [
            np.sin(np.pi * t) * p.ml * constants.g / (15 * n),
            np.cos(np.pi * t) * p.ml * constants.g / (9 * n),
            p.ml * constants.g / n + np.sin(np.pi * t) * p.ml * constants.g / (9 * n),
        ]
    )
    return np.vstack([f_] * n).T


def main() -> None:
    n = 3

    vis = meshcat.Visualizer()
    vis.open()

    params, col, s0 = rp_setup(n)
    visualizer = RPVisualizer(params, col, vis)

    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn = RPDynamics(params, s0, dt)
    for i in range(len(t_seq)):
        f = _rp_nactuators_force_trajectory(params, dyn.state, t_seq[i], n)
        dyn.integrate(f)
        visualizer.update(dyn.state, f, vis)


if __name__ == "__main__":
    main()
