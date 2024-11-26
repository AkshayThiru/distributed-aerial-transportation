import meshcat
import numpy as np
from scipy import constants

from example.setup import rqp_setup
from system.rigid_quadrotor_payload import (RQPDynamics, RQPParameters,
                                            RQPState, RQPVisualizer)


def _rqp_nactuators_force_trajectory(
    p: RQPParameters, s: RQPState, t: float, n: int
) -> np.ndarray:
    f = (1 + np.sin(np.pi * t) / 5) * p.mT * constants.g / n * np.ones((n,))
    M = np.zeros((3, n))
    return (f, M)


def main() -> None:
    n = 3

    vis = meshcat.Visualizer()
    vis.open()

    params, col, s0 = rqp_setup(n)
    visualizer = RQPVisualizer(params, col, vis)

    dt = 5e-3
    t_seq = np.arange(0, 10, dt)
    dyn = RQPDynamics(params, s0, dt)
    for i in range(len(t_seq)):
        w = _rqp_nactuators_force_trajectory(params, dyn.state, t_seq[i], n)
        dyn.integrate(w)
        visualizer.update(dyn.state, w[0], vis)


if __name__ == "__main__":
    main()
