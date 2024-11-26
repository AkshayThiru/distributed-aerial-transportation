import meshcat
import numpy as np

from control.rp_centralized import RPCentralizedController
from example.setup import rp_setup
from system.rigid_payload import RPDynamics, RPState, RPVisualizer


def _desired_acceleration(s: RPState, t: float) -> np.ndarray:
    radius = 1.0
    height = 1.0
    freq = 1 / 2.0
    x_ref = np.array([radius * np.cos(freq * t), radius * np.sin(freq * t), height])
    v_ref = np.array(
        [-radius * freq * np.sin(freq * t), radius * freq * np.cos(freq * t), 0.0]
    )
    a_ref = np.array(
        [
            -radius * freq**2 * np.cos(freq * t),
            -radius * freq**2 * np.sin(freq * t),
            0.0,
        ]
    )
    k_p = 1.0
    k_v = 1.0
    dvl_des = a_ref - k_v * (s.vl - v_ref) - k_p * (s.xl - x_ref)

    dwl_des = np.array([np.sin(t), np.cos(t), np.pi / 12])

    acc_des = (dvl_des, dwl_des)
    return acc_des


def main() -> None:
    n = 3
    dt = 10e-3

    vis = meshcat.Visualizer()
    vis.open()

    params, col, s0 = rp_setup(n)
    controller = RPCentralizedController(params, col, s0, dt, verbose=True)
    visualizer = RPVisualizer(params, col, vis)

    t_seq = np.arange(0, 20, dt)
    dyn = RPDynamics(params, s0, dt)
    for i in range(len(t_seq)):
        acc_des = _desired_acceleration(dyn.state, t_seq[i])
        f = controller.control(dyn.state, acc_des)
        dyn.integrate(f)
        visualizer.update(dyn.state, f, vis)


if __name__ == "__main__":
    main()
