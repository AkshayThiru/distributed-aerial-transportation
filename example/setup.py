import numpy as np

from system.rigid_payload import RPCollision, RPParameters, RPState


def _rp_nactuators_parameters(n: int) -> RPParameters:
    ml = 0.225
    Jl = np.diag([2.1, 1.87, 3.97]) * 1e-2
    if n == 3:
        r = np.array(
            [
                [-0.42, -0.27, 0],
                [0.48, -0.27, 0],
                [-0.06, 0.55, 0],
            ]
        ).T
    else:
        raise NotImplementedError
    return RPParameters(ml, Jl, r)


def _rp_collision() -> RPCollision:
    payload_vertices = np.array(
        [
            [-0.42, -0.27, 0],
            [0.48, -0.27, 0],
            [-0.06, 0.55, 0],
            [-0.42, -0.27, -0.1],
            [0.48, -0.27, -0.1],
            [-0.06, 0.55, -0.1],
        ]
    )
    payload_mesh_vertices = np.array(
        [
            [-0.52, -0.37, 0.1],
            [0.58, -0.37, 0.1],
            [-0.06, 0.65, 0.1],
            [-0.52, -0.37, -0.2],
            [0.58, -0.37, -0.2],
            [-0.06, 0.65, -0.2],
        ]
    )
    return RPCollision(payload_vertices, payload_mesh_vertices)


def _rp_init_state() -> RPState:
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return RPState(xl, vl, Rl, wl)


def rp_setup(n: int) -> tuple[RPParameters, RPCollision, RPState]:
    return (_rp_nactuators_parameters(n), _rp_collision(), _rp_init_state())
