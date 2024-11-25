import numpy as np

from system.rigid_payload import RPCollision, RPParameters, RPState
from system.rigid_quadrotor_payload import (RQPCollision, RQPParameters,
                                            RQPState)


# Rigid-payload system setup.
def _rp_nactuators_parameters(n: int) -> RPParameters:
    if n == 3:
        ml = 0.225
        Jl = np.diag([2.1, 1.87, 3.97]) * 1e-2
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


# Rigid-quadrotor-payload system setup.
def _rqp_nquadrotors_parameters(n: int) -> RQPParameters:
    if n == 3:
        m = np.array([0.5] * 3)
        Jq = np.diag([2.32, 2.32, 4]) * 1e-3
        J = np.empty((3, 3, n))
        for i in range(n):
            J[:, :, i] = Jq
        ml = 0.225
        Jl = np.diag([2.1, 1.87, 3.97]) * 1e-2
        r = np.vstack(
            [
                np.array([-0.42, -0.27, 0]),
                np.array([0.48, -0.27, 0]),
                np.array([-0.06, 0.55, 0]),
            ]
        ).T
    else:
        raise NotImplementedError
    return RQPParameters(m, J, ml, Jl, r)


def _rqp_collision() -> RQPCollision:
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
    return RQPCollision(payload_vertices, payload_mesh_vertices)


def _rqp_nquadrotors_init_state(n: int) -> RQPState:
    R = np.empty((3, 3, n))
    for i in range(n):
        R[:, :, i] = np.eye(3)
    w = np.zeros((3, n))
    xl = np.array([0.0, 0.0, 0.0])
    vl = np.array([0.0, 0.0, 0.0])
    Rl = np.eye(3)
    wl = np.array([0.0, 0.0, 0.0])
    return RQPState(R, w, xl, vl, Rl, wl)


def rqp_setup(n: int) -> tuple[RQPParameters, RQPCollision, RQPState]:
    return (
        _rqp_nquadrotors_parameters(n),
        _rqp_collision(),
        _rqp_nquadrotors_init_state(n),
    )
