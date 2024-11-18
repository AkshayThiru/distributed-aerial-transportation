import numpy as np
from utils.math_utils import random_cone_vector, rotation_matrix_from_z_vector


def _test_random_cone_vector(N: int = 10000) -> None:
    theta = 89.9 * np.pi / 180
    err = 0
    for _ in range(N):
        v = random_cone_vector(theta)
        err = err + np.maximum(np.cos(theta) - v[2], 0)
    err = err / N
    print(f"Average error = {err}")


def _test_rotation_matrix_from_z_vector(N: int = 10000) -> None:
    theta = 89.9 * np.pi / 180
    q = np.empty((3, N))
    for i in range(N):
        q[:, i] = random_cone_vector(theta)
    R = rotation_matrix_from_z_vector(q)
    err = 0
    for i in range(N):
        err = err + np.linalg.norm(R[:, :, i].T @ R[:, :, i] - np.eye(3))
    err = err / N
    print(f"Average error in rotation = {err}")


if __name__ == "__main__":
    _test_random_cone_vector()
    _test_rotation_matrix_from_z_vector()
