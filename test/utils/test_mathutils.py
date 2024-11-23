import numpy as np

from utils.math_utils import (random_cone_vector, rotation_matrix_a_to_b,
                              rotation_matrix_from_z_vector)


def _test_random_cone_vector(N: int = 10000) -> None:
    theta = 89.9 * np.pi / 180
    err = 0
    for _ in range(N):
        v = random_cone_vector(theta)
        err = err + np.maximum(np.cos(theta) - v[2], 0)
    err = err / N
    print(f"(random cone vector) Average error = {err}")


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
    print(f"(rotation matrix from z vector) Average error = {err}")


def _test_rotation_matrix_a_to_b(N: int = 10000) -> None:
    a = np.array([1, 0, 0])
    b = -a
    R = rotation_matrix_a_to_b(a, b)
    rot_err = (
        lambda rot_mat, veca, vecb: np.abs(np.linalg.det(rot_mat) - 1.0)
        + np.linalg.norm(rot_mat @ rot_mat.T - np.eye(3))
        + np.linalg.norm(rot_mat @ veca - vecb)
    )
    print(f"(rotation_matrix_a_to_b) Edge case error: {rot_err(R, a, b)}")

    err = 0.0
    for _ in range(N):
        b = np.random.random((3,))
        normb = np.linalg.norm(b)
        if normb == 0:
            continue
        b = b / normb
        R = rotation_matrix_a_to_b(a, b)
        err += rot_err(R, a, b)
    print(f"(rotation_matrix_a_to_b) Average error: {err / N}")


if __name__ == "__main__":
    _test_random_cone_vector()
    _test_rotation_matrix_from_z_vector()
    _test_rotation_matrix_a_to_b()
