import numpy as np
from scipy.linalg import polar


def random_cone_vector(theta: float) -> np.ndarray:
    """Returns a uniform random unit vector within theta angle around (0,0,1)."""
    assert (theta < 89.99 * np.pi / 180) and (theta > 0)
    R = np.tan(theta)
    r = R * np.sqrt(np.random.random())
    phi = 2 * np.pi * np.random.random()
    v = np.array([r * np.cos(phi), r * np.sin(phi), 1])
    return v / np.linalg.norm(v)


def rotation_matrix_from_z_vector(q: np.ndarray, project: bool = False) -> np.ndarray:
    """Returns a rotation matrix R such that R e_3 = q with 0 yaw (in ZYX).
    Inputs:
    q:  (3, n) ndarray, q[:,i] is a unit vector with q[2,i] > 0.
    Outputs:
    R:  (3, 3, n) ndarray, R[:,:,i] is the rotation matrix corresponding to q[:,i].
    """
    assert q.shape[0] == 3 and q.ndim == 2
    assert all(np.abs(np.linalg.norm(q, axis=0) - 1) < 1e-5)
    assert all(q[2, :] > 0)
    n = q.shape[1]
    R = np.zeros((3, 3, n))
    R[:, 2, :] = q
    sin_x = -q[1, :]
    cos_x = np.sqrt(1 - sin_x**2)
    assert all(cos_x > 0)
    sin_y = q[0, :] / cos_x
    cos_y = q[2, :] / cos_x
    R[0, 0, :] = cos_y
    R[2, 0, :] = -sin_y
    R[0, 1, :] = sin_x * sin_y
    R[1, 1, :] = cos_x
    R[2, 1, :] = cos_y * sin_x
    if project:
        for i in range(n):
            R[:, :, i], _ = polar(R[:, :, i])
    return R


def rotation_matrix_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns a rotation matrix to rotate a to b.
    a and b must be unit vectors.
    """
    u = a + b
    normsq = np.dot(u, u)
    if normsq == 0.0:
        e1 = np.array([1, 0, 0])
        u = np.cross(a, e1)
        normsq = np.dot(u, u)
        if normsq == 0.0:
            e2 = np.array([0, 1, 0])
            u = np.cross(a, e2)
            normsq = np.dot(u, u)
            assert normsq > 0.0
    return 2 * np.outer(u, u) / normsq - np.eye(3)
