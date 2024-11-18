import numpy as np
import polytope as pc
from scipy.spatial import ConvexHull


def faces_from_vertex_rep(vertices: np.ndarray) -> np.ndarray:
    """Returns the face indices of a polytope from its vertex representation.
    Inputs:
    vertices: (n, 3) ndarray of vertices in 3D.
    Outputs:
    faces:    (n, 3) ndarray of face indices.
    """
    assert vertices.shape[1] == 3
    hull = ConvexHull(vertices)
    return hull.simplices


def mesh_from_halfspace_rep(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the vertices and face indices of a polytope from its halfspace representation.
    Inputs:
    A:        (m, 3) ndarray of halfspace normals,
    b:        (m,) ndarray of halfspace intercepts.
    Outputs:
    vertices: (n, 3) ndarray of vertices,
    faces:    (n, 3) ndarray of face indices.
    """
    assert A.shape[1] == 3
    assert b.ndim == 1
    assert A.shape[0] == b.shape[0]
    p = pc.Polytope(A, b)
    vertices = pc.extreme(p)
    faces = faces_from_vertex_rep(vertices)
    return (vertices, faces)
