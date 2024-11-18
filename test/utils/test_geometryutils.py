import meshcat
import meshcat.geometry as geom
import numpy as np
from utils.geometry_utils import faces_from_vertex_rep, mesh_from_halfspace_rep


def _visualize_polytope(vertices: np.ndarray, faces: np.ndarray) -> None:
    vis = meshcat.Visualizer()
    mesh = geom.TriangularMeshGeometry(vertices, faces)
    mesh_material = geom.MeshBasicMaterial(
        color=0xFF22DD, wireframe=False, linewidth=2, opacity=0.75
    )
    vis["polytope"].set_object(mesh, mesh_material)
    vis.open()


def _test_faces_from_vertex_rep(N: int = 10000) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = faces_from_vertex_rep(vertices)
    # _visualize_polytope(vertices, faces)


def _test_mesh_from_halfspace_rep(N: int = 10000) -> None:
    A = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]]
    )
    b = np.array([0.0, 0.0, 0.0, 3.0])
    vertices, faces = mesh_from_halfspace_rep(A, b)
    # _visualize_polytope(vertices, faces)


if __name__ == "__main__":
    _test_faces_from_vertex_rep()
    _test_mesh_from_halfspace_rep()
