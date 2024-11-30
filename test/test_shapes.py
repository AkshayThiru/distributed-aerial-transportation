import os

import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
from meshcat import Visualizer

from utils.geometry_utils import faces_from_vertex_rep
from utils.math_utils import rotation_matrix_a_to_b

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
_QUADROTOR_OBJ_PATH = _DIR_PATH + "/../objs/quadrotor.obj"
_SHAPE_MATERIAL = gm.MeshLambertMaterial(
    color=0x47A7B8, wireframe=False, opacity=0.75, reflectivity=0.5
)
_OBJ_MATERIAL = gm.MeshLambertMaterial(
    color=0xFF22DD, wireframe=False, opacity=0.75, reflectivity=0.5
)
_PC_COLOR = [255 / 255, 34 / 255, 221 / 255]

_SIDE_LEN = 5


def _draw_spheres(vis: Visualizer, nobjs: int) -> None:
    for i in range(nobjs):
        obj = "sphere_" + str(i)
        radius = np.random.random() * 0.5 + 0.25  # [m].
        vis[obj].set_object(
            gm.Sphere(radius),
            _SHAPE_MATERIAL,
        )
        pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
        pos[2] = radius / 2.0
        T = tf.translation_matrix(pos)
        vis[obj].set_transform(T)


def _draw_cylinders(vis: Visualizer, nobjs: int) -> None:
    base_rot = rotation_matrix_a_to_b(
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    for i in range(nobjs):
        obj = "cylinder_" + str(i)
        height = np.random.random() + 1.0  # [m].
        radius_bottom = np.random.random() + 0.1  # [m].
        radius_top = np.random.random() * 0.1  # [m].
        vis[obj].set_object(
            gm.Cylinder(
                height=height, radiusBottom=radius_bottom, radiusTop=radius_top
            ),
            _SHAPE_MATERIAL,
        )
        pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
        pos[2] = height / 2.0
        T = tf.translation_matrix(pos)
        T[:3, :3] = base_rot
        vis[obj].set_transform(T)


def _draw_boxes(vis: Visualizer, nobjs: int) -> None:
    for i in range(nobjs):
        obj = "box_" + str(i)
        length = np.random.random()  # [m].
        width = np.random.random()  # [m].
        height = np.random.random() + 0.5  # [m].
        vis[obj].set_object(
            gm.Box([length, width, height]),
            _SHAPE_MATERIAL,
        )
        pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
        pos[2] = height / 2.0
        T = tf.translation_matrix(pos)
        vis[obj].set_transform(T)


def _draw_polytopes(vis: Visualizer, nobjs: int) -> None:
    for i in range(nobjs):
        obj = "polytope_" + str(i)
        vertices = (np.random.random((10, 3)) - 0.5) * 2.0
        faces = faces_from_vertex_rep(vertices)
        vis[obj].set_object(
            gm.TriangularMeshGeometry(vertices, faces),
            _SHAPE_MATERIAL,
        )
        pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
        pos[2] = 0.0
        T = tf.translation_matrix(pos)
        vis[obj].set_transform(T)


def _draw_obj(vis: Visualizer) -> None:
    obj = "obj"
    vis[obj].set_object(
        gm.ObjMeshGeometry.from_file(_QUADROTOR_OBJ_PATH),
        _OBJ_MATERIAL,
    )
    pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
    pos[2] = 3.0
    T = tf.translation_matrix(pos)
    vis[obj].set_transform(T)


def _draw_point_cloud(vis: Visualizer) -> None:
    obj = "pc"
    npts = 100000
    verts = np.random.rand(3, npts)
    vis[obj].set_object(
        gm.PointCloud(verts, np.array([_PC_COLOR] * npts).T),
    )
    pos = (np.random.random((3,)) - 0.5) * _SIDE_LEN
    pos[2] = 3.0
    T = tf.translation_matrix(pos)
    vis[obj].set_transform(T)


def main() -> None:
    np.random.rand(1)

    vis = Visualizer()
    vis.open()

    nobjs_per_type = 5
    _draw_spheres(vis, nobjs_per_type)
    _draw_cylinders(vis, nobjs_per_type)
    _draw_boxes(vis, nobjs_per_type)
    _draw_polytopes(vis, nobjs_per_type)
    _draw_obj(vis)
    _draw_point_cloud(vis)

    # Camera position.
    vis.set_cam_pos(np.array([-7.0, -7.0, 6.0]))
    vis.set_cam_target(np.array([0.0, 0.0, 0.0]))


if __name__ == "__main__":
    main()
