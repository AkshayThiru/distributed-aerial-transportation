from typing import Any

import hppfcl
import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
from hppfcl import Transform3f
from meshcat import Visualizer

from utils.math_utils import rotation_matrix_a_to_b

_GRASS_MATERIAL = gm.MeshLambertMaterial(
    color=0x70AB94, wireframe=False, opacity=1.0, reflectivity=0.5
)
_BARK_MATERIAL = gm.MeshLambertMaterial(
    color=0x694B37, wireframe=False, opacity=0.4, reflectivity=0.5
)
_TREECONE_MATERIAL = gm.MeshLambertMaterial(
    color=0x5F926A, wireframe=False, opacity=0.1, reflectivity=0.5
)

_MOUNTAIN_CENTER = np.array([30.0, 0.0])  # [m].
_MOUNTAIN_RADIUS = 25.0  # [m].
_MOUNTAIN_HEIGHT = 7.5  # [m]. 0 < (.) << _MOUNTAIN_RADIUS.
_BARK_HEIGHT = 4.0  # [m].
_BARK_RADIUS = 0.3  # [m].
_TREECONE_RADIUS = 2.0  # [m].
_TREECONE_HEIGHT = 4.0  # [m].
_GRASS_HEIGHT = 0.1  # [m].
_MIN_DIST_BETWEEN_TREES = 3.2  # [m].
_MAX_TREES = 200

_DISPLAY_TREECONES = True


class Forest:
    def __init__(self) -> None:
        self._set_constants()
        self._generate_trees()
        self._set_collision_data()

    def _set_constants(self) -> None:
        self.mountain_center = _MOUNTAIN_CENTER
        self.mountain_radius = _MOUNTAIN_RADIUS
        self.bark_radius = _BARK_RADIUS

    def _generate_trees(self) -> None:
        np.random.rand(1)

        max_tries = _MAX_TREES * 50
        tree_xy = _MOUNTAIN_CENTER + np.array([0.5, 0.5])
        tree_xy = tree_xy.reshape((1, 2))
        self.num_trees = 1
        if self.num_trees >= _MAX_TREES:
            return

        for _ in range(max_tries):
            pos = np.random.random((2,)) - 0.5
            norm = np.linalg.norm(pos)
            if norm == 0:
                continue
            radius = np.random.random()
            pos = pos / norm * radius * _MOUNTAIN_RADIUS + _MOUNTAIN_CENTER
            min_dist = np.min(np.linalg.norm(tree_xy - pos, axis=1))
            if min_dist < _MIN_DIST_BETWEEN_TREES:
                continue
            tree_xy = np.vstack((tree_xy, pos))
            self.num_trees += 1
            if self.num_trees >= _MAX_TREES:
                break
        self.tree_pos = np.empty((self.num_trees, 3))
        self.tree_pos[:, :2] = tree_xy

        ang = np.pi / 2.0 - np.arctan2(_MOUNTAIN_RADIUS, _MOUNTAIN_HEIGHT)
        assert ang > 0.0
        self.mountain_sphere_radius = _MOUNTAIN_RADIUS / np.sin(ang)
        self.mountain_center_depth = self.mountain_sphere_radius * np.cos(ang)
        for i in range(self.num_trees):
            pos = self.tree_pos[i, :2] - _MOUNTAIN_CENTER
            norm2 = np.dot(pos, pos)
            height_from_ground = (
                np.sqrt(self.mountain_sphere_radius**2 - norm2)
                - self.mountain_center_depth
            )
            self.tree_pos[i, 2] = (height_from_ground + _BARK_HEIGHT) / 2.0

    def _set_collision_data(self) -> None:
        self.tree = hppfcl.Cylinder(_BARK_RADIUS, _BARK_HEIGHT)

    def visualize_env(self, vis: Visualizer) -> None:
        # Draw forest.
        base_rot = rotation_matrix_a_to_b(
            np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        for i in range(self.num_trees):
            bark = "bark_" + str(i)
            vis[bark].set_object(
                gm.Cylinder(height=self.tree_pos[i, 2] * 2.0, radius=_BARK_RADIUS),
                _BARK_MATERIAL,
            )
            T = tf.translation_matrix(self.tree_pos[i, :])
            T[:3, :3] = base_rot
            vis[bark].set_transform(T)
            if _DISPLAY_TREECONES:
                treecone = "treecone_" + str(i)
                vis[treecone].set_object(
                    gm.Cylinder(
                        height=_TREECONE_HEIGHT,
                        radiusBottom=_TREECONE_RADIUS,
                        radiusTop=0.0,
                    ),
                    _TREECONE_MATERIAL,
                )
                T[2, 3] += self.tree_pos[i, 2] + _TREECONE_HEIGHT / 2.0
                vis[treecone].set_transform(T)
        # Draw ground.
        ground = "ground"
        vis[ground].set_object(
            gm.Box([10 * _MOUNTAIN_RADIUS, 10 * _MOUNTAIN_RADIUS, _GRASS_HEIGHT]),
            _GRASS_MATERIAL,
        )
        pos = np.array([_MOUNTAIN_CENTER[0], _MOUNTAIN_CENTER[1], -0.0])
        T = tf.translation_matrix(pos)
        vis[ground].set_transform(T)
        # Draw mountain.
        mountain = "mountain"
        vis[mountain].set_object(
            gm.Sphere(self.mountain_sphere_radius + _GRASS_HEIGHT), _GRASS_MATERIAL
        )
        pos = np.array(
            [_MOUNTAIN_CENTER[0], _MOUNTAIN_CENTER[1], -self.mountain_center_depth]
        )
        T = tf.translation_matrix(pos)
        vis[mountain].set_transform(T)

        # Disable grid.
        vis["/Grid"].set_property("visible", False)

    def centralized_distance(
        self, obj: Any, tf1: Transform3f, vision_radius: float
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        dist_req = hppfcl.DistanceRequest()
        dist_res = hppfcl.DistanceResult()
        tf2 = hppfcl.Transform3f()

        min_dists = []
        nearest_pts1, nearest_pts2 = [], []
        collision = False
        for i in range(self.num_trees):
            pos = self.tree_pos[i, :]
            if (
                np.linalg.norm(tf1.getTranslation() - pos)
                > vision_radius + _BARK_RADIUS
            ):
                continue
            tf2.setTranslation(pos)
            min_dist = hppfcl.distance(obj, tf1, self.tree, tf2, dist_req, dist_res)
            if min_dist < 1e-4:
                collision = True
            min_dists.append(min_dist)
            nearest_pts1.append(dist_res.getNearestPoint1())
            nearest_pts2.append(dist_res.getNearestPoint2())
        min_dists = np.array(min_dists)
        nearest_pts1 = np.array(nearest_pts1)
        nearest_pts2 = np.array(nearest_pts2)

        return collision, min_dists, nearest_pts1, nearest_pts2

    def distributed_distance(
        self,
        obj: Any,
        tf1: Transform3f,
        vision_radius: float,
        dir: np.ndarray,
        camera_pos: np.ndarray,
        ang: float,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        dist_req = hppfcl.DistanceRequest()
        dist_res = hppfcl.DistanceResult()
        tf2 = hppfcl.Transform3f()

        assert (dir.ndim == 2) and (camera_pos.ndim == 2)
        cos_ang = np.cos(ang)

        min_dists = []
        nearest_pts1, nearest_pts2 = [], []
        collision = False
        for i in range(self.num_trees):
            pos = self.tree_pos[i, :]
            if (
                np.linalg.norm(tf1.getTranslation() - pos)
                > vision_radius + _BARK_RADIUS
            ):
                continue
            dir_tree = pos[:2] - camera_pos
            norm = np.linalg.norm(dir_tree)
            if norm > 0:
                dir_tree = dir_tree / norm
                if np.dot(dir_tree, dir) < cos_ang:
                    continue
            tf2.setTranslation(pos)
            min_dist = hppfcl.distance(obj, tf1, self.tree, tf2, dist_req, dist_res)
            if min_dist < 1e-4:
                collision = True
            min_dists.append(min_dist)
            nearest_pts1.append(dist_res.getNearestPoint1())
            nearest_pts2.append(dist_res.getNearestPoint2())
        min_dists = np.array(min_dists)
        nearest_pts1 = np.array(nearest_pts1)
        nearest_pts2 = np.array(nearest_pts2)

        return collision, min_dists, nearest_pts1, nearest_pts2


def main() -> None:
    env = Forest()

    vis = Visualizer()
    vis.open()

    env.visualize_env(vis)


if __name__ == "__main__":
    main()
