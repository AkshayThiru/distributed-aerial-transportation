from time import perf_counter, sleep

import meshcat.geometry as gm
import meshcat.transformations as tf
import numpy as np
from meshcat import Visualizer
from tqdm import tqdm

from utils.math_utils import rotation_matrix_a_to_b

_ROBOT_MATERIAL = gm.MeshLambertMaterial(
    color=0xFF22DD, wireframe=False, opacity=1, reflectivity=0.5
)
_OBJ_MATERIAL = gm.MeshLambertMaterial(
    color=0x47A7B8, wireframe=False, opacity=0.75, reflectivity=0.5
)

_CAMERA = "/Cameras/default/rotated/<object>"

_ROBOT_TRAJ_HEIGHT = 3.0  # [m].
_ROBOT_TRAJ_RADIUS = 2.5  # [m].


def _setup_objects(vis: Visualizer) -> None:
    np.random.rand(1)

    # Cone static objects.
    nobjs = 20
    side_len = 10  # [m].
    base_rot = rotation_matrix_a_to_b(
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    for i in range(nobjs):
        obj = "obj_" + str(i)
        height = np.random.random() + 1.0  # [m].
        radius = np.random.random() + 0.1  # [m].
        vis[obj].set_object(
            gm.Cylinder(height=height, radiusBottom=radius, radiusTop=0.0),
            _OBJ_MATERIAL,
        )
        pos = (np.random.random((3,)) - 0.5) * side_len
        pos[2] = height / 2.0
        T = tf.translation_matrix(pos)
        T[:3, :3] = base_rot
        vis[obj].set_transform(T)

    # Robot object.
    vis["robot"].set_object(gm.Sphere(0.5), _ROBOT_MATERIAL)
    vis["robot"].set_transform(
        tf.translation_matrix(np.array([0.0, 0.0, _ROBOT_TRAJ_HEIGHT]))
    )

    # Camera position.
    vis.set_cam_pos(np.array([-7.0, -7.0, 6.0]))


def _set_robot_camera_positions(vis: Visualizer, t: float) -> None:
    # Robot position.
    robot_pos = np.array(
        [
            _ROBOT_TRAJ_RADIUS * np.cos(t),
            _ROBOT_TRAJ_RADIUS * np.sin(t),
            _ROBOT_TRAJ_HEIGHT,
        ]
    )
    vis["robot"].set_transform(tf.translation_matrix(robot_pos))

    # Camera target (robot).
    # See:
    #   https://github.com/meshcat-dev/meshcat?tab=readme-ov-file#camera-control,
    #   https://github.com/meshcat-dev/meshcat-python/blob/785bc9d5ba6f8a8bb79ee8b25f523805946c1fbd/src/meshcat/visualizer.py#L160.
    vis.set_cam_target(robot_pos)


def main() -> None:
    T = 20.0  # [s].
    dt = 10e-3  # [s].
    t_seq = np.arange(0.0, T, dt)
    min_fps: int = 24

    vis = Visualizer()
    vis.open()

    _setup_objects(vis)

    vis_rel_freq = int(np.floor(1.0 / (min_fps * dt)))
    spf = dt * vis_rel_freq
    start_time = perf_counter()
    for i in tqdm(range(len(t_seq))):
        ###
        # Dynamics code.
        ###
        if i % vis_rel_freq == 0:
            # Visualization update.
            _set_robot_camera_positions(vis, t_seq[i])
            # Sleep for appropriate time.
            stop_time = perf_counter()
            elapsed_time = stop_time - start_time
            if elapsed_time < spf:
                sleep(spf - elapsed_time)
            start_time = perf_counter()


if __name__ == "__main__":
    main()
