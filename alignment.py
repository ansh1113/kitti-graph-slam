from __future__ import annotations

import numpy as np
from utils.geometry_utils import MapSpec


def world_to_pixel(xy: np.ndarray, map_spec: MapSpec) -> np.ndarray:
    """TODO(Task 0): Convert world coordinates to pixel coordinates.

    Inputs:
        xy: (N, 2) array: Points in world (first frame) coordinates [x, y].
        map_spec: Map specification.
        - map_spec.x_min: Minimum x-coordinate of the map.
        - map_spec.y_max: Maximum y-coordinate of the map.
        - map_spec.resolution: Resolution of the map.

    Returns:
        (N, 2) array: Points in pixel coordinates [row, col].
    """

    # ======= STUDENT TODO START (edit only inside this block) =======

    # cols: shift x so that x_min maps to column 0, then divide by resolution
    cols = np.floor((xy[:, 0] - map_spec.x_min) / map_spec.resolution).astype(np.int64)

    # rows: image y-axis is flipped — y_max is row 0, so subtract y from y_max
    rows = np.floor((map_spec.y_max - xy[:, 1]) / map_spec.resolution).astype(np.int64)

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return np.stack([rows, cols], axis=1)


def rasterize_topdown(points_world: list[np.ndarray], map_spec: MapSpec) -> dict[str, np.ndarray]:
    """TODO(Task 0): Rasterize points into a top-down map.

    Inputs:
        points_world: List of points in world (first frame) coordinates.
        map_spec: Map specification.

    Returns:
        Dictionary containing density map.
    """
    shape = (map_spec.height, map_spec.width)
    density = np.zeros(shape, dtype=np.float32)

    for pts in points_world:
        if pts.size == 0:
            continue

        xy = pts[:, :2].astype(np.float64, copy=False)

        # ======= STUDENT TODO START (edit only inside this block) =======

        # Convert world coordinates to pixel coordinates
        rc = world_to_pixel(xy, map_spec)
        rows = rc[:, 0]
        cols = rc[:, 1]

        # Filter out points that fall outside the image bounds
        valid = (rows >= 0) & (rows < map_spec.height) & \
                (cols >= 0) & (cols < map_spec.width)
        rows = rows[valid]
        cols = cols[valid]

        # ======= STUDENT TODO END (do not change code outside this block) =======

        np.add.at(density, (rows, cols), 1.0)

    return {
        "density": density,
    }


def build_accumulated_map(static_points: list[np.ndarray], poses_se2: np.ndarray) -> list[np.ndarray]:
    """TODO(Task 0): Build accumulated map.

    Inputs:
        static_points: List of point clouds in current frame coordinates.
        poses_se2: List of SE(2) poses [x, y, theta] — each is T_world_lidar[t],
                   i.e. the transform that maps points from frame t into world frame.

    Returns:
        List of accumulated points in world (first frame) coordinates.
    """

    # ======= STUDENT TODO START (edit only inside this block) =======

    world_points = []

    for t, pts in enumerate(static_points):
        if pts.size == 0:
            world_points.append(pts)
            continue

        x, y, theta = poses_se2[t]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Apply SE(2) point transform: p_world = R(theta) * p_local + t
        xy_local = pts[:, :2].astype(np.float64, copy=False)
        x_world = cos_t * xy_local[:, 0] - sin_t * xy_local[:, 1] + x
        y_world = sin_t * xy_local[:, 0] + cos_t * xy_local[:, 1] + y

        # Build output — preserve z and any extra columns as-is
        pts_world = pts.astype(np.float64).copy()
        pts_world[:, 0] = x_world
        pts_world[:, 1] = y_world

        world_points.append(pts_world)

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return world_points


def accumulate_and_rasterize(static_points: list[np.ndarray], poses_se2: np.ndarray, map_spec: MapSpec) -> dict[str, np.ndarray]:
    world_points = build_accumulated_map(static_points, poses_se2)
    return rasterize_topdown(world_points, map_spec)