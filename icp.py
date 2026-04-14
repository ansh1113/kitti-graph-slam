from __future__ import annotations

import math

import numpy as np
import open3d as o3d

from utils.geometry_utils import se3_to_se2, se2_to_matrix, matrix_to_se2, normalize_angle


def _to_open3d_cloud(points_xyz: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    return cloud


def _prepare_cloud(points: np.ndarray, voxel_size: float, estimate_normals: bool = False):
    finite = np.isfinite(points).all(axis=1)
    xyz = points[finite, :3]
    if len(xyz) == 0:
        return _to_open3d_cloud(np.zeros((0, 3), dtype=np.float64))
    cloud = _to_open3d_cloud(xyz)
    cloud = cloud.voxel_down_sample(voxel_size=max(voxel_size, 0.05))
    if estimate_normals and len(cloud.points) > 0:
        radius = max(voxel_size * 2.5, 0.25)
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
    return cloud


def run_pairwise_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_size: float = 0.2,
    max_correspondence_distance: float = 1.5,
    max_iterations: int = 60,
) -> np.ndarray:
    """
    TODO(Task 1): Run pairwise ICP between two point clouds using Open3D.

    Open3D ICP finds T such that T @ source_pts ≈ target_pts.
    So T maps source-frame points into target-frame coordinates,
    i.e.  T = T_{target <- source}.

    Inputs:
        source_points (N, 3): Source point cloud.
        target_points (M, 3): Target point cloud.
        voxel_size: Voxel size for downsampling.
        max_correspondence_distance: Maximum correspondence distance.
        max_iterations: Maximum number of iterations.

    Returns:
        np.ndarray (3,): SE(2) of result.transformation  [dx, dy, dtheta].
    """

    source = _prepare_cloud(source_points, voxel_size, estimate_normals=False)
    target = _prepare_cloud(target_points, voxel_size, estimate_normals=False)
    if len(source.points) == 0 or len(target.points) == 0:
        raise ValueError("ICP received an empty source or target cloud after filtering/downsampling.")

    # ======= STUDENT TODO START (edit only inside this block) =======

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    # ======= STUDENT TODO END (do not change code outside this block) =======

    rel_se3 = np.asarray(result.transformation, dtype=np.float64)
    rel_se2 = se3_to_se2(rel_se3)
    return rel_se2


def compute_icp_chains(
    static_points: list[np.ndarray],
    voxel_size: float,
    max_correspondence_distance: float,
    max_iterations: int,
) -> dict[str, np.ndarray]:
    """
    TODO(Task 1): Compute ICP chains and motion edges between all frames.

    Key insight:
        ICP(source=frame_t, target=frame_{t+1}) returns T such that
        T @ pts_t ≈ pts_{t+1}, meaning T = T_{t+1 <- t}
        (maps frame-t points INTO frame-t+1 coordinates).

    To get the world-frame pose of t+1:
        T_{world <- t+1} = T_{world <- t} @ T_{t <- t+1}
                         = M_world_t @ inv(T_{t+1 <- t})
                         = M_world_t @ inv(M_rel)

    Motion edges store [t, t+1, dx, dy, dtheta] where [dx, dy, dtheta]
    is the relative motion FROM t TO t+1 expressed in t's LOCAL frame.
    For the slam_factors residual  r = x_j - (x_i ⊕ u)  to be zero when
    x_j = x_i ⊕ u, we need u = inv(M_rel) expressed as SE(2),
    because inv(T_{t+1<-t}) = T_{t <- t+1} = "where is t+1 seen from t" = forward motion.
    """

    num_frames = len(static_points)
    edges: list[tuple[int, int, float, float, float]] = []

    # ======= STUDENT TODO START (edit only inside this block) =======

    icp_poses = np.zeros((num_frames, 3), dtype=np.float64)
    # icp_poses[0] = [0,0,0]  — world IS frame 0

    M_world = se2_to_matrix(icp_poses[0])  # 3x3 identity at t=0

    for t in range(num_frames - 1):
        # ICP: source=frame t, target=frame t+1
        # result = T_{t+1 <- t}  (maps t's pts into t+1's frame)
        rel_se2 = run_pairwise_icp(
            source_points=static_points[t],
            target_points=static_points[t + 1],
            voxel_size=voxel_size,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
        )

        M_rel = se2_to_matrix(rel_se2)          # T_{t+1 <- t}
        M_rel_inv = np.linalg.inv(M_rel)         # T_{t <- t+1}  =  forward motion in t's frame

        # Pose chaining: T_{world <- t+1} = T_{world <- t} @ T_{t <- t+1}
        M_world = M_world @ M_rel_inv
        icp_poses[t + 1] = matrix_to_se2(M_world)

        # Motion edge: store the forward relative motion (what slam_factors expects as u)
        # u = [dx, dy, dtheta] such that x_{t+1} = x_t ⊕ u
        # This is exactly M_rel_inv expressed as SE(2)
        u = matrix_to_se2(M_rel_inv)
        edges.append((float(t), float(t + 1), u[0], u[1], u[2]))

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return {
        "motion_edges": np.array(edges, dtype=np.float64),
        "trajectory": icp_poses,
    }