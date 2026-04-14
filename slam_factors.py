from __future__ import annotations

import math

import numpy as np

from utils.geometry_utils import normalize_angle


def motion_error_and_jacobians(
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    T_ji: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO(Task 2.1): Computes the motion residual and Jacobians.

    Forward kinematics (SE(2) composition):
        f(x_i, T_ji) = x_i ⊕ T_ji
                     = [x_i + cos(θ_i)·dx - sin(θ_i)·dy,
                        y_i + sin(θ_i)·dx + cos(θ_i)·dy,
                        θ_i + dθ]

    Residual:
        r = x_j - f(x_i, T_ji)
          = [x_j - (x_i + cos(θ_i)·dx - sin(θ_i)·dy),
             y_j - (y_i + sin(θ_i)·dx + cos(θ_i)·dy),
             normalize(θ_j - (θ_i + dθ))]

    Analytical Jacobians (shape 3×3 each):

        J_i = ∂r/∂x_i =
            [[-1,  0,   dx·sin(θ_i) + dy·cos(θ_i)],
             [ 0, -1,  -dx·cos(θ_i) + dy·sin(θ_i)],
             [ 0,  0,  -1                          ]]

        J_j = ∂r/∂x_j =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]

    Inputs:
        pose_i (3,): Pose i [x, y, theta].
        pose_j (3,): Pose j [x, y, theta].
        T_ji (3,):   ICP measurement [dx, dy, dtheta] — relative transform from i to j,
                     expressed in i's local frame.

    Returns:
        residual (3,): Motion residual.
        J_i (3, 3):   Jacobian of residual w.r.t. pose_i.
        J_j (3, 3):   Jacobian of residual w.r.t. pose_j.
    """

    # ======= STUDENT TODO START (edit only inside this block) =======

    xi, yi, theta_i = pose_i
    xj, yj, theta_j = pose_j
    dx, dy, dtheta   = T_ji

    cos_i = math.cos(theta_i)
    sin_i = math.sin(theta_i)

    # Forward kinematics: predicted pose_j = pose_i ⊕ T_ji
    fx = xi + cos_i * dx - sin_i * dy
    fy = yi + sin_i * dx + cos_i * dy
    fth = theta_i + dtheta

    # Residual = x_j - f(x_i, T_ji)
    residual = np.array([
        xj - fx,
        yj - fy,
        normalize_angle(theta_j - fth),   # wrap to (-π, π]
    ], dtype=np.float64)

    # Jacobian w.r.t. pose_i (3×3)
    # ∂rx/∂xi=-1, ∂rx/∂yi=0, ∂rx/∂θi = dx·sin(θi) + dy·cos(θi)
    # ∂ry/∂xi=0,  ∂ry/∂yi=-1,∂ry/∂θi =-dx·cos(θi) + dy·sin(θi)
    # ∂rθ/∂xi=0,  ∂rθ/∂yi=0, ∂rθ/∂θi=-1
    J_i = np.array([
        [-1.0,  0.0,  dx * sin_i + dy * cos_i],
        [ 0.0, -1.0, -dx * cos_i + dy * sin_i],
        [ 0.0,  0.0, -1.0                     ],
    ], dtype=np.float64)

    # Jacobian w.r.t. pose_j (3×3) — residual is linear in x_j
    J_j = np.eye(3, dtype=np.float64)

    # ======= STUDENT TODO END (do not change code outside this block) =======

    return residual, J_i, J_j


def motion_factor(
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    T_ji: np.ndarray,
    sigma_xy: float,
    sigma_theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Compute the raw global-frame error and Jacobians
    residual, J_i, J_j = motion_error_and_jacobians(pose_i, pose_j, T_ji)

    # Apply covariance weighting (Information matrix)
    residual[:2] /= sigma_xy
    residual[2]  /= sigma_theta
    J_i[:2]      /= sigma_xy
    J_i[2]       /= sigma_theta
    J_j[:2]      /= sigma_xy
    J_j[2]       /= sigma_theta

    return residual, J_i, J_j


def distance_factor(
    pose_t: np.ndarray,
    landmark_i: np.ndarray,
    z_ti: float,
    sigma_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = landmark_i[0] - pose_t[0]
    dy = landmark_i[1] - pose_t[1]
    dist = max(math.hypot(dx, dy), 1e-9)
    residual = np.array([(dist - z_ti) / sigma_d], dtype=np.float64)

    inv_dist = 1.0 / dist
    J_pose = np.array([[-dx * inv_dist, -dy * inv_dist, 0.0]], dtype=np.float64) / sigma_d
    J_land = np.array([[dx * inv_dist,   dy * inv_dist      ]], dtype=np.float64) / sigma_d
    return residual, J_pose, J_land