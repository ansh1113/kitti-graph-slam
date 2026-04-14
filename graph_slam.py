from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam_factors import distance_factor, motion_factor


@dataclass
class GraphSlamProblem:
    initial_poses: np.ndarray
    initial_landmarks: np.ndarray
    motion_edges: np.ndarray
    observations: np.ndarray
    sigma_xy: float
    sigma_theta: float
    sigma_d: float

    @property
    def num_poses(self) -> int:
        return int(self.initial_poses.shape[0])

    @property
    def num_landmarks(self) -> int:
        return int(self.initial_landmarks.shape[0])

    @property
    def state_dim(self) -> int:
        return 3 * (self.num_poses - 1) + 2 * self.num_landmarks

    @property
    def residual_dim(self) -> int:
        return (
            3 * int(self.motion_edges.shape[0])
            # + 3 * max(self.num_poses - 2, 0)
            + int(self.observations.shape[0])
        )

    def pack_state(self, poses: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return np.concatenate([poses[1:].reshape(-1), landmarks.reshape(-1)], axis=0)

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose_block = 3 * (self.num_poses - 1)
        poses = self.initial_poses.copy()
        if pose_block > 0:
            poses[1:] = state[:pose_block].reshape(self.num_poses - 1, 3)
        landmarks = state[pose_block:].reshape(self.num_landmarks, 2).copy()
        return poses, landmarks

    def pose_col(self, pose_idx: int) -> int | None:
        if pose_idx == 0:
            return None
        return 3 * (pose_idx - 1)

    def landmark_col(self, landmark_idx: int) -> int:
        return 3 * (self.num_poses - 1) + 2 * landmark_idx


def build_linear_system(problem: GraphSlamProblem, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poses, landmarks = problem.unpack_state(state)
    J = np.zeros((problem.residual_dim, problem.state_dim), dtype=np.float64)
    r = np.zeros((problem.residual_dim,), dtype=np.float64)

    row = 0

    # Motion edges: [t, t+1, dx, dy, dtheta]
    for edge in problem.motion_edges:
        pose_i = int(edge[0])
        pose_j = int(edge[1])
        T_ji = edge[2:5]
        res, J_i, J_j = motion_factor(
            poses[pose_i],
            poses[pose_j],
            T_ji,
            sigma_xy=problem.sigma_xy,
            sigma_theta=problem.sigma_theta,
        )
        r[row:row + 3] = res
        col_i = problem.pose_col(pose_i)
        col_j = problem.pose_col(pose_j)
        if col_i is not None:
            J[row:row + 3, col_i:col_i + 3] = J_i
        if col_j is not None:
            J[row:row + 3, col_j:col_j + 3] = J_j
        row += 3

    # Observations: [landmark_idx, pose_idx, distance]
    for obs in problem.observations:
        landmark_idx = int(obs[0])
        pose_idx = int(obs[1])
        z_ti = float(obs[2])
        res, J_pose, J_land = distance_factor(
            poses[pose_idx],
            landmarks[landmark_idx],
            z_ti,
            sigma_d=problem.sigma_d,
        )
        r[row] = res[0]
        col_pose = problem.pose_col(pose_idx)
        col_land = problem.landmark_col(landmark_idx)
        if col_pose is not None:
            J[row:row + 1, col_pose:col_pose + 3] = J_pose
        J[row:row + 1, col_land:col_land + 2] = J_land
        row += 1

    return r, J, poses, landmarks


def solve_graph_slam(
    problem: GraphSlamProblem,
    max_iterations: int = 20,
    cost_tol: float = 1e-6,
    damping: float = 1e-6,
) -> dict[str, np.ndarray | list[float]]:
    """
    TODO(Task 2.2): Implement Gauss-Newton Graph-SLAM optimization.

    Gauss-Newton solves the non-linear least-squares problem:
        min  0.5 * ||r(x)||^2

    At each iteration:
        1. Build Jacobian J and residual r.
        2. Form normal equations:  A = J^T J,  b = -J^T r
        3. Damp:  A += damping * I
        4. Solve: (A + λI) Δx = b
        5. Update candidate state:  x_cand = x + Δx
        6. Accept and early-stop if cost change is negligible.

    Inputs:
        problem: GraphSlamProblem instance.
        max_iterations: Maximum Gauss-Newton iterations.
        cost_tol: Relative cost tolerance for early stopping.
        damping: Levenberg-Marquardt style damping factor λ.

    Returns:
        Dictionary with optimized poses, landmarks, cost history, jacobian, final_residual.
    """
    state = problem.pack_state(problem.initial_poses, problem.initial_landmarks)
    cost_history: list[float] = []
    first_jacobian = None

    for _ in range(max_iterations):
        r, J, poses, landmarks = build_linear_system(problem, state)
        if first_jacobian is None:
            first_jacobian = J.copy()

        # Compute cost at current state
        cost = 0.5 * float(np.sum(r * r))
        cost_history.append(cost)

        # ======= STUDENT TODO START (edit only inside this block) =======

        # Step 1: Build normal equations  A = J^T J,  b = -J^T r
        A = J.T @ J
        b = -J.T @ r

        # Step 2: Add damping to the diagonal of A
        A += damping * np.eye(A.shape[0], dtype=np.float64)

        # Step 3: Solve the linear system  A Δx = b
        dx = np.linalg.solve(A, b)

        # Step 4: Form candidate state and evaluate new cost
        candidate_state = state + dx
        r_cand, _, _, _ = build_linear_system(problem, candidate_state)
        new_cost = 0.5 * float(np.sum(r_cand * r_cand))

        # ======= STUDENT TODO END (do not change code outside this block) =======

        state = candidate_state
        if abs(cost - new_cost) / max(cost, 1.0) < cost_tol:
            cost_history.append(new_cost)
            break

    final_r, final_J, final_poses, final_landmarks = build_linear_system(problem, state)
    final_cost = 0.5 * float(np.sum(final_r * final_r))
    if not cost_history or cost_history[-1] != final_cost:
        cost_history.append(final_cost)

    return {
        "poses": final_poses,
        "landmarks": final_landmarks,
        "cost_history": np.array(cost_history, dtype=np.float64),
        "jacobian": first_jacobian if first_jacobian is not None else final_J,
        "final_residual": final_r,
    }