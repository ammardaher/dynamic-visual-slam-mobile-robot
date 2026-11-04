"""Kinematics utilities for the Hiwonder MaxArm charging task.

All lengths are in meters and all angles are in radians.

The DH parameters follow the user's document:
    row 1: theta=q1-pi/2, d=L0, alpha=pi/2,  a=R0
    row 2: theta=pi/2,    d=0,  alpha=0,     a=0
    row 3: theta=q2,      d=0,  alpha=0,     a=L2
    row 4: theta=-pi/2,   d=0,  alpha=0,     a=0
    row 5: theta=q3,      d=0,  alpha=-pi/2, a=L3

Important: verify this FK against Isaac's end-effector position before training.
If Isaac's imported URDF frame differs from this DH frame, add a fixed transform in
FK_TO_ISAAC_POS_OFFSET or use Isaac's native Jacobian instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class MaxArmKinematicParams:
    L0: float = 0.0844
    L1: float = 0.00814
    L2: float = 0.1284
    L3: float = 0.1380
    L4: float = 0.0168

    @property
    def R0(self) -> float:
        return self.L0 + self.L4


DEFAULT_PARAMS = MaxArmKinematicParams()

# Joint limits from the document.
JOINT_LOWER = torch.tensor([-math.pi / 2.0, -math.pi / 2.0, -2.0 * math.pi / 3.0])
JOINT_UPPER = torch.tensor([ math.pi / 2.0,  math.pi / 2.0,  2.0 * math.pi / 3.0])


def _dh_transform(theta: torch.Tensor, d: float, alpha: float, a: float) -> torch.Tensor:
    """Create batched standard DH transform.

    Args:
        theta: shape (N,)
        d, alpha, a: scalar DH parameters.

    Returns:
        T: shape (N, 4, 4)
    """
    n = theta.shape[0]
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    T = torch.zeros((n, 4, 4), dtype=theta.dtype, device=theta.device)
    T[:, 0, 0] = cth
    T[:, 0, 1] = -sth * ca
    T[:, 0, 2] = sth * sa
    T[:, 0, 3] = a * cth

    T[:, 1, 0] = sth
    T[:, 1, 1] = cth * ca
    T[:, 1, 2] = -cth * sa
    T[:, 1, 3] = a * sth

    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = d

    T[:, 3, 3] = 1.0
    return T


def fk_dh(q: torch.Tensor, params: MaxArmKinematicParams = DEFAULT_PARAMS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics using the DH table.

    Args:
        q: shape (N, 3), joint angles [q1, q2, q3] in radians.

    Returns:
        position: shape (N, 3), end-effector position in DH/base frame.
        transform: shape (N, 4, 4), end-effector transform.
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]

    T1 = _dh_transform(q1 - math.pi / 2.0, params.L0, math.pi / 2.0, params.R0)
    T2 = _dh_transform(torch.full_like(q1, math.pi / 2.0), 0.0, 0.0, 0.0)
    T3 = _dh_transform(q2, 0.0, 0.0, params.L2)
    T4 = _dh_transform(torch.full_like(q1, -math.pi / 2.0), 0.0, 0.0, 0.0)
    T5 = _dh_transform(q3, 0.0, -math.pi / 2.0, params.L3)

    T = T1 @ T2 @ T3 @ T4 @ T5
    return T[:, :3, 3], T


def numerical_jacobian(q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Finite-difference position Jacobian J = dp/dq.

    Args:
        q: shape (N, 3)

    Returns:
        J: shape (N, 3, 3)
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    n = q.shape[0]
    J = torch.zeros((n, 3, 3), dtype=q.dtype, device=q.device)
    p0, _ = fk_dh(q)
    for i in range(3):
        dq = torch.zeros_like(q)
        dq[:, i] = eps
        p1, _ = fk_dh(q + dq)
        J[:, :, i] = (p1 - p0) / eps
    return J


def differential_ik_step(
    q_current: torch.Tensor,
    delta_p: torch.Tensor,
    damping: float = 1e-3,
    max_joint_step: float = math.radians(2.0),
) -> torch.Tensor:
    """One damped least-squares IK step for Cartesian displacement.

    Solves delta_q = J^T (J J^T + lambda^2 I)^-1 delta_p.

    Args:
        q_current: shape (N, 3)
        delta_p: shape (N, 3)
        damping: DLS damping factor.
        max_joint_step: absolute clamp per joint per control step.

    Returns:
        q_target: shape (N, 3)
    """
    if q_current.ndim == 1:
        q_current = q_current.unsqueeze(0)
    if delta_p.ndim == 1:
        delta_p = delta_p.unsqueeze(0)

    J = numerical_jacobian(q_current)
    JT = J.transpose(1, 2)
    I = torch.eye(3, dtype=q_current.dtype, device=q_current.device).unsqueeze(0).repeat(q_current.shape[0], 1, 1)
    inv_term = torch.linalg.solve(J @ JT + (damping ** 2) * I, delta_p.unsqueeze(-1))
    delta_q = (JT @ inv_term).squeeze(-1)
    delta_q = torch.clamp(delta_q, -max_joint_step, max_joint_step)

    q_target = q_current + delta_q
    lower = JOINT_LOWER.to(device=q_current.device, dtype=q_current.dtype)
    upper = JOINT_UPPER.to(device=q_current.device, dtype=q_current.dtype)
    q_target = torch.max(torch.min(q_target, upper), lower)
    return q_target


def minimum_jerk(start: torch.Tensor, goal: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Minimum jerk interpolation.

    Args:
        start: (..., D)
        goal: (..., D)
        s: scalar/tensor in [0, 1]
    """
    blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    while blend.ndim < start.ndim:
        blend = blend.unsqueeze(-1)
    return start + (goal - start) * blend


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def analytic_ik_position(
    desired_pos: torch.Tensor,
    q_current: torch.Tensor | None = None,
    params: MaxArmKinematicParams = DEFAULT_PARAMS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic position-only IK for the supplied DH model.

    The FK position equations are:
        rho = R0 - L2 sin(q2) + L3 cos(q2+q3)
        x = rho sin(q1)
        y = -rho cos(q1)
        z = L0 + L2 cos(q2) + L3 sin(q2+q3)

    Args:
        desired_pos: shape (N, 3), target [x, y, z] in the DH/base frame.
        q_current: optional shape (N, 3). If given, the IK branch closest to q_current is chosen.

    Returns:
        q_solution: shape (N, 3)
        reachable: shape (N,), bool
    """
    if desired_pos.ndim == 1:
        desired_pos = desired_pos.unsqueeze(0)
    N = desired_pos.shape[0]
    device = desired_pos.device
    dtype = desired_pos.dtype

    x = desired_pos[:, 0]
    y = desired_pos[:, 1]
    z = desired_pos[:, 2]

    q1 = torch.atan2(x, -y)
    rho = torch.sqrt(x * x + y * y)
    A = rho - params.R0
    B = z - params.L0

    L2 = torch.tensor(params.L2, device=device, dtype=dtype)
    L3 = torch.tensor(params.L3, device=device, dtype=dtype)

    D = (A * A + B * B - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    reachable = (D >= -1.0) & (D <= 1.0)
    D_clamped = torch.clamp(D, -1.0, 1.0)

    # Standard 2-link equivalent: theta3_eff = q3 - pi/2.
    theta3_a = torch.atan2(torch.sqrt(torch.clamp(1.0 - D_clamped * D_clamped, min=0.0)), D_clamped)
    theta3_b = torch.atan2(-torch.sqrt(torch.clamp(1.0 - D_clamped * D_clamped, min=0.0)), D_clamped)

    candidates = []
    for theta3_eff in (theta3_a, theta3_b):
        theta1_eff = torch.atan2(B, A) - torch.atan2(L3 * torch.sin(theta3_eff), L2 + L3 * torch.cos(theta3_eff))
        q2 = wrap_to_pi(theta1_eff - math.pi / 2.0)
        q3 = wrap_to_pi(theta3_eff + math.pi / 2.0)
        q = torch.stack([q1, q2, q3], dim=-1)
        candidates.append(q)

    cand = torch.stack(candidates, dim=1)  # (N, 2, 3)
    lower = JOINT_LOWER.to(device=device, dtype=dtype)
    upper = JOINT_UPPER.to(device=device, dtype=dtype)
    within = torch.all((cand >= lower) & (cand <= upper), dim=-1)  # (N, 2)

    if q_current is None:
        # Prefer branch 0 unless invalid.
        idx = torch.where(within[:, 0], torch.zeros(N, dtype=torch.long, device=device), torch.ones(N, dtype=torch.long, device=device))
    else:
        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)
        diff = wrap_to_pi(cand - q_current.unsqueeze(1))
        cost = torch.sum(diff * diff, dim=-1)
        cost = torch.where(within, cost, torch.full_like(cost, 1e9))
        idx = torch.argmin(cost, dim=1)

    q_sol = cand[torch.arange(N, device=device), idx]
    valid = reachable & within[torch.arange(N, device=device), idx]

    # Clamp as a last safety net.
    q_sol = torch.max(torch.min(q_sol, upper), lower)
    return q_sol, valid


def cartesian_position_to_joint_target(
    q_current: torch.Tensor,
    desired_pos: torch.Tensor,
    max_joint_step: float = math.radians(2.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global analytic IK to desired position, then clamp joint step for safe motion."""
    q_ik, valid = analytic_ik_position(desired_pos, q_current=q_current)
    dq = wrap_to_pi(q_ik - q_current)
    dq = torch.clamp(dq, -max_joint_step, max_joint_step)
    q_target = q_current + dq
    lower = JOINT_LOWER.to(device=q_current.device, dtype=q_current.dtype)
    upper = JOINT_UPPER.to(device=q_current.device, dtype=q_current.dtype)
    q_target = torch.max(torch.min(q_target, upper), lower)
    return q_target, valid
