"""Offline minimum-jerk expert trajectory check.

This is the recommended expert for Phase 1 imitation/RL warm start:
    p_pre -> analytic IK -> q_goal -> minimum-jerk joint trajectory.

Run:
    python scripts/scripted_joint_minjerk_offline.py
"""

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxarm_charger.utils.maxarm_math import analytic_ik_position, fk_dh, minimum_jerk


def main():
    socket = torch.tensor([[0.0, -0.10, 0.13]], dtype=torch.float32)
    p_pre = socket.clone()
    p_pre[:, 1] += 0.03

    # Start 20 mm in front of p_pre along +y.
    p_start = p_pre.clone()
    p_start[:, 1] += 0.02

    q_start, ok_start = analytic_ik_position(p_start)
    q_goal, ok_goal = analytic_ik_position(p_pre, q_current=q_start)

    print("p_start [m] =", p_start.squeeze(0).tolist(), "IK valid =", bool(ok_start.item()))
    print("p_goal  [m] =", p_pre.squeeze(0).tolist(), "IK valid =", bool(ok_goal.item()))
    print("q_start [deg] =", [round(math.degrees(v), 2) for v in q_start.squeeze(0).tolist()])
    print("q_goal  [deg] =", [round(math.degrees(v), 2) for v in q_goal.squeeze(0).tolist()])

    steps = 80
    max_err = 0.0
    for k in range(steps + 1):
        s = torch.tensor(k / steps, dtype=torch.float32)
        q = minimum_jerk(q_start, q_goal, s)
        p, _ = fk_dh(q)
        err = torch.linalg.norm(p - p_pre, dim=-1).item()
        max_err = max(max_err, err)
        if k in [0, 20, 40, 60, 80]:
            print(f"k={k:02d} p={p.squeeze(0).tolist()} dist_to_goal_mm={1000*err:.2f} q_deg={[round(math.degrees(v),2) for v in q.squeeze(0).tolist()]}")

    final_p, _ = fk_dh(q_goal)
    final_err = torch.linalg.norm(final_p - p_pre, dim=-1).item()
    print("Final error [mm] =", 1000 * final_err)


if __name__ == "__main__":
    main()
