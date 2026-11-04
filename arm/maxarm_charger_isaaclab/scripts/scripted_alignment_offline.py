"""Offline scripted alignment check using DH FK and differential IK.

This does not require Isaac. It checks whether the nominal pre-insertion point is
approximately reachable with the DH model.

Run:
    python scripts/scripted_alignment_offline.py
"""

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxarm_charger.utils.maxarm_math import fk_dh, cartesian_position_to_joint_target


def main():
    # Nominal task from your document.
    socket = torch.tensor([[0.0, -0.10, 0.13]], dtype=torch.float32)
    preinsert_distance = 0.03
    target = socket.clone()
    target[:, 1] += preinsert_distance  # because insertion is -y

    q = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    max_dp = torch.tensor([[0.002, 0.0015, 0.002]], dtype=torch.float32)

    print("Target pre-insertion point [m]:", target.squeeze(0).tolist())
    print("Starting q [deg]:", [0.0, 0.0, 0.0])

    for step in range(250):
        p, _ = fk_dh(q)
        err = target - p
        err_norm = torch.linalg.norm(err, dim=-1).item()
        if step % 25 == 0 or err_norm < 0.003:
            print(f"step={step:03d} p={p.squeeze(0).tolist()} err_mm={1000*err_norm:.2f} q_deg={[round(math.degrees(v), 2) for v in q.squeeze(0).tolist()]}")
        if err_norm < 0.003:
            print("SUCCESS: within 3 mm")
            return
        delta_p = torch.clamp(0.25 * err, -max_dp, max_dp)
        desired_pos = p + delta_p
        q, valid = cartesian_position_to_joint_target(q, desired_pos, max_joint_step=math.radians(2.0))
        if not bool(valid.item()):
            print("Warning: IK target was outside exact analytic limits; joint target was clamped.")

    p, _ = fk_dh(q)
    err_norm = torch.linalg.norm(target - p, dim=-1).item()
    print("FAILED to reach within 3 mm using current DH/controller.")
    print("Final p [m]:", p.squeeze(0).tolist())
    print("Final error [mm]:", 1000 * err_norm)
    print("Final q [deg]:", [round(math.degrees(v), 2) for v in q.squeeze(0).tolist()])
    print("If this fails but Isaac can reach it physically, your DH frame differs from Isaac's base frame.")


if __name__ == "__main__":
    main()
