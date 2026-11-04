"""Sample the MaxArm DH workspace and check candidate socket/pre-insertion targets.

Run:
    python scripts/sample_workspace.py

This is only a kinematic check from the DH table. Verify with Isaac's imported URDF.
"""

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxarm_charger.utils.maxarm_math import fk_dh


def main():
    n1, n2, n3 = 91, 91, 121
    q1 = torch.linspace(-math.pi/2, math.pi/2, n1)
    q2 = torch.linspace(-math.pi/2, math.pi/2, n2)
    q3 = torch.linspace(-2*math.pi/3, 2*math.pi/3, n3)

    # For socket centered on x=0, q1 near 0 is usually enough for side-view reach.
    qs = []
    for a in torch.linspace(-math.radians(8), math.radians(8), 9):
        Q2, Q3 = torch.meshgrid(q2, q3, indexing="ij")
        Q1 = torch.full_like(Q2, a)
        q = torch.stack([Q1.reshape(-1), Q2.reshape(-1), Q3.reshape(-1)], dim=-1)
        p, _ = fk_dh(q)
        qs.append(torch.cat([q, p], dim=-1))
    data = torch.cat(qs, dim=0)
    p = data[:, 3:6]

    print("Workspace approximate ranges from DH model:")
    print("x [m]:", float(p[:,0].min()), float(p[:,0].max()))
    print("y [m]:", float(p[:,1].min()), float(p[:,1].max()))
    print("z [m]:", float(p[:,2].min()), float(p[:,2].max()))

    candidates = {
        "socket_doc": torch.tensor([0.0, -0.10, 0.13]),
        "pre_doc_if_insert_minus_y": torch.tensor([0.0, -0.07, 0.13]),
        "pre_alternative_farther_from_base": torch.tensor([0.0, -0.13, 0.13]),
        "rest_approx": torch.tensor([0.0, -0.2392, 0.2128]),
    }

    for name, target in candidates.items():
        dist = torch.linalg.norm(p - target, dim=1)
        idx = torch.argmin(dist)
        q_best = data[idx, :3]
        p_best = data[idx, 3:6]
        print(f"\n{name}: target={target.tolist()}")
        print(f"nearest error [mm] = {1000*float(dist[idx]):.2f}")
        print(f"nearest p [m] = {p_best.tolist()}")
        print(f"q [deg] = {[round(math.degrees(v), 2) for v in q_best.tolist()]}")


if __name__ == "__main__":
    main()
