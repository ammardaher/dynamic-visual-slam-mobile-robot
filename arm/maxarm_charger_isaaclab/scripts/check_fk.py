"""Quick offline FK sanity check for the MaxArm DH table.

Run outside Isaac first:
    python scripts/check_fk.py

Then compare the printed position with Isaac's end_effector_link position for the
same joint angles. If they disagree, fix frame offsets before training.
"""

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maxarm_charger.utils.maxarm_math import fk_dh, numerical_jacobian


def main():
    test_angles_deg = [
        (0.0, 0.0, 0.0),
        (20.0, 20.0, -30.0),
        (-30.0, 35.0, 40.0),
    ]

    for deg in test_angles_deg:
        q = torch.tensor([[math.radians(v) for v in deg]], dtype=torch.float32)
        p, T = fk_dh(q)
        J = numerical_jacobian(q)
        print("\nq [deg] =", deg)
        print("p_tip [m] =", p.squeeze(0).tolist())
        print("p_tip [mm] =", (1000.0 * p.squeeze(0)).tolist())
        print("Jacobian [m/rad] =")
        print(J.squeeze(0))


if __name__ == "__main__":
    main()
