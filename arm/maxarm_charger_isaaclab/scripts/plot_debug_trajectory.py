"""Plot a MaxArm debug trajectory CSV produced by the env's TrajectoryCSVLogger.

Usage:
    python scripts/plot_debug_trajectory.py logs/debug_trajectories/episode_<ts>_ep0000.csv

The script renders four subplots in a single matplotlib window:
  1. Position-error norm vs. step (in mm)
  2. Per-step reward vs. step
  3. Tip XYZ vs. target XYZ vs. step
  4. Policy actions vs. step

It only depends on the standard library plus matplotlib. Run this outside
Isaac Lab; the CSV format is self-contained.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt


def load_csv(path: pathlib.Path) -> Dict[str, List[float]]:
    """Load a debug trajectory CSV into a dict of column -> list of floats."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    cols: Dict[str, List[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except ValueError:
                cols[k].append(float("nan"))
    return cols


def plot_trajectory(cols: Dict[str, List[float]], title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title)

    step = cols["step"]

    # 1. Position-error norm (mm)
    err_mm = [1000.0 * v for v in cols["error_norm"]]
    axes[0, 0].plot(step, err_mm, color="tab:red")
    axes[0, 0].axhline(3.0, color="tab:gray", linestyle="--", linewidth=1, label="success (3 mm)")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("error norm [mm]")
    axes[0, 0].set_title("Position error vs step")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Reward
    axes[0, 1].plot(step, cols["reward"], color="tab:blue")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("reward")
    axes[0, 1].set_title("Per-step reward vs step")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Tip xyz vs target xyz
    axes[1, 0].plot(step, cols["tip_x"], label="tip_x", color="tab:blue")
    axes[1, 0].plot(step, cols["tip_y"], label="tip_y", color="tab:orange")
    axes[1, 0].plot(step, cols["tip_z"], label="tip_z", color="tab:green")
    axes[1, 0].plot(step, cols["target_x"], label="target_x", color="tab:blue", linestyle="--", alpha=0.7)
    axes[1, 0].plot(step, cols["target_y"], label="target_y", color="tab:orange", linestyle="--", alpha=0.7)
    axes[1, 0].plot(step, cols["target_z"], label="target_z", color="tab:green", linestyle="--", alpha=0.7)
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("position [m]")
    axes[1, 0].set_title("Tip (solid) vs target (dashed) per axis")
    axes[1, 0].legend(loc="best", ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Actions
    axes[1, 1].plot(step, cols["action_x"], label="ax", color="tab:blue")
    axes[1, 1].plot(step, cols["action_y"], label="ay", color="tab:orange")
    axes[1, 1].plot(step, cols["action_z"], label="az", color="tab:green")
    axes[1, 1].axhline(0.0, color="tab:gray", linewidth=0.5)
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("policy action [-1, 1]")
    axes[1, 1].set_title("Policy actions vs step")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=pathlib.Path, help="Path to episode_*.csv")
    parser.add_argument(
        "--save",
        type=pathlib.Path,
        default=None,
        help="If set, save the figure to this path instead of (in addition to) showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive window; useful for headless plot generation.",
    )
    args = parser.parse_args()

    try:
        cols = load_csv(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[plot_debug_trajectory] {exc}", file=sys.stderr)
        return 2

    plot_trajectory(cols, title=args.csv.name)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"[plot_debug_trajectory] Saved figure to {args.save}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
