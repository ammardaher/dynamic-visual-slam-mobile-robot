"""Debug helpers for the MaxArm charger environment.

This module contains two small, self-contained utilities used by
``maxarm_charger_align_env`` to make debugging the RL policy tractable:

* :class:`DebugDrawHelper`  - Wraps Isaac Sim's debug draw interface to render
  the tip, the target, and a connecting line each step. Silently disables
  itself when the interface is unavailable (e.g. headless training).
* :class:`TrajectoryCSVLogger` - Buffers per-step trajectory rows for one
  environment and flushes them to a timestamped CSV at episode end.

Both helpers fail closed: if anything in the underlying API misbehaves, they
disable themselves rather than crashing the simulation.
"""

from __future__ import annotations

import csv
import pathlib
import time
from typing import Sequence


# ---------------------------------------------------------------------------
# Viewport debug draw
# ---------------------------------------------------------------------------
class DebugDrawHelper:
    """Draw tip / target markers and a connecting line into the Isaac viewport.

    Colors:
        * tip    : blue   (0, 0, 1)
        * target : green  (0, 1, 0)
        * line   : red    (1, 0, 0)

    The helper attempts both the Isaac Sim 4.x+ module path
    (``isaacsim.util.debug_draw``) and the legacy 2023 path
    (``omni.isaac.debug_draw``). If neither is importable, or the interface
    raises on first use, the helper sets ``enabled = False`` and becomes a
    no-op.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._draw = None
        if not self.enabled:
            return

        try:
            try:
                from isaacsim.util.debug_draw import _debug_draw  # type: ignore
            except ImportError:
                from omni.isaac.debug_draw import _debug_draw  # type: ignore
            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # pragma: no cover - depends on Isaac runtime
            print(f"[MaxArm Debug] Debug draw unavailable, disabling: {exc!r}")
            self.enabled = False
            self._draw = None

    def draw_tip_target(
        self,
        tip_xyz: Sequence[float],
        target_xyz: Sequence[float],
        point_size: float = 15.0,
        line_size: float = 3.0,
    ) -> None:
        """Refresh the viewport markers for a single environment.

        Args:
            tip_xyz: world-frame position of the tip body, length-3 sequence.
            target_xyz: world-frame position of the pre-insertion target,
                length-3 sequence.
            point_size: marker size in pixels for both points.
            line_size: line thickness in pixels.
        """
        if not self.enabled or self._draw is None:
            return

        try:
            tip = (float(tip_xyz[0]), float(tip_xyz[1]), float(tip_xyz[2]))
            tgt = (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]))

            # Clear previous frame to avoid pile-up.
            self._draw.clear_points()
            self._draw.clear_lines()

            self._draw.draw_points(
                [tip, tgt],
                [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0)],  # blue, green
                [point_size, point_size],
            )
            self._draw.draw_lines(
                [tip],
                [tgt],
                [(1.0, 0.0, 0.0, 1.0)],  # red
                [line_size],
            )
        except Exception as exc:  # pragma: no cover - depends on Isaac runtime
            print(f"[MaxArm Debug] Debug draw failed mid-run, disabling: {exc!r}")
            self.enabled = False

    def clear(self) -> None:
        """Wipe all currently drawn primitives."""
        if not self.enabled or self._draw is None:
            return
        try:
            self._draw.clear_points()
            self._draw.clear_lines()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-episode CSV trajectory logger
# ---------------------------------------------------------------------------
class TrajectoryCSVLogger:
    """Buffer per-step trajectory rows and flush to CSV on episode end.

    The logger is intended for a single environment (env 0). Each call to
    :meth:`append` adds one row to an in-memory buffer; :meth:`save` writes
    the buffer to ``<log_dir>/episode_<session>_ep<episode_id>.csv`` and
    clears it. Files are never overwritten because the filename embeds both a
    per-process session timestamp and the running episode id.
    """

    HEADER = [
        "step",
        "tip_x", "tip_y", "tip_z",
        "target_x", "target_y", "target_z",
        "error_norm", "reward",
        "action_x", "action_y", "action_z",
        "q1", "q2", "q3",
        "dq1", "dq2", "dq3",
        "success_counter",
    ]

    def __init__(
        self,
        enabled: bool = True,
        log_dir: str = "logs/debug_trajectories",
    ) -> None:
        self.enabled = bool(enabled)
        self.log_dir = pathlib.Path(log_dir)
        self._rows: list[list[float]] = []
        self._session_ts = time.strftime("%Y%m%d_%H%M%S")

        if self.enabled:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[MaxArm Debug] Could not create CSV log dir {self.log_dir}: {exc!r}")
                self.enabled = False

    def append(
        self,
        step: int,
        tip_xyz: Sequence[float],
        target_xyz: Sequence[float],
        error_norm: float,
        reward: float,
        action_xyz: Sequence[float],
        q: Sequence[float],
        dq: Sequence[float],
        success_counter: int,
    ) -> None:
        """Buffer one trajectory row. Called once per policy step."""
        if not self.enabled:
            return
        self._rows.append([
            int(step),
            float(tip_xyz[0]), float(tip_xyz[1]), float(tip_xyz[2]),
            float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]),
            float(error_norm), float(reward),
            float(action_xyz[0]), float(action_xyz[1]), float(action_xyz[2]),
            float(q[0]), float(q[1]), float(q[2]),
            float(dq[0]), float(dq[1]), float(dq[2]),
            int(success_counter),
        ])

    def save(self, episode_id: int) -> str | None:
        """Flush the current buffer to disk and return the file path.

        Returns ``None`` if logging is disabled or the buffer is empty.
        """
        if not self.enabled or not self._rows:
            self._rows = []
            return None

        filename = self.log_dir / f"episode_{self._session_ts}_ep{episode_id:04d}.csv"
        try:
            with filename.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.HEADER)
                writer.writerows(self._rows)
        except Exception as exc:
            print(f"[MaxArm Debug] CSV save failed for {filename}: {exc!r}")
            self._rows = []
            return None

        self._rows = []
        return str(filename)
