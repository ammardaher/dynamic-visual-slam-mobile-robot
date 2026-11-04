#!/usr/bin/env python3
"""Figure 32 - PPO training convergence (mean terminal positioning error).

Regenerates the convergence figure directly from the RSL-RL TensorBoard event
files (tag ``metric/ep_terminal_err_mm``) for the Reach and Insert runs, with the
title/visual elements required by the Experiment-5 spec. No Isaac Sim needed.

    python3 scripts/plot_fig32_convergence.py

Dependency-free tfevents reader (TFRecord framing + minimal protobuf walk of
tensorflow.Event -> Summary -> Value{tag, simple_value}); needs only numpy+matplotlib.
"""
import collections
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- config: source runs + output -------------------------------------------
REACH_EVENT = "/DataDrive/IsaacLab/logs/rsl_rl/new_arm_reach/2026-06-26_14-06-28/events.out.tfevents.1782471990.salman.24841.0"
INSERT_EVENT = "/DataDrive/IsaacLab/logs/rsl_rl/new_arm_insert/2026-06-26_14-57-26/events.out.tfevents.1782475048.salman.13525.0"
TAG = "metric/ep_terminal_err_mm"
SMOOTH_WINDOW = 15
OUT_PNG = "/DataDrive/new_arm_isaac/report_renders/fig_terminal_error.png"


# --- minimal tfevents scalar reader ------------------------------------------
def _varint(data, i):
    shift = result = 0
    while True:
        b = data[i]; i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _walk(data):
    i, n = 0, len(data)
    while i < n:
        key, i = _varint(data, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(data, i)
        elif wt == 1:
            v = data[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(data, i); v = data[i:i + ln]; i += ln
        elif wt == 5:
            v = data[i:i + 4]; i += 4
        else:
            raise ValueError(f"bad wire type {wt}")
        yield fn, wt, v


def read_scalar(path, tag):
    with open(path, "rb") as f:
        buf = f.read()
    out = []
    i, n = 0, len(buf)
    while i + 12 <= n:
        (length,) = struct.unpack_from("<Q", buf, i); i += 12  # len + len-crc
        if i + length + 4 > n:
            break
        rec = buf[i:i + length]; i += length + 4              # data + data-crc
        step = summary = None
        for fn, wt, v in _walk(rec):
            if fn == 2 and wt == 0:
                step = v
            elif fn == 5 and wt == 2:
                summary = v
        if summary is None:
            continue
        for fn, wt, v in _walk(summary):                      # Summary.value
            if fn != 1 or wt != 2:
                continue
            t = val = None
            for ffn, fwt, fv in _walk(v):                     # Value
                if ffn == 1 and fwt == 2:
                    t = fv.decode("utf-8", "replace")
                elif ffn == 2 and fwt == 5:
                    val = struct.unpack("<f", fv)[0]
            if t == tag and val is not None:
                out.append((step or 0, val))
    out.sort()
    return np.array([s for s, _ in out]), np.array([v for _, v in out])


def smooth(y, w):
    if len(y) < w:
        return y.copy()
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(w - 1 - pad, y[-1])])
    return np.convolve(ypad, k, mode="valid")


def main():
    rx, ry = read_scalar(REACH_EVENT, TAG)
    ix, iy = read_scalar(INSERT_EVENT, TAG)
    rs, is_ = smooth(ry, SMOOTH_WINDOW), smooth(iy, SMOOTH_WINDOW)

    BLUE, RED = "#1f77b4", "#d62728"
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

    # pale raw traces
    ax.plot(rx, ry, color=BLUE, alpha=0.22, lw=1.0, zorder=1)
    ax.plot(ix, iy, color=RED, alpha=0.22, lw=1.0, zorder=1)
    # dark smoothed trends
    ax.plot(rx, rs, color=BLUE, lw=2.4, zorder=3,
            label=f"Reach  (NewArm-Reach-v0)   final {rs[-1]:.1f} mm")
    ax.plot(ix, is_, color=RED, lw=2.4, zorder=3,
            label=f"Insert (NewArm-Insert-v0)   final {is_[-1]:.1f} mm")
    # 10 mm tolerance
    ax.axhline(10.0, color="0.45", ls="--", lw=1.3, zorder=2,
               label="10 mm success tolerance")

    ax.set_yscale("log")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean terminal positioning error (mm)")
    ax.set_title("Mean Terminal Positioning Error During PPO Training")
    ax.set_xlim(left=min(rx.min(), ix.min()) - 5)
    ax.grid(True, which="both", color="0.88", lw=0.6)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"[fig32] wrote {OUT_PNG}")
    print(f"[fig32] reach final smoothed={rs[-1]:.2f} mm  insert final smoothed={is_[-1]:.2f} mm")


if __name__ == "__main__":
    main()
