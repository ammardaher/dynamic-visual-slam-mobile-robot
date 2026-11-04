#!/usr/bin/env python3
"""Experiment 5 — generate Table 16, Table 17, Fig 33 and Fig 34 from held-out eval dumps.

Pure numpy + matplotlib; NO Isaac Sim. Consumes the per-episode .npz files written by
`eval_accuracy.py --dump` (one per task: insert, reach, optional vision) and emits every
quantitative deliverable for Experiment 5 except the qualitative trajectory montage (Fig 35).

    python3 scripts/make_exp5_results.py \
        --insert /DataDrive/new_arm_isaac/report_renders/exp5/eval_insert.npz \
        --reach  /DataDrive/new_arm_isaac/report_renders/exp5/eval_reach.npz \
        --outdir /DataDrive/new_arm_isaac/report_renders/exp5

Each .npz holds: target_xyz (N,3, arm-base frame, m), err_mm (N,), success (N,), tol_mm, task.
Tables are written as both Markdown and LaTeX; figures as 150-dpi PNG.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOLS = (5.0, 10.0, 15.0)


def load(path):
    d = np.load(path, allow_pickle=True)
    T = np.asarray(d["target_xyz"], float)
    e = np.asarray(d["err_mm"], float)
    s = np.asarray(d["success"], float).astype(bool)
    return {"T": T, "e": e, "s": s, "task": str(d["task"]),
            "x": T[:, 0], "y": T[:, 1], "z": T[:, 2],
            "r": np.hypot(T[:, 0], T[:, 1]),
            "az": np.degrees(np.arctan2(T[:, 1], T[:, 0]))}


def succ_at(e, tol):
    return 100.0 * np.mean(e <= tol)


def row_stats(e):
    return (len(e), e.mean(), np.median(e), np.percentile(e, 90),
            succ_at(e, 5), succ_at(e, 10), succ_at(e, 15))


# ---------------------------------------------------------------- Table 16
def table16(rows, outdir):
    """rows: list of (label, target_point, e-array)."""
    hdr = ["Policy / Task", "Controlled target point", "Eval episodes",
           "Mean err (mm)", "Median err (mm)", "90th-pct err (mm)",
           "Succ @5mm (%)", "Succ @10mm (%)", "Succ @15mm (%)"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    tex = [r"\begin{tabular}{llrrrrrrr}", r"\hline",
           " & ".join(hdr).replace("%", r"\%") + r" \\", r"\hline"]
    for label, tgt, e in rows:
        n, mean, med, p90, s5, s10, s15 = row_stats(e)
        cells = [label, tgt, f"{n}", f"{mean:.2f}", f"{med:.2f}", f"{p90:.2f}",
                 f"{s5:.1f}", f"{s10:.1f}", f"{s15:.1f}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    cap = ("Table 16. Terminal positioning performance on the held-out "
           "socket-target evaluation set.")
    _write(outdir, "table16", md, tex, cap)
    print("\n" + cap + "\n" + "\n".join(md))


# ---------------------------------------------------------------- Table 17
def table17(d, outdir):
    az, r, z, e = d["az"], d["r"], d["z"], d["e"]
    r_split, z_split = float(np.median(r)), float(np.median(z))
    regions = [
        ("Central frontal region", "|azimuth| <= 30 deg", np.abs(az) <= 30),
        ("Left lateral region", "azimuth < -30 deg", az < -30),
        ("Right lateral region", "azimuth > +30 deg", az > 30),
        ("Near targets", f"r < {r_split:.3f} m", r < r_split),
        ("Far targets", f"r >= {r_split:.3f} m", r >= r_split),
        ("Low socket targets", f"z < {z_split:.3f} m", z < z_split),
        ("High socket targets", f"z >= {z_split:.3f} m", z >= z_split),
    ]
    hdr = ["Workspace region", "Definition", "N episodes",
           "Mean err (mm)", "90th-pct err (mm)", "Succ @10mm (%)"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    tex = [r"\begin{tabular}{llrrrr}", r"\hline",
           " & ".join(hdr).replace("%", r"\%") + r" \\", r"\hline"]
    for name, defn, mask in regions:
        ee = e[mask]
        if len(ee) == 0:
            cells = [name, defn, "0", "-", "-", "-"]
        else:
            cells = [name, defn, f"{len(ee)}", f"{ee.mean():.2f}",
                     f"{np.percentile(ee, 90):.2f}", f"{succ_at(ee, 10):.1f}"]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}"]
    cap = ("Table 17. Plug-tip insertion-alignment performance across "
           f"target-workspace regions (r_split={r_split:.3f} m, z_split={z_split:.3f} m).")
    _write(outdir, "table17", md, tex, cap)
    print("\n" + cap + "\n" + "\n".join(md))


# ---------------------------------------------------------------- Figure 33
def fig33(insert, vision, outdir):
    tol = np.linspace(0, 30, 301)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    for d, color, lbl in [(insert, "#d62728", "NewArm-Insert-v0"),
                          (vision, "#1f77b4", "NewArm-VisionInsert-v0")]:
        if d is None:
            continue
        cdf = 100.0 * np.mean(d["e"][None, :] <= tol[:, None], axis=1)
        ax.plot(tol, cdf, color=color, lw=2.4, label=lbl)
        s10 = succ_at(d["e"], 10)
        ax.plot([10], [s10], "o", color=color, ms=8, zorder=5)
        ax.annotate(f"{s10:.1f}% @10mm", (10, s10), textcoords="offset points",
                    xytext=(10, -16), color=color, fontsize=11, fontweight="bold")
    ax.axvline(10, color="0.45", ls="--", lw=1.3, label="10 mm threshold")
    ax.set_xlim(0, 30); ax.set_ylim(0, 100)
    ax.set_xlabel("Terminal positioning tolerance (mm)")
    ax.set_ylabel("Successful evaluation episodes (%)")
    ax.set_title("Cumulative Terminal-Success Curve (held-out set)")
    ax.grid(True, color="0.9", lw=0.6); ax.legend(loc="lower right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig33_cumulative_success.png")
    fig.savefig(p); print(f"[fig33] wrote {p}")


# ---------------------------------------------------------------- Figure 34
def fig34(d, outdir):
    x, y, r, z, e, s = d["x"], d["y"], d["r"], d["z"], d["e"], d["s"]
    vmax = float(np.percentile(e, 95))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
    for ax, (hx, hy, xl, yl, ttl) in zip(axes, [
            (x, y, "Socket target x rel. arm base (m)", "Socket target y rel. arm base (m)",
             "(a) Top-view workspace map"),
            (r, z, "Radial distance from arm base (m)", "Socket height z (m)",
             "(b) Vertical workspace map")]):
        for mask, mk, lbl in [(s, "o", "success @10mm"), (~s, "X", "failure")]:
            if not mask.any():
                continue
            sc = ax.scatter(hx[mask], hy[mask], c=e[mask], cmap="viridis_r",
                            vmin=0, vmax=vmax, marker=mk,
                            edgecolors="k", linewidths=0.4, s=42, label=lbl)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(True, color="0.92", lw=0.5); ax.legend(loc="best", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("Terminal positioning error (mm)")
    fig.suptitle("Workspace-Dependent Terminal-Error Map (held-out set)", y=1.0)
    p = os.path.join(outdir, "fig34_workspace_map.png")
    fig.savefig(p, bbox_inches="tight"); print(f"[fig34] wrote {p}")


def _write(outdir, stem, md, tex, cap):
    with open(os.path.join(outdir, stem + ".md"), "w") as f:
        f.write(cap + "\n\n" + "\n".join(md) + "\n")
    with open(os.path.join(outdir, stem + ".tex"), "w") as f:
        f.write("% " + cap + "\n" + "\n".join(tex) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", required=True, help="held-out eval .npz for NewArm-Insert-v0")
    ap.add_argument("--reach", default=None, help="held-out eval .npz for NewArm-Reach-v0")
    ap.add_argument("--vision", default=None, help="held-out eval .npz for NewArm-VisionInsert-v0")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    insert = load(a.insert)
    reach = load(a.reach) if a.reach else None
    vision = load(a.vision) if a.vision else None

    rows = []
    if reach:
        rows.append(("NewArm-Reach-v0", "Wrist / end-effector point", reach["e"]))
    rows.append(("NewArm-Insert-v0", "Plug tip", insert["e"]))
    if vision:
        rows.append(("NewArm-VisionInsert-v0", "Plug tip", vision["e"]))

    table16(rows, a.outdir)
    table17(insert, a.outdir)
    fig33(insert, vision, a.outdir)
    fig34(insert, a.outdir)
    print(f"\n[exp5] all artifacts in {a.outdir}")


if __name__ == "__main__":
    main()
