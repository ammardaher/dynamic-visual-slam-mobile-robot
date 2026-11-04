# Experiment 5 — Results Runbook

Generate Table 16/17 + Fig 33/34 from a held-out evaluation of the **already-trained**
policies. **No retraining.** One short Isaac inference pass (minutes) produces the data;
everything else is offline numpy/matplotlib.

Checkpoints (final, iter 799):
- Insert: `/DataDrive/IsaacLab/logs/rsl_rl/new_arm_insert/2026-06-26_14-57-26/model_799.pt`
- Reach:  `/DataDrive/IsaacLab/logs/rsl_rl/new_arm_reach/2026-06-26_14-06-28/model_799.pt`

Scripts (in `~/Documents/our_work/new_arm_isaac/scripts/`):
- `eval_accuracy.py`  — now has `--dump <npz>` (held-out per-episode dump). cli_args path auto-resolves.
- `make_exp5_results.py` — offline: dumps → Table 16/17 + Fig 33/34 (no Isaac).
- `plot_fig32_convergence.py` — Fig 32 (already run; no Isaac).

Held-out set definition: `--num_envs 1024 --episodes 1` → 1024 distinct FK-sampled socket
targets at **full curriculum difficulty**, fixed `--seed 20260630` (reproducible, and not the
specific targets seen during training). Deterministic (mean) policy actions. 10 mm tolerance
is the env's `success_pos_error` (0.01 m), already baked in.

---

## Step A — Isaac eval dump (the only GPU step) ~ a few minutes

```bash
cd /DataDrive/IsaacLab
mkdir -p /DataDrive/new_arm_isaac/report_renders/exp5

# Insert (plug tip) — drives Table 16 row 2, Table 17, Fig 33, Fig 34
./isaaclab.sh -p /home/salman/Documents/our_work/new_arm_isaac/scripts/eval_accuracy.py \
  --task NewArm-Insert-v0 --headless --num_envs 1024 --episodes 1 --seed 20260630 \
  --checkpoint /DataDrive/IsaacLab/logs/rsl_rl/new_arm_insert/2026-06-26_14-57-26/model_799.pt \
  --dump /DataDrive/new_arm_isaac/report_renders/exp5/eval_insert.npz

# Reach (wrist) — drives Table 16 row 1
./isaaclab.sh -p /home/salman/Documents/our_work/new_arm_isaac/scripts/eval_accuracy.py \
  --task NewArm-Reach-v0 --headless --num_envs 1024 --episodes 1 --seed 20260630 \
  --checkpoint /DataDrive/IsaacLab/logs/rsl_rl/new_arm_reach/2026-06-26_14-06-28/model_799.pt \
  --dump /DataDrive/new_arm_isaac/report_renders/exp5/eval_reach.npz
```
Each prints a one-line sanity summary (N / mean / median / p90 / succ@10).

## Step B — Offline tables + figures (no Isaac, seconds)

```bash
python3 /home/salman/Documents/our_work/new_arm_isaac/scripts/make_exp5_results.py \
  --insert /DataDrive/new_arm_isaac/report_renders/exp5/eval_insert.npz \
  --reach  /DataDrive/new_arm_isaac/report_renders/exp5/eval_reach.npz \
  --outdir /DataDrive/new_arm_isaac/report_renders/exp5
```
Outputs in `report_renders/exp5/`:
- `table16.md` / `table16.tex` — held-out performance (Reach + Insert rows)
- `table17.md` / `table17.tex` — per-workspace-region (splits = medians, printed)
- `fig33_cumulative_success.png` — cumulative terminal-success curve, 10 mm marker
- `fig34_workspace_map.png` — two-panel workspace error map

The validated pipeline (synthetic contract test, 2026-06-29) confirmed both scripts
run end-to-end; the real run only swaps synthetic data for the real dumps.

---

## Already done (2026-06-29)
- **Fig 32** regenerated with required title → `report_renders/fig_terminal_error.png`
  (old saved as `…pre-fig32.bak`).

## Not yet automated
- **Fig 35** (representative trajectory montage, easy/lateral/boundary). Qualitative; needs a
  render-enabled single-env rollout capturing initial/intermediate/final poses with overlaid
  labels. Build as a follow-up render script (`render_scene.py` infra) — flag if wanted.
- **NewArm-VisionInsert-v0**: no usable checkpoint exists (only abandoned `maxarm_charger_align`
  model_0/4). Spec marks it optional ("if trained") → omitted. Training it WOULD require a run.

## If `import cli_args` still fails
`export ISAACLAB_PATH=/DataDrive/IsaacLab` before Step A (the script also auto-checks that path).
