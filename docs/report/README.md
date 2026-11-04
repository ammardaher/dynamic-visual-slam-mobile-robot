# Graduation report — LaTeX source

Source of *Autonomous Differential-Drive Robot for Garage EV Charging: A Dynamic
Visual SLAM Approach with RL-Augmented Precision Docking*.

| File | Section |
|---|---|
| `main.tex` | front matter, introduction, objectives, related work, system-design task, conclusion, bibliography |
| `mechanical_design_after_task.tex` | mechanical design and modelling (kinematics, Jacobian, dynamics, motor ID) |
| `electrical_design_after_task.tex` | electrical design and low-level control |
| `software_architecture_after_task.tex` | software architecture and system integration |
| `methodology_after_system_design.tex` | methodology (perception, SLAM, mapping, navigation, control) |
| `rl_training_after_methodology.tex` | simulation environment and RL training workflow |
| `experiments_and_results.tex` | experiments 1–8 |
| `Figures/` | every image the source includes; `Figures/experiments/` holds the experiment plots also kept in the repo's top-level `figures/` |

## Build

```bash
cd docs/report
pdflatex main.tex
pdflatex main.tex   # second pass for the table of contents and cross-references
```

The bibliography is inline (`thebibliography` in `main.tex`), so no BibTeX run is
needed. Diagrams in `main.tex` use TikZ.

## Data status: illustrative figures

Five figures in `experiments_and_results.tex` are **illustrative, not measured**:

| Figure | Experiment |
|---|---|
| `Figures/report/exp02_pose_error.png` | 2 — socket pose estimation |
| `Figures/report/exp03_nav_results.png` | 3 — navigation and base approach |
| `Figures/report/exp04_arm_results.png` | 4 — arm positioning and wrist leveling |
| `Figures/report/exp06_dr_robustness.png` | 6 — domain-randomization ablation |
| `Figures/report/exp08_timeline.png` | 8 — end-to-end mission |

They come from [`tools/gen_all_missing.py`](tools/gen_all_missing.py), which reads no
logs: the values are hard-coded assumptions, some with added random noise. They
were drawn so the layout could be finished before those experiments were run, and
each caption now says so. Replace them with plots from real logs once the
experiments are done. The published `GRADUATION_REPORT.pdf` predates these labels.

## Version note

This is the latest source revision that was kept (27 June 2026). The published
[`../GRADUATION_REPORT.pdf`](../GRADUATION_REPORT.pdf) was built from a slightly later
edit of it (69 pages; this source builds to about 72), so small wording and layout
differences are expected.
