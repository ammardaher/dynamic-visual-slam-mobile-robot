> **Historical prototype.** This is the first Isaac Lab charging-alignment package
> (`MaxArm-Charger-Align-v0`, versions v2–v8), kept for reference. It was superseded by
> [`../new_arm_isaac`](../new_arm_isaac), which holds the trained Reach / Insert /
> VisionInsert tasks used in the report. The `usd_path` in
> `maxarm_charger/tasks/direct/maxarm_charger_align_env.py` points at the original
> workstation and must be edited before use.

# MaxArm Charger Alignment Isaac Lab Package

This package registers the Isaac Lab task:

```text
MaxArm-Charger-Align-v0
```

The current version is a **stability/debug version**. It fixes the robot root, avoids absolute IK branch jumps, and lets you validate the imported USD before doing serious PPO training.

## What changed in this version

1. The arm root is fixed using `fix_root_link=True`, because the robot/mobile base must be stationary during the charging alignment stage.
2. The controller no longer uses full analytic IK inside each action step. It uses a small damped differential IK update, which avoids sudden 180-degree flips.
3. The fourth joint `end_effector_joint` is held by a position drive and is not controlled by the RL policy.
4. The temporary tip body is now `visuals`, with fallback to `link2_link`.
5. The default reset mode is zero/default pose, not analytic IK. This prevents reset-time jumps while the USD/DH frame convention is still being validated.
6. Debug modes can be selected using environment variables.

## Install

```bash
cd ~/Documents/our_work
rm -rf maxarm_charger_isaaclab
unzip ~/Downloads/maxarm_charger_isaaclab_v6_fixed.zip
cd maxarm_charger_isaaclab
python -m pip install -e .
```

## Required import in Isaac Lab scripts

Make sure both files include this near the imports:

```python
import maxarm_charger
```

Files:

```text
~/Documents/grad_robot/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py
~/Documents/grad_robot/IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py
```

## Important validation commands

### 1. Hold test: robot should not move

```bash
cd ~/Documents/grad_robot/IsaacLab
MAXARM_DEBUG_MODE=hold ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task MaxArm-Charger-Align-v0 \
  --num_envs 1
```

If the robot still falls or rotates in this mode, the problem is the USD/articulation, not PPO.

### 2. Fixed direction test: robot should move smoothly

```bash
MAXARM_DEBUG_MODE=fixed_neg_y ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task MaxArm-Charger-Align-v0 \
  --num_envs 1
```

This bypasses PPO and commands a small fixed Cartesian action in negative Y.

### 3. Normal policy play

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task MaxArm-Charger-Align-v0 \
  --num_envs 1
```

If no checkpoint is loaded, the policy may be random. Use `--load_run` and `--checkpoint` explicitly when needed.

## Training

Start only after the hold test is stable.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task MaxArm-Charger-Align-v0 \
  --headless
```

## Notes

The green point in the viewport debug drawing is the target pre-insertion point. The blue point is the current tip body. The red line connects the tip to the target.

The robot USD still appears to contain extra things such as a ground plane. For serious multi-environment training, create a clean robot USD that contains only robot links, joints, collision meshes, and visual meshes.


## v7 debugging notes

This version separates two tests:

```bash
MAXARM_DEBUG_MODE=hold      # force-freezes joint states every simulation step
MAXARM_DEBUG_MODE=drive_hold # relies only on the simulated servo drives
```

Use `hold` first. If a link still falls in `hold`, that link is probably not part of the articulation, or the USD hierarchy/joint definition is broken. If `hold` is stable but `drive_hold` falls, then the simulated drives were too weak or the inertial properties are unrealistic. This version also uses high-gain idealized servo drives for initial validation.

After the USD is stable, remove debug mode and train/play normally.


## v8 note

`drive_hold` was corrected to hold the frozen reset joint pose instead of continuously using the current falling joint pose as the target. This is the correct test for whether the simulated servo drives can hold the arm against gravity.
