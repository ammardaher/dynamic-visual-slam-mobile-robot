# MaxArm Charging-Arm — Reinforcement-Learning Training (Technical Reference)

> Audience: an AI/engineer who needs to understand *exactly* how this was built and trained.
> Project: `new_arm_isaac` — the **manipulation half** of an autonomous EV-charging robot. A 3-DOF
> arm ("MaxArm") that holds a charging plug and learns, with reinforcement learning in NVIDIA
> **Isaac Sim 5.1 / Isaac Lab v2.3.2**, to (1) reach a target, (2) drive the plug into a wall
> socket, and (3) do the same guided by an eye-in-hand camera. The mobile base + Nav2 navigation
> is a separate subsystem; this document is only the arm.

## 0. Overview

Three Gymnasium tasks are registered, in increasing difficulty, each a thin subclass of the last:

| Task id | Goal | Tracked point | Obs dim |
|---|---|---|---|
| `NewArm-Reach-v0` | wrist tip → 3-D target | `end_effector_link` origin | 12 |
| `NewArm-Insert-v0` | plug tip → socket (radial insertion) | plug tip (EE +X, +0.25 m) | 12 |
| `NewArm-VisionInsert-v0` | insertion from eye-in-hand camera bearings | plug tip | 15 (asym. actor-critic) |

The end-to-end pipeline is: **CAD/URDF → USD (Isaac Sim URDF importer, EE camera + plug baked in) →
vectorized Isaac Lab `DirectRLEnv` (1024 parallel arms) → RSL-RL PPO → trained policy (`.pt`) →
replay / deploy**. Everything below is extracted directly from the source in
`~/Documents/our_work/new_arm_isaac/`.

A central design idea recurs throughout: **the targets are sampled as the forward-kinematics of
random joint configurations, so every target is reachable *by construction*** — this makes the
position-reaching RL problem well-posed (no unreachable goals that would cap the success rate).

---

## Robot Model & Asset Pipeline

The simulated robot is a 3-DOF "MaxArm"-class arm. Its single source of truth for joint
order, prim paths, calibration, asset poses, and output locations is
`new_arm_isaac/new_arm_isaac/config.py`. Three Python entry points consume it:
`scripts/import_urdf.py` (URDF→USD, runs inside Isaac Sim), `new_arm_isaac/usd_camera.py`
(`bake_ee_camera`), and `new_arm_isaac/usd_socket_plug.py` (`bake_plug` / `bake_socket`).

> **Important consistency note.** The prose in `README.md` and several inline comments
> describe *older* values (e.g. "+14 cm forward / +4 cm up", "45° tilt", install on
> `/media/salman/DataDrive1`). The **authoritative numbers are the constants in
> `config.py`**, which differ (camera offset `(0.20, 0.0, 0.05)`, `CAMERA_PITCH_DEG = -60`,
> `DATA_DIR = /DataDrive/new_arm_isaac`). All numbers below are quoted from `config.py` and
> the URDF, not from the README narrative.

### Kinematic chain (joints & links)

The URDF is `WORKSPACE/new_arm_description/urdf/new_arm.urdf`
(`/home/salman/Documents/our_work/new_arm_description/urdf/new_arm.urdf`). From `config.py`:

```python
ACTUATED_JOINTS = ["after_base_full_joint", "link1_joint", "link2_joint"]  # RL acts on these 3
PASSIVE_JOINT   = "end_effector_joint"                                     # commanded, not acted on
ALL_JOINTS      = ACTUATED_JOINTS + [PASSIVE_JOINT]                        # 4 DOF total
TIP_LINK        = "end_effector_link"   # RL/IK target = the wrist pivot (this link's origin)
BASE_LINK       = "arm_base_link"       # fixed root, bolted to world
ROBOT_PRIM      = "/World/NewArm"       # articulation prim path inside the stage
```

The full chain from the URDF (revolute joints; axes and limits verified in the URDF):

| Joint | Type | Axis | Parent link → Child link | Limits (lower/upper rad) | effort / vel |
|---|---|---|---|---|---|
| `after_base_full_joint` | revolute (actuated) | Z `0 0 1` | `arm_base_link` → `after_base_full_link` | −3.14 / 3.14 | 10.0 / 1.0 |
| `link1_joint` | revolute (actuated) | Y `0 1 0` | `after_base_full_link` → `link1_link` | −3.14 / 3.14 | 10.0 / 1.0 |
| `link2_joint` | revolute (actuated) | Y `0 1 0` | `link1_link` → `link2_link` | −3.14 / 3.14 | 10.0 / 1.0 |
| `end_effector_joint` | revolute (**passive**) | Y `0 1 0` | `link2_link` → `end_effector_link` | −3.14 / 3.14 | 5.0 / 1.0 |

The passive wrist is **not** acted on by RL. The real arm couples it mechanically; the sim
keeps a position drive on it and **commands** it every step to its coupled value
`theta4 = 2*pi - theta2 - theta3` (the firmware's `- pi/2` term is intentionally dropped so
the wrist sits **level (0 rad) at home** — see `kinematics.PASSIVE_OFFSET`). RL still only
*acts* on the 3 actuated joints.

### URDF → USD import (`scripts/import_urdf.py`)

Run with the Isaac Sim 5.1.0 python; produces a self-contained articulation at
`config.USD_PATH`. It boots `SimulationApp({"headless": True})`, then configures the importer
via `omni.kit.commands.execute("URDFCreateImportConfig")` with **direct attribute assignment**:

```python
import_config.fix_base = True                  # arm_base_link is the root -> bolted to world
import_config.merge_fixed_joints = False       # keep every link/joint distinct
import_config.import_inertia_tensor = True
import_config.distance_scale = 1.0             # URDF is already metres (mesh scale 0.001)
import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_position_drive_damping = DRIVE_DAMPING   # = 1.0e5
import_config.self_collision = False
import_config.make_default_prim = True         # best-effort (also set explicitly below)
```

Import is done into the **current in-memory stage** (no `dest_path`):
`omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=config.URDF_PATH, import_config=..., get_articulation_root=True)`.
Passing `dest_path` then `save_as_stage` was found to corrupt the asset (empty wrapper); the
script instead follows the canonical `standalone_examples/.../urdf_import.py` pattern: import
live, walk up from the returned ArticulationRoot prim to its top-level ancestor, set that as
the stage `defaultPrim`, bake the camera + plug, set drive gains, then `save_as_stage` **exactly
once** to `config.USD_PATH`.

**Position-drive gains** are baked into the USD (used by the standalone/twin
`SingleArticulation` path; the IsaacLab env overrides them via `ImplicitActuatorCfg`):

```python
DRIVE_STIFFNESS = 1.0e7
DRIVE_DAMPING   = 1.0e5
```

After import, the script traverses the stage and, for **every** `PhysicsRevoluteJoint`, applies
`UsdPhysics.DriveAPI.Apply(prim, "angular")` and sets `CreateStiffnessAttr(1.0e7)` /
`CreateDampingAttr(1.0e5)`. It then plays one timeline step and initializes an `Articulation`
view to print `dof_names` and verify they match `ACTUATED_JOINTS + PASSIVE_JOINT`.

During the same conversion it bakes both the EE camera (`bake_ee_camera`) and the plug
(`bake_plug`) onto `end_effector_link`, so the twin, standalone demo, and all cloned RL envs
inherit them. `scripts/add_ee_camera.py` is a lightweight, idempotent, pure-`pxr` patcher
(no `SimulationApp`) that adds/refreshes only the camera on an existing `new_arm.usd` and can
`--verify` the written intrinsics.

### Baked EE camera (`usd_camera.py`, `bake_ee_camera`)

A Logitech C920 HD Pro RGB webcam is created as a child prim
`<top>/end_effector_link/logitech_camera` (`UsdGeom.Camera`), so it rigidly follows the gripper
through every reference/clone. Values (from `config.py`):

```python
CAMERA_MODEL              = "Logitech C920 HD Pro"
CAMERA_PRIM_NAME          = "logitech_camera"
CAMERA_PARENT_LINK        = "end_effector_link"   # TIP_LINK
CAMERA_OFFSET_XYZ         = (0.20, 0.0, 0.05)     # m, EE frame: +0.20 forward (+X), +0.05 up (+Z)
CAMERA_PITCH_DEG          = -60.0                 # RotateY: look forward + down toward plug tip/socket
CAMERA_ROLL_DEG           = -90.0                 # RotateZ about optical axis -> portrait, upright
CAMERA_FOCAL_LENGTH       = 14.84588              # mm-equivalent
CAMERA_HORIZONTAL_APERTURE= 11.78719              # -> HFOV 2*atan(11.78719/(2*14.84588)) = 43.3 deg
CAMERA_VERTICAL_APERTURE  = 20.955                # -> VFOV 70.4 deg
CAMERA_CLIPPING_RANGE     = (0.01, 1000.0)        # m
CAMERA_FOCUS_DISTANCE     = 40.0                  # irrelevant at f_stop=0
CAMERA_RESOLUTION         = (1080, 1920)          # (w,h) native C920 1080p, portrait
CAMERA_RL_RESOLUTION      = (72, 128)             # (w,h) small 9:16 portrait render for batched RL
```

> Note: the offset and pitch here are the live values **(0.20, 0.0, 0.05)** and **−60°**. The
> README/comment prose ("+14 cm / +4 cm", "45° forward-and-down") is stale; `config.py` wins.

The pose is written in IsaacLab's required canonical `xformOpOrder` **[translate, orient, scale]**
(double precision; euler ops are rejected by IsaacLab). The orientation is a single quaternion
`q_total = q_pitch * q_roll` (matrix `Ry(pitch) @ Rz(roll)`):

```python
py = radians(CAMERA_PITCH_DEG) / 2.0
pz = radians(CAMERA_ROLL_DEG)  / 2.0
q_pitch = Quatd(cos(py), Vec3d(0, sin(py), 0))   # about +Y
q_roll  = Quatd(cos(pz), Vec3d(0, 0, sin(pz)))   # about +Z (optical axis)
quat    = q_pitch * q_roll
```

Intrinsics are pinhole: `CreateFStopAttr(0.0)` (no depth-of-field, only FOV ratio matters),
`projection = perspective`. FOV follows `FOV = 2*atan(aperture / (2*focal))`. The 90° portrait
roll swaps which aperture is horizontal vs vertical so pixels stay square: the wide 70.4° C920
FOV becomes **vertical** (forward-up) and 43.3° is **horizontal** (lateral). RGB-only; depth vs
RGB is a render-product choice made by the consumer (e.g. the RL `TiledCamera`).

### Baked plug geometry (`usd_socket_plug.py`, `bake_plug`)

The plug is a rigid child of `end_effector_link` (same parent as the camera), pointing along
**EE +X** so it can insert. It is **visual-only (no collider)** so analytic IK isn't perturbed
by contact. Built procedurally with `pxr` as an `Xform` with identity orient and these
constants (from `config.py`):

```python
PLUG_PRIM_NAME    = "plug"
PLUG_PARENT_LINK  = "end_effector_link"
PLUG_OFFSET_XYZ   = (0.18, 0.0, 0.0)   # m, EE frame: plug body starts at +0.18
PLUG_BODY_LENGTH  = 0.04               # cylinder, axis = EE +X -> body emerges to 0.22
PLUG_BODY_RADIUS  = 0.013
PLUG_PRONG_LENGTH = 0.03               # two pins -> tip at 0.25 m
PLUG_PRONG_RADIUS = 0.0035
PLUG_PRONG_SPACING= 0.019              # MUST equal SOCKET_HOLE_SPACING
```

Geometry: a body `UsdGeom.Cylinder` (axis `"X"`, height `0.04`, radius `0.013`) translated to
its half-length, colored dark `(0.08, 0.08, 0.10)`; plus two prong cylinders (axis `"X"`,
height `0.03`, radius `0.0035`) at `x = body_len + prong_len/2` and `y = ±spacing/2 = ±0.0095`,
colored bright orange `(0.95, 0.45, 0.05)`. Resulting **tip depth in the EE frame**:

```
tip_x = PLUG_OFFSET_XYZ[0] + PLUG_BODY_LENGTH + PLUG_PRONG_LENGTH
      = 0.18 + 0.04 + 0.03 = 0.25 m  (clearly past the ~0.20 m gripper)
```

### Socket geometry (`usd_socket_plug.py`, `bake_socket`)

The socket is **static world geometry** (a faceplate with two recessed holes on a support
stand) placed in front of the arm, faceplate front face facing **−X** (toward the base).
Constants (from `config.py`):

```python
SOCKET_PRIM           = "/World/Socket"
SOCKET_SEMANTIC_CLASS = "socket"              # Replicator/CenterPose label, best-effort
SOCKET_POSE_XYZ       = (0.334, 0.0, 0.12)    # faceplate CENTRE; front face at 0.334 - 0.004 = 0.330
SOCKET_FACEPLATE_HALF = (0.004, 0.043, 0.043) # half-extents (X,Y,Z) -> 8 x 86 x 86 mm plate
SOCKET_HOLE_RADIUS    = 0.0065                # the two round pin holes
SOCKET_HOLE_SPACING   = 0.019                 # centre-to-centre (EU type-C/F); == PLUG_PRONG_SPACING
SOCKET_HOLE_DEPTH     = 0.012                 # recess depth into the faceplate
SOCKET_STAND_HALF     = (0.02, 0.02, 0.06)    # support post below the faceplate
```

The faceplate is a unit `UsdGeom.Cube` (size `2.0`, half-extent 1) scaled by the half-extents,
colored blue `(0.16, 0.28, 0.55)`, with a **static** `UsdPhysics.CollisionAPI` (no
`RigidBodyAPI`). Two hole cylinders (axis `"X"`, height `0.012`, radius `0.0065`) sit flush with
the front face at `y = ±spacing/2 = ±0.0095`, colored bright green `(0.10, 0.90, 0.20)` as a
vision aid. A support `Cube` stand fills from ground (`z=0`) up to the faceplate bottom, colored
`(0.3, 0.3, 0.32)`, also a static collider. A best-effort `class=socket` semantic label is
applied for Replicator/CenterPose SDG.

`bake_socket(marker=True)` produces a **hole-centred, stand-less, collider-less** variant for
per-env RL `VisualizationMarker`s: geometry is shifted by `+hx` so the front (hole) plane lands
at the prim **origin**, and the prim's own translate is `(0,0,0)` (the marker system supplies
the transform).

### Named poses & calibration

```python
NAMED_POSES = {"home":  [0.0, 0.0, 0.0],
               "ready": [0.0, 0.4, -0.6]}   # 3 actuated values each (from SRDF)

# rad <-> Lobot pulse: pulse = scale*theta + offset, clamped to [pulse_min, pulse_max]
CAL_SCALE     = [238.7324, -238.7324, 238.7324]
CAL_OFFSET    = [625.0, 875.0, 500.0]   # home -> pulses (625, 875, 500)
CAL_PULSE_MIN = [0.0, 0.0, 470.0]
CAL_PULSE_MAX = [1000.0, 700.0, 1000.0]
```

`theta_to_pulse(i, theta)` clamps `scale[i]*theta + offset[i]` to `[min,max]`;
`pulse_to_theta(i, pulse)` inverts it. (Note: per the README, with the seeded calibration the
shoulder's `offset=875` clamps to its `700` max, so commanded URDF-home maps the shoulder to
~0.733 rad, not 0 — a hardware-calibration caveat, faithfully reproduced by the twin.)

### File / directory layout (everything writes to `/DataDrive`)

Because the system disk is full, **all heavy USD/cache output lives on the ext4 `/DataDrive`**
(`config.DATA_DIR`). `config.ensure_dirs()` creates `DATA_DIR`, the USD dir, and `OV_CACHE`.

```python
ISAAC_ROOT            = "/DataDrive/isaac-sim"
WORKSPACE             = "/home/salman/Documents/our_work"
URDF_PATH             = WORKSPACE + "/new_arm_description/urdf/new_arm.urdf"
DATA_DIR              = "/DataDrive/new_arm_isaac"
USD_PATH              = DATA_DIR + "/usd/new_arm.usd"           # baked articulation + camera + plug
SOCKET_USD_PATH       = DATA_DIR + "/usd/socket.usd"           # standalone socket (demo + SDG)
SOCKET_MARKER_USD_PATH= DATA_DIR + "/usd/socket_marker.usd"    # hole-centred per-env RL marker
OV_CACHE              = DATA_DIR + "/ov_cache"
```

Source-tree layout (root `/home/salman/Documents/our_work/new_arm_isaac/`):

```
new_arm_isaac/
  README.md
  scripts/
    import_urdf.py        URDF -> USD (once); bakes EE camera + plug, sets drive gains
    add_ee_camera.py      add/refresh EE camera on existing USD (pure pxr, no SimulationApp)
  new_arm_isaac/
    config.py             paths, joint names, camera/plug/socket geometry, poses, calibration
    usd_camera.py         bake_ee_camera()
    usd_socket_plug.py    bake_plug() (EE child) + bake_socket() (world / marker)
    kinematics.py, kinematics_torch.py, vision_align.py
    tasks/, agents/
  meshes/                 plug.stl, socket.stl, socket_marker.stl, socket_sim.stl, ... (+ DIMENSIONS.md)
```

(Observed live output dirs under `/DataDrive/new_arm_isaac/` include `usd/`, `ov_cache/`,
`renders/`, and several camera/insert capture dirs — confirming the pipeline has been run.)


---
## Kinematics & Math

The MaxArm is a 3-DOF serial chain whose analytic kinematics live in two
dependency-light modules:

- `kinematics.py` — pure NumPy reference (FK, full chain, analytic IK with a
  damped-Newton polish, passive-wrist coupling, plus a faithful firmware port).
- `kinematics_torch.py` — a batched Torch port of FK + passive coupling for the
  RL environments (GPU-friendly, no IsaacLab dependency).

Both use the **exact same URDF joint geometry**, so the simulated twin, the RL
policy, and the firmware agree. Angles are **radians** in the URDF joint
convention (centered at 0); positions are **metres** in the `arm_base_link`
frame. The kinematic target is the **wrist pivot** = the origin of
`end_effector_link`; its position is independent of the passive wrist joint
(rotating that joint only spins the frame in place), which is exactly why the
problem is treated as 3-DOF.

### Chain structure

| Step | Joint | URDF name | Axis | Type |
|------|-------|-----------|------|------|
| q0 | base yaw | `after_base_full_joint` | Z | actuated |
| q1 | shoulder pitch | `link1_joint` | Y | actuated |
| q2 | elbow pitch | `link2_joint` | Y | actuated |
| (passive) | wrist | `end_effector_joint` | Y | passive (coupled) |

`ACTUATED_JOINTS = ["after_base_full_joint", "link1_joint", "link2_joint"]`,
`PASSIVE_JOINT = "end_effector_joint"`. All four joints share limits
`(-3.14, 3.14)` rad.

### Link / joint offset vectors (URDF origins, metres)

Taken verbatim from `new_arm.urdf` (all link `rpy` are zero, so each transform
is `Trans(origin) @ Rot(axis, q)`). NumPy names `T1..T4`; Torch names `_T1.._T4`
(identical values).

```python
T1 = [0.00043239, -0.00184063,  0.08396122]   # base_link  -> after_base_full (axis Z)
T2 = [0.00814224, -0.00043239,  0.00161718]   # after_base -> link1          (axis Y)
T3 = [0.001,      -0.00856,     0.12839612]   # link1      -> link2          (axis Y)
T4 = [0.12531939,  0.01099,    -0.05778452]   # link2      -> end_effector   (axis Y, passive)
```

### Rotation primitives

$$
R_z(q)=\begin{bmatrix} c & -s & 0\\ s & c & 0\\ 0 & 0 & 1\end{bmatrix},\qquad
R_y(q)=\begin{bmatrix} c & 0 & s\\ 0 & 1 & 0\\ -s & 0 & c\end{bmatrix},\qquad
c=\cos q,\; s=\sin q
$$

```python
def _rz(q):
    c, s = cos(q), sin(q)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]
def _ry(q):
    c, s = cos(q), sin(q)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
```

### Forward kinematics (exact, by transform composition)

For actuated joints $q=[q_0,q_1,q_2]$ define the cumulative rotations

$$
R_1 = R_z(q_0),\qquad
R_2 = R_1\,R_y(q_1),\qquad
R_3 = R_2\,R_y(q_2),
$$

then the wrist-pivot ("tip") position is

$$
\boxed{\;\mathbf{p}_{\text{tip}} = T_1 + R_1\,T_2 + R_2\,T_3 + R_3\,T_4\;}
$$

```python
def forward_kinematics(q):
    R1 = _rz(q[0]); R2 = R1 @ _ry(q[1]); R3 = R2 @ _ry(q[2])
    return T1 + R1 @ T2 + R2 @ T3 + R3 @ T4
```

`fk_full_chain` returns all five joint-origin points
`[p0, p1, p2, p3, p4]` with the partial sums:
`p0 = 0`, `p1 = T1`, `p2 = T1 + R1·T2`, `p3 = T1 + R1·T2 + R2·T3`,
`p4 = tip`. The Torch version (`kinematics_torch.forward_kinematics`) implements
the identical formula batched: input `q` shape `(N, 3)`, output `(N, 3)`, using
`_rz`/`_ry` that produce `(N,3,3)` rotations and `torch.bmm` for the products.

Reference sanity values (from the `__main__` self-test): `q=[0,0,0]` →
`tip≈[0.135, 0.0, 0.156]`-class point; `q=[0,0.4,-0.6]` and `q=[0.5,0.3,-0.4]`
also printed with the coupled passive angle.

### Planar reach model and derived constants

Each pitch joint rotates about +Y, so after the base yaw the arm moves in a
single **(radial, z)** plane. A link vector $v=(v_x,v_y,v_z)$ rotated by
$R_y(q)$ lands at planar angle $(q+\varphi)$ measured **from +z toward +radial**,
where $\varphi=\operatorname{atan2}(v_x,v_z)$, keeping planar length
$\lvert(v_x,v_z)\rvert$. The small $v_y$ terms (sub-cm) are dropped for the
analytic seed, then polished out. This yields the 2-link constants:

$$
A=\sqrt{T_{3x}^2+T_{3z}^2},\quad
B=\sqrt{T_{4x}^2+T_{4z}^2},\quad
\varphi_A=\operatorname{atan2}(T_{3x},T_{3z}),\quad
\varphi_B=\operatorname{atan2}(T_{4x},T_{4z})
$$

with exact computed values:

| Constant | Code name | Value |
|----------|-----------|-------|
| Shoulder→elbow length | `_A` | `0.12840001413961916` m |
| Elbow→wrist length | `_B` | `0.13800000094783516` m |
| Intrinsic angle of link a | `_PHI_A` | `0.007788239870681536` rad |
| Intrinsic angle of link b | `_PHI_B` | `2.0028409287283235` rad |
| Max planar reach | `_A + _B` | `0.2664000150874543` m |
| Min planar reach | `abs(_A - _B)` | `0.009599986808215999` m |
| Shoulder pivot (base frame) | `_SHOULDER = T1 + T2` | `[0.00857463, -0.00227302, 0.0855784]` m |
| Shoulder height | `_SH_Z` | `0.0855784` m |

### Inverse kinematics (analytic 3-DOF, position-only)

`inverse_kinematics(target_xyz, elbow_up=True)` returns `(q, ok)`.

**1) Base yaw** aligns the arm plane with the target azimuth:

$$
q_0 = \operatorname{atan2}(y, x)
$$

**2) Project target into the (radial, z) plane relative to the shoulder pivot.**
The shoulder's radial coordinate after yaw is
`sh_r = _SHOULDER_x·cos(q0) + _SHOULDER_y·sin(q0)`:

$$
X = \sqrt{x^2+y^2} - \text{sh\_r},\qquad
Y = z - \text{\_SH\_Z},\qquad
d=\sqrt{X^2+Y^2}
$$

**3) Reach clamping / `ok` flag.** With `reach = _A + _B`:

- if $d>\text{reach}$: scale $(X,Y)$ by $s=(\text{reach}-10^{-9})/d$, set
  $d=\text{reach}-10^{-9}$, **`ok = False`**;
- elif $d<\lvert A-B\rvert$: scale by $s=(\lvert A-B\rvert+10^{-9})/\max(d,10^{-12})$,
  set $d=\lvert A-B\rvert+10^{-9}$, **`ok = False`**;
- otherwise **`ok = True`**.

**4) Two-link law of cosines** (angles measured from +z toward +radial):

$$
\cos\delta = \frac{d^2 - A^2 - B^2}{2AB}\quad(\text{clamped to }[-1,1]),\qquad
\delta=\arccos(\cos\delta)
$$

The **`elbow_up`** branch is selected by the sign of $\delta$: if `not elbow_up`,
$\delta \leftarrow -\delta$. Then the shoulder planar angle is

$$
\alpha_a = \operatorname{atan2}(X, Y) - \operatorname{atan2}\!\big(B\sin\delta,\; A + B\cos\delta\big)
$$

**5) Map planar angles back to URDF joint angles** (subtract the intrinsic link
offsets):

$$
q_1 = \alpha_a - \varphi_A,\qquad
q_2 = \delta + \varphi_A - \varphi_B
$$

All three are wrapped into $(-\pi,\pi]$ via
`wrap_pi(a) = atan2(sin a, cos a)`.

**6) Damped-Newton polish.** Because FK is exact and cheap, the analytic seed is
refined to ~0 residual by `_newton_polish` (the $v_y$ terms dropped in step 2
are recovered here). It runs up to `iters=8` damped least-squares steps
($\lambda=10^{-4}$, `tol=1e-9`) using a numerical central-difference Jacobian
(`eps=1e-6`):

$$
\Delta q = \big(J^\top J + \lambda I\big)^{-1} J^\top\,(\text{target}-\text{FK}(q)),\qquad
q \leftarrow q + \Delta q
$$

stopping early when $\lVert\text{err}\rVert^2 < \text{tol}^2$; each result
component is `wrap_pi`-ed.

### Passive wrist coupling

The wrist joint (`end_effector_joint`, axis Y) has no motor; it is driven by a
linear coupling of the shoulder and elbow angles. With

```python
PASSIVE_OFFSET = 2.0 * math.pi   # = 6.283185307179586
PASSIVE_C_SH   = -1.0
PASSIVE_C_EL   = -1.0
```

the coupled wrist angle is

$$
\theta_4 = \operatorname{wrap}_\pi\!\big(\underbrace{2\pi}_{\text{PASSIVE\_OFFSET}}
+ \underbrace{(-1)}_{C_{sh}}\,q_{1} + \underbrace{(-1)}_{C_{el}}\,q_{2}\big)
= \operatorname{wrap}_\pi(2\pi - q_{sh} - q_{el})
$$

```python
def passive_wrist(q_shoulder, q_elbow):
    v = PASSIVE_OFFSET + PASSIVE_C_SH * q_shoulder + PASSIVE_C_EL * q_elbow
    return wrap_pi(v)          # torch: atan2(sin v, cos v)
```

**Why:** $\theta_4 = -(q_{sh}+q_{el})$ (mod $2\pi$) cancels the cumulative pitch
of the shoulder + elbow, so the end-effector frame stays **level (parallel to
the base plane)** regardless of arm posture — the plug points radially outward
into the socket. The Torch port is identical and lets the env command the
leveled wrist directly.

> **Sim/twin calibration note:** The original ESPMax firmware offset was
> `2*pi - pi/2` (home wrist at $-\pi/2$). The `- pi/2` term was **intentionally
> dropped** so the wrist is level (0) at home (`home EE = 0`). This is a
> deliberate sim/twin change (see `import_urdf.py`); the twin therefore no
> longer matches the un-recalibrated real ESPMax firmware until the hardware
> calibration is updated. Both `kinematics.py` and `kinematics_torch.py` are
> kept in sync on this.

### rad ↔ servo-pulse calibration

Lives in `config.py` (mirrors `new_arm_driver/config/calibration.yaml` so the
twin behaves like the firmware). Per-actuated-joint affine map, then clamp:

$$
\text{pulse}_i = \operatorname{clip}\!\big(\text{CAL\_SCALE}_i\cdot\theta_i + \text{CAL\_OFFSET}_i,\;
\text{PULSE\_MIN}_i,\;\text{PULSE\_MAX}_i\big)
$$

$$
\theta_i = \frac{\text{pulse}_i - \text{CAL\_OFFSET}_i}{\text{CAL\_SCALE}_i}
\quad(\theta_i = 0 \text{ if } \text{CAL\_SCALE}_i = 0)
$$

| Joint i | `CAL_SCALE` (pulse/rad) | `CAL_OFFSET` (pulse) | `CAL_PULSE_MIN` | `CAL_PULSE_MAX` |
|---------|------------------------|----------------------|-----------------|-----------------|
| 0 (base) | `238.7324` | `625.0` | `0.0` | `1000.0` |
| 1 (shoulder) | `-238.7324` | `875.0` | `0.0` | `700.0` |
| 2 (elbow) | `238.7324` | `500.0` | `470.0` | `1000.0` |

```python
CAL_SCALE  = [238.7324, -238.7324, 238.7324]
CAL_OFFSET = [625.0, 875.0, 500.0]
CAL_PULSE_MIN = [0.0, 0.0, 470.0]
CAL_PULSE_MAX = [1000.0, 700.0, 1000.0]
```

The scale magnitude `238.7324 = 1000 / (240° in rad)` is the standard
Hiwonder/Lobot bus-servo mapping (1000-pulse range over a 240° span); joint 1's
**negative** scale flips the shoulder's rotation sense, and the per-joint
offsets place the zero-radian home pose at pulses `[625, 875, 500]`.

### Reference firmware port (degrees + mm)

For real-arm cross-checks `kinematics.py` also ports the manufacturer
`_espmax.cpp` FK/IK using link lengths (mm, from `_espmax.h`):
`L0,L1,L2,L3,L4 = 84.4, 8.14, 128.4, 138.0, 16.8`. `fk_firmware` adds a 150°
base offset (`a1 = q0 + 150°`, minus 2π if it overflows), solves a law-of-cosines
triangle on `L2,L3`, and returns `[-x, y, z]` in mm. `ik_firmware` inverts it,
applying a `if deg <= 30: deg += 360` branch and a final `-150°` base
de-offset. These are reference only — the URDF transform-composition FK above is
authoritative for the sim/RL stack.

### Workspace sampling bounds (RL reach targets)

`kinematics_torch.WORKSPACE` constrains randomly sampled reach targets inside
the planar reach ($A+B\approx0.266$ m), away from the base axis:

```python
WORKSPACE = {"z": (0.02, 0.34), "r_min": 0.05, "r_max": 0.26}
```

---
## RL Environments

All three tasks are IsaacLab **`DirectRLEnv`** environments built on a common base (`NewArmReachEnv` / `NewArmReachEnvCfg`). `Insert` subclasses `Reach`, and `VisionInsert` subclasses `Insert`, so each task inherits the proven training machinery (joint-delta action, bounded tanh-attractor reward, fixed-length episodes, reachable-by-construction target sampling) and overrides only what changes. The robot is the 3-DOF MaxArm: 3 actuated joints `["after_base_full_joint", "link1_joint", "link2_joint"]` (base yaw, shoulder pitch, elbow pitch) plus one **passive** wrist joint `"end_effector_joint"`. The tip body is `"end_effector_link"`.

| Class | Cfg class | `task` id | obs dim | action dim | state (critic) dim |
|---|---|---|---|---|---|
| `NewArmReachEnv` | `NewArmReachEnvCfg` | `NewArm-Reach-v0` | 12 | 3 | 0 |
| `NewArmInsertEnv` | `NewArmInsertEnvCfg` | `NewArm-Insert-v0` | 12 | 3 | 0 |
| `NewArmVisionInsertEnv` | `NewArmVisionInsertEnvCfg` | `NewArm-VisionInsert-v0` | 15 | 3 | 12 (asymmetric AC) |

### Episode / timing structure (all tasks)

- `sim.dt = 1/120 s` (120 Hz physics), `decimation = 4`, `render_interval = decimation` → control runs at **30 Hz** (control dt = 4/120 = 1/30 s).
- `episode_length_s`: **5.0 s** (Reach) → ~150 control steps; **6.0 s** (Insert and VisionInsert, inherited) → ~180 control steps.
- `scene`: `num_envs = 1024`, `env_spacing = 1.0`, `replicate_physics = True`, `clone_in_fabric = False`.
- **Fixed-length truncation only** — episodes never terminate on success. In `_get_dones`: `terminated = zeros(bool)` always; `truncated = episode_length_buf >= max_episode_length - 1`. This makes mean episode reward a clean monotone signal of reach quality.
- Actuators: a single `ImplicitActuatorCfg` on `.*` with `effort_limit_sim=100.0`, `velocity_limit_sim=20.0`, `stiffness=5000.0`, `damping=150.0`; `fix_root_link=True`, `enabled_self_collisions=False`, `soft_joint_pos_limit_factor=0.95`.

### Action space (all tasks)

3-D continuous, interpreted as **per-step joint deltas** (no IK in the hot loop). In `_pre_physics_step`:

```python
self.actions[:] = torch.clamp(actions, -1.0, 1.0)                     # raw action ∈ [-1, 1]^3
q = joint_pos[:, arm_joint_ids]                                       # current actuated angles
q_target = clamp(q + actions * max_joint_step, -joint_limit, joint_limit)
# passive wrist commanded to its mechanical coupling each step:
passive = kt.passive_wrist(q_target[:,1], q_target[:,2])             # shoulder=col1, elbow=col2
q_target_all[:, passive_joint_ids] = passive
robot.set_joint_position_target(q_target_all)
```

- `max_joint_step`: **`radians(6.0)` ≈ 0.10472 rad/step (Reach)**; overridden to **`radians(3.0)` ≈ 0.05236 rad/step (Insert / VisionInsert)** for a finer sub-cm endgame.
- `joint_limit = math.pi` (≈ ±3.14159 rad), matching the URDF actuated-joint limit.
- **Passive wrist** (`end_effector_joint`) is *not* an action — it is set every step from `kt.passive_wrist(q_shoulder, q_elbow)`:
  ```python
  PASSIVE_OFFSET = 2.0 * math.pi   # firmware "- pi/2" dropped so wrist is level (0) at home
  v = 2π + (-1.0)*q_shoulder + (-1.0)*q_elbow
  passive_wrist = atan2(sin(v), cos(v))   # wrapped to (-π, π]
  ```
  This keeps the EE **level**, so the plug always points radially outward (yawed by the base joint).

### Reach: observation, reward, sampling

**Observation (12-D, key `"policy"`)** — `cat([pos_error(3), q(3), dq(3), prev_action(3)])`:
- `pos_error = target_pos_w − tip_pos_w` (3): world-frame vector from the wrist-tip (`end_effector_link` origin) to the target.
- `q` (3): actuated joint positions `[base, shoulder, elbow]`.
- `dq` (3): actuated joint velocities.
- `prev_action` (3): the previous clamped action.

**Reward** (`_get_rewards`), with `d = ‖target_pos_w − tip_pos_w‖`, `action_rate = ‖actions − prev_actions‖`, `dq_norm = ‖dq‖`, `within = (d < success_pos_error)`:

```
r_reach   =  w_reach   * (1 − tanh(d / reach_std))     # coarse bounded tanh attractor
r_fine    =  w_fine    * (1 − tanh(d / fine_std))      # fine tanh attractor (sub-cm)
r_dist    = −w_dist    * d                             # small L2 distance penalty
r_arate   = −w_action_rate * action_rate              # penalize jerky action changes
r_jvel    = −w_joint_vel * dq_norm                    # penalize joint speed
r_bull    =  w_bullseye * within.float()              # dense bonus while within tolerance
r_center  =  w_center  * exp(−(d / center_std)²)       # graded gaussian centering well
reward    = r_reach + r_fine + r_dist + r_arate + r_jvel + r_bull + r_center
```

| Weight | Reach value | std/param |
|---|---|---|
| `w_reach` | 1.0 | `reach_std = 0.10 m` |
| `w_fine` | 0.5 | `fine_std = 0.02 m` |
| `w_dist` | 0.2 | — |
| `w_action_rate` | 0.002 | — |
| `w_joint_vel` | 0.001 | — |
| `w_bullseye` | 2.0 | — |
| `w_center` | **0.0** (off in Reach) | `center_std = 0.004 m` |

**Success criterion**: `success_pos_error = 0.01 m` (1 cm). Used ONLY for the bullseye bonus + logged metrics, **not** for termination.

**Metrics logged each step** (`extras["log"]`, fresh dict per step):
- Per-step (rollout time-average): `metric/mean_pos_error_mm = 1000*d.mean()`, `metric/success_rate = within.mean()`, and the component reward means `reward/{reach,fine,dist,bullseye,center}`.
- Per-episode **terminal** metrics (only on the truncation step, `episode_length_buf >= max_episode_length-1`, when the tip is still at its FINAL pose): `metric/ep_terminal_success = within[term_mask]` and `metric/ep_terminal_err_mm = 1000*d[term_mask]`. These give a true per-episode success rate (the per-step `success_rate` is bounded well below 1, empirically ~0.8, by the unavoidable travel phase).

**Target sampling — reachable by construction** (`_sample_reachable_targets`): sample random joint configs, run batched FK (`kt.forward_kinematics`, returns the wrist-pivot position), and keep tips passing a workspace filter (rejection sampling, oversampled `m = max(4*(n−got), 256)`, up to 64 iterations):
- `sample_q_base = (−π, π)` (full base yaw circle)
- `sample_q_shoulder = (−1.2, 1.2)` rad
- `sample_q_elbow = (−1.2, 1.2)` rad
- Filter: `target_z_range = (0.05, 0.34) m` AND radial `r = ‖t[:, :2]‖ ≥ target_r_min = 0.05 m` (off the base axis, avoid near-base singularity).
- On `_reset_idx`, arm is reset to home (all actuated + passive = 0; offset 2π → level wrist), `actions`/`prev_actions` zeroed.

**Lighting**: `dome_intensity = 2000.0`. Optional eye-in-hand camera `enable_camera = False` by default (`CAMERA_RL_RESOLUTION = (72, 128)`, reuses the prim baked into the USD with `spawn=None`).

### Insert: overrides on Reach

`NewArmInsertEnv(NewArmReachEnv)` keeps the identical 12-D observation, action, reward *shape*, and metrics, but:

**Tracked point = the PLUG TIP** instead of the wrist pivot. `_tip_pos_w` is overridden:
```python
PLUG_TIP_LEN = PLUG_OFFSET_XYZ[0] + PLUG_BODY_LENGTH + PLUG_PRONG_LENGTH
             = 0.18 + 0.04 + 0.03 = 0.25 m   # offset along end_effector_link +X
tip_pos_w = ee_pos + quat_apply(ee_quat, [PLUG_TIP_LEN, 0, 0])
```
All reward/obs `d`, `pos_error`, success use this plug-tip position. Because the passive wrist keeps the EE level, the plug points radially outward, so reducing plug-tip→socket distance drives the plug straight into a radially-facing socket.

**Socket sampling** (`_sample_reachable_targets`, plug-tip aware, with curriculum): sample joint configs, FK to wrist pivot, add the plug-tip offset along the base-yaw radial direction (`off = PLUG_TIP_LEN * [cos(q_base), sin(q_base), 0]`), then filter on the plug-tip `tip = wrist + off`:
- `sample_q_base = (−0.9, 0.9)` (frontal fan), `sample_q_shoulder = (−1.2, 1.2)`, `sample_q_elbow = (−1.2, 1.2)`.
- Filter: `target_z_range = (0.06, 0.30) m`, `target_r_min = 0.12 m`, `target_r_max = 0.42 m` (drops far edge-of-reach sockets).

**Curriculum** (ramps the joint-sample half-ranges from easy→full over env-steps):
```python
frac = min(1.0, common_step_counter / curriculum_steps)   # curriculum_steps = 6000
base = lerp(curr_base_easy=0.3, sample_q_base[1]=0.9)
arm  = lerp(curr_arm_easy=0.4, sample_q_shoulder[1]=1.2)
lo = [-base, -arm, -arm];  hi = [base, arm, arm]
```
The r/z filter is held CONSTANT (reachability is by construction; the plug tip is ~0.25 m radial so r is always large).

**Changed reward weights / timing** (vs Reach):

| Param | Insert value | (Reach) |
|---|---|---|
| `success_pos_error` | 0.01 m | 0.01 m |
| `max_joint_step` | `radians(3.0)` | `radians(6.0)` |
| `w_fine` | **1.5** | 0.5 |
| `fine_std` | **0.008 m** | 0.02 m |
| `w_bullseye` | **1.0** | 2.0 |
| `w_center` | **5.0** (on) | 0.0 |
| `center_std` | 0.004 m | 0.004 m |
| `episode_length_s` | **6.0 s** | 5.0 s |
| `dome_intensity` | **1200.0** | 2000.0 |

So Insert's precision ladder is: coarse tanh (`reach_std` 10 cm, `w_reach` 1.0) → fine tanh (`fine_std` 8 mm, `w_fine` 1.5) → gaussian centering well (`center_std` 4 mm, `w_center` 5.0). The per-env marker is a real socket model (`socket_marker.usd`), yawed so the opening faces the base via `Rz(az)`, `az = atan2(local_y, local_x)`.

### VisionInsert: overrides on Insert

`NewArmVisionInsertEnv(NewArmInsertEnv)` keeps **identical** plug-tip tracking, socket sampling, curriculum, and the ground-truth (privileged) reward. It only changes the policy observation to camera-derived bearings, using an **asymmetric actor-critic**:

- `observation_space = 15` (actor / `"policy"`), `state_space = 12` (critic / `"critic"`).
- `cam_pixel_noise = 0.012` (gaussian σ added to the normalized bearings).

**Actor observation (15-D, key `"policy"`)** — `cat([fL(3), fR(3), q(3), dq(3), prev_action(3)])`:
- `fL`, `fR` = features for the **two socket holes** (left/right), each `[nx, ny, vis]`:
  - Hole world positions: socket centre `± (SOCKET_HOLE_SPACING/2 = 0.0095 m)` along the socket lateral `lat = [−sin(az), cos(az), 0]`, `az = atan2(local_y, local_x)`.
  - Each hole is projected into the wrist camera. Camera pose = EE pose offset by `CAMERA_OFFSET_XYZ = (0.20, 0.0, 0.05)` and rotated by a constant local quaternion `RotateY(CAMERA_PITCH_DEG=−60°) * RotateZ(CAMERA_ROLL_DEG=−90°)` (the −90° roll → portrait image).
  - Projection (USD camera looks down its own −Z): `pc = quat_apply(conj(cam_quat), P − cam_pos)`; `depth = −pc_z`, `d = clamp(depth, 1e-4)`;
    ```
    nx = (pc_x / d) / tan(HFOV/2),   tan_h = CAMERA_HORIZONTAL_APERTURE / (2*focal) = 11.78719/(2*14.84588)
    ny = (pc_y / d) / tan(VFOV/2),   tan_v = CAMERA_VERTICAL_APERTURE   / (2*focal) = 20.955 /(2*14.84588)
    vis = (depth > near=0.01) & (|nx| ≤ 1) & (|ny| ≤ 1)
    ```
    (HFOV ≈ 43.3°, VFOV ≈ 70.4°.)
  - Feature: `v = vis.float()`; `nx = (nx + 0.012*randn) * v`, `ny = (ny + 0.012*randn) * v` — bearings are **zeroed when the hole is not visible** (`vis=0`), forcing active perception (move to bring the socket into FOV).
- `q`, `dq`, `prev_action` (3 each): same proprioception as the other tasks.

**Critic observation (12-D, key `"critic"`)** — privileged ground-truth `cat([pos_err(3), q(3), dq(3), prev_action(3)])`, where `pos_err = target_pos_w − tip_pos_w` (true plug-tip → socket vector). The critic sees true state for stable value estimates while the actor stays camera-only (real-deployable). The **reward is unchanged** from Insert (privileged, ground-truth — standard "observable actor, privileged reward" setup).

---
## Training Algorithm — PPO (RSL-RL)

All three tasks are trained with **RSL-RL's on-policy PPO** runner (`RslRlOnPolicyRunnerCfg` + `RslRlPpoActorCriticCfg` + `RslRlPpoAlgorithmCfg`), configured in `agents/rsl_rl_ppo_cfg.py`. Each gym task id is bound to its runner cfg via the `rsl_rl_cfg_entry_point` kwarg in `new_arm_isaac/__init__.py`.

### Task → runner cfg binding (gym.register)

| Gym id | Runner cfg class | `experiment_name` |
|---|---|---|
| `NewArm-Reach-v0` | `NewArmReachPPORunnerCfg` | `new_arm_reach` |
| `NewArm-Insert-v0` | `NewArmInsertPPORunnerCfg` | `new_arm_insert` |
| `NewArm-VisionInsert-v0` | `NewArmVisionInsertPPORunnerCfg` | `new_arm_vision_insert` |

`NewArmInsertPPORunnerCfg` and `NewArmVisionInsertPPORunnerCfg` both **subclass** `NewArmReachPPORunnerCfg`, so any field not re-declared is inherited from the reach config.

### Runner-level config (rollout / scheduling)

| Field | Reach | Insert | VisionInsert |
|---|---|---|---|
| `num_steps_per_env` (rollout length) | `24` | `24` (inherited) | `24` (inherited) |
| `max_iterations` | `800` | `1200` | `1500` |
| `save_interval` | `50` | `50` (inherited) | `50` (inherited) |
| `experiment_name` | `new_arm_reach` | `new_arm_insert` | `new_arm_vision_insert` |
| `empirical_normalization` | `False` | `False` (inherited) | `False` (inherited) |
| `obs_groups` | (default) | (default) | `{"policy": ["policy"], "critic": ["critic"]}` |

VisionInsert is the only one that overrides `obs_groups`, giving it an **asymmetric actor-critic**: the actor reads the camera-feature `policy` obs group while the critic reads the privileged ground-truth `critic` obs group.

### Actor-Critic network (`RslRlPpoActorCriticCfg`)

| Field | Reach | Insert | VisionInsert (inherited from Reach) |
|---|---|---|---|
| `init_noise_std` | `1.0` | `1.0` | `1.0` |
| `actor_hidden_dims` | `[128, 128, 64]` | `[256, 256, 128]` | `[128, 128, 64]` |
| `critic_hidden_dims` | `[128, 128, 64]` | `[256, 256, 128]` | `[128, 128, 64]` |
| `activation` | `elu` | `elu` | `elu` |

Both actor and critic are **plain MLPs** (3 hidden layers each) with ELU activations. The policy is a Gaussian whose mean comes from the actor MLP and whose log-std is a learnable parameter initialized so the std starts at `init_noise_std = 1.0`. Insert uses a **larger net** (256/256/128) than reach/vision (128/128/64) to add capacity for covering failure states.

### PPO algorithm hyperparameters (`RslRlPpoAlgorithmCfg`)

| Field | Reach | Insert | VisionInsert (inherited from Reach) |
|---|---|---|---|
| `value_loss_coef` | `1.0` | `1.0` | `1.0` |
| `use_clipped_value_loss` | `True` | `True` | `True` |
| `clip_param` (ε) | `0.2` | `0.2` | `0.2` |
| `entropy_coef` | `0.005` | `0.01` | `0.005` |
| `num_learning_epochs` | `5` | `5` | `5` |
| `num_mini_batches` | `4` | `4` | `4` |
| `learning_rate` | `1.0e-3` | `1.0e-3` | `1.0e-3` |
| `schedule` | `adaptive` | `adaptive` | `adaptive` |
| `gamma` (γ) | `0.99` | `0.99` | `0.99` |
| `lam` (λ, GAE) | `0.95` | `0.95` | `0.95` |
| `desired_kl` | `0.01` | `0.01` | `0.01` |
| `max_grad_norm` | `1.0` | `1.0` | `1.0` |

The only algorithm-level difference is `entropy_coef`: Insert raises it to `0.01` (from `0.005`) for more exploration to visit and correct failure states; Reach and VisionInsert keep `0.005`.

> **Note on the Insert tuning comments.** The in-file docstring describes a planned regime (lower base LR + tighter KL, then fine-tune) for breaking a ~94.5% terminal-accuracy plateau, but the **actual committed values** keep `learning_rate=1.0e-3` and `desired_kl=0.01` (same as reach), changing only the network size (256/256/128) and `entropy_coef=0.01`.

### How RSL-RL PPO works here

1. **Rollout collection.** With `num_steps_per_env = 24`, the runner steps all parallel environments for 24 timesteps per iteration, storing observations, actions, sampled action log-probs, rewards, dones, and critic value estimates. Actions are sampled from the Gaussian actor (mean from the actor MLP, std from the learnable `init_noise_std`-initialized log-std).

2. **GAE advantage estimation.** After each 24-step rollout, advantages are computed with Generalized Advantage Estimation using `gamma = 0.99` (discount) and `lam = 0.95` (GAE λ):

   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$$

   Value targets are `returns = advantages + V(s_t)`, and advantages are normalized before the policy update.

3. **Clipped surrogate + value + entropy loss.** The collected batch is shuffled into `num_mini_batches = 4` mini-batches and optimized for `num_learning_epochs = 5` passes. The clipped PPO policy objective uses `clip_param = 0.2`:

   $$L^{\text{clip}} = \mathbb{E}\!\left[\min\!\big(\rho_t\hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\right], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

   The value loss (weight `value_loss_coef = 1.0`) is clipped (`use_clipped_value_loss = True`) around the old value prediction with the same `clip_param`. An entropy bonus (`entropy_coef`, 0.005 or 0.01) encourages exploration. Gradients are clipped to global norm `max_grad_norm = 1.0`.

4. **Adaptive-KL learning-rate schedule.** With `schedule = "adaptive"` and `desired_kl = 0.01`, after each update the empirical KL divergence between the old and new policy is measured: if KL exceeds ~`2 × desired_kl` the learning rate is decreased, and if KL falls below ~`desired_kl / 2` it is increased (bounded around the base `learning_rate = 1.0e-3`). This keeps per-iteration policy changes near the target KL, stabilizing training.

5. **Checkpointing.** The runner saves a model checkpoint every `save_interval = 50` iterations and runs for `max_iterations` (800 reach / 1200 insert / 1500 vision-insert). `empirical_normalization = False`, so observations are not running-mean/std normalized by the runner.

---

## 5. Training Procedure (exactly what was run)

Hardware: laptop RTX 4060 (8 GB). Sim: Isaac Sim 5.1.0 (binary), Isaac Lab v2.3.2, RSL-RL 3.1.2,
PyTorch 2.7.0+cu128, in conda env `env_isaaclab` (Python 3.11).

The custom task package must be importable so the gym ids register — the two RSL-RL runner scripts
have `import new_arm_isaac` (and `import maxarm_charger`) added near their post-AppLauncher imports.

**Train (headless, 1024 parallel envs):**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /DataDrive/IsaacLab
# (headless training does not need a display)
env -u DISPLAY ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task NewArm-Reach-v0  --num_envs 1024 --headless --max_iterations 800
env -u DISPLAY ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task NewArm-Insert-v0 --num_envs 1024 --headless --max_iterations 800
```
Checkpoints + TensorBoard logs land in `/DataDrive/IsaacLab/logs/rsl_rl/<experiment>/<timestamp>/`
(`model_<iter>.pt`, `events.out.tfevents.*`). `<experiment>` = `new_arm_reach` / `new_arm_insert`.

**Replay a trained policy in the GUI** (loads the latest run's highest checkpoint automatically):
```bash
DISPLAY=:1 LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task NewArm-Insert-v0 --num_envs 16
```

> **Two environment gotchas on this machine** (needed to reproduce the GUI/renders):
> 1. **GUI segfault fix.** The conda env's libstdc++ is newer than the system one (it needs
>    `CXXABI_1.3.15`); running the Isaac Sim GUI from the conda env otherwise segfaults at
>    `XOpenDisplay`. Fix: `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` (force the newer libstdc++).
>    Headless runs don't need it.
> 2. **Standalone scripts need torch.** Isaac Sim's bundled `python.sh` lacks `torch`, which
>    `isaacsim.core.api.World` imports. Run standalone scene scripts via
>    `PYTHONPATH=$CONDA_PREFIX/lib/python3.11/site-packages ./python.sh script.py` so they borrow
>    the conda env's torch (both are Python 3.11 → ABI-compatible). No re-download.

---

## 6. Results

Both policies were trained for 800 PPO iterations (1024 envs). Per-**episode terminal** success
(within the 1 cm tolerance) and terminal positioning error:

| Task | Final success | Peak success | Final terminal error |
|---|---|---|---|
| **Reach** (`NewArm-Reach-v0`) | **95.7 %** | 96.8 % | 5.7 mm |
| **Insert** (`NewArm-Insert-v0`) | **94.5 %** | 95.8 % | 8.7 mm |

(`metric/ep_terminal_success`, `metric/ep_terminal_err_mm` from TensorBoard, mean of last 25 logged
points.) The per-step `metric/success_rate` is much lower (~0.8 ceiling even for a perfect policy)
because it time-averages over the unavoidable travel phase of each fixed-length episode — the
terminal metric is the correct "accuracy".

**Reachability note (Insert).** The arm's planar wrist reach is only ≈ 0.266 m, while the socket
sits at radius 0.334 m — so the *wrist* cannot reach the socket, but the **plug extends 0.25 m past
the end-effector**, so the wrist only needs to reach ≈ 0.084 m for the plug tip to seat in the
socket. Analytic IK confirms this: `IK(EE=[0.084,0,0.12]) → q=[-0.002,-0.232,0.518], ok=True`.
The Insert task encodes this by tracking the **plug tip** (not the wrist) and sampling sockets that
are reachable by the plug tip.

Figures rendered for the report (in `/DataDrive/new_arm_isaac/report_renders/`):
`fig_success_rate.png`, `fig_terminal_error.png`, `fig_reward.png`, `fig_{reach,insert}_panel.png`
(training curves); `arm_3quarter.png`, `arm_side.png`, `scene_insert_{3quarter,closeup,top}.png`
(RTX renders of the arm and the seated-plug insertion pose).

---

## 7. Key Files

```
~/Documents/our_work/new_arm_isaac/new_arm_isaac/
  config.py                       paths, joint names, geometry, calibration, named poses
  kinematics.py                   analytic FK/IK (numpy), passive-wrist coupling, rad<->pulse
  kinematics_torch.py             batched FK + passive coupling (torch) for the vectorized envs
  __init__.py                     gym.register(NewArm-Reach/Insert/VisionInsert)
  agents/rsl_rl_ppo_cfg.py        PPO runner cfgs (per task)
  tasks/new_arm_reach_env.py      base DirectRLEnv (NewArm-Reach-v0)
  tasks/new_arm_insert_env.py     plug-tip tracking subclass (NewArm-Insert-v0)
  tasks/new_arm_vision_insert_env.py  camera-bearing subclass (NewArm-VisionInsert-v0)
  scripts/import_urdf.py          URDF -> USD + bake EE camera + plug
  scripts/{play_with_camera,socket_demo,render_usd,render_scene}.py   eval / rendering
/DataDrive/new_arm_isaac/usd/     new_arm.usd, socket.usd, socket_marker.usd  (DATA_DIR assets)
/DataDrive/IsaacLab/logs/rsl_rl/{new_arm_reach,new_arm_insert}/   checkpoints + tb logs
```
