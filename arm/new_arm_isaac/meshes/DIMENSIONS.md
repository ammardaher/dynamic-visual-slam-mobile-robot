# new_arm — Plug & Socket dimensions + STL meshes

All numbers come straight from `new_arm_isaac/config.py` (the single source of truth) and
`new_arm_isaac/usd_socket_plug.py` (how each primitive is placed). Generated/verified by
`meshes/generate_stls.py` — all four STLs are **watertight** and every dimension below is
asserted against config at generation time.

Sim works in **metres**; the STL files are written in **millimetres** (the STL convention).
Both units are listed below.

---

## 1. Plug

A child of `end_effector_link`, pointing forward along the EE **+X** axis. It is the male part
that inserts into the socket. One cylindrical **body** + two cylindrical **prongs**.

**Plug local frame** (used by `plug.stl`): origin = the plug's attachment point, which sits
`PLUG_OFFSET_XYZ = (0.18, 0, 0)` m ahead of `end_effector_link`. So `plug_local_x + 0.18 m`
= distance ahead of the wrist.

| Part | Quantity | Value (m) | Value (mm) | Notes |
|------|----------|-----------|-----------|-------|
| Body | length (along X) | 0.040 | 40 | `PLUG_BODY_LENGTH`; cylinder axis = X |
| Body | radius | 0.013 | 13 (Ø26) | `PLUG_BODY_RADIUS` |
| Body | X span (plug-local) | 0.000 → 0.040 | 0 → 40 | starts at the origin |
| Body | X span (EE frame) | 0.180 → 0.220 | 180 → 220 | + the 0.18 m offset |
| Body | colour | (0.08, 0.08, 0.10) | dark grey | |
| Prong ×2 | length (along X) | 0.030 | 30 | `PLUG_PRONG_LENGTH` |
| Prong ×2 | radius | 0.0035 | 3.5 (Ø7) | `PLUG_PRONG_RADIUS` |
| Prong ×2 | Y position | ±0.0095 | ±9.5 | `PLUG_PRONG_SPACING/2` |
| Prong ×2 | spacing (centre-to-centre) | 0.019 | 19 | `PLUG_PRONG_SPACING` |
| Prong ×2 | X span (plug-local) | 0.040 → 0.070 | 40 → 70 | emerge from the body end |
| Prong ×2 | X span (EE frame) | 0.220 → 0.250 | 220 → 250 | |
| Prong ×2 | colour | (0.95, 0.45, 0.05) | bright orange | high contrast vs blue socket |
| **Tip** | X (plug-local) | **0.070** | **70** | prong ends |
| **Tip** | X (EE frame) | **0.250** | **250** | `PLUG_TIP_LEN = 0.18 + 0.04 + 0.03` |

`PLUG_TIP_LEN = 0.25 m` is the lever used by the RL insert task and the analytic plug-tip IK —
the controller drives this point to the socket.

---

## 2. Socket

Static world geometry: a **faceplate** with two holes on a **support stand**. The faceplate
front face points back toward the arm (**−X**).

**Socket local frame** (used by `socket.stl` / `socket_sim.stl`): origin = faceplate **centre**.
In the world it is placed at `SOCKET_POSE_XYZ = (0.334, 0, 0.12)` m (local-Z = −0.12 is the floor).

| Part | Quantity | Value (m) | Value (mm) | Notes |
|------|----------|-----------|-----------|-------|
| Faceplate centre | world XYZ | (0.334, 0, 0.12) | (334, 0, 120) | `SOCKET_POSE_XYZ` |
| Faceplate | half-extents (X,Y,Z) | (0.004, 0.043, 0.043) | (4, 43, 43) | `SOCKET_FACEPLATE_HALF` |
| Faceplate | full size (X,Y,Z) | 0.008 × 0.086 × 0.086 | **8 × 86 × 86** | thin plate |
| Faceplate | front face X (world) | 0.330 | 330 | centre − 4 mm; faces −X |
| Faceplate | colour | (0.16, 0.28, 0.55) | blue | |
| Holes ×2 | radius | 0.0065 | 6.5 (Ø13) | `SOCKET_HOLE_RADIUS` |
| Holes ×2 | Y position | ±0.0095 | ±9.5 | `SOCKET_HOLE_SPACING/2` |
| Holes ×2 | spacing (centre-to-centre) | 0.019 | 19 | `SOCKET_HOLE_SPACING` (EU type-C/F) |
| Holes ×2 | recess depth | 0.012 | 12 | `SOCKET_HOLE_DEPTH` (see note ▼) |
| Holes ×2 | colour | (0.10, 0.90, 0.20) | bright green | vision-tracking aid |
| Stand | half-extents (X,Y) | (0.02, 0.02) | (20, 20) | `SOCKET_STAND_HALF[0:2]` |
| Stand | half-height Z (computed) | 0.0385 | 38.5 | `max(0.005, (0.12−0.043)/2)` |
| Stand | full size (X,Y,Z) | 0.04 × 0.04 × 0.077 | 40 × 40 × 77 | |
| Stand | Z span (world) | 0.0 → 0.077 | 0 → 77 | floor up to faceplate bottom |
| Stand | colour | (0.3, 0.3, 0.32) | grey | |
| Semantic class | label | `socket` | — | for Replicator/CenterPose |

**▼ Note on hole depth:** in the USD the holes are *solid green cylinders* 12 mm long (not
booleaned out), so they poke 4 mm out the back of the 8 mm plate — a colour/vision device, not
real geometry. `socket_sim.stl` reproduces that faithfully. `socket.stl` / `socket_marker.stl`
instead drill **clean 8 mm through-holes** (better for printing/CAD).

---

## 3. Fit between plug and socket

| Check | Value |
|-------|-------|
| Prong radius vs hole radius | 3.5 mm vs 6.5 mm → **3.0 mm radial clearance per side** (6 mm diametral) |
| Prong spacing == hole spacing | 19 mm == 19 mm ✓ (config enforces this) |
| Insertion axis | plug +X (EE frame) meets socket −X (front face) head-on when the wrist is level |

The EE is kept level by the passive-wrist coupling so the plug stays horizontal and enters the
holes. The eye-in-hand Logitech-C920 camera (baked at the EE, 45° fwd/down — see `config.py`
`CAMERA_*`) is what the vision/CenterPose pipeline uses to see these holes.

---

## 4. STL files

All in **millimetres**, binary STL, watertight (each verified: every edge shared by exactly 2
triangles). To re-import into a metric tool (Isaac/USD, ROS), set import scale **0.001**.

| File | Frame origin | Contents |
|------|--------------|----------|
| `plug.stl` | plug attach point (0.18 m ahead of `end_effector_link`); tip at x=+70 mm | body + 2 prongs (solid) |
| `socket.stl` | faceplate centre (local-Z −120 mm = floor) | faceplate + 2 **drilled** through-holes + stand — *printable* |
| `socket_marker.stl` | faceplate **front face** at origin | faceplate + 2 drilled holes, **no stand** — matches the hole-centred RL marker |
| `socket_sim.stl` | faceplate centre | **faithful to USD**: solid plate + 2 solid hole-cylinders (12 mm, poke through) + stand |

Regenerate any time with:

```
python3 meshes/generate_stls.py
```

(pure numpy; reads `config.py` directly so it never drifts from the sim).

---

## 5. STEP files (`.stp`) — for CATIA V5

Exact **BREP** solids (real cylinders + planes, holes are true boolean cut-outs — not
tessellated like the STLs). Schema **AP214 (AUTOMOTIVE_DESIGN)**, unit **millimetre** — drop
each straight into a CATIA V5 assembly (Insert > Existing Component, or open as a part).

| File | Solid | Faces | Volume (mm³) | Contents |
|------|-------|-------|--------------|----------|
| `plug.stp` | 1 (fused) | 7 | 23 546 | body + 2 prongs, one solid; tip at x=+70 mm |
| `socket.stp` | 1 (fused) | 16 | 180 884 | faceplate + 2 drilled through-holes + stand |
| `socket_marker.stp` | 1 | 8 | 57 044 | faceplate + 2 drilled holes, no stand (front face at origin) |

Frames match the STL files above. Drilled holes are clean **8 mm through-holes** (the USD's
12 mm recess is a render-only solid cylinder; not relevant for CAD).

Built with gmsh's OpenCASCADE kernel (installed under
`/media/salman/DataDrive1/new_arm_isaac/cadenv`, off the full root disk):

```
PYTHONPATH=<cadenv>/site LD_LIBRARY_PATH=<cadenv>/site/lib python3 meshes/build_step.py
```

(also reads `config.py` directly; re-opens every `.stp` to validate bbox/volume/solid count).
