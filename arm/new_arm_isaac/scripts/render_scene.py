#!/usr/bin/env python3
"""Headless RTX scene renders for the report: the arm posed at the INSERTION pose with the
plug seated in the socket, plus a reach pose. Viewport capture (reliable; the EE TiledCamera
path deadlocks headless). Output -> report_renders.

Run:  cd /DataDrive/isaac-sim
      PYTHONPATH=/DataDrive/conda_envs/env_isaaclab/lib/python3.11/site-packages \
        ./python.sh /home/salman/Documents/our_work/new_arm_isaac/scripts/render_scene.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from new_arm_isaac import config            # noqa: E402
from new_arm_isaac import kinematics as kin  # noqa: E402

OUT_DIR = "/DataDrive/new_arm_isaac/report_renders"
os.makedirs(OUT_DIR, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720,
                                "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402


def log(m): print(f"[render_scene] {m}", flush=True)

PLUG_TIP_LEN = config.PLUG_OFFSET_XYZ[0] + config.PLUG_BODY_LENGTH + config.PLUG_PRONG_LENGTH  # 0.25 m
sk = np.array([float(v) for v in config.SOCKET_POSE_XYZ])


def place_socket(stage):
    add_reference_to_stage(usd_path=config.SOCKET_USD_PATH, prim_path=config.SOCKET_PRIM)
    sp = stage.GetPrimAtPath(config.SOCKET_PRIM)
    xf = UsdGeom.Xformable(sp); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(sk[0]), float(sk[1]), float(sk[2])))
    log(f"socket placed at {tuple(round(float(v),3) for v in sk)}")


def pose_arm(robot, ee_xyz):
    q_ee, ok = kin.inverse_kinematics(tuple(ee_xyz))
    dof = list(robot.dof_names)
    act = [dof.index(j) for j in config.ACTUATED_JOINTS]; pas = dof.index(config.PASSIVE_JOINT)
    tgt = np.array(robot.get_joint_positions(), dtype=float)
    for k, i in enumerate(act): tgt[i] = q_ee[k]
    tgt[pas] = kin.passive_wrist(q_ee[1], q_ee[2])
    for _ in range(140):
        robot.apply_action(ArticulationAction(joint_positions=tgt)); world.step(render=True)
    return q_ee, ok


def shoot(vp, name, eye, tgt):
    set_camera_view(eye=list(eye), target=list(tgt))
    for _ in range(70): world.step(render=True)
    path = os.path.join(OUT_DIR, name)
    capture_viewport_to_file(vp, path)
    for _ in range(70): simulation_app.update()
    log(f"wrote {path} ({os.path.getsize(path) if os.path.exists(path) else 0} B)")


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(usd_path=config.USD_PATH, prim_path=config.ROBOT_PRIM)
place_socket(stage)
prim_utils.create_prim("/World/DomeLight", "DomeLight",
                       attributes={"inputs:intensity": 1500.0, "inputs:texture:format": "latlong"})
robot = SingleArticulation(prim_path=config.ROBOT_PRIM, name="new_arm")
world.scene.add(robot); world.reset()
log(f"dof_names = {list(robot.dof_names)}")

# --- INSERTION pose: EE so the plug tip is at the socket face ---
ee_ins = sk - PLUG_TIP_LEN * np.array([1.0, 0.0, 0.0])  # socket is on +X axis
q, ok = pose_arm(robot, ee_ins)
log(f"insert pose q={np.round(q,3)} ok={ok}  (EE {np.round(ee_ins,3)}, plug tip -> socket {tuple(round(float(v),3) for v in sk)})")
vp = get_active_viewport()
shoot(vp, "scene_insert_3quarter.png", (0.55, -0.40, 0.42), (0.22, 0.0, 0.13))
shoot(vp, "scene_insert_closeup.png",  (0.40, -0.22, 0.22), (0.30, 0.0, 0.12))
shoot(vp, "scene_insert_top.png",      (0.22, 0.02, 0.55), (0.22, 0.0, 0.12))

simulation_app.close()
