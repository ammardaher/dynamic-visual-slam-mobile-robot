#!/usr/bin/env python3
"""Build exact-BREP STEP (.stp) solids for the new_arm plug + socket, for CATIA V5.

Uses gmsh's OpenCASCADE kernel: real cylinders/planes + boolean cut for the holes
(NOT tessellated like the STLs), so CATIA imports clean analytic geometry.

Geometry is built in MILLIMETRES, straight from new_arm_isaac/config.py.
Each part is written as its own .stp (separate CATIA parts to drop into an assembly),
then re-opened to validate bounding box / volume / solid count.

Run:  PYTHONPATH=<cadenv>/site LD_LIBRARY_PATH=<cadenv>/site/lib python3 build_step.py
"""
import importlib.util
import math
import os
import sys

import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "new_arm_isaac"))
MM = 1000.0


def load_config():
    spec = importlib.util.spec_from_file_location("na_config", os.path.join(PKG, "config.py"))
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["na_config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg


def new_model(name):
    gmsh.clear()
    gmsh.model.add(name)


def build_plug(cfg):
    """Body cylinder + 2 prongs, fused into one solid. Local origin = plug attach point
    (0.18 m ahead of end_effector_link); tip at x=+70 mm."""
    new_model("plug")
    occ = gmsh.model.occ
    bl = cfg.PLUG_BODY_LENGTH * MM
    br = cfg.PLUG_BODY_RADIUS * MM
    pl = cfg.PLUG_PRONG_LENGTH * MM
    pr = cfg.PLUG_PRONG_RADIUS * MM
    sp = cfg.PLUG_PRONG_SPACING / 2.0 * MM
    ov = 0.5  # mm overlap into body so the boolean fuse is clean (tip stays at bl+pl)
    body = occ.addCylinder(0, 0, 0, bl, 0, 0, br)
    pL = occ.addCylinder(bl - ov, sp, 0, pl + ov, 0, 0, pr)
    pR = occ.addCylinder(bl - ov, -sp, 0, pl + ov, 0, 0, pr)
    occ.fuse([(3, body)], [(3, pL), (3, pR)])
    occ.synchronize()


def _drilled_plate(occ, cfg, x0, with_stand):
    """Faceplate box spanning x in [x0, x0+2*hx] with 2 through-holes; optional stand fused.
    Returns nothing (leaves the result solid in the model)."""
    hx = cfg.SOCKET_FACEPLATE_HALF[0] * MM
    hy = cfg.SOCKET_FACEPLATE_HALF[1] * MM
    hz = cfg.SOCKET_FACEPLATE_HALF[2] * MM
    hr = cfg.SOCKET_HOLE_RADIUS * MM
    hsp = cfg.SOCKET_HOLE_SPACING / 2.0 * MM
    plate = occ.addBox(x0, -hy, -hz, 2 * hx, 2 * hy, 2 * hz)
    holes = []
    for sgn in (+1, -1):
        # cylinder long enough to pierce the whole plate (start 5mm before, length plate+10)
        holes.append((3, occ.addCylinder(x0 - 5.0, sgn * hsp, 0, 2 * hx + 10.0, 0, 0, hr)))
    out, _ = occ.cut([(3, plate)], holes)
    if with_stand:
        sxh = cfg.SOCKET_STAND_HALF[0] * MM
        syh = cfg.SOCKET_STAND_HALF[1] * MM
        faceplate_bottom = (cfg.SOCKET_POSE_XYZ[2] - cfg.SOCKET_FACEPLATE_HALF[2]) * MM  # 77
        stand_top = -hz + 0.5  # overlap 0.5mm into the plate bottom for a clean fuse
        stand_bottom = -hz - faceplate_bottom  # = -120
        stand = occ.addBox(-sxh, -syh, stand_bottom, 2 * sxh, 2 * syh, stand_top - stand_bottom)
        occ.fuse(out, [(3, stand)])
    occ.synchronize()


def build_socket(cfg):
    """Faceplate centre at origin (matches socket.usd frame); drilled holes + stand."""
    new_model("socket")
    hx = cfg.SOCKET_FACEPLATE_HALF[0] * MM
    _drilled_plate(gmsh.model.occ, cfg, -hx, with_stand=True)


def build_socket_marker(cfg):
    """Front face at origin (matches the hole-centred RL marker); drilled holes, no stand."""
    new_model("socket_marker")
    _drilled_plate(gmsh.model.occ, cfg, 0.0, with_stand=False)


def write_and_validate(path):
    gmsh.write(path)
    # re-open fresh to validate the written file
    new_model("verify")
    gmsh.model.occ.importShapes(path)
    gmsh.model.occ.synchronize()
    solids = gmsh.model.getEntities(3)
    faces = gmsh.model.getEntities(2)
    bb = gmsh.model.getBoundingBox(-1, -1)
    vol = sum(gmsh.model.occ.getMass(3, t) for (_, t) in solids)
    name = os.path.basename(path)
    print(f"  {name:24s} solids={len(solids)} faces={len(faces):3d} vol={vol:10.1f} mm^3")
    print(f"      bbox mm: x[{bb[0]:8.2f},{bb[3]:8.2f}] y[{bb[1]:8.2f},{bb[4]:8.2f}] "
          f"z[{bb[2]:8.2f},{bb[5]:8.2f}]")
    return len(solids), len(faces), bb, vol


def main():
    cfg = load_config()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # ensure STEP is written in millimetres (OCCT default, set explicitly to be safe)
    try:
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    except Exception:
        pass

    print("Building STEP (.stp, millimetres) in", HERE)
    out = {}
    build_plug(cfg)
    out["plug.stp"] = write_and_validate(os.path.join(HERE, "plug.stp"))
    build_socket(cfg)
    out["socket.stp"] = write_and_validate(os.path.join(HERE, "socket.stp"))
    build_socket_marker(cfg)
    out["socket_marker.stp"] = write_and_validate(os.path.join(HERE, "socket_marker.stp"))

    gmsh.finalize()

    # ---- expected-value checks (mm) ----
    print("\nVerification vs config:")
    hr = cfg.SOCKET_HOLE_RADIUS * MM
    hd_through = 2 * cfg.SOCKET_FACEPLATE_HALF[0] * MM
    plate_vol = (2 * cfg.SOCKET_FACEPLATE_HALF[0] * MM) * (2 * cfg.SOCKET_FACEPLATE_HALF[1] * MM) * \
                (2 * cfg.SOCKET_FACEPLATE_HALF[2] * MM)
    holes_vol = 2 * math.pi * hr * hr * hd_through
    marker_expected = plate_vol - holes_vol
    checks = [
        ("plug tip X == 70mm", out["plug.stp"][2][3], 70.0),
        ("plug solids == 1 (fused)", out["plug.stp"][0], 1),
        ("socket faceplate+holes volume", out["socket_marker.stp"][3], marker_expected),
        ("socket_marker solids == 1", out["socket_marker.stp"][0], 1),
        ("socket bbox bottom == -120mm (floor)", out["socket.stp"][2][2], -120.0),
    ]
    allok = True
    for label, got, exp in checks:
        ok = abs(got - exp) < (max(1.0, abs(exp) * 0.01))
        allok &= ok
        print(f"  [{'OK' if ok else 'XX'}] {label:38s} got={got:11.2f} expected={exp:11.2f}")
    print(f"\nholes are real cut-outs: marker volume {out['socket_marker.stp'][3]:.1f} == "
          f"plate {plate_vol:.1f} - 2 holes {holes_vol:.1f} = {marker_expected:.1f}")
    print("All checks pass:", allok)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
