#!/usr/bin/env python3
"""Generate STL meshes for the new_arm plug + socket directly from new_arm_isaac.config.

Pure numpy (no trimesh / no Isaac). Builds the procedural primitives that
usd_socket_plug.py bakes into USD, but as triangle meshes, and writes binary STL.

Outputs (written next to this file, in MILLIMETRES — STL convention; multiply the
sim's metres by 1000):
  plug.stl          body cylinder + 2 prongs (solid object, origin = plug local frame
                    = 0.18 m ahead of end_effector_link along +X; tip at +0.07 here).
  socket.stl        printable socket: faceplate with 2 *drilled* through-holes + stand.
                    Origin = faceplate centre (matches the standalone socket.usd frame).
  socket_marker.stl drilled faceplate + holes only (no stand), front face at origin
                    (matches the hole-centred RL VisualizationMarker variant).
  socket_sim.stl    faithful-to-USD: SOLID faceplate box + 2 SOLID hole cylinders
                    (the holes are colored cylinders in the USD, not subtracted) + stand.

Self-verifies: edge-manifold (watertight) check + bounding-box assertions against config.
"""
import importlib.util
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", "new_arm_isaac"))
MM = 1000.0  # metres -> millimetres for STL output


# ---------------------------------------------------------------- load config
def load_config():
    """Import config.py directly by path (avoids the package __init__ / Isaac)."""
    spec = importlib.util.spec_from_file_location("na_config", os.path.join(PKG, "config.py"))
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["na_config"] = cfg
    spec.loader.exec_module(cfg)
    return cfg


# ---------------------------------------------------------------- mesh helpers
def _orient(tris, ref):
    """Force each triangle's winding so its normal points along `ref` (callable or vec)."""
    out = []
    for t in tris:
        a, b, c = [np.asarray(v, float) for v in t]
        n = np.cross(b - a, c - a)
        r = ref(np.mean([a, b, c], axis=0)) if callable(ref) else np.asarray(ref, float)
        if np.dot(n, r) < 0:
            b, c = c, b
        out.append((a, b, c))
    return out


def cylinder_x(cx, length, radius, n=64, ref_axis_point=None):
    """Closed cylinder, axis along X, centred at x=cx (in y,z plane around (0,0) unless
    ref_axis_point given => cylinder centred on that y,z). Returns oriented triangles."""
    y0, z0 = (0.0, 0.0) if ref_axis_point is None else ref_axis_point
    x0, x1 = cx - length / 2.0, cx + length / 2.0
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ys, zs = y0 + radius * np.cos(th), z0 + radius * np.sin(th)
    r0 = [(x0, ys[k], zs[k]) for k in range(n)]
    r1 = [(x1, ys[k], zs[k]) for k in range(n)]
    side, cap0, cap1 = [], [], []
    c0, c1 = (x0, y0, z0), (x1, y0, z0)
    for k in range(n):
        kp = (k + 1) % n
        side += [(r0[k], r1[k], r1[kp]), (r0[k], r1[kp], r0[kp])]
        cap0.append((c0, r0[k], r0[kp]))
        cap1.append((c1, r1[k], r1[kp]))
    side = _orient(side, lambda p: np.array([0.0, p[1] - y0, p[2] - z0]))
    cap0 = _orient(cap0, np.array([-1.0, 0, 0]))
    cap1 = _orient(cap1, np.array([1.0, 0, 0]))
    return side + cap0 + cap1


def box(center, half):
    """Axis-aligned box. Returns oriented triangles (normals point outward)."""
    cx, cy, cz = center
    hx, hy, hz = half
    s = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    v = [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in s]
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    tris = []
    for a, b, c, d in quads:
        tris += [(v[a], v[b], v[c]), (v[a], v[c], v[d])]
    return _orient(tris, lambda p: np.array(p) - np.array(center))


# --------------------------------------- 2D triangulation: rectangle with 1 hole
def _ear_clip(poly):
    """Ear-clip a simple CCW polygon (list of (u,v)). Returns list of index triples."""
    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def in_tri(p, a, b, c):
        d1, d2, d3 = area2(p, a, b), area2(p, b, c), area2(p, c, a)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    def coincident(p, q):
        return abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12

    idx = list(range(len(poly)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 100000:
        guard += 1
        m = len(idx)
        found = False
        for i in range(m):
            i0, i1, i2 = idx[(i - 1) % m], idx[i], idx[(i + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area2(a, b, c) <= 1e-15:
                continue  # reflex / collinear
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                pj = poly[j]
                if coincident(pj, a) or coincident(pj, b) or coincident(pj, c):
                    continue
                if in_tri(pj, a, b, c):
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                del idx[i]
                found = True
                break
        if not found:
            raise RuntimeError("ear-clip stalled")
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def rect_with_hole(u0, u1, v0, v1, hu, hv, r, n=48):
    """Triangulate the rectangle [u0,u1]x[v0,v1] minus a disk (hu,hv,r).
    Returns (verts2d, tris) and (the hole ring points in order)."""
    outer = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # CCW
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = [(hu + r * math.cos(t), hv + r * math.sin(t)) for t in th]
    hole_cw = list(reversed(ring))  # CW for bridging into CCW outer
    # bridge: M = rightmost hole vertex; ray +u hits the right rectangle edge -> corner (u1,v1)
    mi = max(range(len(hole_cw)), key=lambda i: hole_cw[i][0])
    bridge_target = 2  # index of (u1, v1) in `outer`
    hole_loop = [hole_cw[(mi + j) % len(hole_cw)] for j in range(len(hole_cw) + 1)]
    merged = outer[:bridge_target + 1] + hole_loop + outer[bridge_target:]
    tris = _ear_clip(merged)
    return merged, tris, ring


def perforated_plate(hx, hy, hz, holes, x_front, x_back, n=48):
    """Plate spanning x in [x_front,x_back], y in [-hy,hy], z in [-hz,hz] with cylindrical
    through-holes (each: (cy, cz, r)). Splits at y=0 so each half has one hole.
    Returns oriented triangles in 3D. `holes` must be one at y>0 and one at y<0."""
    tris3d = []
    rings_3d = []  # per hole: (front_ring, back_ring) for walls
    for (cy, cz, r) in holes:
        if cy >= 0:
            u0, u1 = 0.0, hy
        else:
            u0, u1 = -hy, 0.0
        merged, t2d, ring = rect_with_hole(u0, u1, -hz, hz, cy, cz, r, n=n)
        # lift to front and back faces (u,v) -> (x, y=u, z=v)
        for (i, j, k) in t2d:
            pf = [(x_front, merged[idx][0], merged[idx][1]) for idx in (i, j, k)]
            pb = [(x_back, merged[idx][0], merged[idx][1]) for idx in (i, j, k)]
            tris3d += _orient([tuple(pf)], np.array([-1.0, 0, 0]))
            tris3d += _orient([tuple(pb)], np.array([1.0, 0, 0]))
        fr = [(x_front, p[0], p[1]) for p in ring]
        br = [(x_back, p[0], p[1]) for p in ring]
        rings_3d.append(((cy, cz), fr, br))
    # hole walls (normal points toward the hole axis = into the void)
    for (cy, cz), fr, br in rings_3d:
        nn = len(fr)
        wall = []
        for k in range(nn):
            kp = (k + 1) % nn
            wall += [(fr[k], br[k], br[kp]), (fr[k], br[kp], fr[kp])]
        tris3d += _orient(wall, lambda p, cy=cy, cz=cz: np.array([0.0, cy - p[1], cz - p[2]]))
    # rim: outer perimeter loop (with the y=0 seam split on top & bottom edges)
    loop = [(-hy, -hz), (0.0, -hz), (hy, -hz), (hy, hz), (0.0, hz), (-hy, hz)]
    rim = []
    for k in range(len(loop)):
        a = loop[k]
        b = loop[(k + 1) % len(loop)]
        af, bf = (x_front, a[0], a[1]), (x_front, b[0], b[1])
        ab, bb = (x_back, a[0], a[1]), (x_back, b[0], b[1])
        rim += [(af, bf, bb), (af, bb, ab)]
    rim = _orient(rim, lambda p: np.array([0.0, p[1], p[2]]))
    return tris3d + rim


# ---------------------------------------------------------------- STL + checks
def dedup(tris, eps=1e-9):
    verts, index, faces = [], {}, []
    for t in tris:
        f = []
        for v in t:
            key = (round(v[0] / eps), round(v[1] / eps), round(v[2] / eps))
            if key not in index:
                index[key] = len(verts)
                verts.append(tuple(float(x) for x in v))
            f.append(index[key])
        faces.append(tuple(f))
    return verts, faces


def manifold_report(verts, faces):
    from collections import defaultdict
    edges = defaultdict(int)
    for a, b, c in faces:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(e))] += 1
    bad = {e: n for e, n in edges.items() if n != 2}
    return len(bad), len(verts), len(edges), len(faces)


def write_stl(path, tris):
    verts, faces = dedup(tris)
    bad, nv, ne, nf = manifold_report(verts, faces)
    with open(path, "wb") as fh:
        fh.write(b"new_arm mesh (mm) ".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for t in tris:
            a, b, c = [np.asarray(v, float) * MM for v in t]
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if ln > 0 else np.zeros(3)
            fh.write(struct.pack("<3f", *nrm))
            for p in (a, b, c):
                fh.write(struct.pack("<3f", *p))
            fh.write(struct.pack("<H", 0))
    allv = np.array(verts)
    bb_lo, bb_hi = allv.min(0) * MM, allv.max(0) * MM
    status = "WATERTIGHT" if bad == 0 else f"NON-MANIFOLD edges={bad}"
    print(f"  {os.path.basename(path):22s} tris={len(tris):5d} verts={nv:5d}  {status}")
    print(f"      bbox mm: x[{bb_lo[0]:8.2f},{bb_hi[0]:8.2f}] "
          f"y[{bb_lo[1]:8.2f},{bb_hi[1]:8.2f}] z[{bb_lo[2]:8.2f},{bb_hi[2]:8.2f}]")
    return bad, bb_lo, bb_hi


# ---------------------------------------------------------------- build parts
def build(cfg):
    out = {}
    # ---- PLUG (local frame: body base at x=0, +X forward) ----
    bl, br = cfg.PLUG_BODY_LENGTH, cfg.PLUG_BODY_RADIUS
    pl, pr = cfg.PLUG_PRONG_LENGTH, cfg.PLUG_PRONG_RADIUS
    sp = cfg.PLUG_PRONG_SPACING / 2.0
    plug = cylinder_x(bl / 2.0, bl, br)                                   # body
    plug += cylinder_x(bl + pl / 2.0, pl, pr, ref_axis_point=(sp, 0))     # prong_l
    plug += cylinder_x(bl + pl / 2.0, pl, pr, ref_axis_point=(-sp, 0))    # prong_r
    out["plug"] = plug

    # ---- SOCKET geometry (local frame: faceplate centre at origin) ----
    hx, hy, hz = cfg.SOCKET_FACEPLATE_HALF
    hr, hd = cfg.SOCKET_HOLE_RADIUS, cfg.SOCKET_HOLE_DEPTH
    hsp = cfg.SOCKET_HOLE_SPACING / 2.0
    holes = [(hsp, 0.0, hr), (-hsp, 0.0, hr)]
    # stand (computed exactly as in usd_socket_plug.bake_socket)
    sxh, syh = cfg.SOCKET_STAND_HALF[0], cfg.SOCKET_STAND_HALF[1]
    faceplate_bottom = cfg.SOCKET_POSE_XYZ[2] - hz
    stand_hz = max(0.005, faceplate_bottom / 2.0)
    stand_cz = faceplate_bottom / 2.0 - cfg.SOCKET_POSE_XYZ[2]

    # socket.stl: drilled through-holes + stand
    socket = perforated_plate(hx, hy, hz, holes, -hx, hx)
    socket += box((0.0, 0.0, stand_cz), (sxh, syh, stand_hz))
    out["socket"] = socket

    # socket_marker.stl: drilled faceplate+holes only, front face at origin (shift +hx)
    marker = perforated_plate(hx, hy, hz, holes, 0.0, 2 * hx)
    out["socket_marker"] = marker

    # socket_sim.stl: faithful USD -> SOLID faceplate box + SOLID hole cylinders + stand
    sim = box((0.0, 0.0, 0.0), (hx, hy, hz))
    for (cy, cz, r) in holes:
        sim += cylinder_x(-hx + hd / 2.0, hd, r, ref_axis_point=(cy, cz))
    sim += box((0.0, 0.0, stand_cz), (sxh, syh, stand_hz))
    out["socket_sim"] = sim
    return out, dict(hx=hx, hy=hy, hz=hz, stand_hz=stand_hz, stand_cz=stand_cz,
                     bl=bl, br=br, pl=pl, pr=pr, sp=sp)


def main():
    cfg = load_config()
    parts, d = build(cfg)
    print("Generating STLs (mm) in", HERE)
    results = {}
    for name, tris in parts.items():
        results[name] = write_stl(os.path.join(HERE, name + ".stl"), tris)

    # ---- assertions vs config (in mm) ----
    print("\nVerification (mesh bbox vs config-derived expected):")
    plug_tip = (d["bl"] + d["pl"]) * MM
    checks = [
        ("plug tip X (mm)", results["plug"][2][0], plug_tip),
        ("plug max radius Y (mm)", results["plug"][2][1], (d["sp"] + d["pr"]) * MM),
        ("faceplate full depth X (mm)", results["socket_marker"][2][0], 2 * d["hx"] * MM),
        ("socket stand half-width X (mm)", results["socket"][2][0], cfg.SOCKET_STAND_HALF[0] * MM),
        ("socket width Y half (mm)", results["socket"][2][1], d["hy"] * MM),
        ("socket top Z (mm)", results["socket"][2][2], d["hz"] * MM),
        ("socket bottom Z=ground (mm)", results["socket"][1][2], (d["stand_cz"] - d["stand_hz"]) * MM),
    ]
    allok = True
    for label, got, exp in checks:
        ok = abs(got - exp) < 1e-3
        allok &= ok
        print(f"  [{'OK' if ok else 'XX'}] {label:34s} got={got:9.3f} expected={exp:9.3f}")
    wt = all(r[0] == 0 for r in results.values())
    print(f"\nAll watertight: {wt}   All dims match: {allok}")
    print("Prong<->hole radial clearance:",
          f"{(cfg.SOCKET_HOLE_RADIUS - cfg.PLUG_PRONG_RADIUS) * MM:.1f} mm/side;",
          f"prong spacing {cfg.PLUG_PRONG_SPACING * MM:.1f} mm == hole spacing "
          f"{cfg.SOCKET_HOLE_SPACING * MM:.1f} mm:",
          cfg.PLUG_PRONG_SPACING == cfg.SOCKET_HOLE_SPACING)
    return 0 if (wt and allok) else 1


if __name__ == "__main__":
    sys.exit(main())
