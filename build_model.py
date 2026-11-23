import bpy
import bmesh
import math
import time
import numpy as np
from mathutils import Vector, Matrix, noise

# ==============================================================================
# 黑洞构筑体 (Black Hole Construct) - v5 Silky-Smooth Fluid Sculpt + Deep Coral Pores
# Fixes all origami/faceting artifacts:
#   - Smooth C2 harmonic waves & fluid rounded basalt ribbons (ZERO jagged creases!)
#   - Uniform clean quad topology on body (NO destructive decimation triangles!)
#   - High-segment smooth fluted gothic tendrils (ZERO per-vertex Voronoi crumpling!)
#   - Deep, smooth-lipped multi-scale circular/oval volcanic coral cavities (pores)
#   - Exact 100 x 48 x 120 m bounding box, Ø40 m crater, Ø26 m pitch-black void, 18 m depth
# ==============================================================================

HOLE_CX = -2.0
HOLE_CZ = 58.0
R_VOID = 14.5        # -> calibrated Ø26.0 m inner pitch-black event horizon
R_CRATER = 22.3      # -> calibrated Ø40.0 m outer glowing magenta crater lip
CAVITY_DEPTH = 20.5  # -> calibrated 18.0 m recess depth


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_main_shell(name="Main_Shell_Raw", n_s=150, n_theta=210):
    """
    Silky-smooth closed organic concave-front / convex-back main body shell.
    Uses C2 smooth harmonic swirling waves (fluid molten obsidian flow, zero paper-fold creases!).
    """
    s_vals = np.linspace(0.0, 1.0, n_s)
    th_vals = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    S, TH = np.meshgrid(s_vals, th_vals, indexing='ij')

    cos_t = np.cos(TH)
    sin_t = np.sin(TH)

    R_outer = (
        35.5
        + 4.8 * sin_t
        + 3.2 * np.cos(TH - 0.35)
        + 2.4 * np.cos(2.0 * TH - 1.05)
        - 5.8 * np.exp(-((np.angle(np.exp(1j * (TH + 2.15)))) / 0.50) ** 2)
        + 5.2 * np.exp(-((np.angle(np.exp(1j * (TH + 0.90)))) / 0.48) ** 2)
        + 1.6 * np.sin(3.0 * TH + 0.55)
    )

    y_rim = (
        -22.5
        + 12.2 * (cos_t ** 2)
        - 1.3 * np.maximum(0.0, sin_t)
    )

    y_edge = (
        -10.5
        + 17.0 * (cos_t ** 2)
        - 4.5 * np.maximum(0.0, sin_t)
        - 4.5 * np.maximum(0.0, -sin_t)
    )

    y_throat = -4.2 + 2.0 * (cos_t ** 2)
    y_floor_center = 1.6
    y_back_apex = 24.8

    R = np.zeros_like(S)
    Y = np.zeros_like(S)

    m1 = S <= 0.18
    t1 = np.clip(S / 0.18, 0.0, 1.0)
    R = np.where(m1, R_VOID * np.sin(t1 * math.pi * 0.5), R)
    Y = np.where(m1, y_floor_center + (y_throat - y_floor_center) * (1.0 - np.cos(t1 * math.pi * 0.5)), Y)

    m2 = (S > 0.18) & (S <= 0.42)
    t2 = np.clip((S - 0.18) / 0.24, 0.0, 1.0)
    r2 = R_VOID + (R_CRATER - R_VOID) * (t2 ** 1.25)
    y2 = y_throat + (y_rim - y_throat) * (np.sin(t2 * math.pi * 0.5) ** 0.92)
    R = np.where(m2, r2, R)
    Y = np.where(m2, y2, Y)

    m3 = (S > 0.42) & (S <= 0.65)
    t3 = np.clip((S - 0.42) / 0.23, 0.0, 1.0)
    r3 = R_CRATER + (R_outer - R_CRATER) * np.sin(t3 * math.pi * 0.5)
    lip_roll = -2.0 * np.sin(t3 * math.pi) * (1.0 - 0.3 * t3)
    y3 = y_rim + (y_edge - y_rim) * (t3 ** 1.35) + lip_roll
    R = np.where(m3, r3, R)
    Y = np.where(m3, y3, Y)

    m4 = S > 0.65
    t4 = np.clip((S - 0.65) / 0.35, 0.0, 1.0)
    r4 = R_outer * np.cos(t4 * math.pi * 0.5)
    y4 = y_edge + (y_back_apex - y_edge) * (np.sin(t4 * math.pi * 0.5) ** 0.72)
    R = np.where(m4, r4, R)
    Y = np.where(m4, y4, Y)

    back_shift_x = -2.6 * smoothstep(0.65, 1.0, S)
    back_shift_z = -3.0 * smoothstep(0.65, 1.0, S)
    X = HOLE_CX + R * cos_t + back_shift_x
    Z = HOLE_CZ + R * sin_t + back_shift_z

    # SILKY-SMOOTH C2 harmonic swirling strata (fluid molten obsidian flow, zero sharp creases!)
    front_mask = smoothstep(0.37, 0.48, S) * (1.0 - smoothstep(0.62, 0.72, S))
    funnel_mask = smoothstep(0.18, 0.24, S) * (1.0 - smoothstep(0.41, 0.46, S))
    rear_mask = smoothstep(0.64, 0.76, S)

    def fluid_wave(phase):
        return np.sin(phase) + 0.22 * np.sin(2.0 * phase + 0.5)

    front_s1 = fluid_wave(6.0 * TH - 0.23 * R + 0.6 * np.sin(2.0 * TH))
    front_s2 = fluid_wave(11.0 * TH - 0.38 * R + 1.0)
    rear_s1  = fluid_wave(3.0 * TH + 0.34 * R + 0.5 * np.cos(TH))
    rear_s2  = fluid_wave(6.0 * TH + 0.52 * R)
    # Smooth fluid spiral filaments inside the glowing magenta crater!
    funnel_ribs = np.sin(16.0 * TH - 0.42 * R)

    disp_r = (
        front_mask * (1.9 * front_s1 + 0.65 * front_s2)
        + rear_mask * (1.5 * rear_s1 + 0.60 * rear_s2)
    )
    disp_y = (
        front_mask * (-2.0 * front_s1 - 0.70 * front_s2)
        + rear_mask * (2.6 * rear_s1 + 0.90 * rear_s2)
        + funnel_mask * (-0.75 * funnel_ribs)
    )

    X = X + disp_r * cos_t
    Z = Z + disp_r * sin_t
    Y = Y + disp_y

    bm = bmesh.new()
    v_front_pole = bm.verts.new((float(np.mean(X[0])), float(np.mean(Y[0])), float(np.mean(Z[0]))))
    v_back_pole = bm.verts.new((float(np.mean(X[-1])), float(np.mean(Y[-1])), float(np.mean(Z[-1]))))

    rings = []
    for i in range(1, n_s - 1):
        ring = [bm.verts.new((float(X[i, j]), float(Y[i, j]), float(Z[i, j]))) for j in range(n_theta)]
        rings.append(ring)

    first_ring = rings[0]
    for j in range(n_theta):
        jn = (j + 1) % n_theta
        bm.faces.new((v_front_pole, first_ring[jn], first_ring[j]))

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))

    last_ring = rings[-1]
    for j in range(n_theta):
        jn = (j + 1) % n_theta
        bm.faces.new((last_ring[j], last_ring[jn], v_back_pole))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_smooth_fluid_tendril(
    name,
    points,
    radii,
    n_seg=22,
    flutes=5,
    flute_amp=0.11,
    twist=2.2,
    oval=1.22,
):
    """
    Create a SILKY-SMOOTH, FLUID, ORGANIC sculpted baroque tendril or flowing ribbon!
    Uses C-infinity smooth cosine/sine harmonic fluting (NO sharp abs(sin) origami cusps!)
    and generous circular segment resolution so light glides across every curve like
    hand-sculpted bronze & obsidian!
    """
    pts = [Vector(p) for p in points]
    n_pts = len(pts)
    if n_pts < 2:
        return None

    tangents = []
    for i in range(n_pts):
        t = (pts[1] - pts[0]) if i == 0 else ((pts[-1] - pts[-2]) if i == n_pts - 1 else (pts[i + 1] - pts[i - 1]))
        t = Vector((0, 0, 1)) if t.length < 1e-6 else t.normalized()
        tangents.append(t)

    up = Vector((0, 1, 0)) if abs(tangents[0].y) < 0.9 else Vector((1, 0, 0))
    n0 = tangents[0].cross(up).normalized()
    b0 = tangents[0].cross(n0).normalized()

    normals, binormals = [n0], [b0]
    for i in range(1, n_pts):
        t_prev, t_curr = tangents[i - 1], tangents[i]
        axis = t_prev.cross(t_curr)
        if axis.length > 1e-6:
            rot = Matrix.Rotation(t_prev.angle(t_curr), 3, axis.normalized())
            n_curr = (rot @ normals[-1]).normalized()
        else:
            n_curr = normals[-1].copy()
        n_curr = n_curr - t_curr * n_curr.dot(t_curr)
        n_curr = normals[-1].copy() if n_curr.length < 1e-6 else n_curr.normalized()
        normals.append(n_curr)
        binormals.append(t_curr.cross(n_curr).normalized())

    bm = bmesh.new()
    rings = []
    for i in range(n_pts):
        u = i / max(1, n_pts - 1)
        r_base = max(0.18, float(radii[i]))
        # Gently fade fluting toward the delicate rounded tip
        f_amp = flute_amp * (1.0 - 0.55 * (u ** 1.2))
        ring = []
        for j in range(n_seg):
            ang = 2.0 * math.pi * j / n_seg
            phase = flutes * ang + twist * u * math.pi * 2.0
            # Pure smooth harmonic fluting — fluid organic sculpting with ZERO folded edges!
            flute = 1.0 + f_amp * math.sin(phase) + (f_amp * 0.25) * math.cos(2.0 * ang + twist * u * 1.4)
            rx = r_base * flute * oval
            ry = r_base * flute / oval
            v = bm.verts.new(pts[i] + normals[i] * (rx * math.cos(ang)) + binormals[i] * (ry * math.sin(ang)))
            ring.append(v)
        rings.append(ring)

    for i in range(n_pts - 1):
        r0, r1 = rings[i], rings[i + 1]
        for j in range(n_seg):
            jn = (j + 1) % n_seg
            bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))

    # Smooth rounded dome caps
    v_start = bm.verts.new(pts[0] - tangents[0] * (radii[0] * 0.35))
    v_end = bm.verts.new(pts[-1] + tangents[-1] * (radii[-1] * 0.75))
    for j in range(n_seg):
        jn = (j + 1) % n_seg
        bm.faces.new((v_start, rings[0][jn], rings[0][j]))
        bm.faces.new((rings[-1][j], rings[-1][jn], v_end))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Assign EXACT ANALYTICAL SEAMLESS TUBE UVs (u along length, v around circumference)!
    # Guarantees ZERO UV splits along curves and silky-smooth normal interpolation!
    uv_layer = bm.loops.layers.uv.new("UVMap")
    # Map ring vertices cleanly using ring/segment indices
    vert_uv_idx = {}
    for i, ring in enumerate(rings):
        u_coord = i / max(1, len(rings) - 1)
        for j, v in enumerate(ring):
            vert_uv_idx[v] = (u_coord, j / n_seg)
    vert_uv_idx[v_start] = (0.0, 0.5)
    vert_uv_idx[v_end] = (1.0, 0.5)

    for f in bm.faces:
        vs_face = [vert_uv_idx[l.vert][1] for l in f.loops]
        wraps = (max(vs_face) - min(vs_face)) > 0.5
        for l in f.loops:
            u_c, v_c = vert_uv_idx[l.vert]
            if wraps and v_c < 0.25:
                v_c = 1.0
            # Right-handed positive UV winding (1.0 - v_c) so normal maps NEVER invert when tangents are computed!
            l[uv_layer].uv = (u_c, 1.0 - v_c)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def generate_baroque_scroll(
    root_pos,
    init_dir,
    curl_plane_normal,
    length,
    total_curl_rad,
    r_root,
    r_tip,
    y_drift=0.0,
    n_steps=72,
    curl_accel=2.62,
    taper_exp=1.62,
):
    """Airy, sweeping baroque logarithmic fiddlehead spiral with high longitudinal smoothness."""
    p = Vector(root_pos)
    d = Vector(init_dir).normalized()
    n = Vector(curl_plane_normal).normalized()
    d_proj = d - n * d.dot(n)
    d_proj = Vector((1, 0, 0)) if d_proj.length < 1e-4 else d_proj.normalized()

    pts = [p.copy()]
    dirs = [d_proj.copy()]
    radii = [r_root]
    ds = length / (n_steps - 1)

    for i in range(1, n_steps):
        u = i / (n_steps - 1)
        theta = total_curl_rad * (u ** curl_accel)
        rot = Matrix.Rotation(theta, 3, n)
        dir_curr = (rot @ d_proj).normalized()
        step_scale = 1.0 - 0.56 * (u ** 1.30)
        step_vec = dir_curr * (ds * step_scale) + Vector((0.0, (y_drift / (n_steps - 1)) * (1.0 - 0.45 * u), 0.0))
        p = p + step_vec
        pts.append(p.copy())
        dirs.append(dir_curr.copy())

        taper = (1.0 - u) ** taper_exp
        root_flare = 0.25 * math.exp(-u * 8.5)
        r = r_tip + (r_root - r_tip) * taper + r_root * root_flare
        radii.append(r)

    return pts, radii, dirs


def spawn_branching_scroll(
    name,
    root_pos,
    init_dir,
    curl_plane_normal,
    length,
    total_curl_rad,
    r_root,
    r_tip,
    y_drift=0.0,
    curl_accel=2.62,
    children=None,
):
    objs = []
    eff_curl = math.copysign(min(7.6, max(6.2, abs(total_curl_rad) * 0.74)), total_curl_rad)
    eff_accel = max(2.48, curl_accel + 0.55)
    pts, radii, dirs = generate_baroque_scroll(
        root_pos, init_dir, curl_plane_normal, length * 0.92, eff_curl,
        r_root * 0.92, r_tip, y_drift=y_drift, n_steps=68, curl_accel=eff_accel,
        taper_exp=1.65 if r_root >= 3.2 else 1.52,
    )
    n_seg = 20 if r_root >= 3.8 else 16
    flutes = 5 if r_root >= 3.8 else 4
    main_obj = create_smooth_fluid_tendril(
        name, pts, radii, n_seg=n_seg, flutes=flutes, flute_amp=0.11, twist=2.2, oval=1.22
    )
    if main_obj:
        objs.append(main_obj)

    if children:
        n_vec = Vector(curl_plane_normal).normalized()
        for idx, (u_att, len_sc, c_sign, c_rad, r_sc, ang_bias) in enumerate(children):
            k = max(3, min(len(pts) - 6, int(u_att * (len(pts) - 1))))
            c_root = pts[k]
            rot_b = Matrix.Rotation(ang_bias, 3, n_vec)
            c_dir = (rot_b @ dirs[k]).normalized()
            c_r_root = max(0.72, radii[k] * r_sc)
            c_r_tip = max(0.20, min(0.28, r_tip * 0.80))
            c_eff_curl = c_sign * min(7.4, max(6.2, abs(c_rad) * 0.74))
            c_pts, c_radii, _ = generate_baroque_scroll(
                c_root, c_dir, curl_plane_normal,
                length * len_sc * 0.92,
                c_eff_curl,
                c_r_root, c_r_tip,
                y_drift=y_drift * 0.30,
                n_steps=46,
                curl_accel=2.55,
                taper_exp=1.50,
            )
            child_obj = create_smooth_fluid_tendril(
                f"{name}_sub{idx}", c_pts, c_radii,
                n_seg=14, flutes=4, flute_amp=0.09, twist=1.8, oval=1.18
            )
            if child_obj:
                objs.append(child_obj)

    return objs


def build_hero_tail_and_filigree():
    body_roots = []
    filigree = []

    def catmull_rom(pts, vals, samples_per_seg=8):
        out_p, out_v = [], []
        P = [Vector(pts[0])] + [Vector(x) for x in pts] + [Vector(pts[-1])]
        V = [vals[0]] + list(vals) + [vals[-1]]
        for i in range(1, len(P) - 2):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            v0, v1, v2, v3 = V[i - 1], V[i], V[i + 1], V[i + 2]
            for s in range(samples_per_seg):
                t = s / samples_per_seg
                t2, t3 = t * t, t * t * t
                pos = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
                val = 0.5 * ((2 * v1) + (-v0 + v2) * t + (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2 + (-v0 + 3 * v1 - 3 * v2 + v3) * t3)
                out_p.append(pos)
                out_v.append(val)
        out_p.append(Vector(pts[-1]))
        out_v.append(vals[-1])
        return out_p, out_v

    ctrl_trunk = [
        (21.5,  1.0, 43.0),
        (22.5, -3.0, 30.0),
        (17.5, -6.5, 19.0),
        ( 8.0, -8.5, 10.8),
        (-4.5, -9.8,  6.0),
        (-19.0, -10.8, 5.8),
        (-32.5, -11.8, 9.8),
        (-42.0, -12.8, 17.5),
        (-45.5, -13.4, 25.5),
    ]
    r_trunk = [9.4, 8.2, 7.0, 5.9, 5.1, 4.4, 3.6, 2.5, 1.5]
    tp, tr = catmull_rom(ctrl_trunk, r_trunk, 7)
    body_roots.append(create_smooth_fluid_tendril(
        "Tail_Trunk_Core", tp, tr, n_seg=24, flutes=6, flute_amp=0.11, twist=2.4, oval=1.20
    ))

    ctrl_back = [
        (16.0, 12.0, 44.0),
        (17.0, 10.5, 29.0),
        (11.0,  6.8, 17.2),
        ( 1.0,  2.2, 10.0),
        (-13.0, -2.5, 7.2),
        (-27.0, -6.5, 9.8),
        (-38.0, -9.5, 16.5),
    ]
    r_back = [8.8, 7.6, 6.4, 5.2, 4.2, 3.1, 1.8]
    bp, br = catmull_rom(ctrl_back, r_back, 7)
    body_roots.append(create_smooth_fluid_tendril(
        "Tail_Back_Core", bp, br, n_seg=20, flutes=5, flute_amp=0.10, twist=2.0, oval=1.18
    ))

    ctrl_whip = [
        (22.5, -5.0, 34.0),
        (17.5, -8.5, 20.5),
        ( 7.5, -10.5, 11.2),
        (-5.5, -11.8,  6.2),
        (-20.0, -12.5, 6.2),
        (-34.0, -13.2, 10.6),
        (-44.0, -13.8, 18.2),
        (-47.8, -14.0, 26.8),
        (-44.2, -14.2, 34.0),
        (-36.5, -14.0, 36.2),
        (-30.0, -13.6, 32.5),
        (-28.4, -13.3, 26.5),
        (-31.8, -13.0, 22.8),
        (-36.2, -12.9, 24.4),
        (-36.8, -12.8, 27.8),
        (-34.4, -12.8, 28.8),
    ]
    r_whip = [4.5, 4.0, 3.6, 3.3, 3.0, 2.7, 2.3, 1.8, 1.35, 0.98, 0.72, 0.54, 0.42, 0.34, 0.27, 0.22]
    wp, wr = catmull_rom(ctrl_whip, r_whip, 8)
    filigree.append(create_smooth_fluid_tendril(
        "Tail_Filigree_Whip", wp, wr, n_seg=22, flutes=5, flute_amp=0.11, twist=3.2, oval=1.24
    ))

    ctrl_braid = [
        (23.0, -10.0, 35.0),
        (14.5, -13.5, 21.0),
        ( 2.0, -15.2, 12.8),
        (-12.5, -15.8, 9.6),
        (-27.0, -16.0, 12.4),
        (-38.5, -16.0, 19.5),
        (-42.5, -15.8, 27.5),
        (-37.0, -15.5, 32.2),
        (-32.2, -15.2, 29.5),
    ]
    r_braid = [3.6, 3.1, 2.6, 2.2, 1.8, 1.35, 0.92, 0.55, 0.28]
    fp, fr = catmull_rom(ctrl_braid, r_braid, 7)
    filigree.append(create_smooth_fluid_tendril(
        "Tail_Filigree_InnerBraid", fp, fr, n_seg=18, flutes=4, flute_amp=0.10, twist=2.8, oval=1.20
    ))

    tail_curls = [
        ("TailCurl_BotRight_Hero",
         (15.5, -4.5, 21.5), (0.60, -0.06, -0.78), (0.10, 0.99, 0.05),
         56.0, 9.80, 4.0, 0.25, -2.5, 2.05,
         [(0.24, 0.52, -1, 9.5, 0.55, 0.45), (0.46, 0.44, 1, 9.2, 0.50, -0.35)]),
        ("TailCurl_BotRight_Mid",
         (21.0, -6.5, 26.0), (0.88, -0.05, -0.44), (-0.08, 0.99, 0.08),
         44.0, 9.60, 3.2, 0.24, -2.5, 2.05,
         [(0.32, 0.50, -1, 9.2, 0.55, 0.50)]),
        ("TailCurl_BotRight_Rear",
         (11.5, 4.5, 16.5), (0.50, 0.14, -0.84), (0.16, 0.97, 0.10),
         42.0, -9.60, 3.1, 0.24, 3.5, 2.05,
         [(0.30, 0.48, 1, 9.0, 0.52, -0.45)]),
        ("TailCurl_BotCenter_Drop",
         ( 4.5, -8.5, 10.5), (0.62, -0.08, -0.78), (-0.10, 0.98, 0.08),
         34.0, 9.80, 2.4, 0.22, -2.0, 2.08, None),

        ("TailCurl_BotLeft_Outer1",
         (-34.5, -11.5, 10.5), (-0.86, -0.05, 0.25), (0.05, 0.99, 0.08),
         38.0, -9.80, 2.8, 0.23, -2.0, 2.05,
         [(0.32, 0.48, 1, 9.2, 0.52, -0.45)]),
        ("TailCurl_BotLeft_Under",
         (-21.5, -7.5, 6.8), (-0.84, 0.10, -0.35), (-0.08, 0.99, 0.08),
         34.0, -9.60, 2.6, 0.22, 3.0, 2.05, None),
        ("TailCurl_BotLeft_Hook",
         (-44.0, -13.0, 22.0), (-0.78, -0.10, -0.60), (0.08, 0.99, 0.06),
         28.0, -9.80, 2.0, 0.21, -1.5, 2.08, None),
        ("TailCurl_BotLeft_Crest",
         (-39.0, -13.2, 33.0), (-0.55, -0.10, 0.82), (-0.08, 0.99, 0.05),
         26.0, 9.60, 1.9, 0.21, -2.0, 2.08, None),
    ]
    for spec in tail_curls:
        filigree.extend(spawn_branching_scroll(*spec))

    return body_roots, filigree


def build_all_filigree_tendrils():
    objs = []
    specs = [
        ("T1_TopRight_HeroCrest",
         (9.5, -3.0, 95.0), (0.34, -0.06, 0.93), (0.06, 0.99, 0.05),
         68.0, -9.40, 5.2, 0.28, -5.0, 1.95,
         [(0.22, 0.45, 1, 9.4, 0.48, -0.45), (0.45, 0.38, -1, 9.2, 0.45, 0.40)]),
        ("T1_UpperRight_Wave",
         (26.5, -1.5, 82.0), (0.94, -0.05, 0.30), (-0.06, 0.99, 0.05),
         58.0, 9.40, 5.0, 0.26, -3.5, 1.98,
         [(0.25, 0.46, -1, 9.4, 0.50, 0.42), (0.48, 0.38, 1, 9.2, 0.46, -0.35)]),
        ("T1_MidRight_Hook",
         (29.5, 3.0, 62.0), (0.96, 0.08, -0.15), (0.08, 0.99, -0.06),
         50.0, 9.20, 4.5, 0.26, 3.0, 2.00,
         [(0.28, 0.48, -1, 9.2, 0.50, 0.45)]),
        ("T1_LowerRight_Spur",
         (25.0, -1.5, 41.0), (0.86, -0.08, -0.45), (-0.08, 0.99, 0.06),
         46.0, 9.40, 4.3, 0.25, -3.0, 2.00,
         [(0.28, 0.46, -1, 9.2, 0.48, 0.45)]),
        ("T1_TopLeft_Crown",
         (-17.0, -2.0, 93.0), (-0.54, -0.08, 0.83), (-0.06, 0.99, 0.05),
         52.0, 9.30, 4.7, 0.26, -3.5, 1.98,
         [(0.25, 0.46, -1, 9.2, 0.48, 0.45), (0.48, 0.38, 1, 9.2, 0.45, -0.38)]),
        ("T1_UpperLeft_Wave",
         (-28.5, 1.5, 79.0), (-0.92, 0.06, 0.36), (0.08, 0.99, -0.05),
         56.0, -9.40, 4.9, 0.26, 3.5, 1.98,
         [(0.24, 0.48, 1, 9.4, 0.50, -0.42), (0.48, 0.38, -1, 9.2, 0.45, 0.35)]),
        ("T1_MidLeft_Scroll",
         (-30.5, -2.5, 58.0), (-0.96, -0.08, -0.14), (-0.08, 0.99, 0.08),
         48.0, -9.20, 4.4, 0.25, -3.5, 2.00,
         [(0.28, 0.48, 1, 9.2, 0.50, -0.45)]),
        ("T1_LowerLeft_Arch",
         (-23.5, -6.0, 40.0), (-0.88, -0.14, -0.40), (0.06, 0.99, 0.05),
         42.0, -9.60, 4.0, 0.24, -3.0, 2.02,
         [(0.30, 0.46, 1, 9.4, 0.48, -0.45)]),

        ("T2_TopCenter_Front",
         (-3.5, -6.0, 94.5), (-0.28, -0.12, 0.94), (0.08, 0.99, -0.05),
         38.0, 9.50, 3.3, 0.24, -3.5, 2.02, [(0.30, 0.46, -1, 9.4, 0.50, 0.45)]),
        ("T2_TopCenter_Back",
         (-4.5, 10.5, 92.5), (-0.32, 0.20, 0.92), (-0.14, 0.96, 0.16),
         40.0, 9.20, 3.5, 0.25, 4.8, 2.00, [(0.32, 0.46, -1, 9.0, 0.50, 0.42)]),
        ("T2_TopLeft_Outer",
         (-26.5, -4.0, 88.0), (-0.78, -0.10, 0.60), (0.10, 0.99, 0.08),
         40.0, 9.50, 3.3, 0.24, -3.0, 2.02, [(0.30, 0.46, -1, 9.2, 0.50, 0.45)]),
        ("T2_TopRight_Outer",
         (21.5, -5.5, 89.0), (0.72, -0.12, 0.68), (-0.10, 0.99, 0.08),
         40.0, -9.50, 3.3, 0.24, -3.5, 2.02, [(0.30, 0.46, 1, 9.2, 0.50, -0.45)]),
        ("T2_TopRight_Back",
         (18.0, 12.0, 86.0), (0.68, 0.24, 0.68), (0.18, 0.95, -0.15),
         38.0, 9.20, 3.4, 0.25, 5.2, 2.00, [(0.32, 0.46, -1, 9.0, 0.50, 0.42)]),

        ("T2_LeftFront_Upper",
         (-22.5, -14.5, 81.0), (-0.70, -0.42, 0.56), (0.24, 0.93, -0.16),
         34.0, -9.50, 2.8, 0.23, -3.5, 2.04, [(0.34, 0.48, 1, 9.2, 0.50, -0.45)]),
        ("T2_LeftFront_Lower",
         (-20.5, -15.5, 40.5), (-0.68, -0.48, -0.54), (-0.24, 0.93, 0.16),
         32.0, 9.50, 2.7, 0.23, -3.0, 2.04, [(0.34, 0.48, -1, 9.2, 0.50, 0.45)]),
        ("T2_RightFront_Upper",
         (21.5, -14.5, 81.0), (0.70, -0.42, 0.56), (-0.24, 0.93, 0.16),
         34.0, 9.50, 2.9, 0.23, -3.5, 2.04, [(0.34, 0.48, -1, 9.2, 0.50, 0.45)]),
        ("T2_RightFront_Lower",
         (21.5, -15.0, 41.0), (0.72, -0.44, -0.50), (0.22, 0.93, -0.16),
         33.0, -9.50, 2.8, 0.23, -3.0, 2.04, [(0.34, 0.48, 1, 9.2, 0.50, -0.45)]),

        ("T2_RearLeft_Upper",
         (-24.5, 14.0, 80.0), (-0.82, 0.35, 0.42), (-0.22, 0.93, 0.18),
         40.0, 9.10, 3.5, 0.25, 4.8, 2.00, [(0.32, 0.46, -1, 9.2, 0.50, 0.45)]),
        ("T2_RearLeft_Mid",
         (-27.5, 15.5, 55.0), (-0.88, 0.32, -0.25), (0.20, 0.94, -0.18),
         38.0, -9.20, 3.4, 0.25, 4.5, 2.00, [(0.32, 0.46, 1, 9.2, 0.50, -0.45)]),
        ("T2_RearRight_Upper",
         (24.5, 13.5, 77.0), (0.85, 0.32, 0.35), (0.20, 0.94, -0.15),
         39.0, -9.10, 3.5, 0.25, 4.8, 2.00, [(0.32, 0.46, 1, 9.2, 0.50, -0.45)]),
        ("T2_RearRight_Mid",
         (26.5, 15.0, 51.0), (0.88, 0.30, -0.30), (-0.18, 0.95, 0.15),
         37.0, 9.20, 3.3, 0.24, 4.5, 2.00, [(0.32, 0.46, -1, 9.2, 0.50, 0.45)]),
        ("T2_RearTop_Swirl",
         (4.0, 18.5, 85.0), (0.72, 0.25, 0.62), (0.14, 0.96, -0.20),
         36.0, -9.00, 3.3, 0.24, 3.5, 2.00, [(0.34, 0.46, 1, 9.0, 0.48, -0.42)]),
        ("T2_RearBot_Swirl",
         (-8.0, 12.0, 24.0), (-0.55, 0.45, -0.68), (0.22, 0.92, -0.22),
         32.0, 9.40, 2.8, 0.23, 3.5, 2.02, [(0.35, 0.46, -1, 9.2, 0.50, 0.42)]),

        ("T3_Top_1", (-11.0, -9.5, 93.0), (-0.38, -0.22, 0.88), (0.10, 0.98, 0.08), 28.0, 9.80, 2.1, 0.21, -2.5, 2.06, None),
        ("T3_Top_2", (  5.0, -10.5, 95.0), (0.35, -0.22, 0.90), (-0.10, 0.98, 0.08), 28.0, -9.80, 2.1, 0.21, -2.5, 2.06, None),
        ("T3_Top_3", (-24.0,  8.5, 90.0), (-0.65, 0.25, 0.70), (0.16, 0.96, -0.12), 29.0, -9.60, 2.2, 0.22, 3.0, 2.05, None),
        ("T3_Top_4", ( 27.0,  7.5, 85.0), (0.78, 0.22, 0.58), (-0.16, 0.96, 0.12), 29.0, 9.60, 2.2, 0.22, 3.0, 2.05, None),
        ("T3_Top_5", ( -2.0, 15.0, 95.0), (0.18, 0.35, 0.92), (0.12, 0.97, -0.12), 26.0, -9.60, 2.0, 0.21, 3.0, 2.06, None),

        ("T3_Left_1", (-33.0, -5.0, 70.0), (-0.92, -0.20, 0.20), (0.14, 0.96, 0.10), 28.0, 9.80, 2.1, 0.21, -2.5, 2.06, None),
        ("T3_Left_2", (-32.0,  8.0, 46.0), (-0.90, 0.25, -0.35), (-0.14, 0.96, -0.10), 28.0, -9.80, 2.2, 0.21, 3.0, 2.06, None),
        ("T3_Left_3", (-27.0, -10.5, 84.0), (-0.75, -0.32, 0.55), (0.18, 0.95, 0.14), 26.0, -9.60, 2.0, 0.21, -2.5, 2.06, None),
        ("T3_Left_4", (-18.0, -13.5, 34.0), (-0.70, -0.32, -0.60), (-0.14, 0.96, 0.10), 25.0, 9.80, 1.9, 0.20, -2.0, 2.08, None),
        ("T3_Left_5", (-24.0, 16.0, 68.0), (-0.78, 0.50, 0.35), (-0.25, 0.92, 0.20), 26.0, 9.50, 2.1, 0.21, 3.5, 2.05, None),
        ("T3_Left_6", (-19.0, 18.0, 38.0), (-0.75, 0.52, -0.40), (0.22, 0.93, -0.20), 25.0, -9.60, 2.0, 0.21, 3.5, 2.05, None),

        ("T3_Right_1", (33.0, -5.0, 72.0), (0.92, -0.20, 0.25), (-0.14, 0.96, -0.10), 28.0, -9.80, 2.1, 0.21, -2.5, 2.06, None),
        ("T3_Right_2", (32.0, -6.5, 54.0), (0.92, -0.20, -0.25), (0.14, 0.96, 0.10), 28.0, 9.80, 2.1, 0.21, -2.5, 2.06, None),
        ("T3_Right_3", (28.0, 10.5, 68.0), (0.88, 0.35, 0.22), (0.20, 0.94, -0.14), 28.0, 9.60, 2.2, 0.21, 3.5, 2.05, None),
        ("T3_Right_4", (27.0, -10.0, 35.0), (0.82, -0.28, -0.45), (-0.16, 0.96, 0.12), 26.0, -9.80, 2.0, 0.20, -2.5, 2.06, None),
        ("T3_Right_5", (22.0, 16.5, 64.0), (0.80, 0.48, 0.32), (0.24, 0.92, -0.20), 26.0, -9.50, 2.1, 0.21, 3.5, 2.05, None),
        ("T3_Right_6", (19.0, 17.5, 36.0), (0.76, 0.50, -0.42), (-0.22, 0.93, 0.20), 25.0, 9.60, 2.0, 0.21, 3.5, 2.05, None),

        ("T3_TopPlane_1", (-15.0,  2.5, 95.0), (-0.88, -0.42, 0.18), (0.10, 0.18, 0.97), 38.0,  9.20, 2.7, 0.24, -2.5, 2.00, [(0.34, 0.46, -1, 9.0, 0.48, 0.42)]),
        ("T3_TopPlane_2", ( 16.0,  4.5, 94.0), ( 0.88, -0.42, 0.16), (-0.10, 0.18, 0.97), 38.0, -9.20, 2.7, 0.24, -2.5, 2.00, [(0.34, 0.46, 1, 9.0, 0.48, -0.42)]),
        ("T3_TopPlane_3", ( -9.0, 14.5, 90.0), (-0.78,  0.58, 0.20), (0.12, -0.22, 0.96), 34.0, -9.20, 2.5, 0.23,  2.5, 2.02, None),
        ("T3_TopPlane_4", ( 11.0, 15.5, 88.0), ( 0.80,  0.54, 0.18), (-0.12, -0.22, 0.96), 34.0,  9.20, 2.5, 0.23,  2.5, 2.02, None),
        ("T3_TopPlane_5", (  0.0, 19.5, 84.0), ( 0.92,  0.30, 0.15), (0.05, -0.25, 0.96), 32.0, -9.00, 2.5, 0.23,  2.0, 2.02, None),
        ("T3_TopPlane_6", ( -4.0, -14.5, 91.0), (-0.88, -0.40, 0.22), (0.10, 0.22, 0.96), 28.0, -9.40, 2.1, 0.21, -2.0, 2.05, None),
        ("T3_TopPlane_7", (  8.0, -14.0, 92.0), ( 0.86, -0.42, 0.24), (-0.10, 0.22, 0.96), 28.0,  9.40, 2.1, 0.21, -2.0, 2.05, None),

        ("T3_SideProfile_1", (-18.0, 18.0, 74.0), (-0.25, 0.85, 0.45), (0.96, 0.18, -0.18), 26.0,  9.40, 2.1, 0.21, 0.0, 2.05, None),
        ("T3_SideProfile_2", ( 18.0, 18.0, 74.0), ( 0.25, 0.85, 0.45), (-0.96, 0.18, -0.18), 26.0, -9.40, 2.1, 0.21, 0.0, 2.05, None),
        ("T3_SideProfile_3", (-14.0, 19.5, 42.0), (-0.20, 0.80, -0.56), (0.96, -0.15, 0.20), 25.0, -9.40, 2.1, 0.21, 0.0, 2.05, None),
        ("T3_SideProfile_4", ( 14.0, 19.5, 42.0), ( 0.20, 0.80, -0.56), (-0.96, -0.15, 0.20), 25.0,  9.40, 2.1, 0.21, 0.0, 2.05, None),
    ]
    for spec in specs:
        objs.extend(spawn_branching_scroll(*spec))
    return objs


def build_surface_strata_and_crater_veins():
    body_ribs = []
    crater_veins = []

    # 1. Smooth flowing front swirling strata
    for k in range(14):
        th0 = 2.0 * math.pi * k / 14.0 + 0.10
        pts, radii = [], []
        for i in range(36):
            u = i / 35.0
            th = th0 + u * 1.95
            r = 21.8 + 15.5 * (u ** 0.90)
            x = HOLE_CX + r * math.cos(th)
            z = HOLE_CZ + r * math.sin(th)
            y_lip = -20.8 + 11.8 * (math.cos(th) ** 2)
            y = y_lip + 16.5 * (u ** 1.35) - 1.2 * math.sin(u * math.pi)
            pts.append((x, y, z))
            radii.append(1.2 + 2.7 * (math.sin(u * math.pi) ** 0.65))
        body_ribs.append(create_smooth_fluid_tendril(
            f"Front_Strata_{k}", pts, radii, n_seg=16, flutes=4, flute_amp=0.10, twist=1.8, oval=1.42
        ))

    # 2. Smooth flowing Rear Conch Whorl Strata
    rear_cx, rear_cz = -4.5, 55.0
    for k in range(12):
        th0 = 2.0 * math.pi * k / 12.0
        pts, radii = [], []
        for i in range(38):
            u = i / 37.0
            th = th0 - u * 3.05
            r = 2.2 + 34.5 * (u ** 1.00)
            x = rear_cx + r * math.cos(th)
            z = rear_cz + r * math.sin(th)
            dome = math.cos(min(1.0, r / 38.0) * math.pi * 0.5)
            y = -1.5 + 26.0 * (dome ** 0.72) + 1.8 * math.sin(u * math.pi)
            pts.append((x, y, z))
            radii.append(1.2 + 3.4 * (math.sin(u * math.pi) ** 0.65))
        body_ribs.append(create_smooth_fluid_tendril(
            f"Rear_WhorlStrata_{k}", pts, radii, n_seg=16, flutes=4, flute_amp=0.10, twist=1.6, oval=1.48
        ))

    eye_pts, eye_r = [], []
    for i in range(56):
        u = i / 55.0
        th = u * 5.2 * math.pi
        r = 1.0 + 12.0 * u
        x = rear_cx + r * math.cos(th)
        z = rear_cz + r * math.sin(th)
        y = 24.8 - 2.6 * (u ** 1.35)
        eye_pts.append((x, y, z))
        eye_r.append(0.9 + 2.1 * u * (1.0 - 0.30 * u))
    body_ribs.append(create_smooth_fluid_tendril(
        "Rear_Whorl_Eye", eye_pts, eye_r, n_seg=16, flutes=3, flute_amp=0.08, twist=1.4, oval=1.28
    ))

    # 3. Top Dome Swirling Strata
    top_cx, top_cy = -1.0, 4.0
    for k in range(8):
        th0 = 2.0 * math.pi * k / 8.0
        pts, radii = [], []
        for i in range(30):
            u = i / 29.0
            th = th0 + u * 2.25
            rx = 3.5 + 27.0 * u
            ry = 2.2 + 15.0 * u
            x = top_cx + rx * math.cos(th)
            y = top_cy + ry * math.sin(th)
            z = 99.5 - 14.5 * (u ** 1.38) + 2.2 * math.sin(u * math.pi)
            pts.append((x, y, z))
            radii.append(1.1 + 2.4 * math.sin(u * math.pi))
        body_ribs.append(create_smooth_fluid_tendril(
            f"Top_Strata_{k}", pts, radii, n_seg=14, flutes=4, flute_amp=0.09, twist=1.5, oval=1.38
        ))

    # 4. Smooth Dark Obsidian Inner Crater Spiral Veins (flowing cleanly down into Ø26m void!)
    for k in range(18):
        th0 = 2.0 * math.pi * k / 18.0
        pts, radii = [], []
        for i in range(28):
            u = i / 27.0
            th = th0 + u * 0.95 + 0.10 * math.sin(k * 2.1 + u * 4.0)
            r = (R_CRATER + 2.4) - (R_CRATER + 2.4 - R_VOID) * (u ** 0.85)
            y_lip = -22.2 + 11.8 * (math.cos(th) ** 2)
            y_th  = -4.0 + 2.0 * (math.cos(th) ** 2)
            y = y_lip + (y_th - y_lip) * (u ** 1.12) - 0.50 * math.sin(u * math.pi)
            x = HOLE_CX + r * math.cos(th)
            z = HOLE_CZ + r * math.sin(th)
            pts.append((x, y, z))
            rad = 0.36 + 0.88 * (1.0 - 0.70 * u) * (math.sin(u * math.pi) ** 0.5 + 0.22)
            radii.append(rad)
        crater_veins.append(create_smooth_fluid_tendril(
            f"Crater_DarkVein_{k}", pts, radii, n_seg=12, flutes=3, flute_amp=0.08, twist=1.2, oval=1.32
        ))

    return body_ribs, crater_veins


def join_and_voxel_remesh(objs, name, voxel_size=0.40, smooth_iters=4):
    """
    Fuse main body at clean uniform quad density (`voxel_size=0.40 m` -> ~265k UNIFORM quads)
    and apply 4 smooth passes so every macro curve is buttery smooth like sculpted bronze/obsidian
    — NO destructive decimation needed during GLB export!
    """
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    merged = bpy.context.active_object
    merged.name = name

    mod = merged.modifiers.new(name="VoxelRemesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    mod.adaptivity = 0.0
    mod.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=mod.name)

    if smooth_iters > 0:
        smod = merged.modifiers.new(name="Smooth", type='SMOOTH')
        smod.factor = 0.55
        smod.iterations = smooth_iters
        bpy.ops.object.modifier_apply(modifier=smod.name)

    return merged


def sculpt_smooth_fluid_with_deep_pores(obj, is_filigree=False):
    """
    Sculpt SILKY-SMOOTH flowing surfaces while carving DEEP, CLEAN, SMOOTH-LIPPED
    volcanic-coral cavities (holes/pores)!
    KEY ANTI-ORIGAMI DESIGN:
      - ZERO high-frequency per-vertex jitter (`n3`)!
      - ZERO sharp `1 - |noise|` paper-fold creases!
      - On `BlackHole_Filigree` (tendrils): ZERO destructive Voronoi vertex poking — keeps
        every tendril 100% silky smooth & fluid geometrically, while still storing smooth
        organic SculptData for color/crevice/glow & letting the 4K shader carve clean micro-pores!
      - On `BlackHole_Body`: smooth C2 bowl-shaped circular coral cavities + gentle post-sculpt
        Laplacian smooth so every cavity rim rounds smoothly like real carved volcanic coral!
    """
    t0 = time.time()
    mesh = obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    normals = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    mesh.vertices.foreach_get("normal", normals)
    coords = coords.reshape((n_verts, 3))
    normals = normals.reshape((n_verts, 3))

    dx = coords[:, 0] - HOLE_CX
    dz = coords[:, 2] - HOLE_CZ
    y = coords[:, 1]
    r_hole = np.sqrt(dx * dx + dz * dz)
    ang = np.arctan2(dz, dx)

    in_front = smoothstep(7.0, -3.5, y)
    void_mask = (1.0 - smoothstep(R_VOID - 1.2, R_VOID + 0.3, r_hole)) * smoothstep(9.0, 4.0, y)
    crater_mask = (1.0 - smoothstep(R_CRATER - 1.5, R_CRATER + 4.0, r_hole)) * in_front * (1.0 - void_mask)
    outer_mask = np.clip(1.0 - crater_mask - void_mask, 0.0, 1.0)

    disp = np.zeros(n_verts, dtype=np.float32)
    pore_attr = np.zeros(n_verts, dtype=np.float32)
    ridge_attr = np.zeros(n_verts, dtype=np.float32)
    fissure_attr = np.zeros(n_verts, dtype=np.float32)

    swirl_a = 0.028 * r_hole
    ca = np.cos(swirl_a)
    sa = np.sin(swirl_a)
    wx = dx * ca - dz * sa
    wz = dx * sa + dz * ca

    n_noise = noise.noise
    n_voronoi = noise.voronoi
    cx_arr, cy_arr, cz_arr = coords[:, 0], coords[:, 1], coords[:, 2]
    om_arr, cm_arr = outer_mask, crater_mask
    ang_arr, rh_arr = ang, r_hole

    def sstep(e0, e1, x):
        t = (x - e0) / (e1 - e0)
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return t * t * (3.0 - 2.0 * t)

    for i in range(n_verts):
        px = float(cx_arr[i])
        py = float(cy_arr[i])
        pz = float(cz_arr[i])
        wxi = float(wx[i])
        wzi = float(wz[i])

        # Smooth, fluid rolling organic undulations (pure C2 smooth Perlin, NO sharp abs folds!)
        n1 = n_noise((wxi * 0.065, py * 0.095, wzi * 0.065))
        n2 = n_noise((wxi * 0.14,  py * 0.15,  wzi * 0.14))
        fluid_undul = 0.5 + 0.35 * n1 + 0.18 * n2

        # Smooth-lipped circular volcanic coral cavities (C2 smoothstep bowls!)
        vd1, _ = n_voronoi((px * 0.24, py * 0.24, pz * 0.24), distance_metric='DISTANCE')
        vd2, _ = n_voronoi((px * 0.46, py * 0.46, pz * 0.46), distance_metric='DISTANCE')
        d1 = vd1[0]
        d2 = vd2[0]

        pzone = sstep(-0.45, 0.35, n1)
        # Smooth rounded bowl cavities (zero sharp polygon spikes!)
        t_pit1 = sstep(0.36, 0.05, d1)
        pit1 = (t_pit1 * t_pit1) * pzone
        rim1 = math.exp(-((d1 - 0.40) / 0.09) ** 2) * pzone

        t_pit2 = sstep(0.34, 0.05, d2)
        pit2 = (t_pit2 * t_pit2) * pzone
        rim2 = math.exp(-((d2 - 0.38) / 0.085) ** 2) * pzone

        om = float(om_arr[i])
        cm = float(cm_arr[i])

        if is_filigree:
            # KEEP TENDRILS 100% SILKY SMOOTH GEOMETRICALLY! Zero vertex poking!
            d_outer = 0.0
        else:
            # Smooth rolling fluid shell + deep rounded volcanic-coral cavities!
            d_outer = (
                0.75 * (fluid_undul - 0.5)
                - 2.15 * pit1 + 0.30 * rim1
                - 0.90 * pit2 + 0.14 * rim2
            )

        # Smooth flowing glowing fissures inside the crater funnel
        if cm > 0.001:
            ai = float(ang_arr[i])
            rhi = float(rh_arr[i])
            f1 = math.sin(22.0 * ai - 0.58 * rhi + 2.2 * n2)
            f2 = math.sin(38.0 * ai - 0.92 * rhi + 1.5 * n1)
            crack1 = (1.0 - abs(f1)) ** 2.6
            crack2 = (1.0 - abs(f2)) ** 2.2
            fissure = min(1.0, 0.78 * crack1 + 0.42 * crack2)
            d_crater = 0.0 if is_filigree else (-0.55 * fissure + 0.25 * (1.0 - fissure))
        else:
            fissure = 0.0
            d_crater = 0.0

        disp[i] = om * d_outer + cm * d_crater

        # Deep dark crevices inside rounded coral pits & undulating valleys
        valley = max(0.0, 0.50 - fluid_undul) * 1.4
        pore_attr[i] = min(1.0, max(0.0, 1.05 * pit1 + 0.65 * pit2 + 0.50 * valley))

        # Smooth oxidized bronze sheen on raised rolling crests & rounded pore lips
        crest = max(0.0, fluid_undul - 0.50) * 2.2 + 0.55 * rim1 + 0.30 * rim2
        ridge_attr[i] = min(1.0, max(0.0, crest * (1.0 - 0.95 * pit1)))
        fissure_attr[i] = fissure

    if not is_filigree:
        new_coords = coords + normals * disp[:, None]
        mesh.vertices.foreach_set("co", new_coords.ravel())
        mesh.update()

        # Gentle post-carve smooth pass rounds every coral cavity lip silky smooth!
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        smod = obj.modifiers.new(name="PolishSmooth", type='SMOOTH')
        smod.factor = 0.38
        smod.iterations = 2
        bpy.ops.object.modifier_apply(modifier=smod.name)

    if "SculptData" in mesh.color_attributes:
        mesh.color_attributes.remove(mesh.color_attributes["SculptData"])
    col_attr = mesh.color_attributes.new(name="SculptData", type='BYTE_COLOR', domain='POINT')

    radial_band = smoothstep(R_VOID - 0.8, R_VOID + 0.8, r_hole) * (1.0 - smoothstep(R_CRATER - 1.2, R_CRATER + 3.6, r_hole))
    front_face = smoothstep(5.5, -3.5, y)
    if is_filigree:
        crater_glow = radial_band * front_face * fissure_attr * 0.22
    else:
        crater_glow = radial_band * front_face * (0.08 + 0.92 * fissure_attr)

    tail_vein = (
        smoothstep(52.0, 8.0, coords[:, 2])
        * smoothstep(4.0, -11.0, y)
        * np.clip(pore_attr * 1.15, 0.0, 1.0)
        * 0.32
    )
    rim_vein = (
        smoothstep(R_CRATER + 10.0, R_CRATER - 0.5, r_hole)
        * smoothstep(1.0, -10.5, y)
        * np.clip(pore_attr, 0.0, 1.0)
        * 0.22
    )
    glow_attr = np.clip(crater_glow + tail_vein + rim_vein, 0.0, 1.0) * (1.0 - void_mask)

    rgba = np.zeros((n_verts, 4), dtype=np.float32)
    rgba[:, 0] = pore_attr
    rgba[:, 1] = glow_attr
    rgba[:, 2] = ridge_attr * (1.0 - void_mask)
    rgba[:, 3] = void_mask
    col_attr.data.foreach_set("color", rgba.ravel())

    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    print(f"  Sculpted smooth {obj.name} ({n_verts:,} verts) in {time.time() - t0:.1f}s")


def calibrate_bounding_box(target_objs):
    bpy.context.view_layer.update()
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in target_objs:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, wc.x); mins.y = min(mins.y, wc.y); mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x); maxs.y = max(maxs.y, wc.y); maxs.z = max(maxs.z, wc.z)

    cur_size = maxs - mins
    cur_center = (mins + maxs) * 0.5
    sx, sy, sz = 100.0 / cur_size.x, 48.0 / cur_size.y, 120.0 / cur_size.z

    for obj in target_objs:
        mesh = obj.data
        n_v = len(mesh.vertices)
        co = np.empty(n_v * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape((n_v, 3))
        if obj.matrix_world != Matrix.Identity(4):
            M = np.array(obj.matrix_world, dtype=np.float32)
            co = (co @ M[:3, :3].T) + M[:3, 3]
            obj.matrix_world = Matrix.Identity(4)

        co[:, 0] = (co[:, 0] - cur_center.x) * sx
        co[:, 1] = (co[:, 1] - cur_center.y) * sy
        co[:, 2] = (co[:, 2] - mins.z) * sz
        mesh.vertices.foreach_set("co", co.ravel())
        mesh.update()

    bpy.context.view_layer.update()
    fmin = Vector((1e9, 1e9, 1e9))
    fmax = Vector((-1e9, -1e9, -1e9))
    for obj in target_objs:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            fmin.x = min(fmin.x, wc.x); fmin.y = min(fmin.y, wc.y); fmin.z = min(fmin.z, wc.z)
            fmax.x = max(fmax.x, wc.x); fmax.y = max(fmax.y, wc.y); fmax.z = max(fmax.z, wc.z)
    fsize = fmax - fmin
    print(f"[Calibrated] min={tuple(round(v,2) for v in fmin)}, max={tuple(round(v,2) for v in fmax)}, size={tuple(round(v,2) for v in fsize)}")
    return fmin, fmax


def create_silky_porous_material():
    """
    Silky-Smooth Sculpted Dark Obsidian-Bronze + Deep Rounded Coral Pores Shader!
    - Smooth satin-obsidian surface (ZERO harsh origami bump noise!)
    - 3-scale CLEAN SMOOTH-RIMMED CIRCULAR PORES carved cleanly via smoothstep Voronoi
      into BaseColor (pitch-black hollows), Roughness, AO, and clean gentle Normal Map!
    - Warm oxidized dark bronze/copper sheen flowing smoothly over sculpted contours!
    - Crisp electric royal-magenta energy fissures + pure unlit black Event Horizon!
    """
    mat = bpy.data.materials.new("BlackHole_Crust_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (1350, 0)

    mix_void = nt.nodes.new("ShaderNodeMixShader")
    mix_void.location = (1100, 0)
    nt.links.new(mix_void.outputs["Shader"], out.inputs["Surface"])

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (760, 140)
    nt.links.new(bsdf.outputs["BSDF"], mix_void.inputs[1])

    void_em = nt.nodes.new("ShaderNodeEmission")
    void_em.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    void_em.inputs["Strength"].default_value = 0.0
    void_em.location = (760, -260)
    nt.links.new(void_em.outputs["Emission"], mix_void.inputs[2])

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "SculptData"
    attr.location = (-1250, 320)

    sep = nt.nodes.new("ShaderNodeSeparateColor")
    sep.location = (-1020, 340)
    nt.links.new(attr.outputs["Color"], sep.inputs["Color"])

    void_ramp = nt.nodes.new("ShaderNodeMapRange")
    void_ramp.inputs["From Min"].default_value = 0.15
    void_ramp.inputs["From Max"].default_value = 0.55
    void_ramp.location = (800, -120)
    nt.links.new(attr.outputs["Alpha"], void_ramp.inputs["Value"])
    nt.links.new(void_ramp.outputs["Result"], mix_void.inputs["Fac"])

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-1450, -220)

    # Very gentle low-freq organic warp so circular coral pores feel natural & fluid
    noise_warp = nt.nodes.new("ShaderNodeTexNoise")
    noise_warp.inputs["Scale"].default_value = 0.22
    noise_warp.inputs["Detail"].default_value = 2.0
    noise_warp.location = (-1220, -140)
    nt.links.new(texcoord.outputs["Object"], noise_warp.inputs["Vector"])

    mix_warp = nt.nodes.new("ShaderNodeMix")
    mix_warp.data_type = 'VECTOR'
    mix_warp.inputs["Factor"].default_value = 0.045
    mix_warp.location = (-1000, -220)
    nt.links.new(texcoord.outputs["Object"], mix_warp.inputs["A"])
    nt.links.new(noise_warp.outputs["Color"], mix_warp.inputs["B"])

    # 3 scales of CLEAN, DEEP, SMOOTH-RIMMED CIRCULAR VOLCANIC CORAL PORES!
    vor_m = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_m.inputs["Scale"].default_value = 0.58   # ~1.7 m coral hollows
    vor_m.location = (-780, -160)
    nt.links.new(mix_warp.outputs["Result"], vor_m.inputs["Vector"])

    vor_s = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_s.inputs["Scale"].default_value = 1.45   # ~0.7 m smooth round pores
    vor_s.location = (-780, -380)
    nt.links.new(mix_warp.outputs["Result"], vor_s.inputs["Vector"])

    vor_xs = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_xs.inputs["Scale"].default_value = 3.20  # ~0.3 m fine coral pinholes
    vor_xs.location = (-780, -600)
    nt.links.new(mix_warp.outputs["Result"], vor_xs.inputs["Vector"])

    def make_smooth_pit(vor_node, cutoff, loc):
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = 'SMOOTHERSTEP'  # C2 smooth transitions!
        mr.inputs["From Min"].default_value = cutoff
        mr.inputs["From Max"].default_value = 0.06
        mr.inputs["To Min"].default_value = 0.0
        mr.inputs["To Max"].default_value = 1.0
        mr.location = loc
        nt.links.new(vor_node.outputs["Distance"], mr.inputs["Value"])
        return mr

    pit_m  = make_smooth_pit(vor_m,  0.34, (-540, -160))
    pit_s  = make_smooth_pit(vor_s,  0.32, (-540, -380))
    pit_xs = make_smooth_pit(vor_xs, 0.28, (-540, -600))

    add_p1 = nt.nodes.new("ShaderNodeMath")
    add_p1.operation = 'MAXIMUM'
    add_p1.location = (-320, -240)
    nt.links.new(pit_m.outputs["Result"], add_p1.inputs[0])
    nt.links.new(pit_s.outputs["Result"], add_p1.inputs[1])

    mul_xs = nt.nodes.new("ShaderNodeMath")
    mul_xs.operation = 'MULTIPLY'
    mul_xs.inputs[1].default_value = 0.45
    mul_xs.location = (-320, -500)
    nt.links.new(pit_xs.outputs["Result"], mul_xs.inputs[0])

    total_micro_pit = nt.nodes.new("ShaderNodeMath")
    total_micro_pit.operation = 'MAXIMUM'
    total_micro_pit.location = (-120, -320)
    nt.links.new(add_p1.outputs["Value"], total_micro_pit.inputs[0])
    nt.links.new(mul_xs.outputs["Value"], total_micro_pit.inputs[1])

    # Soft, velvety fine stone micro-variation (LOW detail = silky smooth, zero crunchy paper noise!)
    noise_soft = nt.nodes.new("ShaderNodeTexNoise")
    noise_soft.inputs["Scale"].default_value = 0.28
    noise_soft.inputs["Detail"].default_value = 3.0
    noise_soft.inputs["Roughness"].default_value = 0.45
    noise_soft.location = (-780, -820)
    nt.links.new(texcoord.outputs["Object"], noise_soft.inputs["Vector"])

    # Clean smooth Cycles Ambient Occlusion (replaces noisy per-vertex Pointiness!)
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 16
    ao.inputs["Distance"].default_value = 7.5
    ao.location = (-950, 620)

    # Smooth color flow: AO * (0.50 + 0.50 * SculptData.Blue) + subtle soft stone flow
    blue_lift = nt.nodes.new("ShaderNodeMapRange")
    blue_lift.inputs["From Min"].default_value = 0.0
    blue_lift.inputs["From Max"].default_value = 1.0
    blue_lift.inputs["To Min"].default_value = 0.36
    blue_lift.inputs["To Max"].default_value = 0.96
    blue_lift.location = (-740, 420)
    nt.links.new(sep.outputs["Blue"], blue_lift.inputs["Value"])

    edge_ao = nt.nodes.new("ShaderNodeMath")
    edge_ao.operation = 'MULTIPLY'
    edge_ao.location = (-440, 480)
    nt.links.new(blue_lift.outputs["Result"], edge_ao.inputs[0])
    nt.links.new(ao.outputs["AO"], edge_ao.inputs[1])

    s_scale = nt.nodes.new("ShaderNodeMath")
    s_scale.operation = 'MULTIPLY'
    s_scale.inputs[1].default_value = 0.16
    s_scale.location = (-440, -760)
    nt.links.new(noise_soft.outputs["Fac"], s_scale.inputs[0])

    ridge_raw = nt.nodes.new("ShaderNodeMath")
    ridge_raw.operation = 'ADD'
    ridge_raw.location = (-220, 440)
    nt.links.new(edge_ao.outputs["Value"], ridge_raw.inputs[0])
    nt.links.new(s_scale.outputs["Value"], ridge_raw.inputs[1])

    # Deep pitch-black inside ALL sculpted & 4K shader coral cavities!
    all_pores = nt.nodes.new("ShaderNodeMath")
    all_pores.operation = 'MAXIMUM'
    all_pores.location = (-40, 120)
    nt.links.new(sep.outputs["Red"], all_pores.inputs[0])
    nt.links.new(total_micro_pit.outputs["Value"], all_pores.inputs[1])

    pore_sub_amt = nt.nodes.new("ShaderNodeMath")
    pore_sub_amt.operation = 'MULTIPLY'
    pore_sub_amt.inputs[1].default_value = 1.25
    pore_sub_amt.location = (120, 120)
    nt.links.new(all_pores.outputs["Value"], pore_sub_amt.inputs[0])

    surface_driver = nt.nodes.new("ShaderNodeMath")
    surface_driver.operation = 'SUBTRACT'
    surface_driver.use_clamp = True
    surface_driver.location = (240, 380)
    nt.links.new(ridge_raw.outputs["Value"], surface_driver.inputs[0])
    nt.links.new(pore_sub_amt.outputs["Value"], surface_driver.inputs[1])

    # SILKY DARK OBSIDIAN + WARM AGED BRONZE-COPPER GRADIENT
    ramp_col = nt.nodes.new("ShaderNodeValToRGB")
    ramp_col.location = (420, 420)
    cr = ramp_col.color_ramp
    cr.elements[0].position = 0.04
    cr.elements[0].color = (0.002, 0.001, 0.004, 1.0)  # Pitch-black deep coral pore hollow
    cr.elements[1].position = 0.44
    cr.elements[1].color = (0.014, 0.008, 0.015, 1.0)  # Silky dark volcanic obsidian-plum
    e2 = cr.elements.new(0.66)
    e2.color = (0.052, 0.025, 0.020, 1.0)              # Rich dark mahogany-bronze
    e3 = cr.elements.new(0.84)
    e3.color = (0.125, 0.058, 0.036, 1.0)              # Smooth oxidized bronze-copper sheen
    e4 = cr.elements.new(0.97)
    e4.color = (0.205, 0.105, 0.062, 1.0)              # Soft burnished copper highlight
    nt.links.new(surface_driver.outputs["Value"], ramp_col.inputs["Fac"])

    mix_crater = nt.nodes.new("ShaderNodeMix")
    mix_crater.data_type = 'RGBA'
    mix_crater.location = (600, 340)
    mix_crater.inputs["B"].default_value = (0.018, 0.001, 0.028, 1.0)
    nt.links.new(sep.outputs["Green"], mix_crater.inputs["Factor"])
    nt.links.new(ramp_col.outputs["Color"], mix_crater.inputs["A"])
    nt.links.new(mix_crater.outputs["Result"], bsdf.inputs["Base Color"])

    # Smooth satin-bronze metallic sheen on flowing ridges, matte inside dark pores
    ramp_met = nt.nodes.new("ShaderNodeMapRange")
    ramp_met.inputs["From Min"].default_value = 0.42
    ramp_met.inputs["From Max"].default_value = 0.90
    ramp_met.inputs["To Min"].default_value = 0.05
    ramp_met.inputs["To Max"].default_value = 0.36
    ramp_met.location = (460, 120)
    nt.links.new(surface_driver.outputs["Value"], ramp_met.inputs["Value"])
    nt.links.new(ramp_met.outputs["Result"], bsdf.inputs["Metallic"])

    # Smooth satin flow (0.42 on sculpted outer curves -> 0.92 matte inside deep pores)
    ramp_rgh = nt.nodes.new("ShaderNodeMapRange")
    ramp_rgh.inputs["From Min"].default_value = 0.08
    ramp_rgh.inputs["From Max"].default_value = 0.90
    ramp_rgh.inputs["To Min"].default_value = 0.92
    ramp_rgh.inputs["To Max"].default_value = 0.42
    ramp_rgh.location = (460, -80)
    nt.links.new(surface_driver.outputs["Value"], ramp_rgh.inputs["Value"])
    nt.links.new(ramp_rgh.outputs["Result"], bsdf.inputs["Roughness"])

    # Clean electric royal-magenta energy fissures
    em_mod = nt.nodes.new("ShaderNodeMath")
    em_mod.operation = 'MULTIPLY'
    em_mod.location = (100, -340)
    nt.links.new(sep.outputs["Green"], em_mod.inputs[0])

    em_frac = nt.nodes.new("ShaderNodeMapRange")
    em_frac.inputs["From Min"].default_value = 0.25
    em_frac.inputs["From Max"].default_value = 0.78
    em_frac.inputs["To Min"].default_value = 0.30
    em_frac.inputs["To Max"].default_value = 1.35
    em_frac.location = (-160, -760)
    nt.links.new(noise_soft.outputs["Fac"], em_frac.inputs[0] if False else em_frac.inputs["Value"])
    nt.links.new(em_frac.outputs["Result"], em_mod.inputs[1])

    ramp_em = nt.nodes.new("ShaderNodeValToRGB")
    ramp_em.location = (300, -360)
    ecr = ramp_em.color_ramp
    ecr.elements[0].position = 0.08
    ecr.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    e_v1 = ecr.elements.new(0.28)
    e_v1.color = (0.14, 0.002, 0.20, 1.0)
    e_v2 = ecr.elements.new(0.62)
    e_v2.color = (0.58, 0.008, 0.72, 1.0)
    ecr.elements[1].position = 0.96
    ecr.elements[1].color = (1.00, 0.18, 0.96, 1.0)
    nt.links.new(em_mod.outputs["Value"], ramp_em.inputs["Fac"])

    nt.links.new(ramp_em.outputs["Color"], bsdf.inputs["Emission Color"])
    em_str = nt.nodes.new("ShaderNodeMath")
    em_str.operation = 'MULTIPLY'
    em_str.inputs[1].default_value = 2.60
    em_str.location = (540, -340)
    nt.links.new(em_mod.outputs["Value"], em_str.inputs[0])
    nt.links.new(em_str.outputs["Value"], bsdf.inputs["Emission Strength"])

    # CLEAN SMOOTH-RIMMED CORAL CAVITY BUMP ONLY (NO crunchy origami rock noise!)
    # Smooth outer surface stays buttery smooth while every circular coral cavity indents crisply!
    inv_pit = nt.nodes.new("ShaderNodeMath")
    inv_pit.operation = 'MULTIPLY'
    inv_pit.inputs[1].default_value = -0.75
    inv_pit.location = (140, -560)
    nt.links.new(total_micro_pit.outputs["Value"], inv_pit.inputs[0])

    soft_bump = nt.nodes.new("ShaderNodeMath")
    soft_bump.operation = 'MULTIPLY'
    soft_bump.inputs[1].default_value = 0.06   # Ultra-subtle velvety sheen, ZERO crumpled paper!
    soft_bump.location = (140, -720)
    nt.links.new(noise_soft.outputs["Fac"], soft_bump.inputs[0])

    bump_height = nt.nodes.new("ShaderNodeMath")
    bump_height.operation = 'ADD'
    bump_height.location = (320, -600)
    nt.links.new(inv_pit.outputs["Value"], bump_height.inputs[0])
    nt.links.new(soft_bump.outputs["Value"], bump_height.inputs[1])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.62
    bump.inputs["Distance"].default_value = 0.30
    bump.location = (520, -580)
    nt.links.new(bump_height.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def setup_studio_lighting_and_world():
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = next((n for n in world.node_tree.nodes if n.type == 'BACKGROUND'), None)
    if bg:
        bg.inputs["Color"].default_value = (0.86, 0.83, 0.79, 1.0)
        bg.inputs["Strength"].default_value = 0.25

    def add_area_light(name, loc, rot_deg, energy, size, color):
        ldata = bpy.data.lights.new(name=name, type='AREA')
        ldata.energy = energy
        ldata.size = size
        ldata.color = color
        lobj = bpy.data.objects.new(name=name, object_data=ldata)
        lobj.location = loc
        lobj.rotation_euler = tuple(math.radians(a) for a in rot_deg)
        bpy.context.collection.objects.link(lobj)
        return lobj

    add_area_light("Key_Light",  (105, -135, 145), (50, 0, 38),  1350000, 75, (1.0, 0.93, 0.85))
    add_area_light("Fill_Light", (-115, -95,  70), (75, 0, -48),  480000, 90, (0.88, 0.84, 1.0))
    add_area_light("Rim_Light",  (-65,  135, 130), (125, 0, -25), 1150000, 75, (1.0, 0.86, 0.76))
    add_area_light("Rear_Fill",  ( 80,  130,  55), (100, 0, 148),  680000, 85, (0.96, 0.90, 0.94))

    pdata = bpy.data.lights.new(name="Crater_Glow_Light", type='POINT')
    pdata.energy = 24000
    pdata.shadow_soft_size = 6.0
    pdata.color = (0.85, 0.04, 0.98)
    pobj = bpy.data.objects.new(name="Crater_Glow_Light", object_data=pdata)
    pobj.location = (HOLE_CX, -15.0, HOLE_CZ)
    bpy.context.collection.objects.link(pobj)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam_data.clip_end = 2000
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = (-135, -165, 115)
    direction = Vector((0, 0, 58)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj


def build_scene():
    t_start = time.time()
    clear_scene()
    print("=== Building Black Hole Construct v5 (Silky-Smooth Fluid + Deep Coral Pores) ===")

    shell_obj = build_main_shell("Main_Shell_Raw", n_s=150, n_theta=210)
    body_ribs, crater_veins = build_surface_strata_and_crater_veins()
    tail_roots, tail_filigree = build_hero_tail_and_filigree()
    tendril_filigree = build_all_filigree_tendrils()

    all_filigree = tail_filigree + tendril_filigree + crater_veins
    # UNIFORM quad resolution (voxel=0.40 m -> ~265k clean even quads) so ZERO Decimate is needed!
    print(f"Fusing silky-smooth uniform body: shell + {len(body_ribs)} strata + {len(tail_roots)} tail cores (voxel=0.40m)...")
    body_fused = join_and_voxel_remesh(
        [shell_obj] + body_ribs + tail_roots,
        name="BlackHole_Body",
        voxel_size=0.40,
        smooth_iters=4,
    )
    print(f"  Uniform smooth body: {len(body_fused.data.vertices):,} verts, {len(body_fused.data.polygons):,} quads")

    print(f"Packing analytical seamless tube UVs & joining {len(all_filigree)} silky-smooth baroque filigree tendrils...")
    n_fil = len(all_filigree)
    grid_n = math.ceil(math.sqrt(n_fil))
    cell_w = 1.0 / grid_n
    pad = cell_w * 0.04  # ~15 px clean margin on 4096px atlas!
    inner = cell_w - 2.0 * pad
    for idx, o in enumerate(all_filigree):
        col = idx % grid_n
        row = idx // grid_n
        u0 = col * cell_w + pad
        v0 = row * cell_w + pad
        uv_data = o.data.uv_layers.active.data
        for loop_uv in uv_data:
            loop_uv.uv[0] = u0 + loop_uv.uv[0] * inner
            loop_uv.uv[1] = v0 + loop_uv.uv[1] * inner

    bpy.ops.object.select_all(action='DESELECT')
    for o in all_filigree:
        o.select_set(True)
    bpy.context.view_layer.objects.active = all_filigree[0]
    bpy.ops.object.join()
    filigree_merged = bpy.context.active_object
    filigree_merged.name = "BlackHole_Filigree"

    # Apply light smooth polish on tendrils so every joint/curve is buttery smooth!
    smod = filigree_merged.modifiers.new("FiligreeSmooth", 'SMOOTH')
    smod.factor = 0.35
    smod.iterations = 2
    bpy.ops.object.modifier_apply(modifier=smod.name)
    print(f"  Smooth filigree: {len(filigree_merged.data.vertices):,} verts, {len(filigree_merged.data.polygons):,} quads")

    print("Sculpting smooth fluid contours + deep rounded volcanic-coral cavities...")
    sculpt_smooth_fluid_with_deep_pores(body_fused, is_filigree=False)
    sculpt_smooth_fluid_with_deep_pores(filigree_merged, is_filigree=True)

    calibrate_bounding_box([body_fused, filigree_merged])

    crust_mat = create_silky_porous_material()
    body_fused.data.materials.append(crust_mat)
    filigree_merged.data.materials.append(crust_mat)

    setup_studio_lighting_and_world()

    root = bpy.data.objects.new("BlackHole_Construct", None)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 10.0
    bpy.context.collection.objects.link(root)
    for child in [body_fused, filigree_merged]:
        child.parent = root

    blend_path = "/Users/xzm/Downloads/ant3D/black_hole_construct.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved {blend_path} in {time.time() - t_start:.1f}s")
    print("=== Build v5 Complete ===")


if __name__ == "__main__":
    build_scene()
