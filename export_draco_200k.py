import bpy
import bmesh
import os
import math
import time
import numpy as np
from mathutils import Vector, Matrix

# ==============================================================================
# Production <200k Triangles + High-to-Low 4K Bake + Max Draco Compression
# Target:
#   - Total triangles  : ~175,000 - 190,000 (< 200k!)
#   - Body_Low         : ~100k uniform smooth tris (Voxel 0.92m + Smooth, ZERO Decimate!)
#                        with 4K High-to-Low Normal/PBR baked from 543k-tri High-Poly Body!
#   - Filigree_Low     : ~80k smooth analytical-UV tris (100% of all 113 tendrils preserved,
#                        rebuilt cleanly with smooth quad rings, ZERO Decimate!)
#   - EventHorizon     : True unlit pitch-black Ø26m core (KHR_materials_unlit)
#   - Compression      : Draco Level 10 (14-bit Pos / 12-bit Normal / 14-bit UV)
#                        + WebP Q90/Q94 4K PBR -> ~8-12 MB total!
# ==============================================================================

TEX_SIZE = 4096
OUT_DIR = "/Users/xzm/Downloads/ant3D/web/assets"
GLB_DRACO_PATH = os.path.join(OUT_DIR, "black_hole_construct_draco200k.glb")
TEX_DIR = "/Users/xzm/Downloads/ant3D/baked_textures_200k"


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEX_DIR, exist_ok=True)


def setup_cycles_gpu(samples=16, margin=4, selected_to_active=False, cage_extrusion=3.2):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    used_gpu = False
    for d in prefs.devices:
        if d.type in ('METAL', 'CUDA', 'OPTIX', 'HIP'):
            d.use = True
            used_gpu = True
        else:
            d.use = False
    scene.cycles.device = 'GPU' if used_gpu else 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.render.bake.margin = margin
    scene.render.bake.use_clear = True
    scene.render.bake.use_selected_to_active = selected_to_active
    scene.render.bake.cage_extrusion = cage_extrusion
    scene.render.bake.max_ray_distance = 7.0 if selected_to_active else 0.0


def bake_pass(active_obj, src_mat, target_mat, source_socket, img_name, colorspace='sRGB',
              is_normal=False, samples=4, selected_objs=None, cage_extrusion=3.2, margin=4):
    t0 = time.time()
    s2a = bool(selected_objs)
    setup_cycles_gpu(samples=samples, margin=margin, selected_to_active=s2a, cage_extrusion=cage_extrusion)

    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    img = bpy.data.images.new(img_name, width=TEX_SIZE, height=TEX_SIZE, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace

    # Target image texture node lives on active_obj's target_mat!
    tnt = target_mat.node_tree
    tex_node = tnt.nodes.new("ShaderNodeTexImage")
    tex_node.name = "__BAKE_TARGET__"
    tex_node.image = img
    for n in tnt.nodes:
        n.select = False
    tex_node.select = True
    tnt.nodes.active = tex_node

    # Temporarily route source_socket on src_mat (the high-poly or self material) -> Emission
    snt = src_mat.node_tree
    out_node = next(n for n in snt.nodes if n.type == 'OUTPUT_MATERIAL')
    old_link = out_node.inputs["Surface"].links[0] if out_node.inputs["Surface"].links else None
    old_from_socket = old_link.from_socket if old_link else None

    bpy.ops.object.select_all(action='DESELECT')
    if selected_objs:
        for so in selected_objs:
            so.select_set(True)
    active_obj.select_set(True)
    bpy.context.view_layer.objects.active = active_obj

    tmp_em = None
    try:
        if is_normal:
            bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT')
        else:
            tmp_em = snt.nodes.new("ShaderNodeEmission")
            tmp_em.inputs["Strength"].default_value = 1.0
            snt.links.new(source_socket, tmp_em.inputs["Color"])
            snt.links.new(tmp_em.outputs["Emission"], out_node.inputs["Surface"])
            bpy.ops.object.bake(type='EMIT')
    finally:
        if tmp_em:
            snt.nodes.remove(tmp_em)
        if old_from_socket:
            snt.links.new(old_from_socket, out_node.inputs["Surface"])
        tnt.nodes.remove(tex_node)

    filepath = os.path.join(TEX_DIR, f"{img_name}.png")
    img.filepath_raw = filepath
    img.file_format = 'PNG'
    img.save()
    img.reload()
    img.colorspace_settings.name = colorspace
    print(f"  Baked 4K {img_name}.png (spp={samples}, s2a={s2a}, {time.time() - t0:.1f}s)")
    return img


def make_bake_src_mat(prefix):
    src_mat = bpy.data.materials["BlackHole_Crust_Mat"]
    bake_mat = src_mat.copy()
    bake_mat.name = f"__BakeSrc_{prefix}__"
    nt = bake_mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    col_sock = bsdf.inputs["Base Color"].links[0].from_socket
    met_sock = bsdf.inputs["Metallic"].links[0].from_socket
    rgh_sock = bsdf.inputs["Roughness"].links[0].from_socket
    em_col_sock = bsdf.inputs["Emission Color"].links[0].from_socket
    em_str_sock = bsdf.inputs["Emission Strength"].links[0].from_socket

    norm_str = nt.nodes.new("ShaderNodeMath")
    norm_str.operation = 'DIVIDE'
    norm_str.inputs[1].default_value = 2.60
    norm_str.use_clamp = True
    nt.links.new(em_str_sock, norm_str.inputs[0])

    em_vec = nt.nodes.new("ShaderNodeVectorMath")
    em_vec.operation = 'SCALE'
    nt.links.new(em_col_sock, em_vec.inputs[0])
    nt.links.new(norm_str.outputs["Value"], em_vec.inputs["Scale"])
    return bake_mat, col_sock, rgh_sock, met_sock, em_vec.outputs["Vector"]


def build_web_mat(mat_name, prefix):
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])
    web_mat = bpy.data.materials.new(mat_name)
    web_mat.use_nodes = True
    wnt = web_mat.node_tree
    wnt.nodes.clear()

    w_out = wnt.nodes.new("ShaderNodeOutputMaterial")
    w_out.location = (600, 0)
    w_bsdf = wnt.nodes.new("ShaderNodeBsdfPrincipled")
    w_bsdf.location = (250, 0)
    wnt.links.new(w_bsdf.outputs["BSDF"], w_out.inputs["Surface"])

    def load_tex(suffix, loc, non_color=False):
        fp = os.path.join(TEX_DIR, f"{prefix}_{suffix}.png")
        img = bpy.data.images.load(fp, check_existing=False)
        img.name = f"{prefix}_{suffix}"
        img.colorspace_settings.name = 'Non-Color' if non_color else 'sRGB'
        t = wnt.nodes.new("ShaderNodeTexImage")
        t.image = img
        t.location = loc
        return t

    t_col = load_tex("basecolor", (-250, 320), non_color=False)
    t_met = load_tex("metallic",  (-250, 80),  non_color=True)
    t_rgh = load_tex("roughness", (-250, -140), non_color=True)
    t_em  = load_tex("emission",  (-250, -360), non_color=False)
    t_nrm = load_tex("normal",    (-420, -600), non_color=True)

    nrm_map = wnt.nodes.new("ShaderNodeNormalMap")
    nrm_map.inputs["Strength"].default_value = 0.95
    nrm_map.location = (-80, -600)
    wnt.links.new(t_nrm.outputs["Color"], nrm_map.inputs["Color"])

    wnt.links.new(t_col.outputs["Color"], w_bsdf.inputs["Base Color"])
    wnt.links.new(t_met.outputs["Color"], w_bsdf.inputs["Metallic"])
    wnt.links.new(t_rgh.outputs["Color"], w_bsdf.inputs["Roughness"])
    wnt.links.new(t_em.outputs["Color"],  w_bsdf.inputs["Emission Color"])
    w_bsdf.inputs["Emission Strength"].default_value = 2.60
    wnt.links.new(nrm_map.outputs["Normal"], w_bsdf.inputs["Normal"])
    return web_mat


def step1_build_and_bake_body_low():
    """
    Step 1: Create silky-smooth uniform low-poly Body (~105k tris, ZERO Decimate!)
    and bake 4K High-to-Low Normal + PBR maps from the 543k-tri High-Poly BlackHole_Body!
    """
    t0 = time.time()
    ensure_dirs()
    body_hi = bpy.data.objects["BlackHole_Body"]
    fil_hi = bpy.data.objects["BlackHole_Filigree"]

    if "Low200k_Body" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Low200k_Body"], do_unlink=True)

    # Create silky-smooth uniform low-poly body via Voxel Remesh (0.90m -> ~52k quads = ~104k tris!)
    body_lo = body_hi.copy()
    body_lo.data = body_hi.data.copy()
    body_lo.parent = None
    body_lo.matrix_world = body_hi.matrix_world.copy()
    body_lo.name = "Low200k_Body"
    body_lo.hide_viewport = False
    body_lo.hide_render = False
    bpy.context.collection.objects.link(body_lo)

    bpy.ops.object.select_all(action='DESELECT')
    body_lo.select_set(True)
    bpy.context.view_layer.objects.active = body_lo

    rm = body_lo.modifiers.new("RemeshLow", 'REMESH')
    rm.mode = 'VOXEL'
    rm.voxel_size = 0.90
    rm.adaptivity = 0.0
    rm.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=rm.name)

    sm = body_lo.modifiers.new("SmoothLow", 'SMOOTH')
    sm.factor = 0.50
    sm.iterations = 3
    bpy.ops.object.modifier_apply(modifier=sm.name)

    # Transfer SculptData (specifically Alpha void_mask) from High-Poly Body for crisp Ø26m void slot!
    dt = body_lo.modifiers.new("TransferSculpt", 'DATA_TRANSFER')
    dt.object = body_hi
    dt.use_loop_data = False
    dt.use_vert_data = True
    dt.data_types_verts = {'COLOR_VERTEX'}
    dt.vert_mapping = 'NEAREST'
    try:
        bpy.ops.object.datalayout_transfer(modifier=dt.name)
    except Exception:
        pass
    bpy.ops.object.modifier_apply(modifier=dt.name)

    tri = body_lo.modifiers.new("Triangulate", 'TRIANGULATE')
    tri.quad_method = 'FIXED'
    bpy.ops.object.modifier_apply(modifier=tri.name)

    mesh_lo = body_lo.data
    mesh_lo.polygons.foreach_set("use_smooth", [True] * len(mesh_lo.polygons))
    mesh_lo.update()
    print(f"Low200k_Body created: {len(mesh_lo.vertices):,} verts, {len(mesh_lo.polygons):,} tris (ZERO Decimate!)")

    # Clean high-density UV unwrap (only ~104k tris on a full 4K map -> ~115 px² per triangle!!)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(66.0),
        island_margin=0.0005,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    uv = mesh_lo.uv_layers.active.data
    areas = []
    for idx in range(0, len(mesh_lo.polygons), 25):
        p = mesh_lo.polygons[idx]
        l0, l1, l2 = p.loop_indices[0], p.loop_indices[1], p.loop_indices[2]
        x0, y0 = uv[l0].uv[0]*TEX_SIZE, uv[l0].uv[1]*TEX_SIZE
        x1, y1 = uv[l1].uv[0]*TEX_SIZE, uv[l1].uv[1]*TEX_SIZE
        x2, y2 = uv[l2].uv[0]*TEX_SIZE, uv[l2].uv[1]*TEX_SIZE
        areas.append(0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)))
    areas = np.array(areas)
    print(f"Low200k_Body 4K UV area: zero_frac={(areas < 0.05).mean():.2%}, p10={np.percentile(areas, 10):.1f} px², median={np.median(areas):.1f} px²")

    # Prepare temp target material on body_lo and temp bake source material on body_hi
    tgt_mat = bpy.data.materials.new("__LowBody_Target__")
    tgt_mat.use_nodes = True
    mesh_lo.materials.clear()
    mesh_lo.materials.append(tgt_mat)

    orig_hi_mats = list(body_hi.data.materials)
    src_mat, col_s, rgh_s, met_s, em_s = make_bake_src_mat("body200k")
    body_hi.data.materials.clear()
    body_hi.data.materials.append(src_mat)

    # Hide filigree during body bake
    body_hi.hide_viewport = False
    body_hi.hide_render = False
    fil_hi.hide_viewport = True
    fil_hi.hide_render = True

    print("High-to-Low 4K Cycles GPU Bake: BlackHole_Body (543k tris) -> Low200k_Body (~104k tris)...")
    bake_pass(body_lo, src_mat, tgt_mat, col_s, "body_basecolor", 'sRGB',      is_normal=False, samples=14, selected_objs=[body_hi], cage_extrusion=3.5, margin=4)
    bake_pass(body_lo, src_mat, tgt_mat, rgh_s, "body_roughness", 'Non-Color', is_normal=False, samples=4,  selected_objs=[body_hi], cage_extrusion=3.5, margin=4)
    bake_pass(body_lo, src_mat, tgt_mat, met_s, "body_metallic",  'Non-Color', is_normal=False, samples=4,  selected_objs=[body_hi], cage_extrusion=3.5, margin=4)
    bake_pass(body_lo, src_mat, tgt_mat, em_s,  "body_emission",  'sRGB',      is_normal=False, samples=4,  selected_objs=[body_hi], cage_extrusion=3.5, margin=4)
    bake_pass(body_lo, src_mat, tgt_mat, None,  "body_normal",    'Non-Color', is_normal=True,  samples=4,  selected_objs=[body_hi], cage_extrusion=3.5, margin=4)

    # Restore body_hi materials & cleanup temp mats
    body_hi.data.materials.clear()
    for m in orig_hi_mats:
        body_hi.data.materials.append(m)
    bpy.data.materials.remove(src_mat)
    bpy.data.materials.remove(tgt_mat)
    fil_hi.hide_viewport = False
    fil_hi.hide_render = False
    print(f"=== Step 1 (Body High-to-Low 4K Bake) Complete in {time.time() - t0:.1f}s ===")


def step2_build_filigree_low_and_export_draco():
    """
    Step 2: Build silky-smooth low-poly Filigree (~80k tris, ALL 113 baroque tendrils +
    18 inner crater veins with exact analytical seamless tube UVs, ZERO Decimate!),
    bake its 4K PBR maps, combine with Low200k_Body (<200k tris total!), and export
    Draco Level 10 + WebP Q92 GLB!
    """
    t0 = time.time()
    ensure_dirs()
    orig_engine = bpy.context.scene.render.engine

    body_hi = bpy.data.objects["BlackHole_Body"]
    fil_hi = bpy.data.objects["BlackHole_Filigree"]
    body_lo = bpy.data.objects["Low200k_Body"]

    if "Low200k_Filigree" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Low200k_Filigree"], do_unlink=True)

    # Build clean smooth low-poly filigree using build_model.py's exact tendril definitions
    # but with sleek smooth segment counts (n_seg ~ 10/8, n_steps ~ 38/26 -> ~80k smooth tris!)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bm_mod", "/Users/xzm/Downloads/ant3D/build_model.py")
    bm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm_mod)

    # Monkey-patch segment & step density cleanly for <85k smooth filigree tris (ZERO Decimate!)
    orig_create = bm_mod.create_smooth_fluid_tendril
    orig_scroll = bm_mod.generate_baroque_scroll

    def create_low_tendril(name, points, radii, n_seg=22, flutes=5, flute_amp=0.11, twist=2.2, oval=1.22):
        # Smooth circular 10-12 segment loops (still 100% smooth normals, ~40% of tris!)
        lo_seg = 12 if n_seg >= 20 else (10 if n_seg >= 15 else 8)
        # Uniformly resample polyline to ~55% steps while preserving exact start & end curl!
        n_pts = len(points)
        target_n = max(16, int(n_pts * 0.56))
        if target_n < n_pts:
            idxs = np.linspace(0, n_pts - 1, target_n)
            new_pts, new_r = [], []
            for fv in idxs:
                i0 = int(math.floor(fv))
                i1 = min(n_pts - 1, i0 + 1)
                frac = fv - i0
                p0, p1 = Vector(points[i0]), Vector(points[i1])
                new_pts.append(p0.lerp(p1, frac))
                new_r.append(radii[i0] * (1.0 - frac) + radii[i1] * frac)
            points, radii = new_pts, new_r
        return orig_create(name, points, radii, n_seg=lo_seg, flutes=flutes, flute_amp=flute_amp, twist=twist, oval=oval)

    bm_mod.create_smooth_fluid_tendril = create_low_tendril

    # Generate all 113 tendrils + 18 inner crater veins (discard body_ribs/tail_roots)
    body_ribs, crater_veins = bm_mod.build_surface_strata_and_crater_veins()
    tail_roots, tail_filigree = bm_mod.build_hero_tail_and_filigree()
    tendril_filigree = bm_mod.build_all_filigree_tendrils()
    for unused in body_ribs + tail_roots:
        bpy.data.objects.remove(unused, do_unlink=True)

    all_fil = tail_filigree + tendril_filigree + crater_veins

    # Pack exact analytical seamless tube UVs into 11x11 tile grid
    n_fil = len(all_fil)
    grid_n = math.ceil(math.sqrt(n_fil))
    cell_w = 1.0 / grid_n
    pad = cell_w * 0.04
    inner = cell_w - 2.0 * pad
    for idx, o in enumerate(all_fil):
        col, row = idx % grid_n, idx // grid_n
        u0, v0 = col * cell_w + pad, row * cell_w + pad
        for loop_uv in o.data.uv_layers.active.data:
            loop_uv.uv[0] = u0 + loop_uv.uv[0] * inner
            loop_uv.uv[1] = v0 + loop_uv.uv[1] * inner

    bpy.ops.object.select_all(action='DESELECT')
    for o in all_fil:
        o.select_set(True)
    bpy.context.view_layer.objects.active = all_fil[0]
    bpy.ops.object.join()
    fil_lo = bpy.context.active_object
    fil_lo.name = "Low200k_Filigree"

    sm = fil_lo.modifiers.new("FilSmooth", 'SMOOTH')
    sm.factor = 0.35
    sm.iterations = 2
    bpy.ops.object.modifier_apply(modifier=sm.name)

    # SculptData vertex attributes (zero vertex displacement on filigree -> 100% silky smooth!)
    bm_mod.sculpt_smooth_fluid_with_deep_pores(fil_lo, is_filigree=True)

    # Align Low200k_Filigree EXACTLY to master BlackHole_Filigree's calibrated world bounding box!
    bpy.context.view_layer.update()
    def get_bounds(obj):
        mn = Vector((1e9, 1e9, 1e9))
        mx = Vector((-1e9, -1e9, -1e9))
        for c in obj.bound_box:
            wc = obj.matrix_world @ Vector(c)
            mn.x = min(mn.x, wc.x); mn.y = min(mn.y, wc.y); mn.z = min(mn.z, wc.z)
            mx.x = max(mx.x, wc.x); mx.y = max(mx.y, wc.y); mx.z = max(mx.z, wc.z)
        return mn, mx

    hi_mn, hi_mx = get_bounds(fil_hi)
    lo_mn, lo_mx = get_bounds(fil_lo)
    hi_sz, lo_sz = hi_mx - hi_mn, lo_mx - lo_mn
    sx, sy, sz = hi_sz.x / lo_sz.x, hi_sz.y / lo_sz.y, hi_sz.z / lo_sz.z

    mesh_f = fil_lo.data
    nv = len(mesh_f.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    mesh_f.vertices.foreach_get("co", co)
    co = co.reshape((nv, 3))
    co[:, 0] = hi_mn.x + (co[:, 0] - lo_mn.x) * sx
    co[:, 1] = hi_mn.y + (co[:, 1] - lo_mn.y) * sy
    co[:, 2] = hi_mn.z + (co[:, 2] - lo_mn.z) * sz
    mesh_f.vertices.foreach_set("co", co.ravel())
    mesh_f.update()

    tri = fil_lo.modifiers.new("Triangulate", 'TRIANGULATE')
    tri.quad_method = 'FIXED'
    bpy.ops.object.modifier_apply(modifier=tri.name)
    mesh_f.polygons.foreach_set("use_smooth", [True] * len(mesh_f.polygons))
    mesh_f.update()

    total_tris = len(body_lo.data.polygons) + len(mesh_f.polygons)
    print(f"Low200k_Filigree: {len(mesh_f.polygons):,} tris | TOTAL = {total_tris:,} tris (<200k target!)")

    # Bake 4K PBR maps for Low200k_Filigree using its seamless analytical tube UVs
    src_mat, col_s, rgh_s, met_s, em_s = make_bake_src_mat("fil200k")
    mesh_f.materials.clear()
    mesh_f.materials.append(src_mat)

    body_hi.hide_viewport = True; body_hi.hide_render = True
    fil_hi.hide_viewport = True;  fil_hi.hide_render = True
    body_lo.hide_viewport = True; body_lo.hide_render = True

    print("Baking 4K PBR maps for Low200k_Filigree (exact analytical tube UVs)...")
    bake_pass(fil_lo, src_mat, src_mat, col_s, "filigree_basecolor", 'sRGB',      is_normal=False, samples=12, selected_objs=None, margin=10)
    bake_pass(fil_lo, src_mat, src_mat, rgh_s, "filigree_roughness", 'Non-Color', is_normal=False, samples=4,  selected_objs=None, margin=10)
    bake_pass(fil_lo, src_mat, src_mat, met_s, "filigree_metallic",  'Non-Color', is_normal=False, samples=4,  selected_objs=None, margin=10)
    bake_pass(fil_lo, src_mat, src_mat, em_s,  "filigree_emission",  'sRGB',      is_normal=False, samples=4,  selected_objs=None, margin=10)
    bake_pass(fil_lo, src_mat, src_mat, None,  "filigree_normal",    'Non-Color', is_normal=True,  samples=4,  selected_objs=None, margin=10)
    bpy.data.materials.remove(src_mat)

    # Assemble final clean Web PBR + unlit pitch-black EventHorizon materials
    if "EventHorizon_Void_200k" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["EventHorizon_Void_200k"])
    void_mat = bpy.data.materials.new("EventHorizon_Void_200k")
    void_mat.use_nodes = True
    vnt = void_mat.node_tree
    vnt.nodes.clear()
    v_out = vnt.nodes.new("ShaderNodeOutputMaterial")
    v_bg = vnt.nodes.new("ShaderNodeBackground")
    v_bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    v_bg.inputs["Strength"].default_value = 0.0
    vnt.links.new(v_bg.outputs["Background"], v_out.inputs["Surface"])

    body_web_mat = build_web_mat("BlackHole_Body_200k_4K", "body")
    fil_web_mat  = build_web_mat("BlackHole_Filigree_200k_4K", "filigree")

    def apply_mats_and_void(obj, main_mat):
        m = obj.data
        m.materials.clear()
        m.materials.append(main_mat)
        m.materials.append(void_mat)
        ca = m.color_attributes.get("SculptData")
        if ca:
            n_v = len(m.vertices)
            rgba = np.empty(n_v * 4, dtype=np.float32)
            ca.data.foreach_get("color", rgba)
            alpha = rgba.reshape((n_v, 4))[:, 3]
            vc = 0
            for p in m.polygons:
                if sum(alpha[vi] for vi in p.vertices) / len(p.vertices) > 0.42:
                    p.material_index = 1
                    vc += 1
                else:
                    p.material_index = 0
            print(f"  {obj.name}: {vc:,} unlit pitch-black EventHorizon faces assigned.")
            m.color_attributes.remove(ca)

    apply_mats_and_void(body_lo, body_web_mat)
    apply_mats_and_void(fil_lo, fil_web_mat)

    body_lo.hide_viewport = False; body_lo.hide_render = False
    fil_lo.hide_viewport = False;  fil_lo.hide_render = False

    bpy.ops.object.select_all(action='DESELECT')
    body_lo.select_set(True)
    fil_lo.select_set(True)
    bpy.context.view_layer.objects.active = body_lo

    print("Exporting <200k Production Draco Level-10 + WebP Q92 GLB...")
    bpy.ops.export_scene.gltf(
        filepath=GLB_DRACO_PATH,
        export_format='GLB',
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials='EXPORT',
        export_image_format='WEBP',
        export_image_quality=92,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=10,
        export_draco_position_quantization=14,
        export_draco_normal_quantization=12,
        export_draco_texcoord_quantization=14,
    )
    size_mb = os.path.getsize(GLB_DRACO_PATH) / (1024 * 1024)
    print(f"Exported <200k Draco GLB: {GLB_DRACO_PATH} ({size_mb:.2f} MB, {total_tris:,} tris) in {time.time() - t0:.1f}s")

    # Cleanup temp low-poly objects & restore master scene
    bpy.context.scene.render.engine = orig_engine
    bpy.data.objects.remove(body_lo, do_unlink=True)
    bpy.data.objects.remove(fil_lo, do_unlink=True)
    body_hi.hide_viewport = False; body_hi.hide_render = False
    fil_hi.hide_viewport = False;  fil_hi.hide_render = False
    print("=== Step 2 (<200k Draco GLB Export) Complete ===")


if __name__ == "__main__":
    step1_build_and_bake_body_low()
    step2_build_filigree_low_and_export_draco()
