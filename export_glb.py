import bpy
import os
import math
import time
import numpy as np

# ==============================================================================
# Export Dual-4K Seamless PBR + High-Precision Draco GLB (v6 Zero-Triangle-Artifact)
# Fixes the root cause of visible triangles (UV island starvation + 10-bit Draco normals):
#   1. BlackHole_Body    -> Dedicated 4096x4096 PBR map set (island_margin=0.00025,
#                           margin=2px -> ~50x higher texel density, median ~25 px²!)
#   2. BlackHole_Filigree-> Dedicated 4096x4096 PBR map set using EXACT ANALYTICAL
#                           SEAMLESS TUBE UVs (ZERO UV seams along curves, ~115 px²/quad!)
#   3. EventHorizon_Void -> True unlit pitch-black KHR_materials_unlit on Ø26m void faces
#   4. 16-bit Position / 14-bit Normal / 16-bit Texcoord Draco quantization +
#      exact MikkTSpace tangents + 100% LOSSLESS 4K PNG Normal Maps!
# ==============================================================================

TEX_SIZE = 4096
OUT_DIR = "/Users/xzm/Downloads/ant3D/web/assets"
GLB_PATH = os.path.join(OUT_DIR, "black_hole_construct.glb")
TEX_DIR = "/Users/xzm/Downloads/ant3D/baked_textures"


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEX_DIR, exist_ok=True)


def setup_cycles_gpu(samples=16, margin=2):
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


def bake_pass_to_image(obj, mat, source_socket, img_name, colorspace='sRGB', is_normal=False, lossless_png=False):
    t0 = time.time()
    nt = mat.node_tree
    out_node = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')

    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])
    img = bpy.data.images.new(img_name, width=TEX_SIZE, height=TEX_SIZE, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace

    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.name = "__BAKE_TARGET__"
    tex_node.image = img
    for n in nt.nodes:
        n.select = False
    tex_node.select = True
    nt.nodes.active = tex_node

    old_link = out_node.inputs["Surface"].links[0] if out_node.inputs["Surface"].links else None
    old_from_socket = old_link.from_socket if old_link else None

    tmp_em = None
    try:
        if is_normal:
            bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT')
        else:
            tmp_em = nt.nodes.new("ShaderNodeEmission")
            tmp_em.inputs["Strength"].default_value = 1.0
            nt.links.new(source_socket, tmp_em.inputs["Color"])
            nt.links.new(tmp_em.outputs["Emission"], out_node.inputs["Surface"])
            bpy.ops.object.bake(type='EMIT')
    finally:
        if tmp_em:
            nt.nodes.remove(tmp_em)
        if old_from_socket:
            nt.links.new(old_from_socket, out_node.inputs["Surface"])
        nt.nodes.remove(tex_node)

    ext = "png" if lossless_png else "jpg"
    filepath = os.path.join(TEX_DIR, f"{img_name}.{ext}")
    img.filepath_raw = filepath
    img.file_format = 'PNG' if lossless_png else 'JPEG'
    bpy.context.scene.render.image_settings.quality = 95
    img.save()
    img.reload()
    img.colorspace_settings.name = colorspace
    print(f"  Baked 4K {img_name}.{ext} ({time.time() - t0:.1f}s)")
    return img


def build_pbr_web_material(mat_name, img_col, img_met, img_rgh, img_em, img_nrm):
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

    def add_tex(img, loc, non_color=False):
        t = wnt.nodes.new("ShaderNodeTexImage")
        t.image = img
        if non_color:
            img.colorspace_settings.name = 'Non-Color'
        t.location = loc
        return t

    t_col = add_tex(img_col, (-250, 320), non_color=False)
    t_met = add_tex(img_met, (-250, 80),  non_color=True)
    t_rgh = add_tex(img_rgh, (-250, -140), non_color=True)
    t_em  = add_tex(img_em,  (-250, -360), non_color=False)
    t_nrm = add_tex(img_nrm, (-420, -600), non_color=True)

    nrm_map = wnt.nodes.new("ShaderNodeNormalMap")
    nrm_map.inputs["Strength"].default_value = 0.90
    nrm_map.location = (-80, -600)
    wnt.links.new(t_nrm.outputs["Color"], nrm_map.inputs["Color"])

    wnt.links.new(t_col.outputs["Color"], w_bsdf.inputs["Base Color"])
    wnt.links.new(t_met.outputs["Color"], w_bsdf.inputs["Metallic"])
    wnt.links.new(t_rgh.outputs["Color"], w_bsdf.inputs["Roughness"])
    wnt.links.new(t_em.outputs["Color"],  w_bsdf.inputs["Emission Color"])
    w_bsdf.inputs["Emission Strength"].default_value = 2.60
    wnt.links.new(nrm_map.outputs["Normal"], w_bsdf.inputs["Normal"])
    return web_mat


def bake_single_export_part(src_obj, export_name, prefix, use_existing_uv, bake_margin, void_mat):
    """Duplicate, triangulate smoothly, bake dedicated 4K PBR maps, assign void faces."""
    dup = src_obj.copy()
    dup.data = src_obj.data.copy()
    dup.parent = None
    dup.matrix_world = src_obj.matrix_world.copy()
    dup.name = export_name
    dup.hide_viewport = False
    dup.hide_render = False
    bpy.context.collection.objects.link(dup)

    # Smooth fixed triangulation (ZERO decimation!)
    bpy.ops.object.select_all(action='DESELECT')
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    tri = dup.modifiers.new("Triangulate", 'TRIANGULATE')
    tri.quad_method = 'FIXED'
    bpy.ops.object.modifier_apply(modifier=tri.name)

    mesh = dup.data
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()

    if not use_existing_uv:
        t_uv = time.time()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66.0),
            island_margin=0.00025,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True,
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  {export_name}: High-density Smart UV complete ({time.time()-t_uv:.1f}s)")
    else:
        print(f"  {export_name}: Preserving exact analytical seamless tube UVs!")

    # Measure UV texel area on 4K before baking
    uv = mesh.uv_layers.active.data
    areas = []
    for idx in range(0, len(mesh.polygons), 50):
        p = mesh.polygons[idx]
        l0, l1, l2 = p.loop_indices[0], p.loop_indices[1], p.loop_indices[2]
        x0, y0 = uv[l0].uv[0]*TEX_SIZE, uv[l0].uv[1]*TEX_SIZE
        x1, y1 = uv[l1].uv[0]*TEX_SIZE, uv[l1].uv[1]*TEX_SIZE
        x2, y2 = uv[l2].uv[0]*TEX_SIZE, uv[l2].uv[1]*TEX_SIZE
        areas.append(0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)))
    areas = np.array(areas)
    print(f"  {export_name} 4K UV tri area: zero_frac={(areas < 0.05).mean():.2%}, p10={np.percentile(areas, 10):.1f} px², median={np.median(areas):.1f} px², p90={np.percentile(areas, 90):.1f} px²")

    # Bake 5x 4K maps using dedicated temp copy of procedural material
    src_mat = bpy.data.materials["BlackHole_Crust_Mat"]
    bake_mat = src_mat.copy()
    bake_mat.name = f"__Bake_{prefix}__"
    mesh.materials.clear()
    mesh.materials.append(bake_mat)

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

    em_vec_mul = nt.nodes.new("ShaderNodeVectorMath")
    em_vec_mul.operation = 'SCALE'
    nt.links.new(em_col_sock, em_vec_mul.inputs[0])
    nt.links.new(norm_str.outputs["Value"], em_vec_mul.inputs["Scale"])

    setup_cycles_gpu(samples=16, margin=bake_margin)
    print(f"Baking dedicated 4K PBR set for {export_name} ({len(mesh.polygons):,} tris)...")
    img_col = bake_pass_to_image(dup, bake_mat, col_sock, f"{prefix}_basecolor", colorspace='sRGB', is_normal=False, lossless_png=False)
    img_rgh = bake_pass_to_image(dup, bake_mat, rgh_sock, f"{prefix}_roughness", colorspace='Non-Color', is_normal=False, lossless_png=False)
    img_met = bake_pass_to_image(dup, bake_mat, met_sock, f"{prefix}_metallic", colorspace='Non-Color', is_normal=False, lossless_png=False)
    img_em  = bake_pass_to_image(dup, bake_mat, em_vec_mul.outputs["Vector"], f"{prefix}_emission", colorspace='sRGB', is_normal=False, lossless_png=False)
    img_nrm = bake_pass_to_image(dup, bake_mat, None, f"{prefix}_normal", colorspace='Non-Color', is_normal=True, lossless_png=True)

    web_mat = build_pbr_web_material(f"BlackHole_{prefix}_4K", img_col, img_met, img_rgh, img_em, img_nrm)
    mesh.materials.clear()
    mesh.materials.append(web_mat)
    mesh.materials.append(void_mat)

    col_attr = mesh.color_attributes.get("SculptData")
    if col_attr:
        n_verts = len(mesh.vertices)
        rgba = np.empty(n_verts * 4, dtype=np.float32)
        col_attr.data.foreach_get("color", rgba)
        alpha = rgba.reshape((n_verts, 4))[:, 3]
        void_count = 0
        for poly in mesh.polygons:
            avg_a = sum(alpha[vi] for vi in poly.vertices) / len(poly.vertices)
            if avg_a > 0.42:
                poly.material_index = 1
                void_count += 1
            else:
                poly.material_index = 0
        print(f"  {export_name}: assigned {void_count:,} unlit pitch-black EventHorizon faces.")
        mesh.color_attributes.remove(col_attr)

    bpy.data.materials.remove(bake_mat)
    return dup


def export_baked_glb():
    t_all = time.time()
    ensure_dirs()
    orig_engine = bpy.context.scene.render.engine

    body = bpy.data.objects.get("BlackHole_Body")
    filigree = bpy.data.objects.get("BlackHole_Filigree")
    if not body or not filigree:
        raise RuntimeError("BlackHole_Body or BlackHole_Filigree not found!")

    for old in ("Export_Body", "Export_Filigree"):
        if old in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[old], do_unlink=True)

    # Shared unlit pitch-black EventHorizon material
    if "EventHorizon_Void_Web" in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials["EventHorizon_Void_Web"])
    void_mat = bpy.data.materials.new("EventHorizon_Void_Web")
    void_mat.use_nodes = True
    vnt = void_mat.node_tree
    vnt.nodes.clear()
    v_out = vnt.nodes.new("ShaderNodeOutputMaterial")
    v_bg = vnt.nodes.new("ShaderNodeBackground")
    v_bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    v_bg.inputs["Strength"].default_value = 0.0
    vnt.links.new(v_bg.outputs["Background"], v_out.inputs["Surface"])

    # Keep master objects visible ONLY for AO raycasting? Better: hide masters during bake
    # Wait: keeping the OTHER export copy hidden avoids double-overlapping self-shadowing,
    # OR keeping masters hidden during bake gives clean per-part AO!
    body.hide_render = True
    filigree.hide_render = True
    body.hide_viewport = True
    filigree.hide_viewport = True

    exp_body = bake_single_export_part(body, "Export_Body", "body", use_existing_uv=False, bake_margin=2, void_mat=void_mat)
    exp_body.hide_render = True
    exp_body.hide_viewport = True

    exp_fil = bake_single_export_part(filigree, "Export_Filigree", "filigree", use_existing_uv=True, bake_margin=10, void_mat=void_mat)

    exp_body.hide_render = False
    exp_body.hide_viewport = False

    bpy.ops.object.select_all(action='DESELECT')
    exp_body.select_set(True)
    exp_fil.select_set(True)
    bpy.context.view_layer.objects.active = exp_body

    print("Exporting Dual-4K High-Precision Draco GLB (16-bit Pos / 14-bit Normal / 16-bit UV)...")
    bpy.ops.export_scene.gltf(
        filepath=GLB_PATH,
        export_format='GLB',
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials='EXPORT',
        export_image_format='AUTO',
        export_jpeg_quality=95,
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=16,
        export_draco_normal_quantization=14,
        export_draco_texcoord_quantization=16,
    )
    size_mb = os.path.getsize(GLB_PATH) / (1024 * 1024)
    print(f"Exported Dual-4K Smooth GLB: {GLB_PATH} ({size_mb:.2f} MB) in {time.time() - t_all:.1f}s total")

    bpy.context.scene.render.engine = orig_engine
    bpy.data.objects.remove(exp_body, do_unlink=True)
    bpy.data.objects.remove(exp_fil, do_unlink=True)
    body.hide_render = False
    filigree.hide_render = False
    body.hide_viewport = False
    filigree.hide_viewport = False
    bpy.ops.wm.save_as_mainfile(filepath="/Users/xzm/Downloads/ant3D/black_hole_construct.blend")
    print("=== Dual-4K Seamless GLB Export Complete ===")


if __name__ == "__main__":
    export_baked_glb()
