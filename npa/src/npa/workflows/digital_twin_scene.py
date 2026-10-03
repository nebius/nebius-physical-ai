"""Author portable industrial scenes and render native CUDA or RTX OptiX viewpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def _material(bpy, name, color, metallic=0.0, roughness=0.35):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _box(bpy, name, position, size, material, bevel=0.04):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel:
        modifier = obj.modifiers.new("Machined edges", "BEVEL")
        modifier.width, modifier.segments = bevel, 3
        obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    return obj


def _link(bpy, name, start, end, radius, material):
    from mathutils import Vector

    direction = Vector(end) - Vector(start)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=radius,
        depth=direction.length,
        location=(Vector(start) + Vector(end)) / 2,
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Rounded edges", "BEVEL")
    bevel.width, bevel.segments = 0.035, 3
    obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    return obj


def _robot(bpy, materials):
    orange, steel, teal = materials["orange"], materials["steel"], materials["teal"]
    _box(bpy, "Robot pedestal", (-0.6, 0, 0.35), (1.3, 1.3, 0.7), steel)
    points = [
        (-0.6, 0, 0.7),
        (-0.6, 0, 1.4),
        (-0.5, 0, 2.8),
        (0.8, 0, 3.3),
        (1.6, 0, 2.6),
        (1.6, 0, 2.25),
    ]
    for index, (start, end) in enumerate(zip(points, points[1:])):
        _link(bpy, f"Robot link {index}", start, end, 0.21, orange)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=0.26, location=start
        )
        bpy.context.object.name = f"Joint housing {index}"
        bpy.context.object.data.materials.append(steel)
    _box(bpy, "End effector", points[-1], (0.55, 0.5, 0.25), teal)
    for offset in (-0.22, 0.22):
        _box(bpy, "Gripper finger", (1.6 + offset, 0, 2.02), (0.08, 0.25, 0.35), steel)


def _conveyor(bpy, materials):
    steel, dark, teal = materials["steel"], materials["dark"], materials["teal"]
    _box(bpy, "Conveyor chassis", (2.15, 0, 0.95), (1.8, 5.0, 0.3), steel)
    _box(bpy, "Belt", (2.15, 0, 1.13), (1.5, 4.9, 0.08), dark)
    for y in (-2.1, 2.1):
        for x in (1.5, 2.8):
            _box(bpy, "Conveyor support", (x, y, 0.45), (0.12, 0.12, 0.9), steel)
    for y in (-1.6, 0, 1.6):
        _box(bpy, "Inspection tray", (2.15, y, 1.28), (1.05, 0.8, 0.2), teal)
        _box(bpy, "Workpiece", (2.15, y, 1.5), (0.55, 0.45, 0.28), materials["white"])
    for y in [index * 0.2 - 2.4 for index in range(25)]:
        _box(bpy, "Belt seam", (2.15, y, 1.178), (1.48, 0.012, 0.005), steel, 0)


def _racks(bpy, materials):
    for x in (-3.1, -1.1, 0.9):
        for z in (0.3, 1.15, 2.0):
            _box(
                bpy, "Storage shelf", (x, 2.65, z), (1.8, 0.75, 0.1), materials["steel"]
            )
            for offset in (-0.45, 0.45):
                _box(
                    bpy,
                    "Parts bin",
                    (x + offset, 2.65, z + 0.25),
                    (0.7, 0.6, 0.4),
                    materials["teal"],
                )
        for offset in (-0.85, 0.85):
            _box(
                bpy,
                "Rack upright",
                (x + offset, 2.65, 1.2),
                (0.1, 0.75, 2.4),
                materials["steel"],
            )


def _floor(bpy, materials):
    _box(bpy, "Factory foundation", (0, 0, -0.17), (9.2, 7.4, 0.3), materials["dark"])
    for x in range(-4, 5):
        _box(
            bpy,
            "Floor joint",
            (x, 0, -0.012),
            (0.014, 7.0, 0.008),
            materials["steel"],
            0,
        )
    for y in range(-3, 4):
        _box(
            bpy,
            "Floor joint",
            (0, y, -0.012),
            (8.8, 0.014, 0.008),
            materials["steel"],
            0,
        )
    for y in (-2.55, 1.35):
        _box(
            bpy,
            "Workcell boundary",
            (-0.6, y, 0.002),
            (3.3, 0.09, 0.008),
            materials["yellow"],
            0,
        )
    for x in (-2.2, 1.05):
        _box(
            bpy,
            "Workcell boundary",
            (x, -0.6, 0.002),
            (0.09, 3.9, 0.008),
            materials["yellow"],
            0,
        )


def _factory(bpy):
    materials = {
        "dark": _material(bpy, "Graphite", (0.055, 0.075, 0.095), 0.3),
        "steel": _material(bpy, "Brushed steel", (0.25, 0.32, 0.38), 0.8),
        "orange": _material(bpy, "Safety orange", (0.95, 0.19, 0.025), 0.25),
        "teal": _material(bpy, "Teal enamel", (0.015, 0.38, 0.38), 0.3),
        "white": _material(bpy, "Ceramic", (0.7, 0.77, 0.8), 0.1),
        "yellow": _material(bpy, "Safety marking", (1, 0.62, 0.04)),
    }
    _floor(bpy, materials)
    _robot(bpy, materials)
    _conveyor(bpy, materials)
    _racks(bpy, materials)
    _box(bpy, "Control cabinet", (-3.1, -0.5, 0.9), (0.8, 0.7, 1.8), materials["white"])
    _box(
        bpy,
        "Control screen",
        (-3.1, -0.859, 1.35),
        (0.55, 0.018, 0.4),
        materials["teal"],
    )


def _lighting(bpy):
    from mathutils import Vector

    scene = bpy.context.scene
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.12,
        0.17,
        0.24,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.45
    for name, position, color, energy in (
        ("Key softbox", (1, -4, 9), (1, 0.88, 0.73), 2400),
        ("Fill softbox", (-5, -1, 5), (0.55, 0.75, 1), 1800),
        ("Rim softbox", (2, 5, 7), (0.7, 1, 1), 2600),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.color, light.shape, light.size = energy, color, "DISK", 5
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (
            (Vector((0, 0, 1)) - obj.location).to_track_quat("-Z", "Y").to_euler()
        )


def _cuda_devices(bpy):
    return _gpu_devices(bpy, "CUDA")


def _gpu_devices(bpy, backend):
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.refresh_devices()
    devices = []
    for device in preferences.devices:
        device.use = device.type == backend
        if device.use:
            devices.append({"name": device.name, "type": device.type})
    if not devices:
        raise RuntimeError(
            f"Cycles {backend} requires a compatible GPU; CPU rendering is disabled"
        )
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    return devices


def _camera_pose(index, views, scene_id):
    if scene_id == "industrial-campus":
        import digital_twin_campus

        return digital_twin_campus._camera(index, views)
    angle = -math.pi / 3 + 2 * math.pi * index / views
    return (
        "Factory cell",
        (14 * math.cos(angle), 14 * math.sin(angle), 11),
        (0, 0, 0.8),
        45,
    )


def _render(bpy, root, views, samples, scene_id):
    from mathutils import Vector

    scene = bpy.context.scene
    scene.cycles.samples, scene.cycles.use_denoising = samples, True
    resolution = (2560, 1440) if scene_id == "industrial-campus" else (1280, 720)
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.object.camera_add(location=(10, -12, 9))
    camera = bpy.context.object
    camera.name, camera.data.lens = "Inspection camera", 45
    scene.camera = camera
    poses = []
    for index in range(views):
        route, position, target, lens = _camera_pose(index, views, scene_id)
        camera.location, camera.data.lens = position, lens
        camera.data.clip_end = 10000
        camera.rotation_euler = (
            (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.render.filepath = str(root / f"frame-{index:03d}.png")
        bpy.ops.render.render(write_still=True)
        poses.append(
            {
                "frame": index,
                "route": route,
                "lens_mm": lens,
                "camera_to_world": [list(row) for row in camera.matrix_world],
            }
        )
    return poses


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument(
        "--scene", choices=("factory-cell", "industrial-campus"), default="factory-cell"
    )
    parser.add_argument("--gpu-backend", choices=("CUDA", "OPTIX"), default="CUDA")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _build_scene(bpy, scene_id):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if scene_id == "industrial-campus":
        import digital_twin_campus

        return digital_twin_campus._build(bpy)
    _factory(bpy)
    _lighting(bpy)
    return {}


def _main():
    import bpy

    sys.path.insert(0, str(Path(__file__).parent))
    args = _arguments()
    devices = _gpu_devices(bpy, args.gpu_backend)
    statistics = _build_scene(bpy, args.scene)
    root = args.output_path
    root.mkdir(parents=True, exist_ok=False)
    poses = _render(bpy, root, args.views, args.samples, args.scene)
    bpy.ops.wm.usd_export(filepath=str(root / "scene.usdc"), export_animation=False)
    bpy.ops.export_scene.gltf(filepath=str(root / "scene.glb"), export_format="GLB")
    receipt = {
        "backend": "Blender Cycles OptiX"
        if args.gpu_backend == "OPTIX"
        else "Blender Cycles CUDA",
        "version": ".".join(str(part) for part in bpy.app.version),
        "devices": devices,
        "cpu_rendering": False,
        "samples": args.samples,
        "scene_id": args.scene,
        "scene_statistics": statistics,
        "resolution": [
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y,
        ],
        "cameras": poses,
    }
    (root / "native-render.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    _main()
