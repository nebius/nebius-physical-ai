"""Render measured Go1 articulations with Cycles GPU materials and mesh shadows."""

import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


def _transform(position, orientation):
    rotation = Quaternion((orientation[3], *orientation[:3])).to_matrix().to_4x4()
    rotation.translation = Vector(position)
    return rotation


def _origin(visual):
    position = [float(v) for v in visual["position"].split()]
    angles = [float(v) for v in visual["angles"].split()]
    result = Euler(angles, "XYZ").to_matrix().to_4x4()
    result.translation = Vector(position)
    return result


def _material(name, color, metallic=0.15, roughness=0.35):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    return material


def _mesh_objects(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    bpy.context.view_layer.update()
    imported = set(bpy.data.objects) - before
    correction = Matrix.Rotation(-math.pi / 2, 4, "X")
    result = []
    for obj in imported:
        if obj.type == "MESH":
            transform = correction @ obj.matrix_world
            obj.parent = None
            obj.matrix_world = transform
            result.append((obj, transform))
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    return result


def _primitive(geometry):
    if geometry["kind"] == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=float(geometry["radius"])
        )
    elif geometry["kind"] == "box":
        bpy.ops.mesh.primitive_cube_add(size=1)
        bpy.context.object.scale = [float(v) for v in geometry["size"].split()]
    elif geometry["kind"] == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=float(geometry["radius"]),
            depth=float(geometry["length"]),
        )
    else:
        raise ValueError("Unsupported robot visual geometry")
    # Blender defers scale evaluation; capture the actual URDF primitive size.
    bpy.context.view_layer.update()
    return [(bpy.context.object, bpy.context.object.matrix_world.copy())]


def _link_material(name, body, dark, rubber):
    if "foot" in name.lower():
        return rubber
    if "trunk" in name.lower() or "thigh" in name.lower():
        return body
    return dark


def _robot(root):
    visuals = json.loads((root / "render-robot.json").read_text())
    body = _material("anodized-silver", (0.52, 0.55, 0.57), 0.4, 0.3)
    dark = _material("motor-housing", (0.028, 0.035, 0.041), 0.2, 0.28)
    rubber = _material("rubber-feet", (0.012, 0.015, 0.019), 0.0, 0.65)
    parts = []
    for visual in visuals:
        name, geometry = visual["link"], visual["geometry"]
        objects = (
            _mesh_objects((root / geometry["filename"]).with_suffix(".glb"))
            if geometry["kind"] == "mesh"
            else _primitive(geometry)
        )
        scale = (
            [float(v) for v in geometry.get("scale", "1 1 1").split()]
            if geometry["kind"] == "mesh"
            else [1, 1, 1]
        )
        material = _link_material(name, body, dark, rubber)
        for obj, transform in objects:
            obj.data.materials.clear()
            obj.data.materials.append(material)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            parts.append(
                (name, obj, _origin(visual) @ Matrix.Diagonal((*scale, 1)) @ transform)
            )
    return parts


def _scene(root, settings):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    gpu = [device for device in preferences.devices if device.type == "CUDA"]
    if not gpu:
        raise RuntimeError(
            "High fidelity capture requires a CUDA GPU; CPU fallback is disabled"
        )
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.samples = settings["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = False
    scene.cycles.use_adaptive_sampling = True
    scene.render.resolution_x, scene.render.resolution_y = (
        settings["width"],
        settings["height"],
    )
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.render.use_persistent_data = True
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.7,
        0.77,
        0.85,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    return scene, [device.name for device in gpu]


def _warehouse(root):
    geometry = json.loads((root / "render-warehouse.json").read_text())
    mesh = bpy.data.meshes.new("original-marble-collider")
    mesh.from_pydata(geometry["vertices"], [], geometry["faces"])
    mesh.update()
    obj = bpy.data.objects.new("warehouse-shadow-catcher", mesh)
    bpy.context.collection.objects.link(obj)
    obj.is_shadow_catcher = True
    obj.data.materials.append(_material("shadow-surface", (0.45, 0.45, 0.45), 0, 0.8))


def _light(name, position, power, size, target):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy, data.shape, data.size = power, "DISK", size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )
    return obj


def _camera(scene, settings):
    data = bpy.data.cameras.new("inspection-camera")
    obj = bpy.data.objects.new("inspection-camera", data)
    bpy.context.collection.objects.link(obj)
    data.type, data.sensor_fit = "PERSP", "HORIZONTAL"
    data.angle = math.radians(settings["field_of_view"])
    data.clip_start, data.clip_end = 0.03, 1000
    scene.camera = obj
    return obj


def _frame(scene, camera, parts, row, destination):
    for name, obj, local in parts:
        obj.matrix_world = _transform(*row["links"][name]) @ local
    # Marble Y-up -> Bullet Z-up; OpenCV camera -> Blender camera axes.
    world_to_bullet = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    camera.matrix_world = (
        world_to_bullet
        @ Matrix(row["observer_camera_to_world"])
        @ Matrix.Diagonal((1, -1, -1, 1))
    )
    scene.render.filepath = str(destination)
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    return time.perf_counter() - started


def main():
    """Render the exact saved articulation sequence on an explicit CUDA device.

    Args: Collection root passed after Blender's argument separator.
    Returns: None; writes RGBA frames and renderer evidence.
    Raises: RuntimeError when no supported GPU or required artifact is available.
    """
    root = Path(sys.argv[sys.argv.index("--") + 1])
    start, stop = map(int, sys.argv[sys.argv.index("--") + 2 :])
    settings = json.loads((root / "render-settings.json").read_text())
    records = json.loads((root / "trajectory.json").read_text())
    scene, devices = _scene(root, settings)
    parts = _robot(root)
    _warehouse(root)
    camera = _camera(scene, settings)
    center = records[len(records) // 2]["position_bullet"]
    _light("overhead-softbox", [center[0] - 1, center[1], 3.5], 700, 5, center)
    _light("aisle-fill", [center[0] + 2, center[1] + 4, 3], 500, 4, center)
    output = root / "actors"
    output.mkdir(exist_ok=True)
    timings = [
        _frame(scene, camera, parts, row, output / f"{i:04d}.png")
        for i, row in enumerate(records[start:stop], start)
    ]
    evidence = {
        "engine": "Blender Cycles",
        "version": bpy.app.version_string,
        "device_type": "CUDA",
        "devices": devices,
        "cpu_render_fallback": False,
        "denoiser": "OpenImageDenoise",
        "denoising_device": "CPU",
        "samples": settings["samples"],
        "frame_start": start,
        "frame_stop": stop,
        "view_transform": scene.view_settings.view_transform,
        "frame_wall_seconds": timings,
        "lighting": "authored industrial area lights; original Marble mesh shadow catcher",
    }
    (root / f"actor-render-{start:04d}.json").write_text(json.dumps(evidence))


if __name__ == "__main__":
    main()
