"""Ray-traced Newton double-pendulum: sim on cuda:0, rendered with Blender Cycles.

The same real physics workload, now with an RTX-style ray-traced render:
Blender 4.2 LTS is fetched in-pod, the scene (metallic links, studio
lighting, shadow-catching ground) is built from the real simulated
trajectory, and Cycles renders on the GPU. The MP4 travels via the
artifact store. Descriptor-only onboarding, as always.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

BLENDER_URL = (
    "https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz"
)
BLENDER_DIR = "/work/blender-4.2.3-linux-x64"
BLENDER_BIN = os.path.join(BLENDER_DIR, "blender")

SCENE_SCRIPT = r"""
import math

import bpy
import numpy as np

d = np.load("/work/traj.npz")
traj = d["traj"]
p1, p2 = traj[:, 0, :2], traj[:, 1, :2]
j1 = 2 * p1
j2 = 2 * p2 - j1
N = traj.shape[0]

import os
STRIDE = int(os.environ.get("RTX_STRIDE", "6"))
frames = list(range(0, N, STRIDE))

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.video_bitrate = 8000
scene.render.filepath = "/work/pendulum_rtx.mp4"
scene.frame_start = 1
scene.frame_end = len(frames)

prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
cuda_devs = [dev for dev in prefs.devices if dev.type == "CUDA"]
if not cuda_devs:
    raise RuntimeError("no CUDA device available for Cycles")
for dev in cuda_devs:
    dev.use = True
# Blender 4.x: scene.cycles.device is CPU|GPU; the CUDA backend comes from
# the preferences compute device type above.
scene.cycles.device = "GPU"
print("Cycles CUDA devices:", [dev.name for dev in cuda_devs], flush=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

world = scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.015, 0.015, 0.02, 1.0)
bg.inputs["Strength"].default_value = 1.0


def metal(name, color, roughness=0.35):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, -2.3))
ground = bpy.context.active_object
gmat = bpy.data.materials.new(name="ground")
gmat.use_nodes = True
bsdf = gmat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.025, 1.0)
bsdf.inputs["Roughness"].default_value = 0.9
bsdf.inputs["Metallic"].default_value = 0.0
ground.data.materials.append(gmat)


def make_link(name, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    # size=1.0 -> 1m cube; scale z by 1.0 for a true 1m link.
    obj.scale = (0.07, 0.07, 1.0)
    obj.data.materials.append(mat)
    return obj


link1 = make_link("link1", metal("blue_metal", (0.12, 0.35, 0.85)))
link2 = make_link("link2", metal("orange_metal", (0.95, 0.45, 0.10)))


def make_joint(name):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.09, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(metal("joint_metal", (0.15, 0.15, 0.17), 0.25))
    return obj


j0o, j1o, j2o = make_joint("j0"), make_joint("j1"), make_joint("j2")


def area_light(name, loc, power, color=(1, 1, 1), size=3.0):
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.active_object
    light.name = name
    light.data.energy = power
    light.data.color = color
    light.data.size = size
    return light


area_light("key", (3.5, -3.5, 4.0), 2500, (1.0, 0.95, 0.9), 3.0)
area_light("rim", (-3.5, 3.5, 2.5), 1500, (0.5, 0.7, 1.0), 2.0)
area_light("fill", (0, -5, 1.0), 400, (1, 1, 1), 4.0)

bpy.ops.object.camera_add(location=(3.2, -4.6, 1.6))
cam = bpy.context.active_object
scene.camera = cam
bpy.ops.object.empty_add(location=(0, 0, -0.9))
target = bpy.context.active_object
con = cam.constraints.new(type="TRACK_TO")
con.target = target
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"

for fi, f in enumerate(frames, start=1):
    b1x, b1z = float(j1[f, 0]), float(j1[f, 1])
    b2x, b2z = float(j2[f, 0]), float(j2[f, 1])
    link1.location = (b1x / 2, 0.0, b1z / 2)
    link1.rotation_euler = (0.0, math.atan2(b1x, b1z), 0.0)
    link2.location = ((b1x + b2x) / 2, 0.0, (b1z + b2z) / 2)
    link2.rotation_euler = (0.0, math.atan2(b2x - b1x, b2z - b1z), 0.0)
    j0o.location = (0, 0, 0)
    j1o.location = (b1x, 0, b1z)
    j2o.location = (b2x, 0, b2z)
    for obj in (link1, link2, j0o, j1o, j2o):
        obj.keyframe_insert(data_path="location", frame=fi)
    for obj in (link1, link2):
        obj.keyframe_insert(data_path="rotation_euler", frame=fi)

bpy.ops.render.render(animation=True)
print("RENDER_DONE", flush=True)
"""


def fetch_blender() -> None:
    if os.path.exists(BLENDER_BIN):
        print("blender already present", flush=True)
        return
    # Preferred: tarball staged via the descriptor's s3 inputs (pod egress
    # to the public internet is restricted; S3 is always reachable).
    tarball = "/work/blender.tar.xz"
    if not os.path.exists(tarball):
        print(f"downloading blender from {BLENDER_URL}", flush=True)
        urllib.request.urlretrieve(BLENDER_URL, tarball)
    print("extracting blender", flush=True)
    with tarfile.open(tarball) as tf:
        tf.extractall("/work")
    try:
        os.remove(tarball)
    except OSError:
        pass
    assert os.path.exists(BLENDER_BIN), "blender binary missing after extract"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    # Multi-file payload: sibling modules live next to this script under the
    # ConfigMap mount (keys preserve basenames).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("/work", exist_ok=True)
    from pendulum_sim import run_sim

    import numpy as np

    dt = 1.0 / 60.0
    sim = run_sim(args.steps, dt)
    traj = sim["traj"]
    np.savez("/work/traj.npz", traj=traj)
    print(f"sim done: {traj.shape[0]} frames in {sim['elapsed_s']:.1f}s", flush=True)

    fetch_blender()
    with open("/work/render_scene.py", "w") as f:
        f.write(SCENE_SCRIPT)

    env = dict(os.environ, RTX_STRIDE=str(args.stride))
    print("rendering with Blender Cycles on CUDA", flush=True)
    r = subprocess.run([BLENDER_BIN, "-b", "-P", "/work/render_scene.py"], env=env)
    if r.returncode != 0:
        raise RuntimeError(f"blender render failed: rc={r.returncode}")
    if not os.path.exists("/work/pendulum_rtx.mp4"):
        raise RuntimeError("blender produced no mp4")
    os.replace("/work/pendulum_rtx.mp4", args.out_mp4)
    mp4_bytes = os.path.getsize(args.out_mp4)

    import torch

    summary = {
        "device_ok": torch.cuda.is_available(),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        ),
        "renderer": "cycles-cuda",
        "blender": "4.2.3",
        "steps": args.steps,
        "stride": args.stride,
        "frames": int(traj.shape[0]),
        "sim_elapsed_s": round(sim["elapsed_s"], 3),
        "mp4_bytes": mp4_bytes,
        "newton": sim["newton"],
        "traj_checksum": float(traj.sum(dtype=np.float64)),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
