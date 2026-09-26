"""Verify all LIBERO-10 initial states and render both cameras for every task."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from evaluate import _check_revision
from simulation_client import _numpy_state_globals


def _setup(root):
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LP_NUM_THREADS="1",
        LIBERO_CONFIG_PATH=str(root / "simulation/libero-config"),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
    )
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    for directory, key in (("framework", "framework"), ("simulation/LIBERO", "libero")):
        _check_revision(root / directory, sources[key])
        sys.path.insert(0, str(root / directory))
    return sources


def _images(evaluation, task, initial, task_id, output):
    import numpy as np
    from PIL import Image

    env, _ = evaluation._get_libero_env(
        task, resolution=256, seed=42, render_gpu_device_id=-1
    )
    try:
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs, _, _, _ = env.step(evaluation._get_libero_dummy_action())
        images = evaluation._get_libero_images(
            obs, ["agentview", "wrist"], flip_images=False, rotate_180=True
        )
        result = {}
        for camera, frame in zip(("front", "wrist"), images, strict=True):
            pixels = np.asarray(frame)
            if pixels.shape != (256, 256, 3) or pixels.max() == pixels.min():
                raise ValueError("simulator produced an invalid camera frame")
            path = output / f"task-{task_id}-{camera}.png"
            Image.fromarray(pixels).save(path)
            result[camera] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            }
        return result
    finally:
        env.close()


def _task(evaluation, suite, task_id, output):
    import numpy as np

    task = suite.get_task(task_id)
    states = np.asarray(suite.get_task_init_states(task_id))[:50]
    if len(states) != 50 or not np.isfinite(states).all():
        raise ValueError("task must provide fifty finite evaluation states")
    return {
        "task_id": task_id,
        "task_description": str(task.language),
        "shape": list(states.shape),
        "dtype": str(states.dtype),
        "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
        "camera_frames": _images(evaluation, task, states[0], task_id, output),
    }


def _main(args):
    sources = _setup(args.shared_root)
    import torch
    from cosmos_framework.simulation.libero import closed_loop_eval as evaluation

    if torch.__version__ != "2.14.0+cpu" or torch.version.cuda is not None:
        raise ValueError("run the probe in the pinned CPU simulation environment")
    args.output_path.mkdir(parents=True, exist_ok=False, mode=0o700)
    evaluation._import_libero()
    suite = evaluation.benchmark.get_benchmark_dict()["libero_10"]()
    with torch.serialization.safe_globals(_numpy_state_globals()):
        tasks = [
            _task(evaluation, suite, index, args.output_path) for index in range(10)
        ]
    packages = (
        "torch",
        "numpy",
        "mujoco",
        "robosuite",
        "pillow",
        "opencv-python",
        "requests",
    )
    result = {
        "schema": "npa.cosmos3.wam-simulation-readiness.v1",
        "scope": "CPU simulator inputs and rendering; no model policy executed",
        "sources": sources,
        "runtime": {name: importlib.metadata.version(name) for name in packages},
        "requirements_sha256": hashlib.sha256(
            Path(__file__).with_name("simulation-requirements.txt").read_bytes()
        ).hexdigest(),
        "seed": 42,
        "warmup_steps": 10,
        "max_simulation_steps": evaluation.TASK_MAX_STEPS["libero_10"],
        "mujoco_gl": "osmesa",
        "tasks": tasks,
    }
    (args.output_path / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified_initial_states": 500, "rendered_camera_frames": 20}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    os.umask(0o077)
    _main(parser.parse_args())
