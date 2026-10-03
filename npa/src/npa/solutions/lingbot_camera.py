"""Native LingBot World v1 camera-conditioned BYOF capability.

The authored camera path is not measured calibration or robot action input.
This preserves the pinned upstream FSDP/Ulysses generation implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from npa.solutions.video_generation import validate_video


SOURCE_REPO = "https://github.com/Robbyant/lingbot-world.git"
SOURCE_REF = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
MODEL_ID = "robbyant/lingbot-world-base-cam"
MODEL_REF = "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd"
TEXT_ENCODER_REF = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
FRAME_COUNT = 161
CONTROL_FILENAMES = ("poses.npy", "intrinsics.npy")


def _checkpoint(cache: Path) -> Path:
    from huggingface_hub import snapshot_download

    model = Path(snapshot_download(MODEL_ID, revision=MODEL_REF))
    tokenizer = Path(
        snapshot_download(
            "google/umt5-xxl",
            revision=TEXT_ENCODER_REF,
            allow_patterns=["*token*", "*.json", "*.model"],
        )
    )
    # Upstream selects camera conditioning from the checkpoint directory name.
    overlay = cache / "lingbot-world-base-cam"
    overlay.mkdir(parents=True, exist_ok=False)
    for item in model.iterdir():
        if item.name != "google":
            (overlay / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (overlay / "google").mkdir()
    (overlay / "google/umt5-xxl").symlink_to(tokenizer, target_is_directory=True)
    return overlay


def create_camera_controls(
    directory: Path,
    *,
    translation_x: float,
    translation_z: float,
    yaw_radians: float,
    frame_count: int = FRAME_COUNT,
) -> dict[str, str]:
    """Create an upstream-compatible authored camera trajectory.

    Args:
        directory: New directory that will contain ``poses.npy`` and
            ``intrinsics.npy``.
        translation_x: Final OpenCV-camera X translation in metres.
        translation_z: Final OpenCV-camera Z translation in metres.
        yaw_radians: Final yaw rotation in radians.
        frame_count: Number of camera control records to emit.

    Returns:
        SHA-256 identities keyed by the emitted control-file names.

    Raises:
        ValueError: A trajectory parameter is nonfinite or frame count is invalid.
        FileExistsError: The output directory already exists.
    """
    import numpy as np

    values = (translation_x, translation_z, yaw_radians)
    if frame_count <= 1 or not all(np.isfinite(value) for value in values):
        raise ValueError("Camera trajectory must have finite values and two frames")
    directory.mkdir()
    time = np.linspace(0, 1, frame_count, dtype=np.float32)
    smooth = time * time * (3 - 2 * time)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    angles = yaw_radians * smooth
    poses[:, 0, 0], poses[:, 0, 2] = np.cos(angles), np.sin(angles)
    poses[:, 2, 0], poses[:, 2, 2] = -np.sin(angles), np.cos(angles)
    poses[:, 0, 3], poses[:, 2, 3] = translation_x * smooth, translation_z * smooth
    intrinsics = np.repeat(
        np.array([[900, 900, 640, 360]], dtype=np.float32), frame_count, axis=0
    )
    np.save(directory / "poses.npy", poses)
    np.save(directory / "intrinsics.npy", intrinsics)
    return _control_hashes(directory)


def _controls(directory: Path) -> dict[str, str]:
    """Create the historical default authored camera trajectory."""

    return create_camera_controls(
        directory,
        translation_x=0.7,
        translation_z=1.5,
        yaw_radians=0.1,
    )


def _control_hashes(directory: Path) -> dict[str, str]:
    """Validate control files and return their byte identities."""

    _validate_controls(directory)
    return {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in CONTROL_FILENAMES
    }


def _validate_controls(directory: Path) -> None:
    """Reject camera controls that do not match the pinned native contract."""

    import numpy as np

    poses = np.load(directory / "poses.npy", allow_pickle=False)
    intrinsics = np.load(directory / "intrinsics.npy", allow_pickle=False)
    if poses.shape != (FRAME_COUNT, 4, 4) or intrinsics.shape != (FRAME_COUNT, 4):
        raise ValueError(
            "LingBot camera controls must match the 161-frame native shape"
        )
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("LingBot camera controls must contain finite values")


def _prepare_controls(output: Path, controls_dir: Path | None) -> dict[str, str]:
    """Create or copy the control files consumed by the native entrypoint."""

    target = output / "controls"
    if controls_dir is None:
        return _controls(target)
    controls_dir = Path(controls_dir)
    _validate_controls(controls_dir)
    target.mkdir()
    for filename in CONTROL_FILENAMES:
        shutil.copy2(controls_dir / filename, target / filename)
    return _control_hashes(target)


def _command(checkpoint, image, output, prompt, seed, degree):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={degree}",
        str(Path(__file__)),
        "--task",
        "i2v-A14B",
        "--size",
        "720*1280",
        "--ckpt_dir",
        str(checkpoint),
        "--image",
        str(image),
        "--action_path",
        str(output / "controls"),
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size",
        str(degree),
        "--frame_num",
        "161",
        "--sample_steps",
        "70",
        "--base_seed",
        str(seed),
        "--prompt",
        prompt,
        "--save_file",
        str(output / "video.mp4"),
    ]


def generate_camera_video(
    image: Path,
    prompt: str,
    seed: int,
    output: Path,
    degree: int = 4,
    controls_dir: Path | None = None,
) -> dict:
    """Generate a world continuation with the native authored-camera path.

    Args:
        image: Local input image whose exact bytes are retained with the output.
        prompt: Native generation prompt.
        seed: Nonnegative generation seed.
        output: New writable output directory.
        degree: Two or four GPUs for FSDP and Ulysses.
        controls_dir: Optional upstream-compatible authored camera controls.

    Returns:
        Model, camera, rank execution, and decoded-media evidence.

    Raises:
        ValueError: Invalid inputs or GPU topology.
        RuntimeError: Native inference or artifact validation fails.
    """
    import torch
    from PIL import Image

    if degree not in (2, 4) or torch.cuda.device_count() != degree:
        raise ValueError(
            "LingBot camera capability requires exactly two or four CUDA GPUs"
        )
    if not prompt.strip() or type(seed) is not int or seed < 0:
        raise ValueError("A nonempty prompt and nonnegative integer seed are required")
    with Image.open(image) as decoded:
        decoded.verify()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "input.png"
    with Image.open(image) as decoded:
        decoded.convert("RGB").save(source)
    controls = _prepare_controls(output, controls_dir)
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-camera-") as cache:
        checkpoint = _checkpoint(Path(cache))
        command = _command(checkpoint, source, output, prompt, seed, degree)
        _run(command, output)
    ranks = [
        json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(degree)
    ]
    if any(
        row["calls"]["all_to_all"] < 1 or row["calls"]["attention"] < 1 for row in ranks
    ):
        raise RuntimeError(
            "Every distributed rank must exercise attention and all-to-all"
        )
    return _evidence(image, source, prompt, seed, output, controls, ranks)


def _run(command, output):
    # FSDP and Ulysses issue collectives on different streams. Preserve host
    # launch order across NCCL communicators to prevent collective deadlocks.
    environment = dict(
        os.environ,
        LINGBOT_OUTPUT=str(output),
        NCCL_CUMEM_ENABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_NVLS_ENABLE="0",
        NCCL_IB_DISABLE="1",
        NCCL_SOCKET_IFNAME="=eth0",
        TORCH_NCCL_USE_COMM_NONBLOCKING="1",
        NCCL_LAUNCH_ORDER_IMPLICIT="1",
    )
    with (output / "native-generation.log").open("w") as log:
        subprocess.run(
            command,
            cwd="/opt/byof",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _evidence(original, image, prompt, seed, output, controls, ranks):
    return {
        "schema": "npa.workbench.byof.lingbot_camera.v1",
        "solution": "lingbot-world",
        "capability": "lingbot_world_camera_conditioned_video",
        "upstream_repo": SOURCE_REPO,
        "upstream_ref": SOURCE_REF,
        "model_id": MODEL_ID,
        "model_ref": MODEL_REF,
        "tokenizer_ref": TEXT_ENCODER_REF,
        "input_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "normalized_input_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "prompt": prompt,
        "seed": seed,
        "controls": controls,
        "ranks": ranks,
        "conditioning": "Authored camera trajectory and approximate intrinsics",
        "collective_launch_order": "NCCL_LAUNCH_ORDER_IMPLICIT=1",
        "weights_baked": False,
        "observed": validate_video(output / "video.mp4", 161),
        "capabilities_exercised": [
            "lingbot_world_camera_conditioned_video",
            "distributed_rank_validation",
            "decoded_mp4_validation",
        ],
        "deferred": [],
        "not_claimed": ["robot_action_conditioning", "training", "real_time"],
    }


def _rank():
    import runpy
    import torch

    sys.path.insert(0, "/opt/byof")
    import wan.modules.attention as attention
    import wan.modules.model as model
    import wan.distributed.ulysses as ulysses

    calls = {"attention": 0, "all_to_all": 0}
    native_attention, native_all_to_all = attention.attention, ulysses.all_to_all

    def sdpa(*args, **kwargs):
        calls["attention"] += 1
        return native_attention(*args, **kwargs)

    def all_to_all(*args, **kwargs):
        calls["all_to_all"] += 1
        return native_all_to_all(*args, **kwargs)

    model.flash_attention = ulysses.flash_attention = sdpa
    ulysses.all_to_all = all_to_all
    runpy.run_path("/opt/byof/generate.py", run_name="__main__")
    rank = int(os.environ["RANK"])
    evidence = dict(
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        calls=calls,
        device=torch.cuda.get_device_name(rank),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        attention_backend="upstream PyTorch SDPA fallback",
    )
    (Path(os.environ["LINGBOT_OUTPUT"]) / f"rank-{rank}.json").write_text(
        json.dumps(evidence)
    )


if __name__ == "__main__":
    _rank()
