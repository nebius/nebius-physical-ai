"""Pinned Diffusers video capabilities for operator-built BYOF images.

Heavy dependencies are imported only inside the GPU worker. Model weights are
fetched at runtime; the result records actual decoded media and CUDA hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import time


SOURCE_REPO = "https://github.com/huggingface/diffusers.git"
SOURCE_REF = "275869dcae4ebcfee6a80253fdabc56033335020"


@dataclass(frozen=True)
class VideoModel:
    """One reviewed native pipeline and its immutable public checkpoint."""

    model_id: str
    revision: str
    pipeline: str
    dtype: str
    frames: int
    fps: int
    steps: int
    width: int
    height: int
    guidance: float


MODELS = {
    "mochi-1": VideoModel(
        "genmo/mochi-1-preview", "14be5fcea23095ed330cb214647916a451e38b6e",
        "MochiPipeline", "bfloat16", 163, 30, 64, 848, 480, 4.5,
    ),
    "cogvideox-2b": VideoModel(
        "zai-org/CogVideoX-2b", "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
        "CogVideoXPipeline", "float16", 49, 8, 50, 720, 480, 6.0,
    ),
    "wan2.1-14b": VideoModel(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        "WanPipeline", "bfloat16", 81, 16, 50, 1280, 720, 5.0,
    ),
}


def validate_request(solution: str, prompt: str, seed: int) -> VideoModel:
    """Validate controls before any CUDA or model acquisition.

    Args:
        solution: Reviewed model key.
        prompt: Nonempty text passed directly to the native pipeline.
        seed: Nonnegative integer generator seed.

    Returns:
        The immutable model configuration.

    Raises:
        ValueError: A model or generation control is invalid.
    """
    if solution not in MODELS:
        raise ValueError(f"Unknown video solution: {solution}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Video prompt must be nonempty text")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Video seed must be an integer in [0, 2**63)")
    return MODELS[solution]


def _pipeline(model: VideoModel):
    import diffusers
    import torch

    options = {"revision": model.revision, "torch_dtype": getattr(torch, model.dtype)}
    if model.pipeline == "WanPipeline":
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

        options["vae"] = AutoencoderKLWan.from_pretrained(
            model.model_id, subfolder="vae", revision=model.revision,
            torch_dtype=torch.float32,
        )
    pipeline = getattr(diffusers, model.pipeline).from_pretrained(model.model_id, **options)
    if model.pipeline == "WanPipeline":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, flow_shift=5.0,
        )
    pipeline.vae.enable_tiling()
    return pipeline.to("cuda")


def cuda_inventory() -> dict:
    """Return actual CUDA device evidence, failing when no GPU is available.

    Returns:
        Torch, CUDA, device names, compute capabilities and memory sizes.

    Raises:
        RuntimeError: CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This capability requires a CUDA GPU")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index, "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
        })
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}


def _decoded_frames(video: Path):
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def validate_video(video: Path, expected_frames: int) -> dict:
    """Decode every frame and reject incomplete, blank, or stationary media.

    Args:
        video: Generated MP4 path.
        expected_frames: Exact requested frame count.

    Returns:
        Full-file hash and observed pixel/frame measurements.

    Raises:
        RuntimeError: The media is incomplete, corrupt, blank, or stationary.
    """
    import numpy as np

    count, spatial, delta, previous, shape = 0, 0.0, 0.0, None, None
    for frame in _decoded_frames(video):
        if shape is not None and frame.shape != shape:
            raise RuntimeError("Generated video changes dimensions")
        shape = frame.shape
        pixels = frame[::8, ::8].astype(np.float32)
        spatial = max(spatial, float(pixels.std()))
        if previous is not None:
            delta += float(np.abs(pixels - previous).mean())
        previous = pixels
        count += 1
    if count != expected_frames or spatial < 1 or delta <= 0.001:
        raise RuntimeError(f"Invalid generated video: frames={count}, std={spatial}, delta={delta}")
    return {
        "frame_count": count, "height": shape[0], "width": shape[1],
        "max_spatial_std": spatial, "mean_temporal_delta": delta / max(1, count - 1),
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "size_bytes": video.stat().st_size,
    }


def generate_video(solution: str, prompt: str, seed: int, output_dir: Path,
                   negative_prompt: str = "") -> dict:
    """Execute native text-to-video inference and validate its resulting MP4.

    Args:
        solution: One of the reviewed MODELS keys.
        prompt: Text describing the requested video.
        seed: Reproducible generator seed.
        output_dir: Writable, run-scoped artifact directory.
        negative_prompt: Optional text passed to native negative conditioning.

    Returns:
        Capability evidence including model identity and decoded output.

    Raises:
        ValueError: Invalid generation controls.
        RuntimeError: GPU execution or decoded-media validation fails.
    """
    model = validate_request(solution, prompt, seed)
    if not isinstance(negative_prompt, str):
        raise ValueError("Negative prompt must be text")
    import torch
    from diffusers.utils import export_to_video

    runtime = cuda_inventory()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _pipeline(model)
    options = dict(
        prompt=prompt, num_frames=model.frames, num_inference_steps=model.steps,
        height=model.height, width=model.width, guidance_scale=model.guidance,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    if negative_prompt:
        options["negative_prompt"] = negative_prompt
    started = time.monotonic()
    with torch.inference_mode():
        frames = pipeline(**options).frames[0]
    video = output_dir / "video.mp4"
    export_to_video(frames, str(video), fps=model.fps)
    observed = validate_video(video, model.frames)
    evidence = _evidence(solution, model, prompt, seed, runtime, observed, time.monotonic() - started)
    evidence["negative_prompt"] = negative_prompt
    return evidence


def _evidence(solution, model, prompt, seed, runtime, observed, elapsed):
    if (observed["width"], observed["height"]) != (model.width, model.height):
        raise RuntimeError("Generated dimensions do not match the requested native capability")
    capability = f"{solution}_text_to_video"
    return {
        "schema": "npa.workbench.byof.video_generation.v1",
        "solution": solution, "capability": capability,
        "upstream_repo": SOURCE_REPO, "upstream_ref": SOURCE_REF,
        "requested": asdict(model), "prompt": prompt, "seed": seed,
        "weights_baked": False, "output_filename": "video.mp4",
        "observed": observed, "runtime": runtime, "elapsed_seconds": elapsed,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("diffusers", "transformers", "torch", "huggingface-hub")
        },
        "capabilities_exercised": [capability, "decoded_mp4_validation"],
        "deferred": [],
        "not_claimed": ["action_conditioning", "training", "real_time_serving"],
    }
