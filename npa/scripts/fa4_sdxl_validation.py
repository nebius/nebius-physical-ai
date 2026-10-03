"""Measure full SDXL image generation with explicit FA4 and stock PyTorch SDPA."""

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import statistics
import time

from fa4_sdxl_attention import FA4SDXLProcessor
import numpy as np
import torch

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
CASES = (
    {
        "name": "robot-workcell",
        "prompt": "A professional photograph of an orange industrial robot arm in a clean "
        "robotics laboratory, holding a small metal component above a workbench, "
        "soft daylight, precise mechanical details, no people, no text",
        "width": 1024,
        "height": 1024,
        "seed": 42,
    },
    {
        "name": "warehouse",
        "prompt": "A wide architectural photograph of an orderly automated warehouse, "
        "small mobile robots between tall shelves of boxes, polished concrete floor, "
        "warm overhead lighting, realistic industrial environment, no people, no text",
        "width": 1536,
        "height": 1024,
        "seed": 123,
    },
    {
        "name": "sensor-rig",
        "prompt": "A detailed studio product photograph of a compact autonomous rover "
        "with cameras and a lidar sensor, four rugged wheels, blue metal body, "
        "neutral grey backdrop, soft rim lighting, no text",
        "width": 1024,
        "height": 1536,
        "seed": 2026,
    },
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    return args


def _load_pipeline(model_path):
    from diffusers import StableDiffusionXLPipeline

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This qualification requires a real SM120 GPU")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        add_watermarker=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _verify_model(model_path):
    manifest_path = Path(__file__).with_name("fa4_sdxl_model.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["model"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("Unexpected model revision in the validation manifest")
    for filename, expected in manifest["files"].items():
        path = model_path / filename
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Pinned model file checksum mismatch: {filename}")
    return manifest


def _select_backend(pipeline, backend):
    from diffusers.models.attention_processor import AttnProcessor2_0

    if backend == "sdpa":
        pipeline.unet.set_attn_processor(AttnProcessor2_0())
        return None
    from flash_attn.cute import flash_attn_func

    processor = FA4SDXLProcessor(flash_attn_func)
    pipeline.unet.set_attn_processor(processor)
    return processor


def _final_latents(steps, captured):
    def capture(pipeline, step, timestep, callback_kwargs):
        if step == steps - 1:
            captured.append(callback_kwargs["latents"].detach().clone())
        return callback_kwargs

    return capture


def _image_metrics(reference, candidate):
    from skimage.metrics import structural_similarity

    reference = np.asarray(reference, dtype=np.float64) / 255
    candidate = np.asarray(candidate, dtype=np.float64) / 255
    error = candidate - reference
    mse = float(np.mean(error**2))
    return {
        "pixel_mae": float(np.mean(np.abs(error))),
        "pixel_psnr_db": float(-10 * np.log10(mse)) if mse else None,
        "pixel_ssim": float(
            structural_similarity(reference, candidate, channel_axis=2, data_range=1)
        ),
    }


def _check_outputs(image, latents, case):
    if image.size != (case["width"], case["height"]):
        raise RuntimeError("Decoded image has the wrong dimensions")
    if not torch.isfinite(latents).all() or np.asarray(image).std() < 1:
        raise RuntimeError("Generation returned non-finite latents or a blank image")


@torch.inference_mode()
def _generate(pipeline, case, backend, steps):
    processor = _select_backend(pipeline, backend)
    captured = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image = pipeline(
        prompt=case["prompt"],
        width=case["width"],
        height=case["height"],
        num_inference_steps=steps,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(case["seed"]),
        callback_on_step_end=_final_latents(steps, captured),
    ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    latents = captured[0].float().cpu()
    _check_outputs(image, latents, case)
    calls = sum(processor.shapes.values()) if processor else 0
    if processor and calls != len(pipeline.unet.attn_processors) * steps:
        raise RuntimeError("FA4 call count does not cover every UNet attention layer")
    measured = {
        "seconds": elapsed,
        "peak_allocated_bytes": peak_memory,
        "fa4_calls": calls,
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
    return measured, image, latents, processor


def _profile_cuda(function, trace_path):
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile,
    ):
        function()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(trace_path))
    kernels = Counter(
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )
    if not kernels:
        raise RuntimeError("Profiler captured no CUDA kernel execution")
    return dict(kernels)


def _profile_attention(processor, output_path, name):
    inputs = processor.profile_inputs

    def fa4():
        return processor.attention_function(
            *inputs, causal=False, pack_gqa=False, num_splits=1
        )

    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            *(tensor.transpose(1, 2) for tensor in inputs)
        )

    return {
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "fa4_kernels": _profile_cuda(fa4, output_path / f"{name}-fa4-trace.json"),
        "sdpa_kernels": _profile_cuda(sdpa, output_path / f"{name}-sdpa-trace.json"),
    }


def _record_generation(pipeline, args, case, backend, repeat):
    measured, image, latents, processor = _generate(pipeline, case, backend, args.steps)
    phase = "warmup" if repeat < 0 else "measured"
    print(
        json.dumps(
            {"case": case["name"], "backend": backend, "phase": phase, **measured}
        ),
        flush=True,
    )
    if repeat == 0:
        filename = f"{case['name']}-{backend}.png"
        image.save(args.output_path / filename)
        measured["image"] = filename
        measured["png_sha256"] = hashlib.sha256(
            (args.output_path / filename).read_bytes()
        ).hexdigest()
    return measured, image, latents, processor


def _measure_case(pipeline, args, case):
    result = {**case, "warmup": {}, "runs": {"fa4": [], "sdpa": []}}
    reference_outputs = {}
    for backend in ("sdpa", "fa4"):
        measured, _, _, _ = _record_generation(pipeline, args, case, backend, -1)
        result["warmup"][backend] = measured
    for repeat in range(args.repeats):
        order = ("fa4", "sdpa") if repeat % 2 == 0 else ("sdpa", "fa4")
        for backend in order:
            measured, image, latents, processor = _record_generation(
                pipeline, args, case, backend, repeat
            )
            result["runs"][backend].append(measured)
            if repeat == 0:
                reference_outputs[backend] = (image, latents)
            if backend == "fa4" and repeat == 0:
                result["shapes"] = [
                    {"qkv": shape, "calls": count}
                    for shape, count in processor.shapes.items()
                ]
                result["profile"] = _profile_attention(
                    processor, args.output_path, case["name"]
                )
            if processor:
                processor.profile_inputs = None
    return _compare_case(result, reference_outputs)


def _compare_case(result, outputs):
    reference, ref_latents = outputs["sdpa"]
    candidate, fa4_latents = outputs["fa4"]
    result["comparison"] = _image_metrics(reference, candidate)
    result["comparison"]["latent_relative_l2"] = float(
        torch.linalg.vector_norm(fa4_latents - ref_latents)
        / torch.linalg.vector_norm(ref_latents)
    )
    medians = {
        backend: statistics.median(run["seconds"] for run in runs)
        for backend, runs in result["runs"].items()
    }
    result["median_seconds"] = medians
    result["sdpa_over_fa4_speed_ratio"] = medians["sdpa"] / medians["fa4"]
    return result


def _environment(pipeline, args, model_manifest):
    packages = (
        "torch",
        "diffusers",
        "transformers",
        "flash-attn-4",
        "nvidia-cutlass-dsl",
        "quack-kernels",
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_manifest": model_manifest,
        "precision": "fp16; Diffusers upcasts VAE decoding to fp32",
        "steps": args.steps,
        "measured_repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "packages": {name: metadata.version(name) for name in packages},
        "unet_parameters": sum(
            parameter.numel() for parameter in pipeline.unet.parameters()
        ),
        "unet_attention_layers": len(pipeline.unet.attn_processors),
        "scheduler": type(pipeline.scheduler).__name__,
        "guidance_scale": 5.0,
        "fa4_scope": "all UNet self/cross attention; text encoders and VAE use stock backends",
        "timing_scope": "full pipeline: text encoding, denoising, VAE decoding and watermark; excludes loading and PNG writes",
        "memory_scope": "total PyTorch peak allocation; FA4 recorder also retains one QKV triple for profiling",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "processor_sha256": hashlib.sha256(
            Path(__file__).with_name("fa4_sdxl_attention.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }


def _run():
    args = _parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    model_manifest = _verify_model(args.model_path)
    pipeline = _load_pipeline(args.model_path)
    report = _environment(pipeline, args, model_manifest)
    for case in CASES:
        report["cases"].append(_measure_case(pipeline, args, case))
        (args.output_path / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print("Completed full SDXL FA4/SDPA qualification", flush=True)


if __name__ == "__main__":
    _run()
