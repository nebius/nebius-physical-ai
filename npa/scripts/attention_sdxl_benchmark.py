"""Compare FA2/FA4 full SDXL generation in separate images with identical processors."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import statistics
import time

from attention_benchmark_backend import attention_backend, benchmark_environment
from fa4_sdxl_attention import FA4SDXLProcessor
from fa4_sdxl_validation import (
    CASES,
    MODEL_ID,
    MODEL_REVISION,
    _check_outputs,
    _final_latents,
    _load_pipeline,
    _verify_model,
)
import numpy as np
import torch


@torch.inference_mode()
def _generate(pipeline, function, case, steps, record_shapes):
    processor = FA4SDXLProcessor(
        function, record_shapes=record_shapes, attention_options={"causal": False}
    )
    pipeline.unet.set_attn_processor(processor)
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
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    latents = captured[0].float().cpu()
    _check_outputs(image, latents, case)
    if record_shapes and sum(processor.shapes.values()) != steps * len(
        pipeline.unet.attn_processors
    ):
        raise RuntimeError(
            "Backend coverage does not include every UNet attention call"
        )
    measured = {
        "seconds": seconds,
        "peak_allocated_bytes": peak,
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
    return measured, image, latents, processor


def _case(pipeline, function, case, args):
    # Coverage and compilation occur in a separate, untimed generation.
    _, _, _, processor = _generate(pipeline, function, case, args.steps, True)
    coverage = [
        {"qkv": shape, "calls": count}
        for shape, count in sorted(processor.shapes.items())
    ]
    del processor
    samples = []
    for repeat in range(args.repeats):
        sample, image, latents, _ = _generate(
            pipeline, function, case, args.steps, False
        )
        samples.append(sample)
        if repeat == 0:
            image.save(args.output_path / f"{case['name']}.png")
            np.save(args.output_path / f"{case['name']}-latents.npy", latents.numpy())
        print(case["name"], repeat, sample["seconds"], flush=True)
    return {
        **case,
        "coverage": coverage,
        "samples": samples,
        "median_seconds": statistics.median(s["seconds"] for s in samples),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fa2", "fa4"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--tile", choices=("64x64", "64x128", "128x64", "128x128"))
    tuning.add_argument("--tuning", choices=("rtx6000-inference",))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.steps, args.repeats) < 1:
        parser.error("steps and repeats must be positive")
    if (args.tile or args.tuning) and args.backend != "fa4":
        parser.error("Inference tuning requires --backend fa4")
    return args


def _environment(backend):
    environment = benchmark_environment(backend)
    environment["packages"].update(
        {
            name: metadata.version(name)
            for name in ("diffusers", "transformers", "invisible-watermark")
        }
    )
    return environment


def main():
    """Measure real SDXL inference after verifying the pinned model bytes.

    Args:
        None; configuration comes from command-line arguments.
    Returns:
        None.
    Raises:
        RuntimeError: Model, backend, GPU or output validation fails.
        OSError: A model file cannot be read or a result cannot be written.
    """
    args = _arguments()
    _verify_model(args.model_path)
    tile = tuple(map(int, args.tile.split("x"))) if args.tile else None
    function = attention_backend(args.backend, tile, args.tuning)
    environment = _environment(args.backend)
    pipeline = _load_pipeline(args.model_path)
    args.output_path.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "tile": tile,
        "tuning": args.tuning,
        "environment": environment,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "steps": args.steps,
        "repeats": args.repeats,
        "instrumented_timing": False,
        "cases": [],
        "status": "incomplete",
    }
    for case in CASES:
        report["cases"].append(_case(pipeline, function, case, args))
        (args.output_path / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    report["status"] = "passed"
    (args.output_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
