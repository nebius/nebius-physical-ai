"""Standalone NVIDIA Cosmos functional smoke checks.

This script loads a Cosmos text-to-world model, runs one text prompt, and
confirms that an output artifact is created. It is intended for GPU VMs.

Run with:
    python -m npa.smoke.test_cosmos_functional
"""

from __future__ import annotations

import atexit
import importlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from npa.smoke._versions import supported_tool_version

DEFAULT_MODEL = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
DEFAULT_PROMPT = "A robot arm gently places a red cube on a table in a clean lab."
DEFAULT_MODEL_DIR = Path("/opt/cosmos/models")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeState:
    root: Path
    output_path: Path
    model_id: str
    model_source: str = ""
    pipe: Any | None = None


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _model_slug(model: str) -> str:
    return model.replace("/", "--").replace(":", "--")


def _resolve_model_source(model: str) -> str:
    model_dir = Path(os.environ.get("COSMOS_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    candidate = model_dir / _model_slug(model)
    return str(candidate) if candidate.exists() else model


def check_cosmos_version(state: SmokeState) -> CheckResult:
    try:
        expected = supported_tool_version("cosmos", __file__)
        importlib.import_module("cosmos_predict2")
        version = metadata.version("cosmos-predict2")
    except Exception as exc:
        return CheckResult("check cosmos version", False, _format_exception(exc))

    if version != expected:
        return CheckResult(
            "check cosmos version",
            False,
            f"expected version: {expected}; found: {version}",
        )
    return CheckResult("check cosmos version", True, f"version: {version}")


def check_load_model(state: SmokeState) -> CheckResult:
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        pipeline_cls = getattr(diffusers, "CosmosTextToWorldPipeline", None)
        if pipeline_cls is None:
            pipeline_cls = getattr(diffusers, "DiffusionPipeline")

        state.model_source = _resolve_model_source(state.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline_cls.from_pretrained(state.model_source, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe.to("cuda")
        state.pipe = pipe
        return CheckResult("load cosmos model", True, f"source: {state.model_source}")
    except Exception as exc:
        return CheckResult("load cosmos model", False, _format_exception(exc))


def _save_result(result: Any, output_path: Path) -> None:
    frames = getattr(result, "frames", None)
    images = getattr(result, "images", None)
    if frames:
        export_to_video = importlib.import_module("diffusers.utils").export_to_video
        export_to_video(frames[0], str(output_path), fps=30)
        return
    if images:
        image_path = output_path.with_suffix(".png")
        images[0].save(image_path)
        output_path.write_text(str(image_path))
        return
    output_path.write_text(str(result))


def check_single_inference(state: SmokeState) -> CheckResult:
    if state.pipe is None:
        return CheckResult("run single cosmos inference", False, "skipped because model load failed")

    prompt = os.environ.get("COSMOS_SMOKE_PROMPT", DEFAULT_PROMPT)
    try:
        kwargs: dict[str, Any] = {"prompt": prompt}
        steps = os.environ.get("COSMOS_SMOKE_STEPS")
        if steps:
            kwargs["num_inference_steps"] = int(steps)
        result = state.pipe(**kwargs)
        _save_result(result, state.output_path)
    except Exception as exc:
        return CheckResult("run single cosmos inference", False, _format_exception(exc))

    if not state.output_path.exists() or state.output_path.stat().st_size == 0:
        return CheckResult("run single cosmos inference", False, f"missing output: {state.output_path}")
    return CheckResult("run single cosmos inference", True, f"output: {state.output_path}")


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="npa_cosmos_functional_"))
    atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
    model_id = os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL)
    state = SmokeState(root=root, output_path=root / "cosmos_output.mp4", model_id=model_id)

    print(f"Temporary workspace: {root}")
    checks: list[Callable[[SmokeState], CheckResult]] = [
        check_cosmos_version,
        check_load_model,
        check_single_inference,
    ]
    results: list[CheckResult] = []

    try:
        for check in checks:
            result = check(state)
            results.append(result)
            _print_result(result)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
