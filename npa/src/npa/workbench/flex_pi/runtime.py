"""Shared real flex-pi policy inference and artifact validation.

The public image contains the pinned MIT source and CUDA runtime, but no model
weights, public sample media, populated cache, or credentials. The CLI, SDK,
HTTP service, golden evaluation, and workflow toolRef all call this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
from urllib.parse import urlparse

from npa.clients.storage import StorageClient

DEFAULT_CHECKPOINT_ID = "flex-pi/flexpi-robotwin"
DEFAULT_CHECKPOINT_REVISION = "87d3833ea3bd89c4922945631db81b346e780785"
DEFAULT_DATASET_ID = "flex-pi/robotwin_3d"
DEFAULT_DATASET_REVISION = "bee164afe94041d8c3d7dd1203725c41163fc3f4"
DEFAULT_INPUT_MANIFEST = "/opt/flex-pi/npa/public_robotwin_sample.json"
DEFAULT_SOURCE_REVISION = "20c1b2b71ea35a415d5d47c39b04443cfadad7a1"
ARTIFACT_SCHEMA = "npa.workbench.flex_pi.inference.v1"
REAL_INFERENCE_MARKER = "FLEX_PI_REAL_INFERENCE_PASSED"


class FlexPiError(RuntimeError):
    """Raised when flex-pi request validation or real inference fails."""


@dataclass(frozen=True)
class FlexPiRequest:
    """A reproducible action-only policy inference request."""

    input_path: str
    output_path: str
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    checkpoint_revision: str = DEFAULT_CHECKPOINT_REVISION
    num_inference_steps: int = 4
    seed: int = 42
    torch_compile: bool = False
    expected_gpu: str = ""
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request: FlexPiRequest) -> None:
    if not request.input_path.strip() or not request.output_path.strip():
        raise FlexPiError("input_path and output_path are required")
    if request.num_inference_steps < 1:
        raise FlexPiError("num_inference_steps must be positive")
    if (
        request.checkpoint_id == DEFAULT_CHECKPOINT_ID
        and request.checkpoint_revision != DEFAULT_CHECKPOINT_REVISION
    ):
        raise FlexPiError(
            "the default checkpoint must use the repository-pinned revision"
        )
    for label, value in (
        ("input_path", request.input_path),
        ("output_path", request.output_path),
    ):
        parsed = urlparse(value)
        if parsed.scheme and parsed.scheme != "s3":
            raise FlexPiError(f"{label} must be a local path or s3:// URI")
        if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
            raise FlexPiError(f"{label} requires an S3 bucket and object/prefix")


def build_inference_argv(
    request: FlexPiRequest, *, input_path: Path, output_path: Path
) -> list[str]:
    """Build the pinned upstream adapter invocation.

    Args:
        request: Validated inference request.
        input_path: Local immutable input-manifest snapshot.
        output_path: Local action artifact destination.
    Returns:
        Argument vector for the real flex-pi runtime.
    """
    argv = [
        os.environ.get("FLEX_PI_PYTHON", "/opt/conda/bin/python"),
        "/opt/flex-pi/npa/inference.py",
        "--input-manifest",
        str(input_path),
        "--output-json",
        str(output_path),
        "--checkpoint-id",
        request.checkpoint_id,
        "--checkpoint-revision",
        request.checkpoint_revision,
        "--num-inference-steps",
        str(request.num_inference_steps),
        "--seed",
        str(request.seed),
    ]
    if request.torch_compile:
        argv.append("--torch-compile")
    return argv


def _materialize_input(input_path: str, destination: Path) -> bytes:
    if input_path.startswith("s3://"):
        parsed = urlparse(input_path)
        StorageClient.from_environment().s3.download_file(
            parsed.netloc, parsed.path.lstrip("/"), str(destination)
        )
    else:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise FlexPiError("input manifest is unavailable")
        shutil.copyfile(source, destination)
    return destination.read_bytes()


def _validate_manifest(contents: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(contents)
        if payload.get("schema") != "npa.flex_pi.public_observation.v1":
            raise ValueError("unexpected schema")
        if payload.get("dataset", {}).get("id") != DEFAULT_DATASET_ID:
            raise ValueError("unexpected dataset")
        if payload.get("dataset", {}).get("revision") != DEFAULT_DATASET_REVISION:
            raise ValueError("unexpected dataset revision")
        if len(payload.get("observation", {}).get("rgb", {})) != 3:
            raise ValueError("three RGB cameras are required")
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FlexPiError("invalid flex-pi input manifest") from exc
    return payload


def _runtime_env(request: FlexPiRequest) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "HF_HOME": env.get("HF_HOME", "/workspace/.cache/huggingface"),
            "DIFFSYNTH_MODEL_BASE_PATH": env.get(
                "DIFFSYNTH_MODEL_BASE_PATH", "/workspace/.cache/flex-pi/diffsynth"
            ),
            "MODELSCOPE_CACHE": env.get(
                "MODELSCOPE_CACHE", "/workspace/.cache/modelscope"
            ),
            # ModelScope otherwise fetches one range at a time. Its supported
            # maximum keeps the roughly 11 GB converted UMT5 shard practical on a
            # cold worker while preserving an explicit operator override.
            "MODELSCOPE_DOWNLOAD_PARALLELS": env.get(
                "MODELSCOPE_DOWNLOAD_PARALLELS", "16"
            ),
            "PYTORCH_CUDA_ALLOC_CONF": env.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "FLEX_PI_SOURCE_REVISION": DEFAULT_SOURCE_REVISION,
        }
    )
    env.pop("NPA_FLEX_PI_TOKEN", None)
    return env


def _execute(
    argv: list[str],
    request: FlexPiRequest,
    runner: Callable[..., Any],
    *,
    compiler_env: dict[str, str] | None = None,
) -> None:
    completed = runner(
        argv,
        cwd="/opt/flex-pi",
        env={**_runtime_env(request), **(compiler_env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        # Preserve a bounded, secret-safe diagnostic tail instead of reducing
        # upstream failures to an exit code. Runtime URLs can be signed and
        # SDKs may echo bearer tokens, so redact those shapes before the
        # message reaches job logs.
        tail = str(completed.stdout or "")[-8_192:]
        tail = re.sub(r"(?i)Bearer\s+\S+", "Bearer <redacted>", tail)
        tail = re.sub(
            r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{12,}\b",
            "<redacted>",
            tail,
        )
        tail = re.sub(r"(?:s3|https?)://\S+", "<uri-ref>", tail)
        detail = tail.strip() or "no diagnostic output"
        raise FlexPiError(
            f"upstream flex-pi inference failed ({completed.returncode}): {detail}"
        )
    if REAL_INFERENCE_MARKER not in str(completed.stdout or ""):
        raise FlexPiError("upstream flex-pi inference did not emit its success marker")


def _validate_action_artifact(path: Path, request: FlexPiRequest) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = payload["actions"]
        if len(actions) != 32 or any(len(row) != 14 for row in actions):
            raise ValueError("action tensor must be 32x14")
        if any(not math.isfinite(float(value)) for row in actions for value in row):
            raise ValueError("actions must be finite")
        if float(payload["metrics"]["inference_seconds"]) <= 0:
            raise ValueError("inference latency must be positive")
        if payload["runtime"].get("cuda") is not True:
            raise ValueError("real inference requires CUDA")
        gpu_name = str(payload["runtime"].get("gpu_name", ""))
        if (
            request.expected_gpu
            and request.expected_gpu.lower() not in gpu_name.lower().replace(" ", "")
        ):
            raise ValueError("runtime GPU does not match expected_gpu")
        if payload.get("regime") != "action-only":
            raise ValueError("unexpected inference regime")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FlexPiError(f"invalid flex-pi action artifact: {exc}") from exc
    return payload


def _publish(local_dir: Path, output_path: str) -> dict[str, str]:
    if output_path.startswith("s3://"):
        base = output_path.rstrip("/") + "/"
        client = StorageClient.from_environment()
        return {
            path.name: client.upload_file(str(path), base + path.name)
            for path in sorted(local_dir.iterdir())
            if path.is_file()
        }
    target = Path(output_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    published = {}
    for source in sorted(local_dir.iterdir()):
        if source.is_file():
            destination = target / source.name
            shutil.copyfile(source, destination)
            published[source.name] = str(destination)
    return published


def _provenance(
    request: FlexPiRequest, manifest: dict[str, Any], argv: list[str]
) -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "source": {
            "repository": "https://github.com/geyan21/flex-pi",
            "revision": DEFAULT_SOURCE_REVISION,
        },
        "checkpoint": {
            "id": request.checkpoint_id,
            "revision": request.checkpoint_revision,
        },
        "dataset": {
            "id": manifest["dataset"]["id"],
            "revision": manifest["dataset"]["revision"],
            "sample": manifest["observation"]["episode"],
            "runtime_fetch": True,
        },
        "request": asdict(request),
        "runtime": {
            "image": request.runtime_image or os.environ.get("NPA_TASK_IMAGE", ""),
            "weights_baked": False,
            "dataset_baked": False,
            "cache_tier": "run-owned-runtime-cache",
        },
        "argv": argv,
    }


def run_inference(
    request: FlexPiRequest,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run real flex-pi action inference and publish validated artifacts.

    Args:
        request: Pinned checkpoint, public observation, and output settings.
        runner: Subprocess execution boundary used by tests.
    Returns:
        Provenance plus published artifact locations.
    Raises:
        FlexPiError: Inputs, execution, or output semantics are invalid.
        OSError: Local artifact publication fails.
    """
    _validate(request)
    with tempfile.TemporaryDirectory(prefix="npa-flex-pi-") as scratch:
        root = Path(scratch)
        contents = _materialize_input(request.input_path, root / "input.json")
        manifest = _validate_manifest(contents)
        argv = build_inference_argv(
            request, input_path=root / "input.json", output_path=root / "actions.json"
        )
        base = _provenance(request, manifest, argv)
        if request.dry_run:
            return {**base, "status": "dry_run", "artifacts": {}}
        compiler_env = {}
        if request.torch_compile and request.expected_gpu.upper() == "B300":
            from zipfile import BadZipFile

            from npa.workbench.flex_pi.compiler import prepare_b300_runtime

            try:
                argv[0], compiler_env, base["compiler"] = prepare_b300_runtime(
                    root / "compiler", base_python=argv[0]
                )
            except (OSError, ValueError, KeyError, BadZipFile) as exc:
                raise FlexPiError(f"B300 compiler preparation failed: {exc}") from exc
        _execute(argv, request, runner, compiler_env=compiler_env)
        action_payload = _validate_action_artifact(root / "actions.json", request)
        result = {
            **base,
            "status": "ok",
            "metrics": action_payload["metrics"],
            "model_runtime": action_payload["runtime"],
        }
        (root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifacts = _publish(root, request.output_path)
        # Emit success only after artifact validation and publication; keep
        # subprocess URLs and credentials out of both CLI output streams.
        print(REAL_INFERENCE_MARKER, file=sys.stderr, flush=True)
        return {**result, "artifacts": artifacts}
