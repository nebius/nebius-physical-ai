"""Pinned RoboLab evaluation for the Cosmos3 Edge FastWAM-K2 derivative.

The derivative model card describes a K=2 serving mode that is absent from its
declared cosmos-framework revision.  This module therefore applies a very small,
hash-recorded *runtime* overlay to that exact source revision.  It never maps K=2
to the upstream K=0 option and fails before inference if the expected upstream
anchors changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from npa.workbench.cosmos.policy_artifacts import (
    file_digest,
    materialize_bundle,
    policy_workspace,
    publish_bundle,
    write_local_json,
)
from npa.workbench.dataset.storage import read_json_uri

PREPARED_SCHEMA = "npa.cosmos3.fastwam-k2.prepared.v1"
VARIANT_SCHEMA = "npa.cosmos3.fastwam-k2.variant.v1"
COMPARISON_SCHEMA = "npa.cosmos3.fastwam-k2.comparison.v1"
VISUALIZATION_SCHEMA = "npa.cosmos3.fastwam-k2.visualization.v1"

FRAMEWORK_REPOSITORY = "https://github.com/NVIDIA/cosmos-framework.git"
FRAMEWORK_REVISION = "4e26181d87878a0b14c37ca021b0e2cd4f28dc5f"
ROBOLAB_REPOSITORY = "https://github.com/NVlabs/RoboLab.git"
ROBOLAB_REVISION = "ad45d4f974725d020f82c2b0d77d78533aeba2b3"
DERIVATIVE_REPOSITORY = "geonmin-kim/Cosmos3-Edge-Policy-DROID-FastWAM-K2"
DERIVATIVE_REVISION = "04cc10f6f790153fa5db1ff90e95ecf9196e88c5"
DERIVATIVE_SUBDIRECTORY = "step12000"
BASE_REPOSITORY = "nvidia/Cosmos3-Edge-Policy-DROID"
BASE_REVISION = "68b17b3c959ccd0999de9a972c5f7c8b57112f86"

KEEP_GENERATED_VISION_FRAMES = 2
CONDITIONING_LATENT_FRAMES = 1
TOKENS_PER_VISION_LATENT_FRAME = 340
SCREENING_TASKS = ("RubiksCubesInBinTask", "StackYellowOnRedTask")
SERVER_PORT = 8000


class FastWamK2Error(RuntimeError):
    """Raised when pinned FastWAM-K2 evaluation evidence is incomplete."""


class EvaluationRequest(BaseModel):
    """The closed-loop task-success protocol shared by both policy variants."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: Literal["screening", "full_suite"] = "screening"
    tasks: list[str] = Field(default_factory=lambda: list(SCREENING_TASKS))
    num_episodes_adaptive: int = Field(default=200, ge=1)
    ci_pp_width: float = Field(default=0.14, gt=0.0, le=1.0)
    num_envs: int = Field(default=1, ge=1)
    instruction_type: str = "default"
    video_mode: Literal["all", "sensor", "viewport"] = "sensor"

    @field_validator("tasks")
    @classmethod
    def unique_tasks(cls, tasks: list[str]) -> list[str]:
        """Require a nonempty, duplicate-free concrete task list."""
        if not tasks or len(tasks) != len(set(tasks)) or any(not task.strip() for task in tasks):
            raise ValueError("tasks must be nonempty, unique task names")
        return tasks


def _sha256_text(value: str) -> str:
    """Return a stable digest for a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_request(input_path: str) -> EvaluationRequest:
    """Load and strictly validate a durable evaluation request."""
    try:
        return EvaluationRequest.model_validate(read_json_uri(input_path))
    except Exception as exc:
        raise FastWamK2Error("invalid FastWAM-K2 evaluation request") from exc


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    """Run one native command with argv boundaries and retained diagnostics."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=stream, check=True)


def _checkout(url: str, revision: str, target: Path, log: Path) -> Path:
    """Fetch one exact upstream revision without materialising LFS payloads."""
    target.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    for argv in (
        ["git", "init", "."],
        ["git", "config", "filter.lfs.process", ""],
        ["git", "config", "filter.lfs.required", "false"],
        ["git", "remote", "add", "origin", url],
        ["git", "fetch", "--depth=1", "origin", revision],
        ["git", "checkout", "--detach", "FETCH_HEAD"],
    ):
        _run(argv, cwd=target, env=env, log=log)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if actual != revision:
        raise FastWamK2Error("upstream checkout revision did not match the pinned identity")
    return target


def _task_source_paths(robolab: Path, tasks: Iterable[str]) -> dict[str, Path]:
    """Resolve each selected task to the source file defining its registered class."""
    paths: dict[str, Path] = {}
    for task in tasks:
        matches = [path for path in robolab.rglob("*.py") if f"class {task}" in path.read_text(errors="ignore")]
        if len(matches) != 1:
            raise FastWamK2Error(f"expected one RoboLab task definition for {task}, found {len(matches)}")
        paths[task] = matches[0]
    return paths


def _prepared_payload(request: EvaluationRequest, robolab: Path) -> dict[str, Any]:
    """Build a hash-bound matched task-input manifest from actual task sources."""
    paths = _task_source_paths(robolab, request.tasks)
    task_sources = {
        task: {
            "path": path.relative_to(robolab).as_posix(),
            "sha256": file_digest(path),
        }
        for task, path in paths.items()
    }
    payload = {
        "schema": PREPARED_SCHEMA,
        "status": "succeeded",
        "protocol": request.protocol,
        "screening_only": request.protocol == "screening",
        "benchmark_claim": False,
        "request": request.model_dump(mode="json"),
        "robolab": {"repository": ROBOLAB_REPOSITORY, "revision": ROBOLAB_REVISION},
        "task_sources": task_sources,
        "model_lineage": _model_lineage(),
    }
    payload["prepared_sha256"] = _sha256_text(json.dumps(payload, sort_keys=True))
    return payload


def _model_lineage() -> dict[str, Any]:
    """Return the immutable policy and framework identities retained in every report."""
    return {
        "derivative": {
            "repository": DERIVATIVE_REPOSITORY,
            "revision": DERIVATIVE_REVISION,
            "subdirectory": DERIVATIVE_SUBDIRECTORY,
        },
        "full_wam_baseline": {"repository": BASE_REPOSITORY, "revision": BASE_REVISION},
        "framework": {"repository": FRAMEWORK_REPOSITORY, "revision": FRAMEWORK_REVISION},
    }


def prepare_evaluation_inputs(*, input_path: str, output_path: str) -> dict[str, Any]:
    """Prepare hash-bound matched RoboLab task definitions for both policy arms."""
    request = _read_request(input_path)
    with policy_workspace(output_path, "fastwam-k2-prepare") as root:
        robolab = _checkout(ROBOLAB_REPOSITORY, ROBOLAB_REVISION, root / "robolab", root / "bootstrap.log")
        payload = _prepared_payload(request, robolab)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        write_local_json(artifacts / "prepared-inputs.json", payload)
        return publish_bundle(artifacts, output_path, payload, "prepared.json")


def _read_prepared(input_path: str, root: Path) -> tuple[dict[str, Any], Path]:
    """Materialize and validate a completed matched-input manifest."""
    materialize_bundle(input_path, root, PREPARED_SCHEMA)
    prepared = root / "prepared-inputs.json"
    if not prepared.is_file():
        raise FastWamK2Error("prepared inputs artifact is missing")
    payload = json.loads(prepared.read_text())
    if payload.get("prepared_sha256") != _sha256_text(
        json.dumps({key: value for key, value in payload.items() if key != "prepared_sha256"}, sort_keys=True)
    ):
        raise FastWamK2Error("prepared input manifest digest mismatch")
    return payload, root


def _ensure_prepared_task_sources(prepared: dict[str, Any], robolab: Path) -> None:
    """Refuse a run if its freshly fetched task definitions differ from preparation."""
    expected = prepared.get("task_sources", {})
    for task, record in expected.items():
        path = robolab / str(record.get("path", ""))
        if not path.is_file() or file_digest(path) != record.get("sha256"):
            raise FastWamK2Error(f"RoboLab task source changed after preparation: {task}")


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    """Replace one expected source anchor or fail closed on upstream drift."""
    if text.count(old) != 1:
        raise FastWamK2Error(f"FastWAM-K2 overlay anchor mismatch: {name}")
    return text.replace(old, new)


def apply_k2_runtime_overlay(framework: Path, artifact_root: Path) -> dict[str, Any]:
    """Add the documented K=2 mode to its exact framework revision at runtime.

    The overlay shortens only the generated vision latent tail.  It leaves the
    sequence plan's conditioning indexes intact, so latent frame zero remains
    clean while latent frames one and two are denoised/generated.
    """
    server = framework / "cosmos_framework/scripts/action_policy_server_robolab.py"
    model = framework / "cosmos_framework/model/generator/omni_mot_model.py"
    artifact_root.mkdir(parents=True, exist_ok=True)
    if not server.is_file() or not model.is_file():
        raise FastWamK2Error("pinned framework lacks the expected RoboLab implementation")
    before = {"server": file_digest(server), "model": file_digest(model)}
    server_text = server.read_text()
    model_text = model.read_text()
    server_text = _replace_once(
        server_text,
        '    format_prompt_as_json: bool | None = None\n    """Serve prompts as structured JSON (matching training ``format_prompt_as_json``)."""\n',
        '    format_prompt_as_json: bool | None = None\n    """Serve prompts as structured JSON (matching training ``format_prompt_as_json``)."""\n    keep_generated_vision_frames: int = 0\n    """Retain this many unconditioned vision latents for FastWAM denoising; zero keeps full WAM."""\n',
        "server argument",
    )
    server_text = _replace_once(
        server_text,
        '        self.model = pipe.model\n        self.model.eval()\n',
        '        self.model = pipe.model\n        self.model.eval()\n        if args.keep_generated_vision_frames < 0:\n            raise ValueError("--keep-generated-vision-frames must be non-negative")\n        self.model.inference_keep_generated_vision_frames = int(args.keep_generated_vision_frames)\n',
        "server model configuration",
    )
    model_text = _replace_once(
        model_text,
        '            full_shape = list(prefix_latent.shape)\n            full_shape[temporal_dim] = num_latent_frames\n',
        '            full_shape = list(prefix_latent.shape)\n            keep_generated = int(getattr(self, "inference_keep_generated_vision_frames", 0))\n            if keep_generated < 0:\n                raise ValueError("inference_keep_generated_vision_frames must be non-negative")\n            kept_latent_frames = min(num_latent_frames, needed_latent_frames + keep_generated)\n            full_shape[temporal_dim] = kept_latent_frames\n',
        "latent-tail length",
    )
    server.write_text(server_text)
    model.write_text(model_text)
    after = {"server": file_digest(server), "model": file_digest(model)}
    _validate_overlay_sources(server_text, model_text)
    record = {
        "schema": "npa.cosmos3.fastwam-k2.overlay.v1",
        "framework_revision": FRAMEWORK_REVISION,
        "patch_kind": "runtime-source-overlay",
        "keep_generated_vision_frames": KEEP_GENERATED_VISION_FRAMES,
        "conditioning_latent_frames": CONDITIONING_LATENT_FRAMES,
        "denoised_generated_latent_frames": KEEP_GENERATED_VISION_FRAMES,
        "expected_vision_tokens": TOKENS_PER_VISION_LATENT_FRAME * 3,
        "expected_mse_target_tokens": TOKENS_PER_VISION_LATENT_FRAME * KEEP_GENERATED_VISION_FRAMES,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "upstream_flag_absent": True,
    }
    write_local_json(artifact_root / "k2-runtime-overlay.json", record)
    return record


def _validate_overlay_sources(server_text: str, model_text: str) -> None:
    """Assert that the overlay cannot silently collapse K=2 to K=0/full WAM."""
    required = (
        "keep_generated_vision_frames: int = 0",
        "self.model.inference_keep_generated_vision_frames",
        "--keep-generated-vision-frames must be non-negative",
        "kept_latent_frames = min(num_latent_frames, needed_latent_frames + keep_generated)",
        "full_shape[temporal_dim] = kept_latent_frames",
    )
    if any(fragment not in (server_text + model_text) for fragment in required):
        raise FastWamK2Error("FastWAM-K2 overlay semantic validation failed")
    if "condition_indexes" not in model_text or "condition_mask" not in model_text:
        raise FastWamK2Error("overlay source does not retain native conditioning semantics")


def _sync_framework(framework: Path, env: dict[str, str], log: Path) -> Path:
    """Create the exact framework policy-server environment on the GPU worker."""
    _run(
        ["uv", "sync", "--frozen", "--extra", "policy-server", "--group", "cu130-train"],
        cwd=framework,
        env=env,
        log=log,
    )
    python = framework / ".venv/bin/python"
    _run([str(python), "-c", "import torch; assert torch.cuda.is_available()"], cwd=framework, env=env, log=log)
    return python


def _sync_robolab(robolab: Path, env: dict[str, str], log: Path) -> Path:
    """Materialize the upstream RoboLab Isaac 5 runtime without baking it."""
    runtime_env = _isaac_runtime_env(env)
    runtime_env["UV_PROJECT_ENVIRONMENT"] = str(robolab / ".venv")
    _run(["uv", "sync", "--extra", "isaac50"], cwd=robolab, env=runtime_env, log=log)
    return robolab / ".venv/bin/python"


def _isaac_runtime_env(env: dict[str, str]) -> dict[str, str]:
    """Derive the upstream wheel variable from NPA's single Isaac EULA policy."""
    raw = env.get("ACCEPT_EULA", "Y").strip().upper()
    if raw in {"", "N", "NO", "0", "FALSE"}:
        raise FastWamK2Error("Isaac runtime was explicitly opted out through ACCEPT_EULA")
    if raw not in {"Y", "YES", "1", "TRUE"}:
        raise FastWamK2Error("ACCEPT_EULA has an invalid value for the Isaac runtime")
    runtime_env = dict(env, ACCEPT_EULA="Y")
    runtime_env["OMNI_KIT_ACCEPT_EULA"] = "Y"
    return runtime_env


def _snapshot_checkpoint(repository: str, revision: str, destination: Path, python: Path, log: Path) -> Path:
    """Fetch one exact Hugging Face checkpoint into run-private scratch."""
    destination.mkdir(parents=True, exist_ok=False)
    code = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
        "local_dir_use_symlinks=False); print('checkpoint materialized')"
    )
    _run([str(python), "-c", code, repository, revision, str(destination)], cwd=destination, env=dict(os.environ), log=log)
    if not any(destination.rglob("*.safetensors")):
        raise FastWamK2Error("checkpoint fetch produced no safetensors payload")
    return destination


def server_argv(framework_python: Path, checkpoint: Path, variant: str) -> list[str]:
    """Return the exact server command for full WAM or the K=2 overlay path."""
    argv = [
        str(framework_python),
        "-m",
        "cosmos_framework.scripts.action_policy_server_robolab",
        "--checkpoint-path",
        str(checkpoint),
        "--port",
        str(SERVER_PORT),
        "--format-prompt-as-json",
        "True",
        "--no-guardrails",
    ]
    if variant == "fastwam-k2":
        argv.extend(["--keep-generated-vision-frames", str(KEEP_GENERATED_VISION_FRAMES)])
    elif variant != "full-wam":
        raise FastWamK2Error(f"unsupported policy variant: {variant}")
    return argv


def _wait_for_server(process: subprocess.Popen[bytes], log: Path) -> None:
    """Wait for the local policy port, retaining the bounded readiness probe log."""
    while process.poll() is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2)
            if probe.connect_ex(("127.0.0.1", SERVER_PORT)) == 0:
                log.write_text("policy port accepted a local connection\n")
                return
        time.sleep(1)
    raise FastWamK2Error("policy server exited before opening its local port")


def _run_robolab(
    robolab_python: Path,
    robolab: Path,
    request: EvaluationRequest,
    variant: str,
    env: dict[str, str],
    log: Path,
) -> Path:
    """Run actual closed-loop RoboLab episodes and retain their native result rows."""
    output_name = f"npa-fastwam-k2-{variant}"
    argv = [
        str(robolab_python),
        "policies/cosmos3/run.py",
        "--remote-host",
        "127.0.0.1",
        "--remote-port",
        str(SERVER_PORT),
        "--headless",
        "--task",
        *request.tasks,
        "--num-envs",
        str(request.num_envs),
        "--num-episodes-adaptive",
        str(request.num_episodes_adaptive),
        "--ci-pp-width",
        str(request.ci_pp_width),
        "--instruction-type",
        request.instruction_type,
        "--video-mode",
        request.video_mode,
        "--output-folder-name",
        output_name,
    ]
    client_env = dict(_isaac_runtime_env(env), CUDA_VISIBLE_DEVICES="1")
    _run(argv, cwd=robolab, env=client_env, log=log)
    output = robolab / "output" / output_name
    if not output.is_dir():
        raise FastWamK2Error("RoboLab completed without its expected output directory")
    return output


def _episode_rows(output: Path, tasks: Iterable[str]) -> list[dict[str, Any]]:
    """Load complete native result rows and reject absent task-success evidence."""
    result_path = output / "episode_results.jsonl"
    if not result_path.is_file():
        raise FastWamK2Error("RoboLab did not write episode_results.jsonl")
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise FastWamK2Error("RoboLab produced no completed episode rows")
    present = {str(row.get("task_name") or row.get("env_name")) for row in rows}
    missing = set(tasks) - present
    if missing or any(type(row.get("success")) is not bool for row in rows):
        raise FastWamK2Error("RoboLab result rows lack requested task-success evidence")
    return rows


def _numeric(values: Iterable[Any]) -> list[float]:
    """Return finite numeric values while excluding booleans and malformed rows."""
    output: list[float] = []
    for value in values:
        if type(value) in (int, float) and math.isfinite(float(value)) and float(value) >= 0.0:
            output.append(float(value))
    return output


def summarize_episode_metrics(rows: list[dict[str, Any]], tasks: Iterable[str]) -> dict[str, Any]:
    """Derive task success and actual policy latency from native episode rows."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_name") or row.get("env_name"))].append(row)
    summary: dict[str, Any] = {"tasks": {}, "overall": {}}
    all_latency: list[float] = []
    for task in tasks:
        task_rows = grouped[task]
        if not task_rows:
            raise FastWamK2Error(f"RoboLab result rows lack requested task-success evidence: {task}")
        latency = _numeric(
            row.get("timing", {}).get("policy_inference_avg_ms")
            for row in task_rows
            if isinstance(row.get("timing"), dict)
        )
        if len(latency) != len(task_rows):
            raise FastWamK2Error(f"RoboLab result rows lack native policy latency evidence: {task}")
        all_latency.extend(latency)
        summary["tasks"][task] = {
            "episodes": len(task_rows),
            "successes": sum(bool(row["success"]) for row in task_rows),
            "success_rate": sum(bool(row["success"]) for row in task_rows) / len(task_rows),
            "policy_inference_avg_ms": statistics.mean(latency) if latency else None,
            "policy_inference_median_ms": statistics.median(latency) if latency else None,
        }
    summary["overall"] = {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "policy_inference_avg_ms": statistics.mean(all_latency) if all_latency else None,
        "policy_inference_median_ms": statistics.median(all_latency) if all_latency else None,
    }
    return summary


def _copy_run_artifacts(source: Path, destination: Path) -> None:
    """Copy native result rows and every recorded MP4, excluding state/cache payloads."""
    destination.mkdir(parents=True, exist_ok=True)
    for result in source.rglob("episode_results.jsonl"):
        target = destination / "episode_results.jsonl"
        shutil.copyfile(result, target)
    videos = [path for path in source.rglob("*.mp4") if path.is_file()]
    if not videos:
        raise FastWamK2Error("closed-loop evaluation produced no MP4 review artifact")
    (destination / "videos").mkdir()
    for index, video in enumerate(sorted(videos)):
        shutil.copyfile(video, destination / "videos" / f"{index:04d}.mp4")


def _variant_checkpoint(variant: str) -> tuple[str, str, str]:
    """Choose the immutable checkpoint identity for one comparison arm."""
    if variant == "full-wam":
        return BASE_REPOSITORY, BASE_REVISION, ""
    if variant == "fastwam-k2":
        return DERIVATIVE_REPOSITORY, DERIVATIVE_REVISION, DERIVATIVE_SUBDIRECTORY
    raise FastWamK2Error(f"unsupported policy variant: {variant}")


def run_variant(*, input_path: str, output_path: str, variant: str, baseline_path: str = "") -> dict[str, Any]:
    """Run one real policy server and matching RoboLab closed-loop protocol."""
    with policy_workspace(output_path, f"fastwam-k2-{variant}") as root:
        prepared, _ = _read_prepared(input_path, root / "prepared")
        request = EvaluationRequest.model_validate(prepared["request"])
        if baseline_path:
            _assert_baseline_matches(prepared, baseline_path, root / "baseline")
        framework = _checkout(FRAMEWORK_REPOSITORY, FRAMEWORK_REVISION, root / "framework", root / "bootstrap.log")
        robolab = _checkout(ROBOLAB_REPOSITORY, ROBOLAB_REVISION, root / "robolab", root / "bootstrap.log")
        _ensure_prepared_task_sources(prepared, robolab)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        overlay = apply_k2_runtime_overlay(framework, artifacts) if variant == "fastwam-k2" else None
        framework_env = dict(os.environ, COSMOS_TRAINING="1", PYTHONPATH=str(framework))
        framework_python = _sync_framework(framework, framework_env, root / "framework-sync.log")
        robolab_python = _sync_robolab(robolab, dict(os.environ), root / "robolab-sync.log")
        repo, revision, subdirectory = _variant_checkpoint(variant)
        checkpoint_root = _snapshot_checkpoint(repo, revision, root / "checkpoint", framework_python, root / "checkpoint.log")
        checkpoint = checkpoint_root / subdirectory if subdirectory else checkpoint_root
        if not checkpoint.is_dir():
            raise FastWamK2Error("checkpoint does not contain the expected immutable subdirectory")
        command = server_argv(framework_python, checkpoint, variant)
        write_local_json(artifacts / "serving-command.json", {"argv": command, "variant": variant})
        server_env = dict(framework_env, CUDA_VISIBLE_DEVICES="0")
        with (root / "policy-server.log").open("wb") as stream:
            process = subprocess.Popen(command, cwd=framework, env=server_env, stdout=stream, stderr=stream)
            try:
                _wait_for_server(process, artifacts / "server-ready.txt")
                output = _run_robolab(
                    robolab_python,
                    robolab,
                    request,
                    variant,
                    dict(os.environ),
                    root / "robolab.log",
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        rows = _episode_rows(output, request.tasks)
        _copy_run_artifacts(output, artifacts / "closed-loop")
        report = _variant_report(variant, prepared, request, rows, overlay)
        write_local_json(artifacts / "variant-result.json", report)
        for name in ("policy-server.log", "robolab.log", "framework-sync.log", "robolab-sync.log", "checkpoint.log"):
            source = root / name
            if source.is_file():
                shutil.copyfile(source, artifacts / name)
        return publish_bundle(artifacts, output_path, report, f"{variant}.json")


def _assert_baseline_matches(prepared: dict[str, Any], baseline_path: str, root: Path) -> None:
    """Require the K=2 arm to consume a baseline from the same prepared inputs."""
    baseline = materialize_bundle(baseline_path, root, VARIANT_SCHEMA)
    result = json.loads((root / "variant-result.json").read_text())
    if baseline.get("prepared_sha256") != prepared.get("prepared_sha256") or result.get("prepared_sha256") != prepared.get("prepared_sha256"):
        raise FastWamK2Error("K=2 evaluation must consume the matching full-WAM prepared inputs")


def _variant_report(
    variant: str,
    prepared: dict[str, Any],
    request: EvaluationRequest,
    rows: list[dict[str, Any]],
    overlay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the durable result report without presenting screening as a benchmark."""
    report = {
        "schema": VARIANT_SCHEMA,
        "status": "succeeded",
        "variant": variant,
        "prepared_sha256": prepared["prepared_sha256"],
        "request": request.model_dump(mode="json"),
        "screening_only": request.protocol == "screening",
        "benchmark_claim": False,
        "physical_robot_tested": False,
        "measurement": "closed-loop RoboLab task success and native policy inference latency",
        "metrics": summarize_episode_metrics(rows, request.tasks),
        "model_lineage": _model_lineage(),
    }
    if overlay is not None:
        report["serving_contract"] = overlay
    else:
        report["serving_contract"] = {"mode": "full-wam", "keep_generated_vision_frames": None}
    return report


def _read_variant(input_path: str, root: Path) -> dict[str, Any]:
    """Materialize and validate one completed policy-arm result bundle."""
    report = materialize_bundle(input_path, root, VARIANT_SCHEMA)
    result_path = root / "variant-result.json"
    if not result_path.is_file():
        raise FastWamK2Error("variant result body is missing")
    result = json.loads(result_path.read_text())
    if result.get("schema") != VARIANT_SCHEMA or result.get("status") != "succeeded":
        raise FastWamK2Error("variant result did not complete")
    return result | {"_manifest": report}


def compare_variants(*, full_wam_path: str, k2_path: str, output_path: str) -> dict[str, Any]:
    """Compare only matched task-success evidence and actual measured latency."""
    with policy_workspace(output_path, "fastwam-k2-compare") as root:
        baseline = _read_variant(full_wam_path, root / "full-wam")
        k2 = _read_variant(k2_path, root / "fastwam-k2")
        if baseline.get("variant") != "full-wam" or k2.get("variant") != "fastwam-k2":
            raise FastWamK2Error("comparison requires one full-WAM and one FastWAM-K2 result")
        if baseline.get("prepared_sha256") != k2.get("prepared_sha256"):
            raise FastWamK2Error("comparison arms do not use matching prepared inputs")
        report = _comparison_report(baseline, k2)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        write_local_json(artifacts / "comparison-result.json", report)
        return publish_bundle(artifacts, output_path, report, "comparison.json")


def _comparison_report(baseline: dict[str, Any], k2: dict[str, Any]) -> dict[str, Any]:
    """Produce a paired descriptive comparison without a proxy-ranking claim."""
    baseline_tasks = baseline["metrics"]["tasks"]
    k2_tasks = k2["metrics"]["tasks"]
    if set(baseline_tasks) != set(k2_tasks):
        raise FastWamK2Error("comparison task sets differ")
    paired = {
        task: {
            "full_wam_success_rate": baseline_tasks[task]["success_rate"],
            "fastwam_k2_success_rate": k2_tasks[task]["success_rate"],
            "success_rate_delta": k2_tasks[task]["success_rate"] - baseline_tasks[task]["success_rate"],
            "full_wam_latency_ms": baseline_tasks[task]["policy_inference_avg_ms"],
            "fastwam_k2_latency_ms": k2_tasks[task]["policy_inference_avg_ms"],
        }
        for task in sorted(baseline_tasks)
    }
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "succeeded",
        "prepared_sha256": baseline["prepared_sha256"],
        "screening_only": bool(baseline["screening_only"] or k2["screening_only"]),
        "benchmark_claim": False,
        "physical_robot_tested": False,
        "decision_basis": "closed-loop task success; open-loop errors are intentionally excluded",
        "paired_task_metrics": paired,
        "overall": {
            "full_wam": baseline["metrics"]["overall"],
            "fastwam_k2": k2["metrics"]["overall"],
        },
        "model_lineage": _model_lineage(),
    }


def _rerun_module() -> Any:
    """Load the Rerun SDK, installing its pinned worker-only dependency if needed."""
    try:
        import rerun as rr
    except ImportError:
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, "rerun-sdk==0.38.1"], check=True)
        import rerun as rr
    return rr


def _rrd_recording_id(comparison: dict[str, Any]) -> str:
    """Bind a workflow RRD to its renderer-provided run identity when available."""
    return os.environ.get("NPA_WORKFLOW_RUN_ID", "").strip() or comparison["prepared_sha256"]


def _log_rrd(rr: Any, destination: Path, comparison: dict[str, Any], videos: list[Path]) -> str:
    """Write a factual Rerun recording from actual comparison metrics and MP4 files."""
    recording_id = _rrd_recording_id(comparison)
    recording = rr.RecordingStream("npa_cosmos3_fastwam_k2", recording_id=recording_id)
    rr.save(str(destination), recording=recording)
    summary = json.dumps(comparison, indent=2, sort_keys=True)
    rr.log("reports/comparison", rr.TextDocument(summary, media_type="application/json"), static=True, recording=recording)
    for index, video in enumerate(videos):
        if hasattr(rr, "AssetVideo"):
            rr.log(f"rollouts/{index}", rr.AssetVideo(path=str(video)), static=True, recording=recording)
    for task, metrics in comparison["paired_task_metrics"].items():
        rr.log(f"metrics/{task}/full_wam_success_rate", rr.Scalars(metrics["full_wam_success_rate"]), recording=recording)
        rr.log(f"metrics/{task}/fastwam_k2_success_rate", rr.Scalars(metrics["fastwam_k2_success_rate"]), recording=recording)
    flush = getattr(recording, "flush", None)
    if callable(flush):
        flush()
    return recording_id


def emit_visualization(*, full_wam_path: str, k2_path: str, comparison_path: str, output_path: str) -> dict[str, Any]:
    """Emit a factual RRD with copied rollout MP4s and measured paired metrics."""
    with policy_workspace(output_path, "fastwam-k2-visualize") as root:
        _read_variant(full_wam_path, root / "full-wam")
        _read_variant(k2_path, root / "fastwam-k2")
        comparison = materialize_bundle(comparison_path, root / "comparison", COMPARISON_SCHEMA)
        comparison_body = json.loads((root / "comparison" / "comparison-result.json").read_text())
        videos = sorted((root / "full-wam").rglob("*.mp4")) + sorted((root / "fastwam-k2").rglob("*.mp4"))
        if not videos:
            raise FastWamK2Error("visualization requires actual rollout MP4 artifacts")
        artifact_root = root / "visualization"
        artifact_root.mkdir()
        for index, video in enumerate(videos):
            shutil.copyfile(video, artifact_root / f"rollout-{index:04d}.mp4")
        rrd_path = artifact_root / "fastwam-k2-screening.rrd"
        recording_id = _log_rrd(_rerun_module(), rrd_path, comparison_body, videos)
        if not rrd_path.is_file() or rrd_path.stat().st_size == 0:
            raise FastWamK2Error("Rerun did not emit a nonempty recording")
        report = {
            "schema": VISUALIZATION_SCHEMA,
            "status": "succeeded",
            "prepared_sha256": comparison["prepared_sha256"],
            "screening_only": comparison_body["screening_only"],
            "benchmark_claim": False,
            "rrd": "fastwam-k2-screening.rrd",
            "rrd_recording_id": recording_id,
            "rollout_mp4_count": len(videos),
        }
        write_local_json(artifact_root / "visualization-result.json", report)
        return publish_bundle(artifact_root, output_path, report, "visualization.json")
