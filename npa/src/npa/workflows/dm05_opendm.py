"""Pinned OpenDM DM05/LIBERO workflow stages.

This module deliberately keeps the meaningful work in Dexmal's native entry
points: ``script/libero_runner.sh``, ``script/dm05_launcher.sh`` and the
Dexbotic LIBERO evaluator.  It owns only the durable NPA hand-offs, immutable
revision checks, and the explicit service-boundary normalization contract.

The five commands are intended for the states in
``workflows/testing/dm05-opendm.yaml``.  They do not make a benchmark claim
without an upstream evaluator ``results.json`` and its actual rollout videos.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient


OPENDM_SOURCE_URL = "https://github.com/dexmal/opendm"
OPENDM_SOURCE_REVISION = "7d52f1591437332cb0157be3303c1c46da811344"
DM05_MODEL_ID = "Dexmal/DM05"
DM05_MODEL_REVISION = "5cd18734814abb075a9ccfd9ad6d16777b5cf10e"
LIBERO_DATASET_ID = "Dexmal/libero"
LIBERO_DATASET_REVISION = "f15a66b3975f8cd210c746991f80adde5ab05ca4"
DEXBOTIC_SOURCE_URL = "https://github.com/dexmal/dexbotic-benchmark"
DEXBOTIC_SOURCE_REVISION = "789b87f50d9fadc7663d2e8bac057941221aab81"

LIBERO_DATASET_NAME = "libero_pi0_all"
LIBERO_IMAGE_KEYS = ("images_1", "images_2")
LIBERO_IMAGE_PROMPTS = ("Head", "Left wrist")
LIBERO_ROBOT_TYPE = "Franka"
LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_CHUNK = 10


class DM05WorkflowError(RuntimeError):
    """An invalid artifact or an unsuccessful native upstream operation."""


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_summary(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in files
            if path.name in {"norm_stats.json", "results.json", "config.yaml"}
        },
    }


def _run(
    command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None
) -> None:
    rendered = " ".join(command)
    print(f"[dm05-opendm] running: {rendered}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(list(command), cwd=cwd, env=merged, check=True)


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def assert_pinned_checkout(root: Path, expected_revision: str, label: str) -> None:
    actual = _git_revision(root)
    if actual != expected_revision:
        raise DM05WorkflowError(
            f"{label} checkout must be pinned to {expected_revision}, got {actual}"
        )


def assert_normalization_contract(
    *,
    state: Sequence[object],
    image_count: int,
    action_dim: int | None = None,
    chunk_size: int = LIBERO_ACTION_CHUNK,
) -> None:
    """Reject an incompatible LIBERO observation/action shape before model use."""
    if image_count != len(LIBERO_IMAGE_PROMPTS):
        raise DM05WorkflowError(
            "LIBERO DM05 requires camera order "
            f"{list(LIBERO_IMAGE_PROMPTS)!r}, got {image_count} images"
        )
    if len(state) != LIBERO_STATE_DIM:
        raise DM05WorkflowError(
            "LIBERO DM05 requires an 8-value Franka state ordered as six joints "
            f"then two gripper values, got {len(state)} values"
        )
    try:
        [float(value) for value in state]
    except (TypeError, ValueError) as exc:
        raise DM05WorkflowError(
            "LIBERO state must contain only numeric values"
        ) from exc
    if action_dim is not None and action_dim != LIBERO_ACTION_DIM:
        raise DM05WorkflowError(
            f"LIBERO DM05 requires {LIBERO_ACTION_DIM} action values, got {action_dim}"
        )
    if chunk_size != LIBERO_ACTION_CHUNK:
        raise DM05WorkflowError(
            f"LIBERO DM05 requires action chunk {LIBERO_ACTION_CHUNK}, got {chunk_size}"
        )


class ArtifactStore:
    """Run-scoped S3 directory transfer helper with no ambient local fallback."""

    def __init__(self) -> None:
        self._storage = StorageClient.from_environment()

    def download_tree(self, uri: str, target: Path) -> Path:
        if not uri.startswith("s3://"):
            raise DM05WorkflowError(f"workflow artifact URI must be s3://, got {uri!r}")
        target.mkdir(parents=True, exist_ok=True)
        self._storage.download_directory(uri, str(target))
        return target

    def upload_tree(self, source: Path, uri: str) -> str:
        if not uri.startswith("s3://"):
            raise DM05WorkflowError(f"workflow artifact URI must be s3://, got {uri!r}")
        if not source.is_dir():
            raise DM05WorkflowError(f"artifact source directory is missing: {source}")
        return self._storage.upload_directory(str(source), uri, require_empty=True)


def _native_python(value: str) -> str:
    path = Path(value)
    if path.is_absolute() and not path.is_file():
        raise DM05WorkflowError(f"configured native Python is missing: {path}")
    return value


def _hf_download(
    *, repo_id: str, revision: str, repo_type: str, destination: Path, cwd: Path
) -> None:
    _run(
        [
            "hf",
            "download",
            repo_id,
            "--repo-type",
            repo_type,
            "--revision",
            revision,
            "--local-dir",
            str(destination),
        ],
        cwd=cwd,
    )


def _compute_norm_stats(
    *, repo_root: Path, data_root: Path, norm_root: Path, native_python: str
) -> Path:
    """Run OpenDM's DM05DataConfig.compute_norm_stats without starting training."""
    program = """
from pathlib import Path
from playground.dm05_libero import DM05DataConfig

data_root = Path(__import__('sys').argv[1])
norm_root = Path(__import__('sys').argv[2])
config = DM05DataConfig(
    jsonl_dir=str(data_root / 'libero_pi0_all'),
    image_dir=str(data_root / 'libero_pi0_all' / 'image'),
    norm_stats_root=str(norm_root),
)
config.compute_norm_stats(action_horizon=10)
print(config.norm_stats_path(10))
"""
    result = subprocess.run(
        [native_python, "-c", program, str(data_root), str(norm_root)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    candidates = sorted(norm_root.glob("*.json"))
    if len(candidates) != 1:
        raise DM05WorkflowError(
            f"OpenDM normalization produced {len(candidates)} JSON files, expected one"
        )
    canonical = norm_root / "norm_stats.json"
    if candidates[0] != canonical:
        candidates[0].replace(canonical)
    if not canonical.is_file() or not canonical.read_text(encoding="utf-8").strip():
        raise DM05WorkflowError("OpenDM normalization did not produce norm_stats.json")
    print(result.stdout, end="")
    return canonical


def _first_libero_observation(data_root: Path) -> tuple[dict[str, Any], list[Path]]:
    jsonl_dir = data_root / LIBERO_DATASET_NAME / "jsonl"
    image_root = data_root / LIBERO_DATASET_NAME / "image"
    jsonl_files = sorted(jsonl_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise DM05WorkflowError(f"no LIBERO JSONL files under {jsonl_dir}")
    for line in jsonl_files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            frame = json.loads(line)
            break
    else:
        raise DM05WorkflowError(f"first LIBERO JSONL file is empty: {jsonl_files[0]}")
    state = frame.get("state")
    if not isinstance(state, list):
        raise DM05WorkflowError("LIBERO frame has no list-valued state")
    image_paths: list[Path] = []
    for key in LIBERO_IMAGE_KEYS:
        entry = frame.get(key)
        if not isinstance(entry, dict) or entry.get("type") != "image":
            raise DM05WorkflowError(f"LIBERO frame {key} is not an image entry")
        value = entry.get("url")
        if not isinstance(value, str):
            raise DM05WorkflowError(f"LIBERO frame {key} has no image URL")
        candidate = (image_root / value).resolve()
        if (
            not candidate.is_relative_to(image_root.resolve())
            or not candidate.is_file()
        ):
            raise DM05WorkflowError(f"LIBERO image escapes or is missing: {value!r}")
        image_paths.append(candidate)
    assert_normalization_contract(state=state, image_count=len(image_paths))
    return frame, image_paths


def _request_payload(
    frame: dict[str, Any], image_paths: Iterable[Path]
) -> dict[str, Any]:
    state = frame["state"]
    images = {
        str(index): base64.b64encode(path.read_bytes()).decode("ascii")
        for index, path in enumerate(image_paths, start=1)
    }
    return {
        "observation": {
            "prompt": str(frame.get("prompt", "")),
            "robot_type": LIBERO_ROBOT_TYPE,
            "state": state,
            "images": images,
        }
    }


def _wait_for_http_server(process: subprocess.Popen[bytes], endpoint: str) -> None:
    """Wait for the owned server until it responds or exits; never mask startup failure."""
    while True:
        if process.poll() is not None:
            raise DM05WorkflowError(
                f"DM05 inference process exited before serving {endpoint}, rc={process.returncode}"
            )
        try:
            with urllib.request.urlopen(endpoint, timeout=5):
                return
        except urllib.error.HTTPError:
            # Flask returns 404 for GET /v1/infer; the port is nevertheless ready.
            return
        except urllib.error.URLError:
            time.sleep(2)


def _start_server(
    *,
    repo_root: Path,
    checkpoint: Path,
    native_python: str,
    port: int,
    cuda_visible_devices: str,
) -> subprocess.Popen[bytes]:
    command = [
        "bash",
        "script/dm05_launcher.sh",
        "--exp",
        "playground/dm05_libero.py",
        "--",
        "--task",
        "inference",
        "--model-config.model-name-or-path",
        str(checkpoint),
        "--model-config.chunk-size",
        str(LIBERO_ACTION_CHUNK),
        "--inference-config.output-action-dim",
        str(LIBERO_ACTION_DIM),
        "--inference-config.image-prompts",
        *LIBERO_IMAGE_PROMPTS,
        "--inference-config.port",
        str(port),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "PYTHON": native_python,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
    }
    # dm05_launcher intentionally uses the image's ``python``.  Put the native
    # interpreter first rather than altering upstream source or its arguments.
    native_dir = (
        str(Path(native_python).parent) if Path(native_python).is_absolute() else ""
    )
    if native_dir:
        env["PATH"] = native_dir + os.pathsep + env.get("PATH", "")
    print(f"[dm05-opendm] running: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=repo_root, env=env)


def _stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _infer_once(*, port: int, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) != LIBERO_ACTION_CHUNK:
        raise DM05WorkflowError(
            "DM05 v1 response has an unexpected action-chunk length"
        )
    for action in actions:
        if not isinstance(action, list):
            raise DM05WorkflowError("DM05 v1 response action is not a vector")
        assert_normalization_contract(
            state=payload["observation"]["state"],
            image_count=len(payload["observation"]["images"]),
            action_dim=len(action),
        )
    return data


def prepare(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    assert_pinned_checkout(repo_root, OPENDM_SOURCE_REVISION, "OpenDM")
    native_python = _native_python(args.opendm_python)
    store = ArtifactStore()
    with tempfile.TemporaryDirectory(prefix="npa-dm05-prepare-") as temporary:
        work = Path(temporary)
        raw = work / "hf-dataset"
        data = work / "data" / "libero"
        norm = work / "norm_stats"
        _hf_download(
            repo_id=LIBERO_DATASET_ID,
            revision=LIBERO_DATASET_REVISION,
            repo_type="dataset",
            destination=raw,
            cwd=repo_root,
        )
        _run(
            [
                "bash",
                "script/libero_runner.sh",
                "dataset",
                "--dataset-repo",
                LIBERO_DATASET_ID,
                "--data-root",
                str(data),
                "--dataset-download-dir",
                str(raw),
            ],
            cwd=repo_root,
        )
        norm_path = _compute_norm_stats(
            repo_root=repo_root,
            data_root=data,
            norm_root=norm,
            native_python=native_python,
        )
        _first_libero_observation(data)
        output = work / "output"
        shutil.copytree(data, output / "data" / "libero")
        (output / "norm_stats").mkdir(parents=True, exist_ok=True)
        shutil.copy2(norm_path, output / "norm_stats" / "norm_stats.json")
        _json_dump(
            output / "manifest.json",
            {
                "schema": "npa.dm05-opendm.prepare.v1",
                "opendm": {
                    "url": OPENDM_SOURCE_URL,
                    "revision": OPENDM_SOURCE_REVISION,
                },
                "dataset": {
                    "id": LIBERO_DATASET_ID,
                    "revision": LIBERO_DATASET_REVISION,
                },
                "normalization": normalization_contract(),
                "artifacts": _tree_summary(output),
            },
        )
        store.upload_tree(output, args.prepared_uri)


def train(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    assert_pinned_checkout(repo_root, OPENDM_SOURCE_REVISION, "OpenDM")
    store = ArtifactStore()
    with tempfile.TemporaryDirectory(prefix="npa-dm05-train-") as temporary:
        work = Path(temporary)
        prepared = store.download_tree(args.prepared_uri, work / "prepared")
        data = prepared / "data" / "libero"
        norm = prepared / "norm_stats"
        norm_stats = norm / "norm_stats.json"
        if not norm_stats.is_file():
            raise DM05WorkflowError("prepared stage has no norm_stats/norm_stats.json")
        _first_libero_observation(data)
        base_model = work / "base-model"
        _hf_download(
            repo_id=DM05_MODEL_ID,
            revision=DM05_MODEL_REVISION,
            repo_type="model",
            destination=base_model,
            cwd=repo_root,
        )
        checkpoint = work / "checkpoint"
        _run(
            [
                "bash",
                "script/libero_runner.sh",
                "train",
                "--data-root",
                str(data),
                "--nproc-per-node",
                str(args.nproc_per_node),
                "--",
                "--model-config.model-name-or-path",
                str(base_model),
                "--model-config.chunk-size",
                str(LIBERO_ACTION_CHUNK),
                "--data-config.norm-stats-root",
                str(norm),
                "--trainer-config.output-dir",
                str(checkpoint),
                "--trainer-config.num-train-steps",
                str(args.train_steps),
            ],
            cwd=repo_root,
        )
        if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
            raise DM05WorkflowError(
                "OpenDM training completed without a checkpoint directory"
            )
        shutil.copy2(norm_stats, checkpoint / "norm_stats.json")
        output = work / "output"
        shutil.copytree(checkpoint, output / "checkpoint")
        _json_dump(
            output / "manifest.json",
            {
                "schema": "npa.dm05-opendm.train.v1",
                "opendm": {
                    "url": OPENDM_SOURCE_URL,
                    "revision": OPENDM_SOURCE_REVISION,
                },
                "base_model": {"id": DM05_MODEL_ID, "revision": DM05_MODEL_REVISION},
                "prepared_uri": args.prepared_uri,
                "train_steps": int(args.train_steps),
                "nproc_per_node": int(args.nproc_per_node),
                "normalization": normalization_contract(),
                "artifacts": _tree_summary(output),
            },
        )
        store.upload_tree(output, args.checkpoint_uri)


def serve_rollout(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    assert_pinned_checkout(repo_root, OPENDM_SOURCE_REVISION, "OpenDM")
    native_python = _native_python(args.opendm_python)
    store = ArtifactStore()
    with tempfile.TemporaryDirectory(prefix="npa-dm05-serve-") as temporary:
        work = Path(temporary)
        prepared = store.download_tree(args.prepared_uri, work / "prepared")
        trained = store.download_tree(args.checkpoint_uri, work / "trained")
        checkpoint = trained / "checkpoint"
        if not (checkpoint / "norm_stats.json").is_file():
            raise DM05WorkflowError(
                "trained checkpoint lacks its exact norm_stats.json"
            )
        frame, images = _first_libero_observation(prepared / "data" / "libero")
        payload = _request_payload(frame, images)
        server = _start_server(
            repo_root=repo_root,
            checkpoint=checkpoint,
            native_python=native_python,
            port=args.port,
            cuda_visible_devices=args.server_cuda_visible_devices,
        )
        try:
            _wait_for_http_server(server, f"http://127.0.0.1:{args.port}/v1/infer")
            response = _infer_once(port=args.port, payload=payload)
        finally:
            _stop_owned_process(server)
        output = work / "output"
        _json_dump(
            output / "rollout.json",
            {
                "schema": "npa.dm05-opendm.http-rollout.v1",
                "checkpoint_uri": args.checkpoint_uri,
                "normalization": normalization_contract(),
                "request": {
                    "prompt": payload["observation"]["prompt"],
                    "camera_slots": ["1", "2"],
                },
                "response": response,
            },
        )
        _json_dump(
            output / "manifest.json",
            {
                "schema": "npa.dm05-opendm.serve-rollout.v1",
                "prepared_uri": args.prepared_uri,
                "checkpoint_uri": args.checkpoint_uri,
                "normalization": normalization_contract(),
                "artifacts": _tree_summary(output),
            },
        )
        store.upload_tree(output, args.rollout_uri)


def evaluate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    evaluator_root = Path(args.evaluator_root).resolve()
    assert_pinned_checkout(repo_root, OPENDM_SOURCE_REVISION, "OpenDM")
    assert_pinned_checkout(
        evaluator_root, DEXBOTIC_SOURCE_REVISION, "Dexbotic benchmark"
    )
    native_python = _native_python(args.opendm_python)
    evaluator_python = _native_python(args.evaluator_python)
    store = ArtifactStore()
    with tempfile.TemporaryDirectory(prefix="npa-dm05-evaluate-") as temporary:
        work = Path(temporary)
        prepared = store.download_tree(args.prepared_uri, work / "prepared")
        trained = store.download_tree(args.checkpoint_uri, work / "trained")
        rollout = store.download_tree(args.rollout_uri, work / "rollout")
        if not (rollout / "rollout.json").is_file():
            raise DM05WorkflowError("serve-rollout stage did not publish rollout.json")
        checkpoint = trained / "checkpoint"
        if not (checkpoint / "norm_stats.json").is_file():
            raise DM05WorkflowError(
                "trained checkpoint lacks its exact norm_stats.json"
            )
        _first_libero_observation(prepared / "data" / "libero")
        results = work / "evaluation"
        server = _start_server(
            repo_root=repo_root,
            checkpoint=checkpoint,
            native_python=native_python,
            port=args.port,
            cuda_visible_devices=args.server_cuda_visible_devices,
        )
        try:
            _wait_for_http_server(server, f"http://127.0.0.1:{args.port}/v1/infer")
            _run(
                [
                    evaluator_python,
                    "evaluation/run_libero_evaluation.py",
                    "--config",
                    "evaluation/configs/libero/example_dm05_libero.yaml",
                    "--set",
                    "base_url",
                    f"http://127.0.0.1:{args.port}",
                    "--set",
                    "api_style",
                    "v1",
                    "--set",
                    "output_dir",
                    str(results),
                    "--set",
                    "num_trails_per_task",
                    str(args.eval_trials),
                ],
                cwd=evaluator_root,
                env={
                    "EGL_PLATFORM": "device",
                    "PYOPENGL_PLATFORM": "egl",
                    "CUDA_VISIBLE_DEVICES": args.evaluator_cuda_visible_devices,
                },
            )
        finally:
            _stop_owned_process(server)
        results_file = results / "results.json"
        if not results_file.is_file():
            raise DM05WorkflowError(
                "upstream Dexbotic evaluation produced no results.json"
            )
        parsed = json.loads(results_file.read_text(encoding="utf-8"))
        for field in (
            "total_tasks",
            "total_episodes",
            "successful_episodes",
            "success_rate",
        ):
            if field not in parsed:
                raise DM05WorkflowError(f"Dexbotic results.json has no {field!r}")
        videos = sorted((results / "videos").glob("*.mp4"))
        if not videos:
            raise DM05WorkflowError(
                "Dexbotic evaluation produced no factual rollout MP4"
            )
        output = work / "output"
        shutil.copytree(results, output / "evaluation")
        _json_dump(
            output / "manifest.json",
            {
                "schema": "npa.dm05-opendm.closed-loop-evaluation.v1",
                "dexbotic": {
                    "url": DEXBOTIC_SOURCE_URL,
                    "revision": DEXBOTIC_SOURCE_REVISION,
                },
                "prepared_uri": args.prepared_uri,
                "checkpoint_uri": args.checkpoint_uri,
                "rollout_uri": args.rollout_uri,
                "eval_trials_per_task": int(args.eval_trials),
                "metrics": {
                    key: parsed[key]
                    for key in (
                        "total_tasks",
                        "total_episodes",
                        "successful_episodes",
                        "success_rate",
                    )
                },
                "rollout_mp4_count": len(videos),
                "normalization": normalization_contract(),
                "evaluator_observation_contract": evaluator_observation_contract(),
                "artifacts": _tree_summary(output),
            },
        )
        store.upload_tree(output, args.evaluation_uri)


def report(args: argparse.Namespace) -> None:
    store = ArtifactStore()
    with tempfile.TemporaryDirectory(prefix="npa-dm05-report-") as temporary:
        work = Path(temporary)
        prepared = store.download_tree(args.prepared_uri, work / "prepared")
        trained = store.download_tree(args.checkpoint_uri, work / "trained")
        rollout = store.download_tree(args.rollout_uri, work / "rollout")
        evaluation = store.download_tree(args.evaluation_uri, work / "evaluation")
        results_file = evaluation / "evaluation" / "results.json"
        if not results_file.is_file():
            raise DM05WorkflowError("evaluation artifact lacks upstream results.json")
        results = json.loads(results_file.read_text(encoding="utf-8"))
        videos = sorted((evaluation / "evaluation" / "videos").glob("*.mp4"))
        if not videos:
            raise DM05WorkflowError("evaluation artifact lacks factual MP4 rollouts")
        output = work / "output"
        output.mkdir()
        report_doc = {
            "schema": "npa.dm05-opendm.provenance.v1",
            "opendm": {"url": OPENDM_SOURCE_URL, "revision": OPENDM_SOURCE_REVISION},
            "base_model": {
                "id": DM05_MODEL_ID,
                "revision": DM05_MODEL_REVISION,
                "license": "Gemma",
            },
            "dataset": {"id": LIBERO_DATASET_ID, "revision": LIBERO_DATASET_REVISION},
            "dexbotic": {
                "url": DEXBOTIC_SOURCE_URL,
                "revision": DEXBOTIC_SOURCE_REVISION,
            },
            "inputs": {
                "prepared_uri": args.prepared_uri,
                "checkpoint_uri": args.checkpoint_uri,
                "rollout_uri": args.rollout_uri,
                "evaluation_uri": args.evaluation_uri,
            },
            "normalization": normalization_contract(),
            "evaluator_observation_contract": evaluator_observation_contract(),
            "metrics": {
                key: results[key]
                for key in (
                    "total_tasks",
                    "total_episodes",
                    "successful_episodes",
                    "success_rate",
                )
            },
            "claims": {
                "operational_smoke": False,
                "full_benchmark": False,
                "benchmark_scope": "Dexbotic LIBERO libero_spatial only",
                "convergence": False,
                "physical_robot_success": False,
            },
        }
        _json_dump(output / "provenance.json", report_doc)
        (output / "rollouts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(videos[0], output / "rollouts" / videos[0].name)
        _write_rrd(output / "dm05-opendm.rrd", report_doc, videos[0])
        # Retain exact upstream manifests in the report bundle for an independent
        # inspector without duplicating datasets or model weights.
        for name, source in {
            "prepared-manifest.json": prepared / "manifest.json",
            "trained-manifest.json": trained / "manifest.json",
            "rollout-manifest.json": rollout / "manifest.json",
            "evaluation-manifest.json": evaluation / "manifest.json",
        }.items():
            if not source.is_file():
                raise DM05WorkflowError(
                    f"missing required upstream-stage manifest: {source}"
                )
            (output / "stage-manifests").mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output / "stage-manifests" / name)
        _json_dump(
            output / "manifest.json",
            {
                "schema": "npa.dm05-opendm.report.v1",
                "rrd": "dm05-opendm.rrd",
                "mp4": f"rollouts/{videos[0].name}",
                "artifacts": _tree_summary(output),
            },
        )
        store.upload_tree(output, args.report_uri)


def _write_rrd(path: Path, provenance: dict[str, Any], video: Path) -> None:
    """Write a factual RRD containing measured evaluator metrics and source provenance."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - exercised in the runtime image
        raise DM05WorkflowError(
            "rerun-sdk is required to emit the requested factual RRD"
        ) from exc
    rr.init(
        "npa.dm05-opendm",
        recording_id=f"dm05-{hashlib.sha256(str(path).encode()).hexdigest()[:16]}",
    )
    rr.save(str(path))
    metrics = provenance["metrics"]
    rr.log("metrics/success_rate", rr.Scalars(float(metrics["success_rate"])))
    rr.log("metrics/total_episodes", rr.Scalars(float(metrics["total_episodes"])))
    rr.log(
        "metrics/successful_episodes", rr.Scalars(float(metrics["successful_episodes"]))
    )
    rr.log(
        "provenance",
        rr.TextDocument(
            json.dumps(provenance, sort_keys=True), media_type="application/json"
        ),
    )
    # The MP4 is kept alongside the RRD as the evaluator wrote it.  Its path and
    # checksum make the relationship inspectable without pretending it was made
    # by Rerun or re-encoding it into an unrelated visualization format.
    rr.log(
        "rollouts/reference",
        rr.TextDocument(
            json.dumps({"file": video.name, "sha256": _sha256(video)}, sort_keys=True),
            media_type="application/json",
        ),
    )
    rr.disconnect()
    if not path.is_file() or path.stat().st_size == 0:
        raise DM05WorkflowError("Rerun did not write an RRD")


def normalization_contract() -> dict[str, Any]:
    return {
        "dataset": LIBERO_DATASET_NAME,
        "robot_type": LIBERO_ROBOT_TYPE,
        "camera_keys": list(LIBERO_IMAGE_KEYS),
        "camera_prompts_in_order": list(LIBERO_IMAGE_PROMPTS),
        "state": {"dimension": LIBERO_STATE_DIM, "order": "six_joint_then_two_gripper"},
        "action": {"dimension": LIBERO_ACTION_DIM, "mode": "absolute"},
        "action_chunk": LIBERO_ACTION_CHUNK,
    }


def evaluator_observation_contract() -> dict[str, Any]:
    """Describe the pinned evaluator input without conflating it with training data.

    Dexbotic's LIBERO evaluator emits end-effector pose (position plus axis-angle)
    and two gripper values. OpenDM's ``dm05_libero`` registration records six
    joints plus two gripper values. Its published configuration sets
    ``add_state=False``; state remains required and shape-normalized at the HTTP
    boundary, but is not tokenized into this model's conditioning. This record
    makes that limited compatibility explicit rather than relabeling evaluator
    pose as the training state.
    """
    return {
        "source": "Dexbotic LIBERO evaluator",
        "dimension": LIBERO_STATE_DIM,
        "order": "eef_position_3_then_axis_angle_3_then_gripper_2",
        "camera_order": ["agentview_image", "robot0_eye_in_hand_image"],
        "http_api": "v1",
        "training_state_equivalent": False,
        "model_add_state": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo-root", required=True)

    prepare_parser = commands.add_parser("prepare")
    common(prepare_parser)
    prepare_parser.add_argument("--opendm-python", required=True)
    prepare_parser.add_argument("--prepared-uri", required=True)
    prepare_parser.set_defaults(handler=prepare)

    train_parser = commands.add_parser("train")
    common(train_parser)
    train_parser.add_argument("--prepared-uri", required=True)
    train_parser.add_argument("--checkpoint-uri", required=True)
    train_parser.add_argument("--nproc-per-node", type=int, required=True)
    train_parser.add_argument("--train-steps", type=int, required=True)
    train_parser.set_defaults(handler=train)

    serve_parser = commands.add_parser("serve-rollout")
    common(serve_parser)
    serve_parser.add_argument("--opendm-python", required=True)
    serve_parser.add_argument("--prepared-uri", required=True)
    serve_parser.add_argument("--checkpoint-uri", required=True)
    serve_parser.add_argument("--rollout-uri", required=True)
    serve_parser.add_argument("--port", type=int, required=True)
    serve_parser.add_argument("--server-cuda-visible-devices", default="0")
    serve_parser.set_defaults(handler=serve_rollout)

    evaluate_parser = commands.add_parser("evaluate")
    common(evaluate_parser)
    evaluate_parser.add_argument("--evaluator-root", required=True)
    evaluate_parser.add_argument("--opendm-python", required=True)
    evaluate_parser.add_argument("--evaluator-python", required=True)
    evaluate_parser.add_argument("--prepared-uri", required=True)
    evaluate_parser.add_argument("--checkpoint-uri", required=True)
    evaluate_parser.add_argument("--rollout-uri", required=True)
    evaluate_parser.add_argument("--evaluation-uri", required=True)
    evaluate_parser.add_argument("--eval-trials", type=int, required=True)
    evaluate_parser.add_argument("--port", type=int, required=True)
    evaluate_parser.add_argument("--server-cuda-visible-devices", default="0")
    evaluate_parser.add_argument("--evaluator-cuda-visible-devices", default="1")
    evaluate_parser.set_defaults(handler=evaluate)

    report_parser = commands.add_parser("report")
    report_parser.add_argument("--prepared-uri", required=True)
    report_parser.add_argument("--checkpoint-uri", required=True)
    report_parser.add_argument("--rollout-uri", required=True)
    report_parser.add_argument("--evaluation-uri", required=True)
    report_parser.add_argument("--report-uri", required=True)
    report_parser.set_defaults(handler=report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (
        DM05WorkflowError,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[dm05-opendm] failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
