"""Closed-loop, task-disjoint GR00T N1.7 LIBERO-X evaluation.

This module is deliberately separate from :mod:`npa.workflows.groot_learning`.
That workflow measures held-out action prediction only.  This one materializes
a hash-bound evaluation protocol, invokes Isaac-GR00T's native LIBERO rollout
entrypoint, retains its videos, measures matching open-loop action error, and
then emits an independently inspected Rerun recording.  It never manufactures
task identities from the LIBERO-X dataset card: an operator-supplied manifest
must prove that the simulator tasks are disjoint from the derivative's
documented 60-task training cohort.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from npa.workflows.groot_learning import (
    _download_prefix,
    _set_rerun_time,
)
from npa.workflows.groot_visualization import (
    GrootVisualizationError,
    _download,
    _head_artifact,
    _list_objects,
    _put_bytes,
    _put_json,
    _read_s3_json,
    _s3_client,
    _split_s3,
    inspect_rrd,
)


TRAINING_TASKS_SCHEMA = "npa.groot_libero_x.training_tasks.v1"
EVALUATION_TASKS_SCHEMA = "npa.groot_libero_x.evaluation_tasks.v1"
OBSERVED_TASKS_SCHEMA = "npa.groot_libero_x.observed_tasks.v1"
DATASET_SCHEMA = "npa.groot_libero_x.evaluation_dataset.v1"
PROTOCOL_SCHEMA = "npa.groot_libero_x.evaluation_protocol.v1"
ROLLOUT_SCHEMA = "npa.groot_libero_x.closed_loop_rollout.v1"
COMPARISON_SCHEMA = "npa.groot_libero_x.closed_loop_comparison.v1"
EVIDENCE_SCHEMA = "npa.groot_libero_x.closed_loop_evidence.v1"
RERUN_APPLICATION_ID = "npa_groot_libero_x_closed_loop"
RERUN_TIMELINE = "evaluation_task"
DERIVATIVE_REPO = "rohansiva/gr00t-libero-x"
DERIVATIVE_REVISION = "b9dfbdcce8da61950db4f34199fff30b286b8f13"
DERIVATIVE_LICENSE = "Apache-2.0"
DERIVATIVE_CARD = "https://huggingface.co/rohansiva/gr00t-libero-x"
BASELINE_REPO = "nvidia/GR00T-N1.7-LIBERO"
BASELINE_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
BASELINE_LICENSE = "NVIDIA Open Model License Agreement"
BASELINE_CARD = "https://huggingface.co/nvidia/GR00T-N1.7-LIBERO"
LIBERO_X_REPO = "meituan/LIBERO-X"
LIBERO_X_REVISION = "73053111f932d4dbaee995e3f06c2f42b3ad4adc"
LIBERO_X_LICENSE = "CC-BY-4.0"
LIBERO_X_CARD = "https://huggingface.co/datasets/meituan/LIBERO-X"
IMAGE_GROOT_REF = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
IMAGE_GROOT_REPOSITORY = "https://github.com/NVIDIA/Isaac-GR00T.git"
IMAGE_GROOT_LICENSE = "Apache-2.0"
LIBERO_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_LICENSE = "MIT"
LIBERO_X_EVALUATOR_REPOSITORY = "https://github.com/meituan/LIBERO-X.git"
LIBERO_X_EVALUATOR_REVISION = "f528726421c7211d8eb05fe48e9e5e2535ccc813"
LIBERO_X_EVALUATOR_LICENSE = "MIT"
LIBERO_X_EVALUATOR_SOURCE = "https://github.com/meituan/LIBERO-X"
RUNTIME_READY_SCHEMA = "npa.groot_libero_x.runtime_ready.v2"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LIBERO_X_BDDL_LEVELS = frozenset({"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"})
STRICT_HELD_OUT_MODE = "strict_held_out"
OBSERVED_PAIRED_MODE = "observed_paired_coverage_unknown"


# This overlay is appended to the private, runtime-fetched Apache-2.0 Isaac-GR00T
# checkout.  Its only job is to let GR00T's existing ``register_libero_envs``
# function register an operator-selected LIBERO-X BDDL task in every
# Sync/AsyncVectorEnv worker.  Keeping it in the native module avoids claiming
# that unmodified Isaac-GR00T already supports LIBERO-X, while preserving the
# upstream ``run_gr00t_sim_policy`` rollout entrypoint.
LIBERO_X_RUNTIME_OVERLAY = r"""

# NPA_GROOT_LIBERO_X_RUNTIME_OVERLAY_V1
def _npa_groot_libero_x_register_envs():
    import json as _npa_json
    import re as _npa_re
    from pathlib import Path as _NpaPath

    manifest_raw = os.environ.get("NPA_GROOT_LIBERO_X_REGISTRATION_MANIFEST", "")
    source_raw = os.environ.get("NPA_GROOT_LIBERO_X_SOURCE", "")
    if not manifest_raw and not source_raw:
        return
    if not manifest_raw or not source_raw:
        raise RuntimeError("LIBERO-X registration requires both source and manifest")
    source = _NpaPath(source_raw).resolve()
    manifest = _NpaPath(manifest_raw).resolve()
    allowed_root = (source / "libero" / "libero_x" / "bddl").resolve()
    if not allowed_root.is_dir() or not manifest.is_file():
        raise RuntimeError("LIBERO-X source or registration manifest is unavailable")
    payload = _npa_json.loads(manifest.read_text())
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "npa.groot_libero_x.registration.v1"
        or not isinstance(tasks, list)
    ):
        raise RuntimeError("LIBERO-X registration manifest is invalid")
    from libero.libero.utils.parse_bddl import parse_bddl_file as _npa_parse_bddl

    for task in tasks:
        if not isinstance(task, dict):
            raise RuntimeError("LIBERO-X registration task is invalid")
        native_env_name = str(task.get("native_env_name") or "")
        relative = str(task.get("libero_x_bddl_path") or "")
        parts = tuple(relative.split("/"))
        if (
            not _npa_re.fullmatch(r"libero_sim/npa_groot_libero_x_[a-f0-9]{32}", native_env_name)
            or len(parts) != 5
            or parts[:3] != ("libero", "libero_x", "bddl")
            or parts[3] not in {"LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4"}
            or not parts[4].endswith(".bddl")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise RuntimeError("LIBERO-X registration task has an unsafe environment or BDDL path")
        bddl_path = (source / relative).resolve()
        if allowed_root not in bddl_path.parents or not bddl_path.is_file():
            raise RuntimeError("LIBERO-X BDDL path escapes the reviewed evaluator source")
        parsed = _npa_parse_bddl(str(bddl_path))
        task_description = str(parsed.get("language") or "").strip()
        if not task_description:
            raise RuntimeError("LIBERO-X BDDL has no language instruction")
        if native_env_name in gym.registry:
            raise RuntimeError(f"LIBERO-X environment is already registered: {native_env_name}")
        register(
            id=native_env_name,
            entry_point="gr00t.eval.sim.LIBERO.libero_env:LiberoEnv",
            kwargs={
                "task_bddl_file": str(bddl_path),
                "task_description": task_description,
            },
        )


_npa_groot_libero_x_original_register_libero_envs = register_libero_envs


def register_libero_envs():
    _npa_groot_libero_x_original_register_libero_envs()
    _npa_groot_libero_x_register_envs()
"""


def _json_hash(payload: Mapping[str, Any]) -> str:
    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise GrootVisualizationError(f"missing required field: {key}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    parsed = str(value or "").strip().lower()
    if not SHA256.fullmatch(parsed):
        raise GrootVisualizationError(f"{field} must be a sha256 hex digest")
    return parsed


def _libero_x_bddl_path(payload: Mapping[str, Any], *, task_id: str) -> str:
    """Accept only a reviewed LIBERO-X BDDL path inside the evaluator tree."""

    raw = _require_string(payload, "libero_x_bddl_path")
    path = PurePosixPath(raw)
    parts = path.parts
    if (
        path.is_absolute()
        or len(parts) != 5
        or parts[:3] != ("libero", "libero_x", "bddl")
        or parts[3] not in LIBERO_X_BDDL_LEVELS
        or path.suffix != ".bddl"
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise GrootVisualizationError(
            f"evaluation task {task_id} has an unsafe LIBERO-X BDDL path"
        )
    return path.as_posix()


def _native_libero_x_env_name(task_id: str) -> str:
    """Map a logical task ID to a portable Gymnasium environment identifier."""

    digest = hashlib.sha256(task_id.encode()).hexdigest()[:32]
    return f"libero_sim/npa_groot_libero_x_{digest}"


def _task_rows(
    payload: Mapping[str, Any], *, training: bool, schema: str | None = None
) -> list[dict[str, Any]]:
    expected = schema or (
        TRAINING_TASKS_SCHEMA if training else EVALUATION_TASKS_SCHEMA
    )
    if payload.get("schema") != expected:
        raise GrootVisualizationError(f"expected manifest schema {expected}")
    rows = payload.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise GrootVisualizationError("task manifest requires a non-empty tasks list")
    parsed: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in rows:
        if not isinstance(item, dict):
            raise GrootVisualizationError("task entries must be mappings")
        task_id = _require_string(item, "task_id")
        if not TASK_ID.fullmatch(task_id):
            raise GrootVisualizationError(
                "task_id must be a safe logical identifier (letters, digits, dot, underscore, hyphen)"
            )
        if task_id in ids:
            raise GrootVisualizationError(f"duplicate task_id in manifest: {task_id}")
        ids.add(task_id)
        row = dict(item)
        row["task_id"] = task_id
        if not training:
            env_name = _require_string(row, "env_name")
            if env_name.startswith("libero_x/"):
                row["libero_x_bddl_path"] = _libero_x_bddl_path(row, task_id=task_id)
            elif not env_name.startswith("libero_sim/"):
                raise GrootVisualizationError(
                    f"evaluation task {task_id} is not a native LIBERO or LIBERO-X env"
                )
            trajectories = row.get("trajectory_ids")
            if not isinstance(trajectories, list) or not trajectories:
                raise GrootVisualizationError(
                    f"evaluation task {task_id} requires trajectory_ids"
                )
            if any(not isinstance(value, int) or value < 0 for value in trajectories):
                raise GrootVisualizationError(
                    f"evaluation task {task_id} has invalid trajectory_ids"
                )
        parsed.append(row)
    if training and len(parsed) != 60:
        raise GrootVisualizationError(
            "the derivative training manifest must identify its documented 60 tasks"
        )
    return parsed


def _protocol_upstream() -> dict[str, dict[str, str]]:
    """Return pinned source credit and terms shared by both protocol modes."""

    return {
        "derivative": {
            "repo": DERIVATIVE_REPO,
            "revision": DERIVATIVE_REVISION,
            "license": DERIVATIVE_LICENSE,
            "model_card": DERIVATIVE_CARD,
            "attribution": "rohansiva; fine-tuned from NVIDIA GR00T-N1.7-LIBERO libero_10",
        },
        "baseline": {
            "repo": BASELINE_REPO,
            "revision": BASELINE_REVISION,
            "license": BASELINE_LICENSE,
            "model_card": BASELINE_CARD,
            "attribution": "NVIDIA",
        },
        "libero_x": {
            "repo": LIBERO_X_REPO,
            "revision": LIBERO_X_REVISION,
            "license": LIBERO_X_LICENSE,
            "dataset_card": LIBERO_X_CARD,
            "attribution": "Meituan; Wang et al., LIBERO-X (2026)",
        },
        "isaac_groot": {
            "repository": IMAGE_GROOT_REPOSITORY,
            "revision": IMAGE_GROOT_REF,
            "license": IMAGE_GROOT_LICENSE,
            "attribution": "NVIDIA Isaac-GR00T",
        },
    }


def _protocol_claims(mode: str) -> dict[str, Any]:
    """Return immutable claim labels for strict or coverage-unknown evidence."""

    if mode == STRICT_HELD_OUT_MODE:
        return {
            "training_coverage": "documented_60_task_manifest",
            "task_disjointness": "verified",
            "labels": {"held_out": True, "generalization": False},
            "scope": "simulator closed-loop held-out evaluation only; no physical-robot or benchmark-convergence claim",
        }
    if mode == OBSERVED_PAIRED_MODE:
        return {
            "training_coverage": "unknown",
            "task_disjointness": "unverified",
            "labels": {"held_out": False, "generalization": False},
            "scope": "simulator paired observed-task comparison only; training coverage and task disjointness are unknown, so this is not a held-out or generalization result",
        }
    raise GrootVisualizationError(f"unsupported evaluation protocol mode: {mode}")


def _object_inventory(client: Any, uri: str) -> dict[str, Any]:
    ref = _split_s3(uri, require_key=False)
    prefix = ref.key.rstrip("/") + "/" if ref.key else ""
    rows = []
    for item in _list_objects(client, uri):
        key = str(item.get("key") or "")
        size = int(item.get("size") or 0)
        if not key.startswith(prefix) or size <= 0:
            continue
        rows.append(
            {"key": key[len(prefix) :], "bytes": size, "etag": item.get("etag", "")}
        )
    if not rows:
        raise GrootVisualizationError(f"evaluation dataset prefix is empty: {uri}")
    return {"uri": uri, "objects": rows, "sha256": _json_hash({"objects": rows})}


def _validate_dataset_manifest(
    client: Any, payload: Mapping[str, Any], *, evaluation_hash: str
) -> dict[str, Any]:
    if payload.get("schema") != DATASET_SCHEMA:
        raise GrootVisualizationError(
            f"expected dataset manifest schema {DATASET_SCHEMA}"
        )
    dataset = payload.get("source")
    materialized = payload.get("materialized")
    if not isinstance(dataset, dict) or not isinstance(materialized, dict):
        raise GrootVisualizationError(
            "dataset manifest requires source and materialized mappings"
        )
    if (
        dataset.get("repo") != LIBERO_X_REPO
        or dataset.get("revision") != LIBERO_X_REVISION
    ):
        raise GrootVisualizationError(
            "dataset manifest does not pin the LIBERO-X source revision"
        )
    if str(dataset.get("license") or "").upper() != LIBERO_X_LICENSE:
        raise GrootVisualizationError(
            "dataset manifest does not preserve LIBERO-X CC-BY-4.0"
        )
    if (
        _require_sha256(
            payload.get("evaluation_task_manifest_sha256"),
            field="evaluation_task_manifest_sha256",
        )
        != evaluation_hash
    ):
        raise GrootVisualizationError(
            "dataset and evaluation task manifests are not bound together"
        )
    dataset_uri = _require_string(materialized, "uri")
    inventory = _object_inventory(client, dataset_uri)
    declared = _require_sha256(
        materialized.get("object_inventory_sha256"), field="object_inventory_sha256"
    )
    if inventory["sha256"] != declared:
        raise GrootVisualizationError("evaluation dataset S3 object inventory changed")
    return {"source": dict(dataset), "materialized": inventory}


def prepare_evaluation(
    training_task_manifest_uri: str,
    evaluation_task_manifest_uri: str,
    evaluation_dataset_manifest_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Validate task disjointness and bind exact evaluation bytes to this run."""

    client = _s3_client(s3_client)
    training = _read_s3_json(client, training_task_manifest_uri)
    evaluation = _read_s3_json(client, evaluation_task_manifest_uri)
    train_rows = _task_rows(training, training=True)
    eval_rows = _task_rows(evaluation, training=False)
    train_ids = {row["task_id"] for row in train_rows}
    eval_ids = {row["task_id"] for row in eval_rows}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise GrootVisualizationError(
            "evaluation tasks overlap derivative training: " + ", ".join(overlap)
        )
    evaluation_hash = _json_hash(evaluation)
    dataset = _validate_dataset_manifest(
        client,
        _read_s3_json(client, evaluation_dataset_manifest_uri),
        evaluation_hash=evaluation_hash,
    )
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "status": "prepared",
        "run_id": run_id,
        "evaluation_mode": STRICT_HELD_OUT_MODE,
        "task_disjoint": True,
        "training_task_count": len(train_rows),
        "evaluation_task_count": len(eval_rows),
        "training_task_manifest": {
            "uri": training_task_manifest_uri,
            "sha256": _json_hash(training),
        },
        "evaluation_task_manifest": {
            "uri": evaluation_task_manifest_uri,
            "sha256": evaluation_hash,
        },
        "evaluation_tasks": eval_rows,
        "dataset": dataset,
        "claims": _protocol_claims(STRICT_HELD_OUT_MODE),
        "upstream": _protocol_upstream(),
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    _put_json(client, output_uri, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return protocol


def prepare_observed_paired_evaluation(
    observed_task_manifest_uri: str,
    evaluation_dataset_manifest_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Bind paired observed tasks without inventing derivative training coverage."""

    client = _s3_client(s3_client)
    observed = _read_s3_json(client, observed_task_manifest_uri)
    tasks = _task_rows(observed, training=False, schema=OBSERVED_TASKS_SCHEMA)
    task_hash = _json_hash(observed)
    dataset = _validate_dataset_manifest(
        client,
        _read_s3_json(client, evaluation_dataset_manifest_uri),
        evaluation_hash=task_hash,
    )
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "status": "prepared",
        "run_id": run_id,
        "evaluation_mode": OBSERVED_PAIRED_MODE,
        "evaluation_task_count": len(tasks),
        "observed_task_manifest": {
            "uri": observed_task_manifest_uri,
            "sha256": task_hash,
        },
        "evaluation_tasks": tasks,
        "dataset": dataset,
        "claims": _protocol_claims(OBSERVED_PAIRED_MODE),
        "upstream": _protocol_upstream(),
    }
    protocol["protocol_sha256"] = _json_hash(protocol)
    _put_json(client, output_uri, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return protocol


def _load_protocol(client: Any, uri: str, run_id: str) -> dict[str, Any]:
    protocol = _read_s3_json(client, uri)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "prepared"
    ):
        raise GrootVisualizationError("closed-loop protocol is not prepared")
    if protocol.get("run_id") != run_id:
        raise GrootVisualizationError(
            "closed-loop protocol does not belong to this run"
        )
    bound = str(protocol.get("protocol_sha256") or "")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    if bound != _json_hash(unsigned):
        raise GrootVisualizationError("closed-loop protocol hash is invalid")
    if not protocol.get("evaluation_tasks"):
        raise GrootVisualizationError(
            "closed-loop protocol contains no evaluation tasks"
        )
    mode = str(protocol.get("evaluation_mode") or STRICT_HELD_OUT_MODE)
    claims = protocol.get("claims")
    if not isinstance(claims, Mapping) or claims != _protocol_claims(mode):
        raise GrootVisualizationError(
            "closed-loop protocol claim classification is invalid"
        )
    if mode == STRICT_HELD_OUT_MODE and protocol.get("task_disjoint") is not True:
        raise GrootVisualizationError(
            "strict held-out protocol lacks task disjointness"
        )
    if mode == OBSERVED_PAIRED_MODE and "task_disjoint" in protocol:
        raise GrootVisualizationError(
            "observed paired protocol must not assert task disjointness"
        )
    return protocol


def _snapshot_model(
    repo: str, revision: str, subdir: str
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=repo, revision=revision)).resolve()
    model = (root / subdir).resolve() if subdir else root
    if root not in model.parents and model != root:
        raise GrootVisualizationError(
            "model subdirectory escapes its downloaded repository"
        )
    if not model.is_dir():
        raise GrootVisualizationError(
            f"model path is absent after exact download: {model}"
        )
    files = [item for item in model.rglob("*") if item.is_file()]
    weights = [item for item in files if item.suffix in {".bin", ".safetensors"}]
    if not weights:
        raise GrootVisualizationError("downloaded model has no checkpoint weight file")
    return model, {
        "repo": repo,
        "revision": revision,
        "subdir": subdir,
        "files": len(files),
        "weight_bytes": sum(item.stat().st_size for item in weights),
    }


def _upload_videos(
    client: Any, directory: Path, output_uri: str
) -> list[dict[str, Any]]:
    ref = _split_s3(output_uri, require_key=False)
    prefix = ref.key.rstrip("/")
    artifacts: list[dict[str, Any]] = []
    for source in sorted(directory.rglob("*.mp4")):
        relative = source.relative_to(directory).as_posix()
        key = "/".join(part for part in (prefix, relative) if part)
        uri = f"s3://{ref.bucket}/{key}"
        artifacts.append(
            _put_bytes(client, uri, source.read_bytes(), content_type="video/mp4")
        )
    if not artifacts:
        raise GrootVisualizationError("native LIBERO rollout produced no MP4 evidence")
    return artifacts


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    mse, mae = float(metrics.get("mse")), float(metrics.get("mae"))
    if not math.isfinite(mse) or not math.isfinite(mae) or mse < 0 or mae < 0:
        raise GrootVisualizationError(
            "open-loop action metrics are not finite non-negative values"
        )
    result = {
        "mse": mse,
        "mae": mae,
        "samples": int(metrics.get("sample_count") or 0),
        "forward_calls": int(metrics.get("forward_calls") or 0),
    }
    if result["samples"] < 1 or result["forward_calls"] < 1:
        raise GrootVisualizationError(
            "open-loop action metrics require at least one sample and forward call"
        )
    return result


def _runtime_cache_root(temporary_root: Path) -> Path:
    """Use the managed data mount when available, otherwise this run's temp root."""

    configured = os.environ.get("NPA_GROOT_LIBERO_X_RUNTIME_CACHE") or os.environ.get(
        "GROOT_DATA_MOUNT"
    )
    if not configured:
        return temporary_root / "runtime-cache"
    root = Path(configured).expanduser().resolve() / "runtime-fetch"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GrootVisualizationError(
            f"cannot create configured GR00T runtime cache: {root}"
        ) from exc
    return root


def _run_runtime_command(
    command: list[str], *, cwd: Path, env: Mapping[str, str]
) -> None:
    completed = subprocess.run(command, cwd=cwd, env=dict(env), check=False)
    if completed.returncode:
        raise GrootVisualizationError(
            "pinned native runtime setup failed: " + " ".join(command[:4])
        )


def _git_without_lfs(git: str, *arguments: str) -> list[str]:
    """Return a Git command that cannot invoke an unavailable LFS filter.

    ``GIT_LFS_SKIP_SMUDGE`` is insufficient when a repository's configured LFS
    filter executable is absent: Git can still try to start ``git-lfs`` during
    checkout.  The runtime only needs source code, never LFS-managed payloads,
    so disable that filter explicitly for clone and checkout.
    """

    return [
        git,
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        *arguments,
    ]


def _overlay_libero_env(source: Path) -> dict[str, str]:
    """Install the deterministic worker-side LIBERO-X registration bridge."""

    target = source / "gr00t/eval/sim/LIBERO/libero_env.py"
    try:
        original = target.read_text()
    except OSError as exc:
        raise GrootVisualizationError(
            "reviewed Isaac-GR00T checkout lacks its native LIBERO environment"
        ) from exc
    if "NPA_GROOT_LIBERO_X_RUNTIME_OVERLAY_V1" in original:
        raise GrootVisualizationError(
            "runtime source unexpectedly contains a pre-existing LIBERO-X overlay"
        )
    target.write_text(original.rstrip() + LIBERO_X_RUNTIME_OVERLAY + "\n")
    return {
        "target": "gr00t/eval/sim/LIBERO/libero_env.py",
        "sha256": hashlib.sha256(LIBERO_X_RUNTIME_OVERLAY.encode()).hexdigest(),
        "version": "v1",
    }


def _runtime_ready(target: Path) -> dict[str, Any] | None:
    marker = target / "runtime-ready.json"
    source = target / "Isaac-GR00T"
    libero_x_source = target / "LIBERO-X"
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != RUNTIME_READY_SCHEMA:
        return None
    if payload.get("isaac_groot_revision") != IMAGE_GROOT_REF:
        return None
    evaluator = payload.get("libero_x_evaluator")
    overlay = payload.get("npa_libero_x_overlay")
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("repository") != LIBERO_X_EVALUATOR_REPOSITORY
        or evaluator.get("revision") != LIBERO_X_EVALUATOR_REVISION
        or evaluator.get("license") != LIBERO_X_EVALUATOR_LICENSE
        or not isinstance(overlay, dict)
        or overlay.get("sha256")
        != hashlib.sha256(LIBERO_X_RUNTIME_OVERLAY.encode()).hexdigest()
    ):
        return None
    source_python = source / ".venv/bin/python"
    libero_python = source / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
    if not source_python.is_file() or not libero_python.is_file():
        return None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode or revision.stdout.strip() != IMAGE_GROOT_REF:
        return None
    evaluator_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=libero_x_source,
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        evaluator_revision.returncode
        or evaluator_revision.stdout.strip() != LIBERO_X_EVALUATOR_REVISION
        or not (libero_x_source / "libero/libero_x/bddl").is_dir()
    ):
        return None
    try:
        overlay_text = (source / "gr00t/eval/sim/LIBERO/libero_env.py").read_text()
    except OSError:
        return None
    if not overlay_text.endswith(LIBERO_X_RUNTIME_OVERLAY + "\n"):
        return None
    return {
        "root": target,
        "source": source,
        "server_python": source_python,
        "sim_python": libero_python,
        "libero_x_source": libero_x_source,
        "home": target / "home",
        "provenance": payload,
    }


def _materialize_native_runtime(temporary_root: Path) -> dict[str, Any]:
    """Fetch, verify, and cache current Apache-2.0 upstream code at runtime.

    The published bootstrap image bundles an older Isaac-GR00T revision.  A
    file lock and atomically published ready marker prevent one policy stage
    from observing another stage's partially built Python 3.12/LIBERO runtime.
    """

    cache_root = _runtime_cache_root(temporary_root)
    cache_identity = (
        f"isaac-groot-{IMAGE_GROOT_REF}-libero-x-{LIBERO_X_EVALUATOR_REVISION}"
    )
    target = cache_root / cache_identity
    lock_path = cache_root / f"{cache_identity}.lock"
    cache_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ready = _runtime_ready(target)
        if ready is not None:
            return ready
        if target.exists():
            raise GrootVisualizationError(
                "native Isaac-GR00T cache exists without a verified ready marker; refusing to overwrite it"
            )
        git = shutil.which("git")
        uv = shutil.which("uv")
        if not git or not uv:
            raise GrootVisualizationError(
                "native Isaac-GR00T runtime requires git and uv in the image"
            )
        build = cache_root / f".{cache_identity}.build-{os.getpid()}"
        if build.exists():
            raise GrootVisualizationError(
                f"native runtime build path already exists: {build}"
            )
        build.mkdir(parents=True)
        source = build / "Isaac-GR00T"
        libero_x_source = build / "LIBERO-X"
        home = build / "home"
        home.mkdir()
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_LFS_SKIP_SMUDGE": "1",
                "HOME": str(home),
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
            }
        )
        _run_runtime_command(
            _git_without_lfs(
                git,
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                IMAGE_GROOT_REPOSITORY,
                str(source),
            ),
            cwd=build,
            env=environment,
        )
        _run_runtime_command(
            _git_without_lfs(git, "checkout", "--detach", IMAGE_GROOT_REF),
            cwd=source,
            env=environment,
        )
        verified = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=source,
            check=False,
            capture_output=True,
            text=True,
        )
        if verified.returncode or verified.stdout.strip() != IMAGE_GROOT_REF:
            raise GrootVisualizationError(
                "runtime-fetched Isaac-GR00T revision differs from reviewed pin"
            )
        _run_runtime_command(
            _git_without_lfs(
                git,
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                LIBERO_X_EVALUATOR_REPOSITORY,
                str(libero_x_source),
            ),
            cwd=build,
            env=environment,
        )
        _run_runtime_command(
            _git_without_lfs(git, "checkout", "--detach", LIBERO_X_EVALUATOR_REVISION),
            cwd=libero_x_source,
            env=environment,
        )
        evaluator_verified = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=libero_x_source,
            check=False,
            capture_output=True,
            text=True,
        )
        if (
            evaluator_verified.returncode
            or evaluator_verified.stdout.strip() != LIBERO_X_EVALUATOR_REVISION
            or not (libero_x_source / "libero/libero_x/bddl").is_dir()
        ):
            raise GrootVisualizationError(
                "runtime-fetched LIBERO-X evaluator differs from reviewed source"
            )
        overlay = _overlay_libero_env(source)
        _run_runtime_command(
            [uv, "sync", "--python", "3.12"], cwd=source, env=environment
        )
        _run_runtime_command(
            ["bash", "gr00t/eval/sim/LIBERO/setup_libero.sh"],
            cwd=source,
            env=environment,
        )
        source_python = source / ".venv/bin/python"
        libero_python = source / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
        if not source_python.is_file() or not libero_python.is_file():
            raise GrootVisualizationError(
                "upstream LIBERO setup did not create both documented Python runtimes"
            )
        libero_revision = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=source / "external_dependencies/LIBERO",
            check=False,
            capture_output=True,
            text=True,
        )
        if libero_revision.returncode:
            raise GrootVisualizationError(
                "upstream LIBERO submodule was not materialized by native setup"
            )
        ready_marker = {
            "schema": RUNTIME_READY_SCHEMA,
            "delivery": "runtime-fetch",
            "repository": IMAGE_GROOT_REPOSITORY,
            "isaac_groot_revision": IMAGE_GROOT_REF,
            "libero_submodule_revision": libero_revision.stdout.strip(),
            "libero_x_evaluator": {
                "repository": LIBERO_X_EVALUATOR_REPOSITORY,
                "revision": LIBERO_X_EVALUATOR_REVISION,
                "license": LIBERO_X_EVALUATOR_LICENSE,
                "source": LIBERO_X_EVALUATOR_SOURCE,
                "attribution": "Meituan LIBERO-X authors; Wang et al. (2026)",
            },
            "npa_libero_x_overlay": overlay,
            "python": "3.12",
            "entrypoint": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
        }
        marker_temp = build / "runtime-ready.pending.json"
        marker_temp.write_text(
            json.dumps(ready_marker, indent=2, sort_keys=True) + "\n"
        )
        marker_temp.replace(build / "runtime-ready.json")
        build.replace(target)
        ready = _runtime_ready(target)
        if ready is None:  # pragma: no cover - defensive filesystem invariant
            raise GrootVisualizationError(
                "atomically materialized native runtime did not verify"
            )
        return ready


def _run_native_evaluator(
    *,
    runtime: Mapping[str, Any],
    root: Path,
    model_path: Path,
    dataset_path: Path,
    tasks: list[dict[str, Any]],
    episodes_per_task: int,
    n_envs: int,
    max_episode_steps: int,
    n_action_steps: int,
    seed: int,
) -> dict[str, Any]:
    runner = Path(__file__).with_name("groot_libero_x_native.py")
    if not runner.is_file():
        raise GrootVisualizationError(
            "native LIBERO evaluator module is absent from source overlay"
        )
    config_path = root / "native-evaluator-config.json"
    result_path = root / "native-evaluator-result.json"
    libero_x_tasks = [
        {
            "task_id": task["task_id"],
            "native_env_name": _native_libero_x_env_name(task["task_id"]),
            "libero_x_bddl_path": task["libero_x_bddl_path"],
        }
        for task in tasks
        if str(task.get("env_name") or "").startswith("libero_x/")
    ]
    registration_manifest = root / "libero-x-registration.json"
    if libero_x_tasks:
        registration_manifest.write_text(
            json.dumps(
                {
                    "schema": "npa.groot_libero_x.registration.v1",
                    "tasks": libero_x_tasks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    config = {
        "server_python": str(runtime["server_python"]),
        "server_script": str(
            Path(runtime["source"]) / "gr00t/eval/run_gr00t_server.py"
        ),
        "server_log": str(root / "native-server.log"),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "evaluation_tasks": tasks,
        "episodes_per_task": episodes_per_task,
        "n_envs": n_envs,
        "max_episode_steps": max_episode_steps,
        "n_action_steps": n_action_steps,
        "seed": seed,
        "video_dir": str(root / "videos"),
        "plot_dir": str(root / "open-loop-plots"),
        "libero_x_source": str(runtime["libero_x_source"]) if libero_x_tasks else "",
        "libero_x_registration_manifest": str(registration_manifest)
        if libero_x_tasks
        else "",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(runtime["home"]),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    if libero_x_tasks:
        environment.update(
            {
                "NPA_GROOT_LIBERO_X_SOURCE": str(runtime["libero_x_source"]),
                "NPA_GROOT_LIBERO_X_REGISTRATION_MANIFEST": str(registration_manifest),
            }
        )
    _run_runtime_command(
        [
            str(runtime["sim_python"]),
            str(runner),
            "--config",
            str(config_path),
            "--output",
            str(result_path),
        ],
        cwd=root,
        env=environment,
    )
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GrootVisualizationError(
            "native LIBERO evaluator did not emit a readable result"
        ) from exc
    if not isinstance(result, dict) or not isinstance(result.get("tasks"), list):
        raise GrootVisualizationError(
            "native LIBERO evaluator emitted an invalid result"
        )
    return result


def run_policy(
    protocol_uri: str,
    output_uri: str,
    rollout_media_uri: str,
    run_id: str,
    *,
    policy_name: str,
    model_repo: str,
    model_revision: str,
    model_subdir: str,
    episodes_per_task: int,
    n_envs: int,
    max_episode_steps: int,
    n_action_steps: int,
    seed: int,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Run real GR00T open-loop forwards and native LIBERO closed-loop rollouts."""

    expected = {
        "baseline": (BASELINE_REPO, BASELINE_REVISION, BASELINE_LICENSE, BASELINE_CARD),
        "derivative": (
            DERIVATIVE_REPO,
            DERIVATIVE_REVISION,
            DERIVATIVE_LICENSE,
            DERIVATIVE_CARD,
        ),
    }
    if (
        policy_name not in expected
        or (model_repo, model_revision) != expected[policy_name][:2]
    ):
        raise GrootVisualizationError(
            "policy identity differs from this evaluation contract"
        )
    if min(episodes_per_task, n_envs, max_episode_steps, n_action_steps) < 1:
        raise GrootVisualizationError("rollout settings must be positive")
    client = _s3_client(s3_client)
    protocol = _load_protocol(client, protocol_uri, run_id)
    bootstrap_ref = os.environ.get("GROOT_REPO_REF", "").strip()
    with tempfile.TemporaryDirectory(prefix="npa-groot-libero-x-") as temporary:
        root = Path(temporary)
        model, model_identity = _snapshot_model(
            model_repo, model_revision, model_subdir
        )
        model_identity.update(
            {
                "license": expected[policy_name][2],
                "model_card": expected[policy_name][3],
                "attribution": "NVIDIA" if policy_name == "baseline" else "rohansiva",
            }
        )
        dataset = root / "evaluation-dataset"
        _download_prefix(client, protocol["dataset"]["materialized"]["uri"], dataset)
        runtime = _materialize_native_runtime(root)
        native = _run_native_evaluator(
            runtime=runtime,
            root=root,
            model_path=model,
            dataset_path=dataset,
            tasks=protocol["evaluation_tasks"],
            episodes_per_task=episodes_per_task,
            n_envs=n_envs,
            max_episode_steps=max_episode_steps,
            n_action_steps=n_action_steps,
            seed=seed,
        )
        action = _numeric_metrics(native.get("open_loop_action_error", {}))
        task_rows: list[dict[str, Any]] = []
        native_tasks = native.get("tasks")
        if not isinstance(native_tasks, list) or len(native_tasks) != len(
            protocol["evaluation_tasks"]
        ):
            raise GrootVisualizationError(
                "native LIBERO evaluator task count differs from prepared protocol"
            )
        video_root = (root / "videos").resolve()
        for expected_task, native_task in zip(
            protocol["evaluation_tasks"], native_tasks, strict=True
        ):
            if not isinstance(native_task, dict):
                raise GrootVisualizationError(
                    "native LIBERO evaluator task result is not an object"
                )
            if (
                native_task.get("task_id") != expected_task["task_id"]
                or native_task.get("env_name") != expected_task["env_name"]
                or native_task.get("trajectory_ids") != expected_task["trajectory_ids"]
            ):
                raise GrootVisualizationError(
                    "native LIBERO evaluator did not preserve the prepared task"
                )
            video_dir = Path(str(native_task.get("video_dir") or "")).resolve()
            if video_root not in video_dir.parents:
                raise GrootVisualizationError(
                    "native LIBERO evaluator video directory escapes run workspace"
                )
            artifacts = _upload_videos(
                client,
                video_dir,
                f"{rollout_media_uri.rstrip('/')}/{expected_task['task_id']}",
            )
            task_rows.append(
                {key: value for key, value in native_task.items() if key != "video_dir"}
                | {"videos": artifacts}
            )
    completed = sum(int(row["completed_episodes"]) for row in task_rows)
    successes = sum(sum(row["successes"]) for row in task_rows)
    result = {
        "schema": ROLLOUT_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "closed_loop_verified": True,
        "policy": policy_name,
        "protocol_uri": protocol_uri,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_mode": protocol["evaluation_mode"],
        "claims": protocol["claims"],
        "model": model_identity,
        "runtime": {
            "bootstrap_image_groot_source_revision": bootstrap_ref or "unreported",
            "native_runtime": runtime["provenance"],
            "native_entrypoint": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
        },
        "open_loop_action_error": action,
        "closed_loop": {
            "native_entrypoint": "gr00t.eval.rollout_policy.run_gr00t_sim_policy",
            "task_count": len(task_rows),
            "completed_episodes": completed,
            "successful_episodes": successes,
            "success_rate": successes / completed,
            "tasks": task_rows,
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_rollout(client: Any, uri: str, run_id: str, policy: str) -> dict[str, Any]:
    report = _read_s3_json(client, uri)
    if report.get("schema") != ROLLOUT_SCHEMA or report.get("status") != "completed":
        raise GrootVisualizationError("rollout report is incomplete")
    if report.get("run_id") != run_id or report.get("policy") != policy:
        raise GrootVisualizationError("rollout report identity mismatch")
    if report.get("closed_loop_verified") is not True or not report.get(
        "closed_loop", {}
    ).get("tasks"):
        raise GrootVisualizationError(
            "rollout does not contain verified closed-loop evidence"
        )
    return report


def compare_closed_loop(
    protocol_uri: str,
    baseline_uri: str,
    derivative_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Compare matched actual episodes and action errors without a proxy score."""

    client = _s3_client(s3_client)
    protocol = _load_protocol(client, protocol_uri, run_id)
    baseline = _load_rollout(client, baseline_uri, run_id, "baseline")
    derivative = _load_rollout(client, derivative_uri, run_id, "derivative")
    if {baseline["protocol_sha256"], derivative["protocol_sha256"]} != {
        protocol["protocol_sha256"]
    }:
        raise GrootVisualizationError("policies did not use the same prepared protocol")
    rows = {}
    for report in (baseline, derivative):
        if (
            report.get("evaluation_mode") != protocol["evaluation_mode"]
            or report.get("claims") != protocol["claims"]
        ):
            raise GrootVisualizationError(
                "rollout claim classification differs from prepared protocol"
            )
        names = [task["task_id"] for task in report["closed_loop"]["tasks"]]
        if names != [task["task_id"] for task in protocol["evaluation_tasks"]]:
            raise GrootVisualizationError(
                "rollout task ordering differs from prepared protocol"
            )
        rows[report["policy"]] = report
    baseline_metrics = baseline["closed_loop"]
    derivative_metrics = derivative["closed_loop"]
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "protocol_uri": protocol_uri,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_mode": protocol["evaluation_mode"],
        "claims": protocol["claims"],
        "baseline_uri": baseline_uri,
        "derivative_uri": derivative_uri,
        "closed_loop_success": {
            "baseline": {
                key: baseline_metrics[key]
                for key in ("completed_episodes", "successful_episodes", "success_rate")
            },
            "derivative": {
                key: derivative_metrics[key]
                for key in ("completed_episodes", "successful_episodes", "success_rate")
            },
            "derivative_minus_baseline": derivative_metrics["success_rate"]
            - baseline_metrics["success_rate"],
        },
        "open_loop_action_error": {
            "baseline": baseline["open_loop_action_error"],
            "derivative": derivative["open_loop_action_error"],
            "derivative_minus_baseline": {
                key: derivative["open_loop_action_error"][key]
                - baseline["open_loop_action_error"][key]
                for key in ("mse", "mae")
            },
        },
        "claim_scope": protocol["claims"]["scope"],
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _decode_rollout_video(video_module: Any, path: Path, uri: str) -> dict[str, Any]:
    """Return factual decode metadata for one native rollout MP4."""

    try:
        with video_module.open(str(path)) as container:
            if not container.streams.video:
                raise GrootVisualizationError(
                    f"native rollout has no video stream: {uri}"
                )
            stream = container.streams.video[0]
            frame_count = sum(1 for _ in container.decode(stream))
            width, height = int(stream.width), int(stream.height)
            codec = str(stream.codec_context.name or "")
    except video_module.FFmpegError as exc:
        raise GrootVisualizationError(
            f"native rollout MP4 is unreadable: {uri}"
        ) from exc
    if frame_count < 1 or width < 1 or height < 1:
        raise GrootVisualizationError(
            f"native rollout MP4 has no decodable frames: {uri}"
        )
    return {
        "uri": uri,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "codec": codec,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def _inspect_rollout_videos(
    client: Any, videos: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Download, hash, and decode each native rollout MP4 before evidence publication."""

    try:
        import av
    except ModuleNotFoundError as exc:  # pragma: no cover - renderer installs it
        raise GrootVisualizationError(
            "PyAV is required to inspect native LIBERO rollout MP4 evidence"
        ) from exc
    inspections: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="npa-groot-libero-x-mp4-") as temporary:
        for index, item in enumerate(videos):
            uri = _require_string(item, "uri")
            expected_hash = _require_sha256(item.get("sha256"), field="video sha256")
            path = Path(temporary) / f"rollout-{index}.mp4"
            _download(client, uri, path)
            inspection = _decode_rollout_video(av, path, uri)
            if inspection["sha256"] != expected_hash:
                raise GrootVisualizationError(
                    f"native rollout MP4 changed after upload: {uri}"
                )
            inspections.append(inspection)
    return inspections


def emit_evidence(
    comparison_uri: str,
    baseline_uri: str,
    derivative_uri: str,
    rrd_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Write and independently inspect RRD metrics linked to native MP4 rollouts."""

    import rerun as rr

    client = _s3_client(s3_client)
    comparison = _read_s3_json(client, comparison_uri)
    if (
        comparison.get("schema") != COMPARISON_SCHEMA
        or comparison.get("run_id") != run_id
    ):
        raise GrootVisualizationError("comparison report does not belong to this run")
    mode = str(comparison.get("evaluation_mode") or "")
    claims = comparison.get("claims")
    if not isinstance(claims, Mapping) or claims != _protocol_claims(mode):
        raise GrootVisualizationError("comparison claim classification is invalid")
    baseline = _load_rollout(client, baseline_uri, run_id, "baseline")
    derivative = _load_rollout(client, derivative_uri, run_id, "derivative")
    expected_entities = [
        "provenance",
        "metrics/success_rate/baseline",
        "metrics/success_rate/derivative",
        "metrics/action_mse/baseline",
        "metrics/action_mse/derivative",
    ]
    with tempfile.TemporaryDirectory(prefix="npa-groot-libero-x-rrd-") as temporary:
        rrd_path = Path(temporary) / "groot-libero-x.rrd"
        recording = rr.RecordingStream(RERUN_APPLICATION_ID, recording_id=run_id)
        rr.save(rrd_path, recording=recording)
        rr.log(
            "provenance",
            rr.TextDocument(
                json.dumps(
                    {
                        "comparison_uri": comparison_uri,
                        "protocol_sha256": comparison["protocol_sha256"],
                        "claims": claims,
                        "scope": comparison["claim_scope"],
                    },
                    sort_keys=True,
                )
            ),
            static=True,
            recording=recording,
        )
        for policy, report in (("baseline", baseline), ("derivative", derivative)):
            metrics = report["closed_loop"]
            _set_rerun_time(rr, recording, RERUN_TIMELINE, 0.0)
            rr.log(
                f"metrics/success_rate/{policy}",
                rr.Scalars(float(metrics["success_rate"])),
                recording=recording,
            )
            rr.log(
                f"metrics/action_mse/{policy}",
                rr.Scalars(float(report["open_loop_action_error"]["mse"])),
                recording=recording,
            )
            rr.log(
                f"metrics/action_mae/{policy}",
                rr.Scalars(float(report["open_loop_action_error"]["mae"])),
                recording=recording,
            )
            for index, task in enumerate(metrics["tasks"]):
                _set_rerun_time(rr, recording, RERUN_TIMELINE, float(index + 1))
                rr.log(
                    f"rollouts/{policy}/task_{index}/success_rate",
                    rr.Scalars(float(task["success_rate"])),
                    recording=recording,
                )
        recording.flush(timeout_sec=60.0)
        recording.disconnect()
        inspection = inspect_rrd(
            rrd_path,
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
            expected_entities=expected_entities,
            timeline=RERUN_TIMELINE,
        )
        rrd_artifact = _put_bytes(client, rrd_uri, rrd_path.read_bytes())
    videos = [
        video
        for report in (baseline, derivative)
        for task in report["closed_loop"]["tasks"]
        for video in task["videos"]
    ]
    for item in videos:
        _head_artifact(client, item["uri"])
    video_inspections = _inspect_rollout_videos(client, videos)
    result = {
        "schema": EVIDENCE_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "comparison_uri": comparison_uri,
        "evaluation_mode": mode,
        "claims": claims,
        "rrd": {**rrd_artifact, "inspect": inspection},
        "native_rollout_mp4s": videos,
        "native_rollout_mp4_inspection": video_inspections,
        "mp4_count": len(videos),
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-evaluation")
    prepare.add_argument("--training-task-manifest-uri", required=True)
    prepare.add_argument("--evaluation-task-manifest-uri", required=True)
    prepare.add_argument("--evaluation-dataset-manifest-uri", required=True)
    prepare.add_argument("--output-uri", required=True)
    prepare.add_argument("--run-id", required=True)
    observed = commands.add_parser("prepare-observed-paired-evaluation")
    observed.add_argument("--observed-task-manifest-uri", required=True)
    observed.add_argument("--evaluation-dataset-manifest-uri", required=True)
    observed.add_argument("--output-uri", required=True)
    observed.add_argument("--run-id", required=True)
    policy = commands.add_parser("run-policy")
    policy.add_argument("--protocol-uri", required=True)
    policy.add_argument("--output-uri", required=True)
    policy.add_argument("--rollout-media-uri", required=True)
    policy.add_argument("--run-id", required=True)
    policy.add_argument(
        "--policy-name", choices=("baseline", "derivative"), required=True
    )
    policy.add_argument("--model-repo", required=True)
    policy.add_argument("--model-revision", required=True)
    policy.add_argument("--model-subdir", default="")
    policy.add_argument("--episodes-per-task", type=int, required=True)
    policy.add_argument("--n-envs", type=int, required=True)
    policy.add_argument("--max-episode-steps", type=int, required=True)
    policy.add_argument("--n-action-steps", type=int, required=True)
    policy.add_argument("--seed", type=int, required=True)
    compare = commands.add_parser("compare-closed-loop")
    compare.add_argument("--protocol-uri", required=True)
    compare.add_argument("--baseline-uri", required=True)
    compare.add_argument("--derivative-uri", required=True)
    compare.add_argument("--output-uri", required=True)
    compare.add_argument("--run-id", required=True)
    evidence = commands.add_parser("emit-evidence")
    evidence.add_argument("--comparison-uri", required=True)
    evidence.add_argument("--baseline-uri", required=True)
    evidence.add_argument("--derivative-uri", required=True)
    evidence.add_argument("--rrd-uri", required=True)
    evidence.add_argument("--output-uri", required=True)
    evidence.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = vars(build_parser().parse_args(argv)).copy()
    command = values.pop("command")
    if command == "prepare-evaluation":
        prepare_evaluation(**values)
    elif command == "prepare-observed-paired-evaluation":
        prepare_observed_paired_evaluation(**values)
    elif command == "run-policy":
        run_policy(**values)
    elif command == "compare-closed-loop":
        compare_closed_loop(**values)
    elif command == "emit-evidence":
        emit_evidence(**values)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
