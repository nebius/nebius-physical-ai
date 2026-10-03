"""Executable OpenVLA-OFT / LIBERO workflow stages.

This module deliberately targets the OFT fork, not the stock OpenVLA decoder.
OFT checkpoints contain an LoRA adapter plus a continuous action head and (for
LIBERO) an eight-dimensional proprioception projector. A manifest from this
module is the portable contract consumed by sibling LIBERO-Plus, Sylvest, and
LoRAFleet work: it proves all component families and binds upstream identities.

The image contains neither OFT source nor model/checkpoint/data bytes.
``bootstrap-runtime`` fetches the pinned public source into an operator-owned
writable cache; stages publish hash-bound local or S3 bundles. No licence
acceptance variable is invented here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from npa.clients.storage import StorageClient
from npa.workbench.cosmos.policy_artifacts import (
    file_digest,
    materialize_bundle,
    policy_workspace,
    publish_bundle,
    write_local_json,
)

SOURCE_REPOSITORY = "https://github.com/moojink/openvla-oft.git"
SOURCE_REVISION = "e4287e94541f459edc4feabc4e181f537cd569a8"
SOURCE_LICENSE = "MIT"
LIBERO_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_LICENSE = "MIT"
DEFAULT_MODEL_ID = "openvla/openvla-7b"
# Hub revision probed from the model API on 2026-10-03. This is the repository
# snapshot, not the README blob digest.
MODEL_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
UPSTREAM_FINETUNE_SCRIPT = "vla-scripts/finetune.py"
UPSTREAM_LIBERO_EVAL_SCRIPT = "experiments/robot/libero/run_libero_eval.py"

DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_STEPS = 150_005
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_LORA_RANK = 32
DEFAULT_SEED = 7
DEFAULT_TRIALS = 50

# The pinned OFT project declares this direct Git dependency without a revision.
# Its current upstream repository does not carry a license file.  Do not resolve
# a moving, unlicensed runtime dependency merely because the top-level OFT source
# is MIT licensed.  The runtime can resume after the upstream owners publish an
# authoritative license/permission for a specific revision.
_UNLICENSED_RUNTIME_DEPENDENCY = "moojink/dlimp_openvla"

PREPARE_SCHEMA = "npa.workbench.openvla-oft.prepare.v1"
TRAIN_SCHEMA = "npa.workbench.openvla-oft.train.v1"
ROLLOUT_SCHEMA = "npa.workbench.openvla-oft.rollout.v1"
EVALUATION_SCHEMA = "npa.workbench.openvla-oft.evaluation.v1"
COMPARISON_SCHEMA = "npa.workbench.openvla-oft.comparison.v1"

# From the official OFT LIBERO guide. These identities are intentionally not
# interchangeable across sibling capabilities.
OFFICIAL_SUITE_CHECKPOINTS: dict[str, dict[str, str]] = {
    "libero_spatial": {
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "revision": "6d0231af0e48c5985f1ff86908f4674b84bc049b",
        "component_step": "150000",
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
        "lora_adapter": "lora_adapter",
    },
    "libero_object": {
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-object",
        "revision": "4c89574e1c538b6c102f43f0526d60a9d3650148",
        "component_step": "150000",
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
        "lora_adapter": "lora_adapter",
    },
    "libero_goal": {
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-goal",
        "revision": "c2d0f9fbbd82674683b397ff923168a12f6a307b",
        "component_step": "50000",
        "action_head": "action_head--50000_checkpoint.pt",
        "proprio_projector": "proprio_projector--50000_checkpoint.pt",
        "lora_adapter": "lora_adapter",
    },
    "libero_10": {
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-10",
        "revision": "95220f9a3421a7ff12d4218e73d09ade830fa9a3",
        "component_step": "150000",
        "action_head": "action_head--150000_checkpoint.pt",
        "proprio_projector": "proprio_projector--150000_checkpoint.pt",
        "lora_adapter": "lora_adapter",
    },
    "libero_90": {
        "repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "revision": "638918f3d1c2e43a39a8a20772bdb8b91835e4b7",
        "component_step": "300000",
        "action_head": "action_head--300000_checkpoint.pt",
        "proprio_projector": "proprio_projector--300000_checkpoint.pt",
        "lora_adapter": "lora_adapter",
    },
}

_HF_REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HF_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_OVERALL_RESULT = re.compile(
    r"Overall success rate:\s*(?P<rate>[0-9.]+)\s*\((?P<pct>[0-9.]+)%\)"
)
_TOTAL_EPISODES = re.compile(r"Total episodes:\s*(?P<count>\d+)")
_TOTAL_SUCCESSES = re.compile(r"Total successes:\s*(?P<count>\d+)")


class OpenVLAPipelineError(RuntimeError):
    """Raised for an OFT component, provenance, or execution invariant."""


def _require(value: str, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise OpenVLAPipelineError(f"{name} must not be empty")
    return result


def _require_hf_repo(value: str, name: str) -> str:
    result = _require(value, name)
    if not _HF_REPO_ID.fullmatch(result):
        raise OpenVLAPipelineError(f"{name} must be a Hugging Face owner/name id")
    return result


def _require_hf_commit(value: str, name: str) -> str:
    result = _require(value, name)
    if not _HF_COMMIT.fullmatch(result):
        raise OpenVLAPipelineError(f"{name} must be an immutable 40-character Hub commit")
    return result


def _copy_or_download_tree(uri: str, target: Path) -> Path:
    """Materialize an exact local/S3 tree without trusting object paths."""
    source = _require(uri, "URI")
    target.mkdir(parents=True, exist_ok=True)
    if source.startswith("s3://"):
        StorageClient.from_environment().download_directory(source, str(target))
        return target
    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise OpenVLAPipelineError(
            f"dataset/checkpoint directory does not exist: {source}"
        )
    shutil.copytree(source_path, target, dirs_exist_ok=True)
    return target


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    files = [item for item in sorted(root.rglob("*")) if item.is_file()]
    if not files:
        raise OpenVLAPipelineError(f"no files found under {root}")
    return [
        {
            "path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_digest(item),
        }
        for item in files
    ]


def _manifest_digest(report: dict[str, Any]) -> str:
    """Bind provenance to the verified manifest content, for local and S3 inputs."""
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dataset_statistics(root: Path) -> dict[str, Any] | None:
    candidates = sorted(root.rglob("dataset_statistics.json"))
    if not candidates:
        return None
    try:
        loaded = json.loads(candidates[0].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenVLAPipelineError("dataset_statistics.json is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise OpenVLAPipelineError("dataset_statistics.json must be a JSON object")
    return loaded


def _upstream_command(
    runtime: Path, script: str, arguments: Sequence[str]
) -> list[str]:
    python = runtime / "venv" / "bin" / "python"
    path = runtime / "source" / script
    if not python.is_file() or not path.is_file():
        raise OpenVLAPipelineError("pinned OFT runtime is not bootstrapped")
    return [str(python), str(path), *arguments]


def _upstream_torchrun_command(
    runtime: Path, script: str, processes: int, arguments: Sequence[str]
) -> list[str]:
    """Use the upstream single-node torchrun shape rather than a Python shim."""
    torchrun = runtime / "venv" / "bin" / "torchrun"
    path = runtime / "source" / script
    if not torchrun.is_file() or not path.is_file():
        raise OpenVLAPipelineError("pinned OFT runtime is not bootstrapped")
    return [
        str(torchrun),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={processes}",
        str(path),
        *arguments,
    ]


@contextmanager
def _runtime_cache_lock(root: Path):
    """Serialize a cache identity while preserving atomic ready-marker semantics."""
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run(command: Sequence[str], *, cwd: Path, log: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(cwd) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["WANDB_MODE"] = "disabled"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise OpenVLAPipelineError(
            f"upstream OFT command failed with exit code {completed.returncode}; see {log.name}"
        )


def _runtime_provenance() -> dict[str, dict[str, str]]:
    return {
        "oft": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
        },
        "libero": {
            "repository": LIBERO_REPOSITORY,
            "revision": LIBERO_REVISION,
            "license": LIBERO_LICENSE,
        },
    }


def _runtime_identity(runtime: Path) -> dict[str, Any]:
    ready = runtime / "ready.json"
    try:
        payload = json.loads(ready.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenVLAPipelineError("OFT runtime ready marker is unreadable") from exc
    if payload.get("status") != "ready" or not payload.get(
        "dependency_inventory_sha256"
    ):
        raise OpenVLAPipelineError("OFT runtime ready marker lacks dependency provenance")
    return payload


def bootstrap_runtime(runtime_root: str) -> Path:
    """Fetch/install OFT after its direct-dependency terms are resolved.

    The ready marker is written only after the complete source/runtime install.
    """
    _reject_unlicensed_runtime_dependency()
    root = Path(_require(runtime_root, "runtime_root")).expanduser().resolve()
    with _runtime_cache_lock(root):
        return _bootstrap_runtime_unlocked(root)


def _bootstrap_runtime_unlocked(root: Path) -> Path:
    ready, source, libero, venv, inventory = (
        root / "ready.json",
        root / "source",
        root / "libero",
        root / "venv",
        root / "runtime-dependencies.txt",
    )
    if ready.is_file():
        payload = json.loads(ready.read_text())
        if (
            payload.get("source_revision") == SOURCE_REVISION
            and payload.get("libero_revision") == LIBERO_REVISION
            and (source / UPSTREAM_FINETUNE_SCRIPT).is_file()
            and (libero / "libero").is_dir()
            and (venv / "bin" / "python").is_file()
            and inventory.is_file()
            and payload.get("dependency_inventory_sha256") == file_digest(inventory)
        ):
            return root
        raise OpenVLAPipelineError("OFT runtime cache identity is stale or incomplete")
    if root.exists() and any(root.iterdir()):
        raise OpenVLAPipelineError(
            "OFT runtime cache exists without an atomic ready marker"
        )
    root.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                SOURCE_REPOSITORY,
                str(source),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", SOURCE_REVISION],
            check=True,
        )
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != SOURCE_REVISION:
            raise OpenVLAPipelineError(
                "downloaded OFT source revision does not match contract"
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                LIBERO_REPOSITORY,
                str(libero),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(libero), "checkout", "--detach", LIBERO_REVISION],
            check=True,
        )
        libero_actual = subprocess.check_output(
            ["git", "-C", str(libero), "rev-parse", "HEAD"], text=True
        ).strip()
        if libero_actual != LIBERO_REVISION:
            raise OpenVLAPipelineError(
                "downloaded LIBERO source revision does not match contract"
            )
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
        # Upstream owns this dependency graph. Runtime installation prevents source,
        # custom transformers, Torch/CUDA, and weights from becoming image bytes.
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", str(source)], check=True
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-e", str(libero)], check=True
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(source / "experiments/robot/libero/libero_requirements.txt"),
            ],
            check=True,
        )
        inventory.write_text(
            subprocess.check_output(
                [str(python), "-m", "pip", "freeze", "--all"], text=True
            ),
            encoding="utf-8",
        )
        write_local_json(
            ready,
            {
                "schema": "npa.workbench.openvla-oft.runtime.v1",
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_license": SOURCE_LICENSE,
                "libero_repository": LIBERO_REPOSITORY,
                "libero_revision": LIBERO_REVISION,
                "libero_license": LIBERO_LICENSE,
                "dependency_inventory": inventory.name,
                "dependency_inventory_sha256": file_digest(inventory),
                "status": "ready",
            },
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return root


def _reject_unlicensed_runtime_dependency() -> None:
    """Fail before fetching OFT's unresolved direct dependency."""
    raise OpenVLAPipelineError(
        "OFT runtime is blocked before download: "
        f"{_UNLICENSED_RUNTIME_DEPENDENCY} is an unpinned direct dependency "
        "with no authoritative repository license at the resolved revision"
    )


def _materialize_model_snapshot(
    runtime: Path, model_id: str, model_revision: str
) -> Path:
    """Fetch the base model at its immutable Hub revision into a writable cache."""
    identity = hashlib.sha256(f"{model_id}@{model_revision}".encode()).hexdigest()
    root = runtime / "models" / identity
    ready = root / "ready.json"
    with _runtime_cache_lock(root):
        if ready.is_file():
            payload = json.loads(ready.read_text())
            if (
                payload.get("repo_id") == model_id
                and payload.get("revision") == model_revision
                and (root / "config.json").is_file()
            ):
                return root
            raise OpenVLAPipelineError("OpenVLA model cache identity is stale or incomplete")
        if root.exists() and any(root.iterdir()):
            raise OpenVLAPipelineError(
                "OpenVLA model cache exists without an atomic ready marker"
            )
        root.mkdir(parents=True, exist_ok=False)
        script = (
            "from huggingface_hub import snapshot_download\n"
            "import sys\n"
            "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
            "local_dir=sys.argv[3])\n"
        )
        try:
            subprocess.run(
                [
                    str(runtime / "venv" / "bin" / "python"),
                    "-c",
                    script,
                    model_id,
                    model_revision,
                    str(root),
                ],
                check=True,
            )
            if not (root / "config.json").is_file():
                raise OpenVLAPipelineError(
                    "immutable OpenVLA Hub snapshot lacks config.json"
                )
            write_local_json(
                ready,
                {
                    "schema": "npa.workbench.openvla-oft.model.v1",
                    "repo_id": model_id,
                    "revision": model_revision,
                    "status": "ready",
                },
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
    return root


@dataclass(frozen=True)
class PrepareConfig:
    dataset_uri: str
    output_uri: str
    dataset_name: str
    task_suite: str

    def validate(self) -> None:
        _require(self.dataset_uri, "dataset_uri")
        _require(self.output_uri, "output_uri")
        _require(self.dataset_name, "dataset_name")
        if self.task_suite not in OFFICIAL_SUITE_CHECKPOINTS:
            raise OpenVLAPipelineError(
                f"unsupported OFT LIBERO suite: {self.task_suite}"
            )


def prepare(cfg: PrepareConfig) -> dict[str, Any]:
    """Inspect real RLDS files and publish a normalization/provenance handoff."""
    cfg.validate()
    with policy_workspace(cfg.output_uri, "openvla-oft-prepare") as workspace:
        materialized = _copy_or_download_tree(cfg.dataset_uri, workspace / "dataset")
        inventory = _tree_inventory(materialized)
        rlds_files = [
            entry for entry in inventory if "tfrecord" in entry["path"].lower()
        ]
        if not rlds_files:
            raise OpenVLAPipelineError("prepared data contains no RLDS TFRecord files")
        normalization = {
            "schema": "npa.workbench.openvla-oft.normalization.v1",
            "dataset_uri": cfg.dataset_uri,
            "dataset_name": cfg.dataset_name,
            "task_suite": cfg.task_suite,
            "unnorm_key_preference": f"{cfg.task_suite}_no_noops",
            "proprio_dim": 8,
            "action_mode": "continuous_l1",
            "rlds_files": rlds_files,
            "dataset_statistics": _dataset_statistics(materialized),
        }
        write_local_json(workspace / "normalization.json", normalization)
        report = {
            "schema": PREPARE_SCHEMA,
            "status": "succeeded",
            "source_dataset_uri": cfg.dataset_uri,
            "dataset_name": cfg.dataset_name,
            "task_suite": cfg.task_suite,
            "rlds_file_count": len(rlds_files),
            "normalization": "normalization.json",
            "dataset_inventory_sha256": hashlib.sha256(
                json.dumps(inventory, sort_keys=True).encode()
            ).hexdigest(),
        }
        return publish_bundle(workspace, cfg.output_uri, report, "preparation.json")


@dataclass(frozen=True)
class TrainConfig:
    prepared_manifest_uri: str
    output_uri: str
    runtime_root: str
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = MODEL_REVISION
    batch_size: int = DEFAULT_BATCH_SIZE
    max_steps: int = DEFAULT_MAX_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = DEFAULT_LORA_RANK
    processes: int = 8
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        _require(self.prepared_manifest_uri, "prepared_manifest_uri")
        _require(self.output_uri, "output_uri")
        _require(self.runtime_root, "runtime_root")
        _require_hf_repo(self.model_id, "model_id")
        _require_hf_commit(self.model_revision, "model_revision")
        if (
            min(self.batch_size, self.max_steps, self.lora_rank, self.processes) <= 0
            or self.learning_rate <= 0
        ):
            raise OpenVLAPipelineError(
                "batch_size, max_steps, lora_rank, processes, and learning_rate must be positive"
            )


def _validate_oft_components(root: Path) -> dict[str, str]:
    paths = [path for path in root.rglob("*") if path.is_file()]
    action = next(
        (path for path in paths if path.name.startswith("action_head--")), None
    )
    proprio = next(
        (path for path in paths if path.name.startswith("proprio_projector--")), None
    )
    adapter = next((path for path in paths if "lora_adapter" in path.parts), None)
    if action is None or proprio is None or adapter is None:
        raise OpenVLAPipelineError(
            "OFT checkpoint must contain lora_adapter, action_head, and proprio_projector; a stock OpenVLA decoder checkpoint is not accepted"
        )
    return {
        "action_head": action.relative_to(root).as_posix(),
        "proprio_projector": proprio.relative_to(root).as_posix(),
        "lora_adapter_member": adapter.relative_to(root).as_posix(),
    }


def _finetune_arguments(
    cfg: TrainConfig,
    model_snapshot: Path,
    data_root: Path,
    dataset_name: str,
    results: Path,
) -> list[str]:
    """Return the documented LIBERO OFT fine-tuning arguments."""
    return [
        "--vla_path",
        str(model_snapshot),
        "--data_root_dir",
        str(data_root),
        "--dataset_name",
        dataset_name,
        "--run_root_dir",
        str(results),
        "--use_l1_regression",
        "True",
        "--use_diffusion",
        "False",
        "--use_film",
        "False",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--batch_size",
        str(cfg.batch_size),
        "--learning_rate",
        str(cfg.learning_rate),
        "--num_steps_before_decay",
        "100000",
        "--max_steps",
        str(cfg.max_steps),
        "--save_freq",
        "10000",
        "--save_latest_checkpoint_only",
        "False",
        "--image_aug",
        "True",
        "--lora_rank",
        str(cfg.lora_rank),
        "--seed",
        str(cfg.seed),
    ]


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run upstream OFT LoRA fine-tuning and publish its component bundle."""
    cfg.validate()
    with policy_workspace(cfg.output_uri, "openvla-oft-train") as workspace:
        prepared = workspace / "prepared"
        preparation = materialize_bundle(
            cfg.prepared_manifest_uri, prepared, PREPARE_SCHEMA
        )
        normalization_path = prepared / "normalization.json"
        if not normalization_path.is_file():
            raise OpenVLAPipelineError("prepared bundle is missing normalization.json")
        normalization = json.loads(normalization_path.read_text())
        dataset_uri = _require(
            str(normalization.get("dataset_uri") or ""), "prepared dataset_uri"
        )
        dataset_name = _require(
            str(normalization.get("dataset_name") or ""), "prepared dataset_name"
        )
        data_root = _copy_or_download_tree(dataset_uri, workspace / "rlds")
        runtime, results = bootstrap_runtime(cfg.runtime_root), workspace / "results"
        runtime_identity = _runtime_identity(runtime)
        results.mkdir()
        model_snapshot = _materialize_model_snapshot(
            runtime, cfg.model_id, cfg.model_revision
        )
        command = _upstream_torchrun_command(
            runtime,
            UPSTREAM_FINETUNE_SCRIPT,
            cfg.processes,
            _finetune_arguments(
                cfg, model_snapshot, data_root, dataset_name, results
            ),
        )
        _run(command, cwd=runtime / "source", log=workspace / "train.log")
        components = _validate_oft_components(results)
        shutil.copytree(results, workspace / "checkpoint")
        write_local_json(
            workspace / "training-command.json",
            {"argv": command, "upstream_source_revision": SOURCE_REVISION},
        )
        report = {
            "schema": TRAIN_SCHEMA,
            "status": "succeeded",
            "prepared_manifest_sha256": _manifest_digest(preparation),
            "model": {"repo_id": cfg.model_id, "revision": cfg.model_revision},
            "upstream": _runtime_provenance(),
            "runtime_cache": runtime_identity,
            "algorithm": {
                "continuous_action_head": "l1_regression",
                "proprioception_projector": {"enabled": True, "dimension": 8},
                "lora_rank": cfg.lora_rank,
                "processes": cfg.processes,
                "num_images_in_input": 2,
            },
            "components": {
                name: f"checkpoint/{path}" for name, path in components.items()
            },
        }
        return publish_bundle(workspace, cfg.output_uri, report, "training.json")


@dataclass(frozen=True)
class RolloutConfig:
    training_manifest_uri: str
    output_uri: str
    runtime_root: str
    task_suite: str
    trials_per_task: int = DEFAULT_TRIALS
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        _require(self.training_manifest_uri, "training_manifest_uri")
        _require(self.output_uri, "output_uri")
        _require(self.runtime_root, "runtime_root")
        if self.task_suite not in OFFICIAL_SUITE_CHECKPOINTS:
            raise OpenVLAPipelineError(
                f"unsupported OFT LIBERO suite: {self.task_suite}"
            )
        if self.trials_per_task <= 0:
            raise OpenVLAPipelineError("trials_per_task must be positive")


def _parse_rollout_log(log: Path) -> dict[str, Any]:
    text = log.read_text(errors="replace")
    rate, episodes, successes = (
        _OVERALL_RESULT.search(text),
        _TOTAL_EPISODES.search(text),
        _TOTAL_SUCCESSES.search(text),
    )
    if rate is None or episodes is None or successes is None:
        raise OpenVLAPipelineError(
            "upstream LIBERO log lacks a complete final success result"
        )
    result = {
        "success_rate": float(rate.group("rate")),
        "episodes": int(episodes.group("count")),
        "successes": int(successes.group("count")),
    }
    if result["episodes"] <= 0 or result["successes"] > result["episodes"]:
        raise OpenVLAPipelineError(
            "upstream LIBERO result has impossible episode/success counts"
        )
    if not math.isclose(
        result["success_rate"], result["successes"] / result["episodes"], abs_tol=5e-4
    ):
        raise OpenVLAPipelineError(
            "upstream LIBERO reported success rate disagrees with counts"
        )
    return result


def _training_lora_rank(training: dict[str, Any]) -> int:
    """Extract and validate the training bundle's LoRA-rank compatibility key."""
    algorithm = training.get("algorithm")
    if not isinstance(algorithm, dict):
        raise OpenVLAPipelineError("training bundle is missing algorithm provenance")
    rank = algorithm.get("lora_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise OpenVLAPipelineError("training bundle has an invalid LoRA rank")
    return rank


def rollout(cfg: RolloutConfig) -> dict[str, Any]:
    """Run closed-loop upstream LIBERO rollouts using OFT-only components."""
    cfg.validate()
    with policy_workspace(cfg.output_uri, "openvla-oft-rollout") as workspace:
        trained = workspace / "trained"
        training = materialize_bundle(cfg.training_manifest_uri, trained, TRAIN_SCHEMA)
        checkpoint = trained / "checkpoint"
        components = _validate_oft_components(checkpoint)
        lora_rank = _training_lora_rank(training)
        runtime, logs = bootstrap_runtime(cfg.runtime_root), workspace / "rollouts"
        runtime_identity = _runtime_identity(runtime)
        logs.mkdir()
        command = _upstream_command(
            runtime,
            UPSTREAM_LIBERO_EVAL_SCRIPT,
            [
                "--pretrained_checkpoint",
                str(checkpoint),
                "--task_suite_name",
                cfg.task_suite,
                "--num_trials_per_task",
                str(cfg.trials_per_task),
                "--seed",
                str(cfg.seed),
                "--use_l1_regression",
                "True",
                "--use_diffusion",
                "False",
                "--use_film",
                "False",
                "--num_images_in_input",
                "2",
                "--use_proprio",
                "True",
                "--center_crop",
                "True",
                "--num_open_loop_steps",
                "8",
                "--lora_rank",
                str(lora_rank),
                "--local_log_dir",
                str(logs),
                "--use_wandb",
                "False",
            ],
        )
        _run(command, cwd=runtime / "source", log=workspace / "rollout-command.log")
        upstream_logs = sorted(logs.glob("*.txt"))
        if len(upstream_logs) != 1:
            raise OpenVLAPipelineError(
                "expected exactly one upstream LIBERO evaluation log"
            )
        result, videos = (
            _parse_rollout_log(upstream_logs[0]),
            sorted(logs.rglob("*.mp4")),
        )
        if not videos:
            raise OpenVLAPipelineError(
                "upstream closed-loop rollout produced no MP4 evidence"
            )
        write_local_json(
            workspace / "rollout.json",
            {
                "schema": ROLLOUT_SCHEMA,
                "status": "succeeded",
                "training_manifest_sha256": _manifest_digest(training),
                "task_suite": cfg.task_suite,
                "trials_per_task": cfg.trials_per_task,
                "seed": cfg.seed,
                "result": result,
                "video_count": len(videos),
                "components": components,
                "training_lora_rank": lora_rank,
                "official_suite_checkpoint_contract": OFFICIAL_SUITE_CHECKPOINTS[
                    cfg.task_suite
                ],
                "checkpoint_source": "training bundle; not an official suite adapter",
            },
        )
        report = {
            "schema": ROLLOUT_SCHEMA,
            "status": "succeeded",
            "task_suite": cfg.task_suite,
            "closed_loop": True,
            "rollout": "rollout.json",
            "upstream": _runtime_provenance(),
            "runtime_cache": runtime_identity,
        }
        return publish_bundle(
            workspace, cfg.output_uri, report, "rollout-manifest.json"
        )


@dataclass(frozen=True)
class EvaluateConfig:
    rollout_manifest_uri: str
    output_uri: str

    def validate(self) -> None:
        _require(self.rollout_manifest_uri, "rollout_manifest_uri")
        _require(self.output_uri, "output_uri")


def _wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    z, p = 1.959963984540054, successes / episodes
    denominator = 1 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denominator
    margin = (
        z * math.sqrt((p * (1 - p) + z * z / (4 * episodes)) / episodes) / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate(cfg: EvaluateConfig) -> dict[str, Any]:
    """Numerically verify held-out success from immutable rollout evidence."""
    cfg.validate()
    with policy_workspace(cfg.output_uri, "openvla-oft-evaluate") as workspace:
        materialized = workspace / "rollout"
        rollout_report = materialize_bundle(
            cfg.rollout_manifest_uri, materialized, ROLLOUT_SCHEMA
        )
        raw = json.loads((materialized / "rollout.json").read_text())
        result = raw.get("result") or {}
        successes, episodes = (
            int(result.get("successes", -1)),
            int(result.get("episodes", -1)),
        )
        if episodes <= 0 or not 0 <= successes <= episodes:
            raise OpenVLAPipelineError("rollout evidence has invalid success counts")
        observed, reported = successes / episodes, float(result.get("success_rate", -1))
        if not math.isclose(observed, reported, abs_tol=5e-4):
            raise OpenVLAPipelineError(
                "rollout evidence's reported rate does not match its counts"
            )
        low, high = _wilson_interval(successes, episodes)
        metrics = {
            "schema": EVALUATION_SCHEMA,
            "status": "succeeded",
            "task_suite": raw["task_suite"],
            "metric": "closed_loop_success_rate",
            "successes": successes,
            "episodes": episodes,
            "success_rate": observed,
            "confidence_interval_95": [low, high],
            "rollout_manifest_sha256": _manifest_digest(rollout_report),
            "note": "Measured simulator evidence only; not a physical-robot or convergence claim.",
        }
        write_local_json(workspace / "metrics.json", metrics)
        report = {
            "schema": EVALUATION_SCHEMA,
            "status": "succeeded",
            "metrics": "metrics.json",
            "source_rollout_manifest_sha256": metrics["rollout_manifest_sha256"],
        }
        return publish_bundle(workspace, cfg.output_uri, report, "evaluation.json")


@dataclass(frozen=True)
class VisualizeConfig:
    evaluation_manifest_uri: str
    output_uri: str

    def validate(self) -> None:
        _require(self.evaluation_manifest_uri, "evaluation_manifest_uri")
        _require(self.output_uri, "output_uri")


def _comparison_svg(metrics: dict[str, Any]) -> str:
    rate = float(metrics["success_rate"])
    low, high = (float(item) for item in metrics["confidence_interval_95"])
    left, scale = 110, 560
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="700" height="180" role="img">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="20" y="30" font-size="18">{metrics["task_suite"]} closed-loop success</text>',
            f'<line x1="{left}" y1="90" x2="{left + scale}" y2="90" stroke="#555"/>',
            f'<line x1="{left + low * scale:.1f}" y1="90" x2="{left + high * scale:.1f}" y2="90" stroke="#2563eb" stroke-width="8"/>',
            f'<circle cx="{left + rate * scale:.1f}" cy="90" r="9" fill="#0f172a"/>',
            f'<text x="20" y="145" font-size="16">{metrics["successes"]}/{metrics["episodes"]} = {rate:.4f}; 95% Wilson CI [{low:.4f}, {high:.4f}]</text>',
            "</svg>",
        ]
    )


def visualize(cfg: VisualizeConfig) -> dict[str, Any]:
    """Create factual SVG/CSV comparison artifacts from verified metrics."""
    cfg.validate()
    with policy_workspace(cfg.output_uri, "openvla-oft-visualize") as workspace:
        materialized = workspace / "evaluation"
        evaluation_report = materialize_bundle(
            cfg.evaluation_manifest_uri, materialized, EVALUATION_SCHEMA
        )
        metrics = json.loads((materialized / "metrics.json").read_text())
        if (
            metrics.get("schema") != EVALUATION_SCHEMA
            or metrics.get("status") != "succeeded"
        ):
            raise OpenVLAPipelineError(
                "evaluation metrics are not a completed OFT evaluation"
            )
        (workspace / "comparison.svg").write_text(
            _comparison_svg(metrics), encoding="utf-8"
        )
        (workspace / "comparison.csv").write_text(
            "task_suite,metric,successes,episodes,success_rate,ci95_low,ci95_high\n"
            f"{metrics['task_suite']},{metrics['metric']},{metrics['successes']},{metrics['episodes']},{metrics['success_rate']},{metrics['confidence_interval_95'][0]},{metrics['confidence_interval_95'][1]}\n",
            encoding="utf-8",
        )
        report = {
            "schema": COMPARISON_SCHEMA,
            "status": "succeeded",
            "evaluation_manifest_sha256": _manifest_digest(evaluation_report),
            "metrics": metrics,
            "visualizations": ["comparison.svg", "comparison.csv"],
        }
        return publish_bundle(workspace, cfg.output_uri, report, "comparison.json")


def _print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openvla_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    boot = commands.add_parser(
        "bootstrap-runtime", help="Fetch the pinned OFT runtime into a writable cache."
    )
    boot.add_argument("--runtime-root", required=True)
    prep = commands.add_parser(
        "prepare", help="Inspect RLDS data and publish normalization provenance."
    )
    prep.add_argument("--dataset-uri", required=True)
    prep.add_argument("--dataset-name", required=True)
    prep.add_argument(
        "--task-suite", required=True, choices=sorted(OFFICIAL_SUITE_CHECKPOINTS)
    )
    prep.add_argument("--output-uri", required=True)
    train_p = commands.add_parser("train", help="Run upstream OFT fine-tuning.")
    train_p.add_argument("--prepared-manifest-uri", required=True)
    train_p.add_argument("--output-uri", required=True)
    train_p.add_argument("--runtime-root", required=True)
    train_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    train_p.add_argument("--model-revision", default=MODEL_REVISION)
    train_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    train_p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train_p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    train_p.add_argument("--processes", type=int, default=8)
    train_p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    roll = commands.add_parser(
        "rollout", help="Run upstream closed-loop LIBERO rollouts."
    )
    roll.add_argument("--training-manifest-uri", required=True)
    roll.add_argument("--output-uri", required=True)
    roll.add_argument("--runtime-root", required=True)
    roll.add_argument(
        "--task-suite", required=True, choices=sorted(OFFICIAL_SUITE_CHECKPOINTS)
    )
    roll.add_argument("--trials-per-task", type=int, default=DEFAULT_TRIALS)
    roll.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ev = commands.add_parser("evaluate", help="Verify held-out numerical success.")
    ev.add_argument("--rollout-manifest-uri", required=True)
    ev.add_argument("--output-uri", required=True)
    vis = commands.add_parser("visualize", help="Create factual comparison artifacts.")
    vis.add_argument("--evaluation-manifest-uri", required=True)
    vis.add_argument("--output-uri", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = vars(args)
    command = values.pop("command")
    if command == "bootstrap-runtime":
        print(bootstrap_runtime(values["runtime_root"]))
        return 0
    if command == "prepare":
        _print_result(prepare(PrepareConfig(**values)))
        return 0
    if command == "train":
        _print_result(train(TrainConfig(**values)))
        return 0
    if command == "rollout":
        _print_result(rollout(RolloutConfig(**values)))
        return 0
    if command == "evaluate":
        _print_result(evaluate(EvaluateConfig(**values)))
        return 0
    if command == "visualize":
        _print_result(visualize(VisualizeConfig(**values)))
        return 0
    raise AssertionError(f"unhandled command {command}")


if __name__ == "__main__":
    raise SystemExit(main())
