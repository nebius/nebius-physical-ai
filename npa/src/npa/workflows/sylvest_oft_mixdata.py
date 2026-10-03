"""Run a source-pinned, paired LIBERO-Plus comparison for the Sylvest checkpoint.

The workflow intentionally does not treat a mixed-data checkpoint as an
improvement.  It selects only task names outside a supplied training inventory,
runs the upstream OpenVLA-OFT LIBERO evaluator once per matched task/initial-state pair,
then reports paired success differences and a factual Rerun recording.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from npa.clients.storage import StorageClient

WORKFLOW_SCHEMA = "npa.sylvest-oft-mixdata.v1"
OFT_SOURCE_REPOSITORY = "https://github.com/moojink/openvla-oft"
LIBERO_PLUS_SOURCE_REPOSITORY = "https://github.com/sylvestf/LIBERO-plus"
_CHECKPOINT_READY_FILE = ".npa-checkpoint-ready.json"
_CHECKPOINT_MUTABLE_FILES = {
    "config.json",
    "configuration_prismatic.py",
    "modeling_prismatic.py",
}


class SylvestComparisonError(RuntimeError):
    """Raised when paired checkpoint evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ArtifactLocation:
    """Describe a local or S3 workflow artifact location.

    Args:
        value: Original user-supplied URI or filesystem path.

    Returns:
        None.

    Raises:
        None.
    """

    value: str

    @property
    def is_s3(self) -> bool:
        """Return whether the artifact uses the supported S3 scheme.

        Args:
            None.

        Returns:
            True when the location is an S3 URI.

        Raises:
            None.
        """

        return self.value.startswith("s3://")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with temporary:
            temporary.write(_canonical_bytes(payload))
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SylvestComparisonError(
            f"invalid required JSON artifact: {path}: {exc}"
        ) from exc


def _storage() -> StorageClient:
    return StorageClient.from_environment()


def _materialize(location: str, destination: Path) -> Path:
    source = ArtifactLocation(location)
    if source.is_s3:
        if source.value.endswith("/"):
            _storage().download_directory(source.value, str(destination))
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        _storage().download_file(source.value, str(destination))
        return destination
    local = Path(source.value)
    if not local.exists():
        raise SylvestComparisonError(f"required local artifact does not exist: {local}")
    if local.is_dir():
        shutil.copytree(local, destination, dirs_exist_ok=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, destination)
    return destination


def _publish_tree(local_dir: Path, output_uri: str) -> None:
    target = ArtifactLocation(output_uri)
    if target.is_s3:
        _storage().upload_directory(str(local_dir), output_uri)
        return
    destination = Path(output_uri)
    shutil.copytree(local_dir, destination, dirs_exist_ok=True)


def _parse_initial_state_indices(value: str) -> list[int]:
    try:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SylvestComparisonError(
            f"invalid comma-separated initial-state list: {value!r}"
        ) from exc
    if indices != list(range(len(indices))):
        raise SylvestComparisonError(
            "initial-state indices must be contiguous zero-based upstream trial indices"
        )
    return indices


def _task_names(payload: Any) -> set[str]:
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return set(payload)
    if isinstance(payload, dict) and isinstance(payload.get("task_names"), list):
        values = payload["task_names"]
        if all(isinstance(item, str) for item in values):
            return set(values)
    raise SylvestComparisonError(
        "training-task inventory must be a JSON string list or {'task_names': [...]}"
    )


def _verified_source_root(root: str, revision: str, repository: str) -> Path:
    path = Path(root).resolve()
    if not (path / ".git").exists():
        raise SylvestComparisonError(
            f"runtime source must be an operator-provided git checkout: {path}"
        )
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    actual = completed.stdout.strip()
    if completed.returncode or actual != revision:
        raise SylvestComparisonError(
            f"source identity mismatch for {repository}: expected {revision}, found {actual or 'unreadable'}"
        )
    return path


def _require_libero_plus_license(root: Path) -> Path:
    candidates = [
        root / "LICENSE",
        root / "LICENSE.md",
        root / "COPYING",
        root / "NOTICE",
    ]
    license_file = next((path for path in candidates if path.is_file()), None)
    if license_file is None:
        raise SylvestComparisonError(
            "LIBERO-Plus source has no license file. Do not execute it until the "
            "upstream authors publish or identify an applicable license; this is "
            "not an NPA EULA requirement."
        )
    return license_file


def _prepare_protocol(args: argparse.Namespace, output: Path) -> None:
    libero_root = _verified_source_root(
        args.libero_plus_root, args.libero_plus_revision, LIBERO_PLUS_SOURCE_REPOSITORY
    )
    license_file = _require_libero_plus_license(libero_root)
    classification_path = (
        libero_root / "libero/libero/benchmark/task_classification.json"
    )
    classification = _read_json(classification_path)
    suite_cases = (
        classification.get(args.task_suite)
        if isinstance(classification, dict)
        else None
    )
    if not isinstance(suite_cases, list):
        raise SylvestComparisonError(
            f"classification has no task list for {args.task_suite!r}"
        )
    with tempfile.TemporaryDirectory(prefix="sylvest-training-inventory-") as temporary:
        inventory = _materialize(
            args.training_task_ids_uri, Path(temporary) / "training.json"
        )
        training_names = _task_names(_read_json(inventory))
    held_out = [
        row for row in suite_cases if str(row.get("name") or "") not in training_names
    ]
    if not training_names:
        raise SylvestComparisonError("training-task inventory must not be empty")
    if not held_out:
        raise SylvestComparisonError(
            "training inventory leaves no held-out LIBERO-Plus tasks"
        )
    protocol_cases = _protocol_cases(
        held_out, _parse_initial_state_indices(args.initial_state_indices)
    )
    source_hash = _sha256_file(classification_path)
    protocol = {
        "schema": WORKFLOW_SCHEMA,
        "kind": "held_out_libero_plus_protocol",
        "task_suite": args.task_suite,
        "initial_state_indices": _parse_initial_state_indices(
            args.initial_state_indices
        ),
        "cases": protocol_cases,
        "held_out_task_count": len(held_out),
        "training_inventory_count": len(training_names),
        "source": {
            "repository": LIBERO_PLUS_SOURCE_REPOSITORY,
            "revision": args.libero_plus_revision,
            "classification_sha256": source_hash,
            "license_file": license_file.name,
            "license_sha256": _sha256_file(license_file),
        },
        "held_out_definition": "task name absent from the supplied mix-SFT training inventory",
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical_bytes(protocol)).hexdigest()
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "provenance.json", _prepare_provenance(args, protocol))


def _protocol_cases(
    rows: Iterable[Mapping[str, Any]], initial_state_indices: Iterable[int]
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in sorted(
        rows, key=lambda item: (int(item.get("id", 0)), str(item.get("name", "")))
    ):
        name = str(row.get("name") or "")
        if not name:
            raise SylvestComparisonError(
                "classification contains a task without a name"
            )
        for initial_state_index in initial_state_indices:
            cases.append(
                {
                    "case_id": f"{name}::initial-state={initial_state_index}",
                    "task_name": name,
                    "category": str(row.get("category") or "unspecified"),
                    "difficulty_level": int(row.get("difficulty_level") or 0),
                    "initial_state_index": initial_state_index,
                }
            )
    return cases


def _prepare_provenance(
    args: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "producer": "npa.workflows.sylvest_oft_mixdata.prepare",
        "protocol_sha256": protocol["protocol_sha256"],
        "limitations": [
            "A task is held out only relative to the supplied training inventory.",
            "The published checkpoint card does not itself provide a training-task split.",
            "Initial-state indices are upstream LIBERO trial indices, not independently sampled seeds.",
            "No benchmark improvement is inferred at preparation time.",
        ],
        "sources": {
            "libero_plus_revision": args.libero_plus_revision,
            "training_task_ids_uri": args.training_task_ids_uri,
        },
    }


def _snapshot_checkpoint(repo_id: str, revision: str, cache_root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SylvestComparisonError(
            "huggingface_hub is required for checkpoint retrieval"
        ) from exc
    safe_id = repo_id.replace("/", "--")
    repository_cache = cache_root / safe_id
    target = repository_cache / revision
    ready = target / _CHECKPOINT_READY_FILE
    repository_cache.mkdir(parents=True, exist_ok=True)
    lock_path = repository_cache / f".{revision}.lock"
    # A run-shared filesystem cache must never expose an unmarked partial
    # snapshot. A lock serializes the fetch; the ready marker is atomically
    # replaced only after its required payload is present. Hugging Face resumes
    # an interrupted local-dir fetch on a later retry.
    import fcntl

    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _checkpoint_is_ready(target, ready, repo_id, revision):
            return target
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(target))
        _write_json(ready, _checkpoint_metadata(target, repo_id, revision))
    return target


def _checkpoint_is_ready(
    target: Path, ready: Path, repo_id: str, revision: str
) -> bool:
    if not ready.is_file():
        return False
    metadata = _read_json(ready)
    if not isinstance(metadata, dict):
        return False
    if metadata.get("repo_id") != repo_id or metadata.get("revision") != revision:
        return False
    return metadata.get("inventory") == _checkpoint_inventory(target)


def _checkpoint_metadata(target: Path, repo_id: str, revision: str) -> dict[str, Any]:
    required_files = _required_checkpoint_files(target)
    inventory = _checkpoint_inventory(target)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "required_files": required_files,
        "inventory": inventory,
        "inventory_sha256": hashlib.sha256(_canonical_bytes(inventory)).hexdigest(),
    }


def _required_checkpoint_files(target: Path) -> list[str]:
    required = [target / "config.json", target / "dataset_statistics.json"]
    action_heads = sorted(target.glob("action_head--*_checkpoint.pt"))
    proprio_projectors = sorted(target.glob("proprio_projector--*_checkpoint.pt"))
    adapter = target / "lora_adapter"
    required.extend(action_heads + proprio_projectors)
    if len(action_heads) != 1 or len(proprio_projectors) != 1 or not adapter.is_dir():
        raise SylvestComparisonError(
            "checkpoint is not an OpenVLA-OFT package with one action head, one proprio projector, and lora_adapter"
        )
    adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
    if not adapter_files:
        raise SylvestComparisonError(
            "OpenVLA-OFT checkpoint has an empty lora_adapter directory"
        )
    required.extend(adapter_files)
    missing = [path for path in required if not path.is_file() or path.is_symlink()]
    if missing:
        names = ", ".join(str(path.relative_to(target)) for path in missing)
        raise SylvestComparisonError(
            f"checkpoint is missing required regular files: {names}"
        )
    return [str(path.relative_to(target)) for path in required]


def _checkpoint_inventory(target: Path) -> list[dict[str, Any]]:
    files = [
        path
        for path in target.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != _CHECKPOINT_READY_FILE
        and ".cache" not in path.relative_to(target).parts
    ]
    return [
        {
            "path": str(path.relative_to(target)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]


def _checkpoint_provenance(checkpoint: Path) -> Mapping[str, Any]:
    metadata = _read_json(checkpoint / _CHECKPOINT_READY_FILE)
    if not isinstance(metadata, dict) or "inventory_sha256" not in metadata:
        raise SylvestComparisonError(
            "checkpoint cache has no verified provenance marker"
        )
    return metadata


def _copy_checkpoint_workspace(snapshot: Path, destination: Path) -> Path:
    for source in sorted(path for path in snapshot.rglob("*") if path.is_file()):
        relative = source.relative_to(snapshot)
        if source.name == _CHECKPOINT_READY_FILE or ".cache" in relative.parts:
            continue
        if source.is_symlink():
            raise SylvestComparisonError(
                f"checkpoint contains a symlinked payload: {relative}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(relative) in _CHECKPOINT_MUTABLE_FILES:
            shutil.copy2(source, target)
        else:
            _link_or_copy(source, target)
    return destination


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _runtime_modules(openvla_root: Path, libero_root: Path) -> Any:
    importlib.invalidate_caches()
    paths = [str(libero_root / "libero"), str(openvla_root)]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("experiments.robot.libero.run_libero_eval")


def _rollout(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-protocol-") as temporary:
        protocol_path = _materialize(
            args.protocol_uri, Path(temporary) / "protocol.json"
        )
        protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    libero_root = _verified_source_root(
        args.libero_plus_root, args.libero_plus_revision, LIBERO_PLUS_SOURCE_REPOSITORY
    )
    _require_libero_plus_license(libero_root)
    openvla_root = _verified_source_root(
        args.openvla_oft_root, args.openvla_oft_revision, OFT_SOURCE_REPOSITORY
    )
    checkpoint_snapshot = _snapshot_checkpoint(
        args.checkpoint_id, args.checkpoint_revision, Path(args.model_cache_root)
    )
    checkpoint_provenance = _checkpoint_provenance(checkpoint_snapshot)
    with tempfile.TemporaryDirectory(
        prefix="sylvest-checkpoint-workspace-"
    ) as temporary:
        runtime_workspace = Path(temporary)
        checkpoint = _copy_checkpoint_workspace(
            checkpoint_snapshot, runtime_workspace / "checkpoint"
        )
        (runtime_workspace / "prismatic").symlink_to(
            openvla_root / "prismatic", target_is_directory=True
        )
        results = _run_upstream_rollouts(
            protocol,
            checkpoint,
            openvla_root,
            libero_root,
            output,
            runtime_workspace,
        )
        _collect_rollout_clips(runtime_workspace, output, expected_count=len(results))
    payload = {
        "schema": WORKFLOW_SCHEMA,
        "kind": "openvla_oft_libero_plus_rollouts",
        "checkpoint": {
            "id": args.checkpoint_id,
            "revision": args.checkpoint_revision,
            "inventory_sha256": checkpoint_provenance["inventory_sha256"],
            "required_files": checkpoint_provenance["required_files"],
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "task_suite": protocol["task_suite"],
        "episodes": results,
        "summary": _rollout_summary(results),
    }
    _write_json(output / "rollouts.json", payload)
    _write_json(output / "clips" / "manifest.json", _clip_manifest(output / "clips"))
    _write_json(output / "provenance.json", _rollout_provenance(args, payload))


def _validate_protocol(protocol: Any) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema") != WORKFLOW_SCHEMA:
        raise SylvestComparisonError("unsupported protocol schema")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SylvestComparisonError("protocol has no held-out cases")
    expected = dict(protocol)
    actual = str(expected.pop("protocol_sha256", ""))
    if not actual or hashlib.sha256(_canonical_bytes(expected)).hexdigest() != actual:
        raise SylvestComparisonError(
            "protocol hash does not match its declared content"
        )


def _run_upstream_rollouts(
    protocol: Mapping[str, Any],
    checkpoint: Path,
    openvla_root: Path,
    libero_root: Path,
    output: Path,
    runtime_workspace: Path,
) -> list[dict[str, Any]]:
    evaluator = _runtime_modules(openvla_root, libero_root)
    prior_cwd = Path.cwd()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime_workspace)
    try:
        cfg = evaluator.GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            task_suite_name=protocol["task_suite"],
            num_trials_per_task=len(protocol["initial_state_indices"]),
            center_crop=True,
            use_wandb=False,
            local_log_dir=str(output / "logs"),
        )
        evaluator.validate_config(cfg)
        model, action_head, proprio, noisy, processor = evaluator.initialize_model(cfg)
        task_suite = evaluator.benchmark.get_benchmark_dict()[protocol["task_suite"]]()
        task_ids = {
            name: index for index, name in enumerate(task_suite.get_task_names())
        }
        log_file, _, _ = evaluator.setup_logging(cfg)
        try:
            results = _run_cases(
                evaluator,
                cfg,
                protocol["cases"],
                task_ids,
                model,
                action_head,
                proprio,
                noisy,
                processor,
                log_file,
            )
        finally:
            log_file.close()
        _raise_if_episode_errors(output / "logs")
        return results
    finally:
        os.chdir(prior_cwd)


def _run_cases(
    evaluator: Any,
    cfg: Any,
    cases: Sequence[Mapping[str, Any]],
    task_ids: Mapping[str, int],
    model: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    processor: Any,
    log_file: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0
    resize_size = evaluator.get_image_resize_size(cfg)
    for task_name, task_cases in _cases_by_task(cases).items():
        if task_name not in task_ids:
            raise SylvestComparisonError(
                f"held-out task is absent from runtime suite: {task_name}"
            )
        task_results, total_episodes, total_successes = _run_task_cases(
            evaluator,
            cfg,
            task_cases,
            task_ids[task_name],
            model,
            resize_size,
            processor,
            action_head,
            proprio,
            noisy,
            total_episodes,
            total_successes,
            log_file,
        )
        results.extend(task_results)
    return results


def _cases_by_task(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["task_name"]), []).append(case)
    for task_cases in grouped.values():
        task_cases.sort(key=lambda case: int(case["initial_state_index"]))
        indices = [int(case["initial_state_index"]) for case in task_cases]
        if indices != list(range(len(indices))):
            raise SylvestComparisonError(
                "each task must use contiguous upstream initial-state indices"
            )
    return dict(sorted(grouped.items()))


def _run_task_cases(
    evaluator: Any,
    cfg: Any,
    cases: Sequence[Mapping[str, Any]],
    task_id: int,
    model: Any,
    resize_size: Any,
    processor: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    total_episodes: int,
    total_successes: int,
    log_file: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    cfg.num_trials_per_task = len(cases)
    evaluator.set_seed_everywhere(cfg.seed)
    outcomes, total_episodes, total_successes = _capture_upstream_task(
        evaluator,
        cfg,
        task_id,
        model,
        resize_size,
        processor,
        action_head,
        proprio,
        noisy,
        total_episodes,
        total_successes,
        log_file,
    )
    if len(outcomes) != len(cases):
        raise SylvestComparisonError(
            "upstream evaluator did not complete every requested initial-state trial"
        )
    results = [
        {**dict(case), "success": int(success)}
        for case, success in zip(cases, outcomes, strict=True)
    ]
    return results, total_episodes, total_successes


def _capture_upstream_task(
    evaluator: Any,
    cfg: Any,
    task_id: int,
    model: Any,
    resize_size: Any,
    processor: Any,
    action_head: Any,
    proprio: Any,
    noisy: Any,
    total_episodes: int,
    total_successes: int,
    log_file: Any,
) -> tuple[list[bool], int, int]:
    outcomes: list[bool] = []
    original_run_episode = evaluator.run_episode

    def record_episode(*args: Any, **kwargs: Any) -> tuple[bool, Any]:
        success, replay_images = original_run_episode(*args, **kwargs)
        outcomes.append(bool(success))
        return success, replay_images

    evaluator.run_episode = record_episode
    try:
        totals = evaluator.run_task(
            cfg,
            evaluator.benchmark.get_benchmark_dict()[cfg.task_suite_name](),
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio,
            noisy,
            total_episodes,
            total_successes,
            log_file,
        )
    finally:
        evaluator.run_episode = original_run_episode
    if totals[0] != total_episodes + len(outcomes) or totals[
        1
    ] != total_successes + sum(outcomes):
        raise SylvestComparisonError(
            "upstream evaluator returned inconsistent task totals"
        )
    return outcomes, *totals


def _rollout_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in results)
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
    }


def _raise_if_episode_errors(logs: Path) -> None:
    errors = []
    for path in sorted(logs.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Episode error:" in line:
                errors.append(f"{path.name}: {line}")
    if errors:
        raise SylvestComparisonError(
            "upstream evaluator recorded episode errors; refusing to score them as failures: "
            + " | ".join(errors)
        )


def _clip_manifest(clips: Path) -> dict[str, Any]:
    files = (
        sorted(path for path in clips.rglob("*.mp4") if path.is_file())
        if clips.exists()
        else []
    )
    return {
        "schema": WORKFLOW_SCHEMA,
        "clips": [
            {
                "path": str(path.relative_to(clips)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def _collect_rollout_clips(
    runtime_workspace: Path, output: Path, expected_count: int
) -> None:
    """Move upstream-generated rollout MP4s into the declared artifact directory.

    Args:
        runtime_workspace: Temporary upstream evaluator work directory.
        output: Stage-local output root for durable workflow artifacts.
        expected_count: Number of completed upstream simulator episodes.

    Returns:
        None.

    Raises:
        OSError: A generated video cannot be moved into the output contract.
    """

    upstream = runtime_workspace / "rollouts"
    clips = output / "clips"
    if not upstream.is_dir():
        raise SylvestComparisonError(
            "upstream evaluator did not produce a rollout-video directory"
        )
    clips.mkdir(parents=True, exist_ok=True)
    for source in sorted(upstream.rglob("*.mp4")):
        target = clips / source.name
        if target.exists():
            raise SylvestComparisonError(
                f"duplicate upstream rollout filename: {source.name}"
            )
        shutil.move(str(source), target)
    videos = sorted(clips.glob("*.mp4"))
    if len(videos) != expected_count:
        raise SylvestComparisonError(
            f"expected {expected_count} upstream rollout MP4s but found {len(videos)}"
        )
    for video in videos:
        _validate_rollout_video(video)


def _validate_rollout_video(video: Path) -> None:
    try:
        import imageio.v3 as imageio

        first_frame = imageio.imread(video, index=0)
    except Exception as exc:
        raise SylvestComparisonError(
            f"upstream rollout MP4 cannot be decoded: {video.name}: {exc}"
        ) from exc
    if video.stat().st_size <= 0 or getattr(first_frame, "size", 0) <= 0:
        raise SylvestComparisonError(f"upstream rollout MP4 is empty: {video.name}")


def _rollout_provenance(
    args: argparse.Namespace, payload: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "producer": "upstream OpenVLA-OFT experiments.robot.libero.run_libero_eval.run_task",
        "openvla_oft": {
            "repository": OFT_SOURCE_REPOSITORY,
            "revision": args.openvla_oft_revision,
        },
        "libero_plus": {
            "repository": LIBERO_PLUS_SOURCE_REPOSITORY,
            "revision": args.libero_plus_revision,
        },
        "checkpoint": payload["checkpoint"],
        "protocol_sha256": payload["protocol_sha256"],
        "limitations": [
            "Results are simulator rollouts, not physical-robot success.",
            "Every scored episode has one decoded upstream save_rollout_video MP4.",
        ],
    }


def _compare(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-compare-") as temporary:
        root = Path(temporary)
        baseline = _read_json(
            _materialize(args.baseline_uri, root / "baseline" / "rollouts.json")
        )
        candidate = _read_json(
            _materialize(args.candidate_uri, root / "candidate" / "rollouts.json")
        )
    comparison = _paired_comparison(baseline, candidate)
    _write_json(output / "comparison.json", comparison)
    _write_json(
        output / "provenance.json",
        {
            "schema": WORKFLOW_SCHEMA,
            "producer": "paired exact success comparison",
            "protocol_sha256": comparison["protocol_sha256"],
        },
    )


def _paired_comparison(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_rollouts(baseline)
    _validate_rollouts(candidate)
    if baseline["protocol_sha256"] != candidate["protocol_sha256"]:
        raise SylvestComparisonError(
            "baseline and candidate used different held-out protocols"
        )
    baseline_rows = {str(row["case_id"]): row for row in baseline["episodes"]}
    candidate_rows = {str(row["case_id"]): row for row in candidate["episodes"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise SylvestComparisonError(
            "baseline and candidate completed different paired cases"
        )
    rows = [
        _paired_row(baseline_rows[key], candidate_rows[key])
        for key in sorted(baseline_rows)
    ]
    deltas = [int(row["delta_success"]) for row in rows]
    wins = sum(delta == 1 for delta in deltas)
    losses = sum(delta == -1 for delta in deltas)
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "paired_robustness_difference",
        "protocol_sha256": baseline["protocol_sha256"],
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "paired_cases": rows,
        "overall": _delta_summary(deltas, wins, losses),
        "by_category": _category_summaries(rows),
        "limitations": [
            "Improvement is supported only when the paired interval and discordant-pair test are reported.",
            "This result does not establish convergence, a full benchmark claim, or physical-robot success.",
        ],
    }


def _validate_rollouts(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != WORKFLOW_SCHEMA or not isinstance(
        payload.get("episodes"), list
    ):
        raise SylvestComparisonError("invalid rollout artifact")
    if not payload.get("protocol_sha256") or not payload["episodes"]:
        raise SylvestComparisonError("rollout artifact lacks paired evidence")


def _paired_row(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    for key in ("task_name", "category", "difficulty_level", "initial_state_index"):
        if baseline.get(key) != candidate.get(key):
            raise SylvestComparisonError(
                f"paired case disagrees on {key}: {baseline['case_id']}"
            )
    return {
        "case_id": baseline["case_id"],
        "task_name": baseline["task_name"],
        "category": baseline["category"],
        "difficulty_level": baseline["difficulty_level"],
        "initial_state_index": baseline["initial_state_index"],
        "baseline_success": int(baseline["success"]),
        "candidate_success": int(candidate["success"]),
        "delta_success": int(candidate["success"]) - int(baseline["success"]),
    }


def _delta_summary(deltas: Sequence[int], wins: int, losses: int) -> dict[str, Any]:
    count = len(deltas)
    mean = sum(deltas) / count
    variance = (
        sum((delta - mean) ** 2 for delta in deltas) / (count - 1) if count > 1 else 0.0
    )
    margin = 1.96 * math.sqrt(variance / count) if count > 1 else 0.0
    return {
        "n": count,
        "mean_delta_success": mean,
        "normal_95_ci": [mean - margin, mean + margin],
        "candidate_only_successes": wins,
        "baseline_only_successes": losses,
        "mcnemar_exact_pvalue": _mcnemar_pvalue(wins, losses),
    }


def _mcnemar_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, value) for value in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _category_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(int(row["delta_success"]))
    return {
        category: _delta_summary(values, values.count(1), values.count(-1))
        for category, values in sorted(grouped.items())
    }


def _report(args: argparse.Namespace, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sylvest-report-") as temporary:
        comparison = _read_json(
            _materialize(args.comparison_uri, Path(temporary) / "comparison.json")
        )
    _validate_comparison(comparison)
    report = _report_payload(args.run_id, comparison)
    _write_json(output / "report.json", report)
    rrd = output / "comparison.rrd"
    _write_rrd(rrd, args.run_id, comparison)
    checksums = {
        "report.json": _sha256_file(output / "report.json"),
        "comparison.rrd": _sha256_file(rrd),
    }
    _write_json(
        output / "checksums.json", {"schema": WORKFLOW_SCHEMA, "files": checksums}
    )


def _validate_comparison(comparison: Mapping[str, Any]) -> None:
    if comparison.get("schema") != WORKFLOW_SCHEMA or not comparison.get(
        "paired_cases"
    ):
        raise SylvestComparisonError("invalid paired comparison artifact")


def _report_payload(run_id: str, comparison: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": WORKFLOW_SCHEMA,
        "kind": "verified_sylvest_oft_mixdata_comparison_report",
        "run_id": run_id,
        "protocol_sha256": comparison["protocol_sha256"],
        "baseline_checkpoint": comparison["baseline_checkpoint"],
        "candidate_checkpoint": comparison["candidate_checkpoint"],
        "overall": comparison["overall"],
        "by_category": comparison["by_category"],
        "rollout_video_source": "upstream OpenVLA-OFT save_rollout_video outputs from both rollout stages",
        "limitations": comparison["limitations"],
    }


def _write_rrd(path: Path, run_id: str, comparison: Mapping[str, Any]) -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise SylvestComparisonError(
            "rerun-sdk is required to emit comparison.rrd"
        ) from exc
    recording = rr.RecordingStream("npa_sylvest_oft_mixdata", recording_id=run_id)
    rr.save(path, recording=recording)
    rr.log(
        "provenance/protocol_sha256",
        rr.TextLog(str(comparison["protocol_sha256"])),
        static=True,
        recording=recording,
    )
    for index, row in enumerate(comparison["paired_cases"]):
        rr.set_time("paired_case", sequence=index, recording=recording)
        rr.log(
            "metrics/baseline_success",
            rr.Scalars(float(row["baseline_success"])),
            recording=recording,
        )
        rr.log(
            "metrics/candidate_success",
            rr.Scalars(float(row["candidate_success"])),
            recording=recording,
        )
        rr.log(
            "metrics/delta_success",
            rr.Scalars(float(row["delta_success"])),
            recording=recording,
        )
    recording.flush()
    recording.disconnect()
    if not path.is_file() or path.stat().st_size <= 0:
        raise SylvestComparisonError(
            "Rerun did not write a non-empty comparison recording"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for each stateless workflow stage.

    Args:
        None.

    Returns:
        Configured command-line parser.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--libero-plus-root", required=True)
    prepare.add_argument("--libero-plus-revision", required=True)
    prepare.add_argument("--training-task-ids-uri", required=True)
    prepare.add_argument("--task-suite", default="libero_spatial")
    prepare.add_argument("--initial-state-indices", default="0,1,2")
    rollout = commands.add_parser("rollout")
    rollout.add_argument("--protocol-uri", required=True)
    rollout.add_argument("--checkpoint-id", required=True)
    rollout.add_argument("--checkpoint-revision", required=True)
    rollout.add_argument("--model-cache-root", required=True)
    rollout.add_argument("--openvla-oft-root", required=True)
    rollout.add_argument("--openvla-oft-revision", required=True)
    rollout.add_argument("--libero-plus-root", required=True)
    rollout.add_argument("--libero-plus-revision", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline-uri", required=True)
    compare.add_argument("--candidate-uri", required=True)
    report = commands.add_parser("report")
    report.add_argument("--comparison-uri", required=True)
    report.add_argument("--run-id", required=True)
    for command in (prepare, rollout, compare, report):
        command.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one source-pinned comparison stage and publish its artifacts.

    Args:
        argv: Explicit CLI arguments or process arguments when omitted.

    Returns:
        Zero after the declared stage output is published.

    Raises:
        SylvestComparisonError: A required source, input, or evidence invariant fails.
        OSError: Local artifact I/O fails.
    """

    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix=f"sylvest-{args.stage}-") as temporary:
        output = Path(temporary) / "output"
        if args.stage == "prepare":
            _prepare_protocol(args, output)
        elif args.stage == "rollout":
            _rollout(args, output)
        elif args.stage == "compare":
            _compare(args, output)
        else:
            _report(args, output)
        _publish_tree(output, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
