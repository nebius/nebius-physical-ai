"""Prepare and run pinned full-DROID pi0.5 fine-tuning on eight GPU nodes."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from npa.workflows.byof.openpi_pipeline import (
    OpenPIPipelineError,
    _download_checkpoint,
    _load_upstream_train_module,
    _read_json_uri,
    _read_bytes_uri,
    _redistribution_evidence,
    _source_build_evidence,
    _upload_checkpoint,
    _validate_runtime_image,
    _uri_exists,
    _write_bytes_uri,
    _write_json_uri,
)

SOURCE_REF = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
CONFIG_NAME = "pi05_full_droid_finetune"
DATASET_URI = "gs://gresearch/robotics/droid/1.0.1"
FILTER_DICTIONARY_URI = (
    "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"
)
FILTER_DICTIONARY_SHA256 = (
    "5046049ab62a2df2f802df89cf0888b720f852ce2557849417d40899c9a38bc8"
)
FILTER_DICTIONARY_BYTES = 28_573_266
EXPECTED_STEPS = 100_000
EXPECTED_BATCH_SIZE = 256
EXPECTED_PROCESSES = 8
EXPECTED_LOCAL_DEVICES = 1
EXPECTED_DEVICES = EXPECTED_PROCESSES * EXPECTED_LOCAL_DEVICES
EXPECTED_FSDP_DEVICES = EXPECTED_DEVICES
NORM_MAX_FRAMES = 10_000_000
PINNED_UPSTREAM_NORM_SHUFFLE_BUFFER_SIZE = 250_000
NORM_SHUFFLE_BUFFER_SIZE = 50_000
COORDINATOR_PORT = 29601
TELEMETRY_SCHEMA = "npa.workbench.openpi.pi05-full-droid-telemetry.v1"
PREPARATION_TELEMETRY_SCHEMA = (
    "npa.workbench.openpi.pi05-full-droid-preparation-telemetry.v1"
)
MILESTONE_MANIFEST_SCHEMA = (
    "npa.workbench.openpi.pi05-full-droid-rerun-milestone.v1"
)
CHECKPOINT_COMPLETION_SCHEMA = (
    "npa.workbench.openpi.pi05-full-droid-checkpoint-completion.v1"
)
RERUN_APPLICATION_ID = "npa_openpi_pi05_full_droid"
DEFAULT_RERUN_WORKER_PYTHON = Path("/opt/rerun-venv/bin/python")
RERUN_TIMELINE = "optimizer_step"
PREPARATION_TIMELINE = "normalization_batch"
RERUN_SCHEMA = "application/vnd.rerun.rrd"
QUALIFICATION_STEPS = 100
OPERATOR_PAUSE_UPDATES = 1_000
FULL_PROGRESS_MILESTONES = (1_000, 10_000, 25_000, 50_000, 75_000, 100_000)
REQUIRED_RRD_ENTITIES = (
    "metrics/loss",
    "metrics/learning_rate",
    "health/gradient_norm",
    "health/param_norm",
    "health/gradient_to_parameter_ratio",
    "health/nonfinite",
    "timing/interval_seconds",
    "throughput/optimizer_steps_per_second",
    "throughput/global_samples_per_second",
    "checkpoint/save_requested",
    "checkpoint/materialized",
    "health/distributed/process_count",
    "health/distributed/global_devices",
    "health/distributed/local_devices_per_process",
    "health/distributed/distinct_nodes",
    "health/device/sm120_ranks",
    "provenance/source_telemetry_sha256",
    "provenance/run",
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclasses.dataclass(frozen=True)
class _WritableActionsTransform:
    """Copy the DROID action view before upstream delta conversion mutates it."""

    def __call__(self, data: Mapping[str, object]) -> dict[str, object]:
        import numpy as np

        if "actions" not in data:
            raise OpenPIPipelineError(
                "pinned full-DROID transform input is missing actions"
            )
        source = np.asarray(data["actions"])
        actions = np.array(source, copy=True, order="K", subok=False)
        if (
            not actions.flags.writeable
            or actions.shape != source.shape
            or actions.dtype != source.dtype
            or np.shares_memory(actions, source)
        ):
            raise OpenPIPipelineError(
                "failed to isolate a writable full-DROID action tensor"
            )
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class _ReadOnlySafeDroidDataFactory:
    """Preserve the pinned factory while isolating its in-place action update."""

    delegate: object

    def create(self, assets_dirs: Path, model_config: object):
        data_config = self.delegate.create(assets_dirs, model_config)
        transforms = tuple(data_config.data_transforms.inputs)
        transform_names = tuple(type(transform).__name__ for transform in transforms)
        if transform_names != ("DroidInputs", "DeltaActions"):
            raise OpenPIPipelineError(
                "pinned full-DROID data transform sequence drifted"
            )
        data_transforms = dataclasses.replace(
            data_config.data_transforms,
            inputs=(transforms[0], _WritableActionsTransform(), transforms[1]),
        )
        return dataclasses.replace(data_config, data_transforms=data_transforms)


def _scalar(value: object, *, name: str) -> float:
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise OpenPIPipelineError(f"telemetry {name} is not scalar")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise OpenPIPipelineError(f"telemetry {name} is not finite")
    return result


def _load_telemetry_records(path: Path, *, run_id: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenPIPipelineError(
                f"telemetry journal line {number} is invalid JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("schema") != TELEMETRY_SCHEMA
            or value.get("run_id") != run_id
        ):
            raise OpenPIPipelineError(
                f"telemetry journal line {number} has incompatible provenance"
            )
        optimizer_step = value.get("optimizer_step")
        segment = value.get("segment")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or isinstance(segment, bool)
            or not isinstance(segment, int)
            or segment < 1
        ):
            raise OpenPIPipelineError(
                f"telemetry journal line {number} has invalid step or segment"
            )
        records.append(value)
    return records


class _TrainingTelemetryJournal:
    """Durable rank-zero facts captured around the pinned upstream trainer."""

    def __init__(self, path: Path, *, run_id: str, config: object) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise OpenPIPipelineError(f"unsafe telemetry run id {run_id!r}")
        self.path = path
        self.run_id = run_id
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = _load_telemetry_records(path, run_id=run_id)
        self._metric_steps = {
            int(record["optimizer_step"])
            for record in records
            if record.get("record_type") == "metrics"
        }
        self._checkpoint_events = {
            (int(record["optimizer_step"]), str(record["event"]))
            for record in records
            if record.get("record_type") == "checkpoint"
        }
        self._has_provenance = any(
            record.get("record_type") == "provenance" for record in records
        )
        self._segment = 1 + max(
            (int(record.get("segment", 0)) for record in records), default=0
        )
        self._started = time.perf_counter()
        self._last_metric_step: int | None = None
        self._last_metric_time: float | None = None
        self._handle = self.path.open("a", encoding="utf-8")
        if not self._has_provenance:
            self._append(
                {
                    "record_type": "provenance",
                    "optimizer_step": 0,
                    "source_ref": SOURCE_REF,
                    "config_name": CONFIG_NAME,
                    "dataset_uri": DATASET_URI,
                    "optimizer_steps": int(config.num_train_steps),
                    "global_batch_size": int(config.batch_size),
                    "log_interval": int(config.log_interval),
                    "save_interval": int(config.save_interval),
                    "held_out_policy_comparison": "not_produced_by_training_run",
                }
            )
            self._has_provenance = True

    def _append(self, value: Mapping[str, object]) -> None:
        record = {
            "schema": TELEMETRY_SCHEMA,
            "run_id": self.run_id,
            "segment": self._segment,
            **value,
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def record_metrics(
        self, *, step: int, values: Mapping[str, object], learning_rate: object
    ) -> None:
        if step in self._metric_steps:
            return
        required = {"loss", "grad_norm", "param_norm"}
        if not required <= values.keys():
            return
        metrics = {
            name: _scalar(values[name], name=name) for name in sorted(required)
        }
        lr = _scalar(learning_rate, name="learning_rate")
        now = time.perf_counter()
        interval: dict[str, object] | None = None
        if self._last_metric_time is not None and self._last_metric_step is not None:
            seconds = now - self._last_metric_time
            steps = step - self._last_metric_step
            if seconds <= 0 or steps <= 0:
                raise OpenPIPipelineError("telemetry interval is not monotonic")
            interval = {
                "optimizer_steps": steps,
                "seconds": seconds,
                "optimizer_steps_per_second": steps / seconds,
                "global_samples_per_second": (
                    steps * int(self.config.batch_size) / seconds
                ),
            }
        gradient_ratio = metrics["grad_norm"] / max(metrics["param_norm"], 1e-30)
        self._append(
            {
                "record_type": "metrics",
                "optimizer_step": step,
                "elapsed_segment_seconds": now - self._started,
                "metrics": {**metrics, "learning_rate": lr},
                "health": {
                    "all_finite": True,
                    "gradient_to_parameter_ratio": gradient_ratio,
                },
                "interval": interval,
            }
        )
        self._metric_steps.add(step)
        self._last_metric_step = step
        self._last_metric_time = now

    def record_checkpoint(self, *, step: int, event: str) -> None:
        identity = (step, event)
        if identity in self._checkpoint_events:
            return
        if step < 0 or event not in {"save_requested", "materialized"}:
            raise OpenPIPipelineError("invalid checkpoint telemetry event")
        self._append(
            {
                "record_type": "checkpoint",
                "optimizer_step": step,
                "event": event,
                "relative_path": f"{step}/",
                "final": step == int(self.config.num_train_steps) - 1,
            }
        )
        self._checkpoint_events.add(identity)

    def close(self) -> None:
        self._handle.close()


def _require_terms() -> None:
    if os.environ.get("NPA_OPENPI_ACCEPT_GEMMA_TERMS") != "YES":
        raise OpenPIPipelineError(
            "full-DROID pi0.5 fine-tuning requires exact run-scoped Gemma terms acceptance"
        )


def _run(
    command: Sequence[str], *, cwd: Path | None = None, stdout: object = None
) -> None:
    subprocess.run(command, cwd=cwd, check=True, stdout=stdout)  # noqa: S603


def _validate_source(repo_root: Path, runtime_image: str) -> dict[str, object]:
    _validate_runtime_image(runtime_image)
    build = _source_build_evidence(repo_root)
    source_metadata = build.get("source_metadata")
    if (
        not isinstance(source_metadata, dict)
        or source_metadata.get("ref") != SOURCE_REF
    ):
        raise OpenPIPipelineError(
            "runtime image does not contain the pinned OpenPI source"
        )
    return build


def _remote_inventory(gsutil: str, manifest_path: Path) -> dict[str, object]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        _run([gsutil, "ls", "-l", "-r", DATASET_URI + "/**"], stdout=handle)
    content = manifest_path.read_bytes()
    count = 0
    total = 0
    for line in content.decode("utf-8", errors="strict").splitlines():
        match = re.match(r"^\s*(\d+)\s+\d{4}-\d{2}-\d{2}T\S+\s+gs://", line)
        if match:
            count += 1
            total += int(match.group(1))
    if count < 1 or total < 1:
        raise OpenPIPipelineError("DROID 1.0.1 GCS inventory was empty or unreadable")
    return {
        "uri": DATASET_URI,
        "version": "1.0.1",
        "object_count": count,
        "total_size_bytes": total,
        "listing_sha256": hashlib.sha256(content).hexdigest(),
    }


def _local_inventory(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
    }


def _stage_dataset(gsutil: str, data_root: Path, work_root: Path) -> dict[str, object]:
    destination = data_root / "droid" / "1.0.1"
    destination.mkdir(parents=True, exist_ok=True)
    inventory_started = time.perf_counter()
    remote = _remote_inventory(gsutil, work_root / "droid-1.0.1-gcs-listing.txt")
    inventory_seconds = time.perf_counter() - inventory_started
    sync_started = time.perf_counter()
    _run([gsutil, "-m", "rsync", "-r", "-c", DATASET_URI, str(destination)])
    sync_seconds = time.perf_counter() - sync_started
    local_started = time.perf_counter()
    local = _local_inventory(destination)
    local_inventory_seconds = time.perf_counter() - local_started
    if local["file_count"] != remote["object_count"]:
        raise OpenPIPipelineError(
            "DROID object count differs after checksum-verified GCS synchronization"
        )
    if local["total_size_bytes"] != remote["total_size_bytes"]:
        raise OpenPIPipelineError(
            "DROID byte count differs after checksum-verified GCS synchronization"
        )
    return {
        **remote,
        **local,
        "remote_total_size_bytes": remote["total_size_bytes"],
        "local_total_size_bytes": local["total_size_bytes"],
        "local_path_role": "run_owned_durable_pvc",
        "timings_seconds": {
            "remote_inventory": inventory_seconds,
            "checksum_sync": sync_seconds,
            "local_inventory": local_inventory_seconds,
        },
        "checksum_verification_bytes_per_second": (
            remote["total_size_bytes"] / sync_seconds
        ),
    }


def _configure_openpi_cache(work_root: Path) -> Path:
    cache_root = work_root / "openpi-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["OPENPI_DATA_HOME"] = str(cache_root)
    return cache_root


def _validate_filter_dictionary(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenPIPipelineError(
            "pinned DROID filter dictionary is absent or invalid JSON"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != FILTER_DICTIONARY_BYTES or digest != FILTER_DICTIONARY_SHA256:
        raise OpenPIPipelineError(
            "pinned DROID filter dictionary failed byte identity validation"
        )
    if not isinstance(decoded, dict) or not decoded:
        raise OpenPIPipelineError(
            "pinned DROID filter dictionary must be a non-empty mapping"
        )
    return {
        "source_uri": FILTER_DICTIONARY_URI,
        "sha256": digest,
        "size_bytes": len(payload),
        "entry_count": len(decoded),
        "source_ref": SOURCE_REF,
        "path_role": "run_owned_durable_openpi_cache",
    }


def _stage_filter_dictionary(gsutil: str, cache_root: Path) -> dict[str, object]:
    """Fetch upstream's single JSON object without its broken wildcard helper."""

    target = (
        cache_root
        / "openpi-assets"
        / "droid"
        / "droid_sample_ranges_v1_0_1.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            evidence = _validate_filter_dictionary(target)
        except OpenPIPipelineError:
            partial = target.with_name(target.name + f".{os.getpid()}.partial")
            partial.unlink(missing_ok=True)
            try:
                command = [gsutil, "cp", FILTER_DICTIONARY_URI, str(partial)]
                if any("*" in item or "?" in item for item in command):
                    raise OpenPIPipelineError(
                        "filter dictionary download must address one exact object"
                    )
                _run(command)
                evidence = _validate_filter_dictionary(partial)
                with partial.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(partial, target)
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                partial.unlink(missing_ok=True)
            evidence = _validate_filter_dictionary(target)
            return {**evidence, "cache_reused": False}
        return {**evidence, "cache_reused": True}


def _configured_upstream(
    data_root: Path,
    work_root: Path,
    experiment: str,
    *,
    qualification: bool = False,
    pause_after_updates: int = 0,
):
    from openpi.training import config as openpi_config

    config = openpi_config.get_config(CONFIG_NAME)
    if (
        config.num_train_steps != EXPECTED_STEPS
        or config.batch_size != EXPECTED_BATCH_SIZE
    ):
        raise OpenPIPipelineError("pinned upstream full-DROID recipe drifted")
    data = _ReadOnlySafeDroidDataFactory(
        dataclasses.replace(config.data, rlds_data_dir=str(data_root))
    )
    configured = dataclasses.replace(
        config,
        data=data,
        assets_base_dir=str(work_root / "assets"),
        checkpoint_base_dir=str(work_root / "checkpoints"),
        exp_name=experiment,
        fsdp_devices=EXPECTED_FSDP_DEVICES,
        wandb_enabled=False,
    )
    if qualification and pause_after_updates:
        raise OpenPIPipelineError("qualification cannot use a full-run pause")
    if qualification:
        configured = dataclasses.replace(
            configured,
            num_train_steps=QUALIFICATION_STEPS,
            log_interval=1,
            save_interval=QUALIFICATION_STEPS,
        )
    elif pause_after_updates:
        if pause_after_updates != OPERATOR_PAUSE_UPDATES:
            raise OpenPIPipelineError(
                "the explicit full-DROID pause must be exactly 1,000 updates"
            )
        configured = dataclasses.replace(
            configured,
            num_train_steps=pause_after_updates,
            log_interval=1,
            save_interval=pause_after_updates,
        )
    return configured


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_preparation_telemetry(
    path: Path, *, run_id: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenPIPipelineError(
                f"preparation journal line {number} is invalid JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("schema") != PREPARATION_TELEMETRY_SCHEMA
            or value.get("run_id") != run_id
        ):
            raise OpenPIPipelineError(
                f"preparation journal line {number} has incompatible provenance"
            )
        records.append(value)
    return records


def _record_dataset_verification_once(
    journal_path: Path, *, run_id: str, dataset: Mapping[str, object]
) -> None:
    identity = {
        "remote_object_count": dataset["object_count"],
        "local_file_count": dataset["file_count"],
        "remote_size_bytes": dataset["remote_total_size_bytes"],
        "local_size_bytes": dataset["local_total_size_bytes"],
        "listing_sha256": dataset["listing_sha256"],
    }
    records = (
        _load_preparation_telemetry(journal_path, run_id=run_id)
        if journal_path.is_file()
        else []
    )
    if any(
        record.get("record_type") == "dataset_verified"
        and all(record.get(key) == value for key, value in identity.items())
        for record in records
    ):
        return
    _append_jsonl(
        journal_path,
        {
            "schema": PREPARATION_TELEMETRY_SCHEMA,
            "run_id": run_id,
            "record_type": "dataset_verified",
            "normalization_batch": 0,
            **identity,
            "checksum_verification_seconds": dataset["timings_seconds"][
                "checksum_sync"
            ],
            "checksum_verification_bytes_per_second": dataset[
                "checksum_verification_bytes_per_second"
            ],
        },
    )


def _reuse_verified_dataset(
    journal_path: Path, *, run_id: str, data_root: Path, work_root: Path
) -> dict[str, object] | None:
    """Reuse a durable checksum result only after revalidating its local identity."""

    if not journal_path.is_file():
        return None
    records = _load_preparation_telemetry(journal_path, run_id=run_id)
    verified = [
        record for record in records if record.get("record_type") == "dataset_verified"
    ]
    if not verified:
        return None
    record = verified[-1]
    listing_path = work_root / "droid-1.0.1-gcs-listing.txt"
    if not listing_path.is_file():
        raise OpenPIPipelineError(
            "durable dataset verification exists but its source listing is absent"
        )
    listing_sha256 = hashlib.sha256(listing_path.read_bytes()).hexdigest()
    if listing_sha256 != record.get("listing_sha256"):
        raise OpenPIPipelineError(
            "durable dataset verification listing identity changed"
        )
    started = time.perf_counter()
    local = _local_inventory(data_root / "droid" / "1.0.1")
    local_seconds = time.perf_counter() - started
    if (
        local["file_count"] != record.get("local_file_count")
        or local["total_size_bytes"] != record.get("local_size_bytes")
        or record.get("remote_object_count") != record.get("local_file_count")
        or record.get("remote_size_bytes") != record.get("local_size_bytes")
    ):
        raise OpenPIPipelineError(
            "durable checksum state no longer matches the local DROID dataset"
        )
    return {
        "uri": DATASET_URI,
        "version": "1.0.1",
        "object_count": record["remote_object_count"],
        "file_count": local["file_count"],
        "total_size_bytes": local["total_size_bytes"],
        "remote_total_size_bytes": record["remote_size_bytes"],
        "local_total_size_bytes": local["total_size_bytes"],
        "listing_sha256": listing_sha256,
        "local_path_role": "run_owned_durable_pvc",
        "verification_reused": True,
        "verification_source": "durable_factual_preparation_journal",
        "timings_seconds": {
            "remote_inventory": 0.0,
            "checksum_sync": record["checksum_verification_seconds"],
            "local_inventory": local_seconds,
        },
        "checksum_verification_bytes_per_second": record[
            "checksum_verification_bytes_per_second"
        ],
    }


class _FactualNormalizationBatchTracker:
    """Count only batches yielded by the pinned normalization loader."""

    def __init__(self, expected_batches: int, on_batch) -> None:
        self.expected_batches = expected_batches
        self.on_batch = on_batch
        self.processed_batches = 0
        self._wrapped = False

    def wrap(self, loader):
        if self._wrapped:
            raise OpenPIPipelineError(
                "normalization loader was constructed more than once"
            )
        self._wrapped = True
        for batch in loader:
            self.processed_batches += 1
            self.on_batch(self.processed_batches)
            yield batch

    def assert_complete(self) -> None:
        if not self._wrapped or self.processed_batches != self.expected_batches:
            raise OpenPIPipelineError(
                "normalization did not consume the pinned number of factual batches"
            )


def _peak_rss_bytes() -> int:
    """Return Linux ru_maxrss in bytes for preparation resource telemetry."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _peak_memory_observation() -> dict[str, int]:
    """Capture both process RSS and container/process-tree cgroup memory."""

    process_rss = _peak_rss_bytes()
    cgroup_values: list[int] = []
    for path in (
        Path("/sys/fs/cgroup/memory.peak"),
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
    ):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if value.isdecimal():
            cgroup_values.append(int(value))
    cgroup_peak = max(cgroup_values, default=0)
    return {
        "peak_rss_bytes": process_rss,
        "cgroup_peak_memory_bytes": cgroup_peak,
        "peak_memory_bytes": max(process_rss, cgroup_peak),
    }


@contextlib.contextmanager
def _normalization_dataset_memory_override(data_loader_module: object):
    """Bound only normalization's decoded-frame shuffle buffer.

    The full training and qualification data paths retain the pinned upstream
    default.  Fail closed if the pinned constructor or its caller contract
    changes, and always restore the module before returning to either path.
    """

    original = getattr(data_loader_module, "DroidRldsDataset", None)
    if original is None:
        raise OpenPIPipelineError(
            "pinned normalization dataset constructor is unavailable"
        )
    parameter = inspect.signature(original).parameters.get("shuffle_buffer_size")
    if (
        parameter is None
        or parameter.default != PINNED_UPSTREAM_NORM_SHUFFLE_BUFFER_SIZE
    ):
        raise OpenPIPipelineError(
            "pinned normalization shuffle-buffer contract changed"
        )

    @functools.wraps(original)
    def bounded_dataset(*args, **kwargs):
        requested = kwargs.get(
            "shuffle_buffer_size", PINNED_UPSTREAM_NORM_SHUFFLE_BUFFER_SIZE
        )
        if requested != PINNED_UPSTREAM_NORM_SHUFFLE_BUFFER_SIZE:
            raise OpenPIPipelineError(
                "pinned normalization caller shuffle-buffer contract changed"
            )
        kwargs["shuffle_buffer_size"] = NORM_SHUFFLE_BUFFER_SIZE
        return original(*args, **kwargs)

    setattr(data_loader_module, "DroidRldsDataset", bounded_dataset)
    try:
        yield
    finally:
        changed = (
            getattr(data_loader_module, "DroidRldsDataset", None)
            is not bounded_dataset
        )
        setattr(data_loader_module, "DroidRldsDataset", original)
        if changed:
            raise OpenPIPipelineError(
                "normalization dataset override was unexpectedly mutated"
            )


def _compute_norm_stats(
    config: object, repo_root: Path, *, run_id: str, journal_path: Path
) -> dict[str, object]:
    from openpi.shared import normalize
    from openpi.training import config as openpi_config
    from openpi.training import data_loader as openpi_data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    stats_path = config.assets_dirs / data_config.repo_id / "norm_stats.json"
    expected_batches = NORM_MAX_FRAMES // int(config.batch_size)
    records = (
        _load_preparation_telemetry(journal_path, run_id=run_id)
        if journal_path.is_file()
        else []
    )
    completed = [
        record
        for record in records
        if record.get("record_type") == "normalization_complete"
        and record.get("normalization_batch") == expected_batches
    ]
    if stats_path.is_file() and not completed:
        # A prior interrupted attempt may have written numerical output before
        # its factual coverage checks ran. Preserve those bytes for private
        # diagnosis, but never accept or overwrite an unlinked result.
        digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        quarantine = stats_path.with_name(f"norm_stats.unverified.{digest}.json")
        if quarantine.exists() and quarantine.read_bytes() != stats_path.read_bytes():
            raise OpenPIPipelineError(
                "normalization quarantine collision has different bytes"
            )
        if not quarantine.exists():
            stats_path.replace(quarantine)
        else:
            stats_path.unlink()
    normalization_attempt = 1 + max(
        (
            int(record.get("normalization_attempt", 0))
            for record in records
            if str(record.get("record_type", "")).startswith("normalization_")
        ),
        default=0,
    )
    if not stats_path.is_file():
        module_path = repo_root / "scripts" / "compute_norm_stats.py"
        namespace: dict[str, object] = {
            "__file__": str(module_path),
            "__name__": "npa_openpi_norm",
        }
        original = openpi_config.get_config
        processed_batches = 0
        started = time.perf_counter()
        create_rlds_dataloader = None
        original_create_rlds_dataloader = None
        target_loader_calls = 0

        def record_progress(batch: int) -> None:
            nonlocal processed_batches
            processed_batches = batch
            if processed_batches % 1_000 == 0:
                elapsed = time.perf_counter() - started
                memory = _peak_memory_observation()
                _append_jsonl(
                    journal_path,
                    {
                        "schema": PREPARATION_TELEMETRY_SCHEMA,
                        "run_id": run_id,
                        "record_type": "normalization_progress",
                        "normalization_attempt": normalization_attempt,
                        "normalization_batch": processed_batches,
                        "frames_processed": processed_batches
                        * int(config.batch_size),
                        "elapsed_seconds": elapsed,
                        "frames_per_second": (
                            processed_batches * int(config.batch_size) / elapsed
                        ),
                        "normalization_shuffle_buffer_size": NORM_SHUFFLE_BUFFER_SIZE,
                        "normalization_resume_mode": "full_restart",
                        **memory,
                    },
                )

        tracker = _FactualNormalizationBatchTracker(
            expected_batches, record_progress
        )
        try:
            openpi_config.get_config = lambda name: (
                config if name == CONFIG_NAME else original(name)
            )
            exec(compile(module_path.read_bytes(), str(module_path), "exec"), namespace)  # noqa: S102
            if namespace.get("_data_loader") is not openpi_data_loader:
                raise OpenPIPipelineError(
                    "pinned normalization script data-loader alias changed"
                )
            create_rlds_dataloader = namespace["create_rlds_dataloader"]
            original_create_rlds_dataloader = create_rlds_dataloader
            create_rlds_dataset = getattr(
                openpi_data_loader, "create_rlds_dataset", None
            )
            if (
                not callable(create_rlds_dataset)
                or getattr(create_rlds_dataset, "__globals__", {}).get(
                    "DroidRldsDataset"
                )
                is not openpi_data_loader.DroidRldsDataset
            ):
                raise OpenPIPipelineError(
                    "pinned normalization dataset call target changed"
                )

            def tracking_create_rlds_dataloader(*args, **kwargs):
                nonlocal target_loader_calls
                loader, num_batches = original_create_rlds_dataloader(
                    *args, **kwargs
                )
                target_loader_calls += 1
                if target_loader_calls != 1 or num_batches != expected_batches:
                    raise OpenPIPipelineError(
                        "pinned normalization loader contract changed"
                    )

                return tracker.wrap(loader), num_batches

            namespace["create_rlds_dataloader"] = tracking_create_rlds_dataloader
            with _normalization_dataset_memory_override(openpi_data_loader):
                namespace["main"](CONFIG_NAME, max_frames=NORM_MAX_FRAMES)  # type: ignore[operator]
            if target_loader_calls != 1:
                raise OpenPIPipelineError(
                    "pinned normalization loader contract changed"
                )
            tracker.assert_complete()
            elapsed = time.perf_counter() - started
            memory = _peak_memory_observation()
            _append_jsonl(
                journal_path,
                {
                    "schema": PREPARATION_TELEMETRY_SCHEMA,
                    "run_id": run_id,
                    "record_type": "normalization_complete",
                    "normalization_attempt": normalization_attempt,
                    "normalization_batch": processed_batches,
                    "frames_processed": processed_batches * int(config.batch_size),
                    "elapsed_seconds": elapsed,
                    "frames_per_second": (
                        processed_batches * int(config.batch_size) / elapsed
                    ),
                    "normalization_shuffle_buffer_size": NORM_SHUFFLE_BUFFER_SIZE,
                    "normalization_resume_mode": "full_restart",
                    **memory,
                },
            )
        except Exception as exc:
            memory = _peak_memory_observation()
            _append_jsonl(
                journal_path,
                {
                    "schema": PREPARATION_TELEMETRY_SCHEMA,
                    "run_id": run_id,
                    "record_type": "normalization_incomplete",
                    "normalization_attempt": normalization_attempt,
                    "normalization_batch": processed_batches,
                    "frames_processed": processed_batches * int(config.batch_size),
                    "failure_type": type(exc).__name__,
                    "normalization_shuffle_buffer_size": NORM_SHUFFLE_BUFFER_SIZE,
                    "normalization_resume_mode": "full_restart",
                    **memory,
                },
            )
            raise
        finally:
            openpi_config.get_config = original
            if (
                create_rlds_dataloader is not None
                and original_create_rlds_dataloader is not None
            ):
                namespace["create_rlds_dataloader"] = (
                    original_create_rlds_dataloader
                )
        records = _load_preparation_telemetry(journal_path, run_id=run_id)
    if not stats_path.is_file():
        raise OpenPIPipelineError("normalization statistics were not materialized")
    loaded = normalize.load(stats_path.parent)
    if not loaded:
        raise OpenPIPipelineError("normalization statistics are empty")
    completed = [
        record
        for record in records
        if record.get("record_type") == "normalization_complete"
        and record.get("normalization_batch") == expected_batches
    ]
    if not completed:
        raise OpenPIPipelineError(
            "normalization statistics lack matching factual progress telemetry"
        )
    final = completed[-1]
    return {
        "path_role": "run_owned_durable_pvc",
        "max_frames": NORM_MAX_FRAMES,
        "batches_processed": expected_batches,
        "frames_processed": expected_batches * int(config.batch_size),
        "normalization_attempt": final.get("normalization_attempt", 0),
        "elapsed_seconds": final["elapsed_seconds"],
        "frames_per_second": final["frames_per_second"],
        "normalization_shuffle_buffer_size": final[
            "normalization_shuffle_buffer_size"
        ],
        "peak_rss_bytes": final["peak_rss_bytes"],
        "cgroup_peak_memory_bytes": final["cgroup_peak_memory_bytes"],
        "peak_memory_bytes": final["peak_memory_bytes"],
        "resume_mode": final["normalization_resume_mode"],
        "sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
    }


def _prepare(args: argparse.Namespace) -> int:
    _require_terms()
    # Preparation is intentionally CPU-only.  The runtime image also carries
    # CUDA-enabled JAX for the later qualification/training stages, and JAX's
    # plugin discovery otherwise calls cuInit while normalization imports the
    # upstream trainer.  On a CPU Kubernetes pod that fails before the factual
    # normalization iterator can start.  Scope the platform override to this
    # preparation process; GPU stages run in separate pods and retain CUDA.
    os.environ["JAX_PLATFORMS"] = "cpu"
    repo_root = Path(args.repo_root)
    build = _validate_source(repo_root, args.runtime_image)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    cache_root = _configure_openpi_cache(work_root)
    started = time.perf_counter()
    filter_dictionary = _stage_filter_dictionary(args.gsutil, cache_root)
    preparation_journal = work_root / "telemetry" / "preparation.jsonl"
    dataset = _reuse_verified_dataset(
        preparation_journal,
        run_id=args.run_id,
        data_root=Path(args.data_root),
        work_root=work_root,
    ) or _stage_dataset(args.gsutil, Path(args.data_root), work_root)
    _record_dataset_verification_once(
        preparation_journal, run_id=args.run_id, dataset=dataset
    )
    config = _configured_upstream(Path(args.data_root), work_root, args.experiment)
    normalization = _compute_norm_stats(
        config,
        repo_root,
        run_id=args.run_id,
        journal_path=preparation_journal,
    )
    result: dict[str, object] = {
        "schema": "npa.workbench.openpi.pi05-full-droid-prepare.v1",
        "status": "passed",
        "source": {
            "repository": "https://github.com/Physical-Intelligence/openpi",
            "ref": SOURCE_REF,
            "license": "Apache-2.0",
            **build,
        },
        "dataset": dataset,
        "filter_dictionary": filter_dictionary,
        "normalization": normalization,
        "recipe": {
            "config_name": CONFIG_NAME,
            "global_batch_size": EXPECTED_BATCH_SIZE,
            "optimizer_steps": EXPECTED_STEPS,
            "normalization_max_frames": NORM_MAX_FRAMES,
        },
        "terms": {"forwarded": True, "persisted": False},
        "timings_seconds": {"total": round(time.perf_counter() - started, 3)},
    }
    result["rerun"] = _publish_preparation_rrd(
        preparation_journal,
        rrd_uri=args.rrd_uri,
        manifest_uri=args.milestone_manifest_uri,
        run_id=args.run_id,
        result=result,
        runtime_image=args.runtime_image,
    )
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _multihost_environment() -> tuple[int, list[str]]:
    try:
        rank = int(os.environ["SKYPILOT_NODE_RANK"])
        num_nodes = int(os.environ["SKYPILOT_NUM_NODES"])
    except (KeyError, ValueError) as exc:
        raise OpenPIPipelineError(
            "SkyPilot multi-node rank metadata is required"
        ) from exc
    node_ips = [
        value.strip()
        for value in os.environ.get("SKYPILOT_NODE_IPS", "").splitlines()
        if value.strip()
    ]
    if num_nodes != EXPECTED_PROCESSES or len(node_ips) != EXPECTED_PROCESSES:
        raise OpenPIPipelineError(
            f"full-DROID requires {EXPECTED_PROCESSES} SkyPilot nodes, got {num_nodes} and {len(node_ips)} IPs"
        )
    if rank not in range(EXPECTED_PROCESSES):
        raise OpenPIPipelineError(f"invalid SkyPilot node rank {rank}")
    visible = os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", "")
    if visible and int(float(visible)) != EXPECTED_LOCAL_DEVICES:
        raise OpenPIPipelineError(
            f"expected one GPU per node, SkyPilot reported {visible}"
        )
    return rank, node_ips


def _initialize_multihost(rank: int, node_ips: Sequence[str]) -> object:
    import jax
    from jax.experimental import multihost_utils

    jax.distributed.initialize(
        coordinator_address=f"{node_ips[0]}:{COORDINATOR_PORT}",
        num_processes=EXPECTED_PROCESSES,
        process_id=rank,
        local_device_ids=[0],
    )
    if jax.process_count() != EXPECTED_PROCESSES or jax.process_index() != rank:
        raise OpenPIPipelineError(
            "JAX process topology differs from the SkyPilot topology"
        )
    if (
        jax.device_count() != EXPECTED_DEVICES
        or jax.local_device_count() != EXPECTED_LOCAL_DEVICES
    ):
        raise OpenPIPipelineError(
            f"expected {EXPECTED_DEVICES} global and one local GPU, got "
            f"{jax.device_count()} global and {jax.local_device_count()} local"
        )
    multihost_utils.sync_global_devices("npa-openpi-initialized")
    return multihost_utils


def _install_distributed_rlds_adapter(rank: int) -> None:
    """Adapt the pinned RLDS loader without changing the upstream recipe."""

    import dlimp as dl
    import jax
    import tensorflow as tf
    from openpi.training import data_loader

    tf.random.set_seed(42 + rank)
    original_sample = dl.DLataset.sample_from_datasets

    def sharded_sample(*args, **kwargs):
        dataset = original_sample(*args, **kwargs)
        return dataset.shard(EXPECTED_PROCESSES, rank)

    dl.DLataset.sample_from_datasets = sharded_sample
    original_create = data_loader.create_rlds_dataset

    def create_local_rlds(data_config, action_horizon, batch_size, *, shuffle=False):
        if batch_size % EXPECTED_PROCESSES:
            raise OpenPIPipelineError(
                "global RLDS batch is not divisible by the process count"
            )
        return original_create(
            data_config,
            action_horizon,
            batch_size // EXPECTED_PROCESSES,
            shuffle=shuffle,
        )

    data_loader.create_rlds_dataset = create_local_rlds

    def rlds_init(self, dataset, *, sharding=None, num_batches=None):
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._dataset = dataset
        self._sharding = sharding
        self._num_batches = num_batches

    data_loader.RLDSDataLoader.__init__ = rlds_init


def _checkpoint_root(config: object) -> Path:
    return Path(config.checkpoint_base_dir) / config.name / config.exp_name


def _checkpoint_completion_marker(checkpoint_root: Path, step: int) -> Path:
    return checkpoint_root / str(step) / ".npa-checkpoint-complete.json"


def _write_checkpoint_completion_marker(
    checkpoint_root: Path, *, step: int, run_id: str
) -> None:
    step_path = checkpoint_root / str(step)
    if not step_path.is_dir() or not any(
        path.name != ".npa-checkpoint-complete.json"
        for path in step_path.rglob("*")
    ):
        raise OpenPIPipelineError(
            "milestone checkpoint did not materialize before RRD emission"
        )
    marker = _checkpoint_completion_marker(checkpoint_root, step)
    payload = (
        json.dumps(
            {
                "schema": CHECKPOINT_COMPLETION_SCHEMA,
                "run_id": run_id,
                "optimizer_step": step,
                "upstream_source_ref": SOURCE_REF,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    if marker.is_file():
        if marker.read_bytes() != payload:
            raise OpenPIPipelineError("checkpoint completion marker provenance differs")
        return
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        directory_fd = os.open(step_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_completion_is_valid(
    checkpoint_root: Path, *, step: int, run_id: str
) -> bool:
    marker = _checkpoint_completion_marker(checkpoint_root, step)
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value == {
        "schema": CHECKPOINT_COMPLETION_SCHEMA,
        "run_id": run_id,
        "optimizer_step": step,
        "upstream_source_ref": SOURCE_REF,
    }


def _prepare_distributed_checkpoint_root(
    checkpoint_root: Path, *, rank: int, multihost_utils: object
) -> bool:
    """Prepare one shared checkpoint root without concurrent deletion.

    Upstream applies ``overwrite`` independently on every process.  On RWX
    storage that makes all eight ranks recursively delete the same directory.
    A fresh attempt instead has rank zero remove any attempt-scoped leftovers,
    then every rank crosses the same barrier before Orbax opens the root.
    """

    import numpy as np

    rank_zero_resuming = rank == 0 and checkpoint_root.is_dir() and any(
        path.is_dir() and path.name.isdigit() for path in checkpoint_root.iterdir()
    )
    decision = multihost_utils.broadcast_one_to_all(
        np.asarray([int(rank_zero_resuming)], dtype=np.int32), is_source=rank == 0
    )
    resuming = bool(int(np.asarray(decision).reshape(-1)[0]))
    if rank == 0 and not resuming:
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices("npa-openpi-checkpoint-root-cleaned")
    # RWX metadata visibility can lag the first barrier on a peer. mkdir with
    # exist_ok is safe on every rank and never removes rank-zero's state.
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices("npa-openpi-checkpoint-root-ready")
    if not checkpoint_root.is_dir():
        raise OpenPIPipelineError(
            "shared checkpoint root is absent after distributed preparation"
        )
    return resuming


def _non_destructive_checkpoint_config(config: object) -> object:
    """Let Orbax decide resume state without allowing any rank to delete."""

    return dataclasses.replace(config, resume=True, overwrite=False)


def _wrap_wandb_initializer(
    upstream_train: object,
    *,
    telemetry_log: object,
    active_log: list[object],
) -> tuple[object, object]:
    """Keep telemetry installed when wandb.init replaces its pre-init log."""

    original_init_wandb = upstream_train.init_wandb

    def telemetry_init_wandb(*positional, **keywords):
        result = original_init_wandb(*positional, **keywords)
        initialized_log = upstream_train.wandb.log
        if initialized_log is not telemetry_log:
            active_log[0] = initialized_log
        upstream_train.wandb.log = telemetry_log
        return result

    return original_init_wandb, telemetry_init_wandb


def _run_training(
    config: object,
    repo_root: Path,
    *,
    rank: int,
    run_id: str,
    multihost_utils: object,
    milestone_publisher: _TrainingMilestonePublisher | None = None,
) -> tuple[Path, bool, Path | None]:
    checkpoint_root = _checkpoint_root(config)
    resuming = _prepare_distributed_checkpoint_root(
        checkpoint_root, rank=rank, multihost_utils=multihost_utils
    )
    configured = _non_destructive_checkpoint_config(config)
    upstream_train = _load_upstream_train_module(repo_root)
    journal: _TrainingTelemetryJournal | None = None
    original_log = None
    original_init_wandb = None
    original_save = None
    original_initialize = upstream_train._checkpoints.initialize_checkpoint_dir
    if rank == 0:
        journal = _TrainingTelemetryJournal(
            Path(config.checkpoint_base_dir)
            / config.name
            / config.exp_name
            / "npa-training-telemetry.jsonl",
            run_id=run_id,
            config=configured,
        )
        if checkpoint_root.is_dir():
            for path in checkpoint_root.iterdir():
                if (
                    path.is_dir()
                    and path.name.isdigit()
                    and _checkpoint_completion_is_valid(
                        checkpoint_root, step=int(path.name), run_id=run_id
                    )
                ):
                    journal.record_checkpoint(
                        step=int(path.name), event="materialized"
                    )
        if milestone_publisher is not None:
            milestone_publisher.reconcile_available()
        learning_rate = configured.lr_schedule.create()
        original_log = upstream_train.wandb.log
        original_save = upstream_train._checkpoints.save_state
        save_parameters = tuple(inspect.signature(original_save).parameters)
        if save_parameters != ("checkpoint_manager", "state", "data_loader", "step"):
            raise OpenPIPipelineError(
                "pinned upstream checkpoint callback signature drifted"
            )

        active_log = [original_log]

        def telemetry_log(data, *positional, **keywords):
            result = active_log[0](data, *positional, **keywords)
            step = keywords.get("step")
            if step is None and positional:
                step = positional[0]
            if isinstance(data, Mapping) and step is not None:
                journal.record_metrics(
                    step=int(step),
                    values=data,
                    learning_rate=learning_rate(int(step)),
                )
                if (
                    milestone_publisher is not None
                    and milestone_publisher.is_log_only(int(step))
                ):
                    milestone_publisher.publish_for_optimizer_step(int(step))
            return result

        def telemetry_save(checkpoint_manager, state, data_loader, step):
            journal.record_checkpoint(step=int(step), event="save_requested")
            result = original_save(checkpoint_manager, state, data_loader, step)
            if (
                milestone_publisher is not None
                and milestone_publisher.requires_checkpoint(int(step))
            ):
                checkpoint_manager.wait_until_finished()
                _write_checkpoint_completion_marker(
                    checkpoint_root, step=int(step), run_id=run_id
                )
                journal.record_checkpoint(step=int(step), event="materialized")
                milestone_publisher.publish_for_optimizer_step(int(step))
            return result

        original_init_wandb, telemetry_init_wandb = _wrap_wandb_initializer(
            upstream_train,
            telemetry_log=telemetry_log,
            active_log=active_log,
        )
        upstream_train.wandb.log = telemetry_log
        upstream_train.init_wandb = telemetry_init_wandb
        upstream_train._checkpoints.save_state = telemetry_save

    def telemetry_initialize(*positional, **keywords):
        checkpoint_manager, upstream_resuming = original_initialize(
            *positional, **keywords
        )
        # Orbax all_steps(read=True) enumerates finalized checkpoints from
        # storage rather than trusting a partially populated directory. This
        # closes the crash window between async finalization and our marker.
        finalized_steps = {int(step) for step in checkpoint_manager.all_steps(read=True)}
        if rank == 0 and journal is not None and milestone_publisher is not None:
            checkpoint_aligned = {
                actual
                for actual in milestone_publisher.milestones.values()
                if milestone_publisher.requires_checkpoint(actual)
            }
            for step in sorted(finalized_steps & checkpoint_aligned):
                _write_checkpoint_completion_marker(
                    checkpoint_root, step=step, run_id=run_id
                )
                journal.record_checkpoint(step=step, event="materialized")
            milestone_publisher.reconcile_available()
        return checkpoint_manager, upstream_resuming

    upstream_train._checkpoints.initialize_checkpoint_dir = telemetry_initialize
    try:
        upstream_train.main(configured)
    finally:
        if original_log is not None:
            upstream_train.wandb.log = original_log
        if original_init_wandb is not None:
            upstream_train.init_wandb = original_init_wandb
        if original_save is not None:
            upstream_train._checkpoints.save_state = original_save
        upstream_train._checkpoints.initialize_checkpoint_dir = original_initialize
        if journal is not None:
            journal.close()
    final_step = int(configured.num_train_steps) - 1
    final_step_dir = checkpoint_root / str(final_step)
    if not final_step_dir.is_dir():
        raise OpenPIPipelineError(
            f"upstream trainer returned without final checkpoint step {final_step}"
        )
    return checkpoint_root, resuming, None if journal is None else journal.path


def _set_rerun_step(rr: object, recording: object, step: int) -> None:
    _set_rerun_time(rr, recording, RERUN_TIMELINE, step)


def _rerun_executable() -> str:
    worker_python = _rrd_worker_python()
    if worker_python is not None:
        worker_cli = worker_python.with_name("rerun")
        if worker_cli.is_file() and os.access(worker_cli, os.X_OK):
            return str(worker_cli)
        raise OpenPIPipelineError(
            "the isolated Rerun worker CLI is unavailable for RRD verification"
        )
    sibling = Path(sys.executable).with_name("rerun")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    value = shutil.which("rerun")
    if value:
        return value
    raise OpenPIPipelineError("rerun CLI is unavailable for RRD verification")


def _inspect_training_rrd(
    path: Path,
    *,
    run_id: str,
    source_telemetry_sha256: str,
    require_checkpoint: bool = True,
    expected_metric_steps: Sequence[int] | None = None,
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise OpenPIPipelineError("Rerun recording is absent or empty")
    executable = _rerun_executable()
    verified = subprocess.run(
        [executable, "rrd", "verify", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode:
        raise OpenPIPipelineError(
            f"Rerun could not verify recording: {verified.stderr[-1000:]}"
        )
    printed = subprocess.run(
        [executable, "rrd", "print", "-vv", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if printed.returncode:
        raise OpenPIPipelineError(
            f"Rerun could not inspect recording: {printed.stderr[-1000:]}"
        )
    provenance = subprocess.run(
        [
            executable,
            "rrd",
            "print",
            "-vvv",
            "--entity",
            "provenance/source_telemetry_sha256",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if provenance.returncode:
        raise OpenPIPipelineError(
            f"Rerun could not inspect recording provenance: {provenance.stderr[-1000:]}"
        )
    decoded_loss = subprocess.run(
        [
            executable,
            "rrd",
            "print",
            "-vvv",
            "--entity",
            "metrics/loss",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if decoded_loss.returncode:
        raise OpenPIPipelineError(
            "Rerun could not decode the optimizer timeline: "
            f"{decoded_loss.stderr[-1000:]}"
        )
    decoded = (
        f"{printed.stdout}\n{printed.stderr}\n"
        f"{provenance.stdout}\n{provenance.stderr}"
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if source_telemetry_sha256 not in provenance.stdout:
        raise OpenPIPipelineError(
            "decoded RRD source-hash entity does not identify its journal bytes"
        )
    required_entities = list(REQUIRED_RRD_ENTITIES)
    if not require_checkpoint:
        required_entities = [
            entity
            for entity in required_entities
            if not entity.startswith("checkpoint/")
        ]
    required = [
        RERUN_APPLICATION_ID,
        run_id,
        RERUN_TIMELINE,
        *required_entities,
    ]
    missing = [value for value in required if value not in decoded]
    if missing:
        raise OpenPIPipelineError(
            "decoded RRD is missing required identity, timeline, or entities: "
            + ", ".join(missing)
        )
    decoded_steps = [
        int(value)
        for value in re.findall(
            r"┆\s*(\d+)\s*┆\s*\[[^\]]+\]", decoded_loss.stdout
        )
    ]
    if expected_metric_steps is not None and decoded_steps != list(
        expected_metric_steps
    ):
        raise OpenPIPipelineError(
            "decoded RRD optimizer coverage differs from its source journal"
        )
    return {
        "parseable": True,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "application_id": RERUN_APPLICATION_ID,
        "recording_id": run_id,
        "timelines": [RERUN_TIMELINE],
        "entities": [
            entity for entity in required_entities if entity in decoded
        ],
        "source_telemetry_sha256": source_telemetry_sha256,
        "decoded_metric_steps": decoded_steps,
    }


PREPARATION_RRD_ENTITIES = (
    "dataset/remote_object_count",
    "dataset/local_file_count",
    "dataset/remote_size_bytes",
    "dataset/local_size_bytes",
    "dataset/object_coverage_ratio",
    "dataset/byte_coverage_ratio",
    "normalization/frames_processed",
    "normalization/frames_per_second",
    "normalization/progress_ratio",
    "normalization/statistics_materialized",
    "normalization/shuffle_buffer_size",
    "resources/normalization_peak_rss_bytes",
    "resources/normalization_peak_memory_bytes",
    "timing/normalization_elapsed_seconds",
    "throughput/checksum_verification_bytes_per_second",
    "provenance/source_telemetry_sha256",
    "provenance/run",
)


def _inspect_preparation_rrd(
    path: Path, *, run_id: str, source_telemetry_sha256: str
) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise OpenPIPipelineError("preparation Rerun recording is absent or empty")
    executable = _rerun_executable()
    verified = subprocess.run(
        [executable, "rrd", "verify", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    printed = subprocess.run(
        [executable, "rrd", "print", "-vv", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    provenance = subprocess.run(
        [
            executable,
            "rrd",
            "print",
            "-vvv",
            "--entity",
            "provenance/source_telemetry_sha256",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode or printed.returncode or provenance.returncode:
        raise OpenPIPipelineError("Rerun could not verify preparation recording")
    decoded = printed.stdout + printed.stderr + provenance.stdout + provenance.stderr
    required = [
        RERUN_APPLICATION_ID,
        run_id,
        PREPARATION_TIMELINE,
        source_telemetry_sha256,
        *PREPARATION_RRD_ENTITIES,
    ]
    missing = [value for value in required if value not in decoded]
    if missing:
        raise OpenPIPipelineError(
            "decoded preparation RRD lacks required factual content: "
            + ", ".join(missing)
        )
    payload = path.read_bytes()
    return {
        "parseable": True,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "application_id": RERUN_APPLICATION_ID,
        "recording_id": run_id,
        "timelines": [PREPARATION_TIMELINE],
        "entities": list(PREPARATION_RRD_ENTITIES),
        "source_telemetry_sha256": source_telemetry_sha256,
    }


def _build_preparation_rrd_direct(
    journal_path: Path,
    output_path: Path,
    *,
    run_id: str,
    result: Mapping[str, object],
    runtime_image: str,
) -> dict[str, object]:
    import rerun as rr
    import rerun.blueprint as rrb

    records = _load_preparation_telemetry(journal_path, run_id=run_id)
    completed = [
        record
        for record in records
        if record.get("record_type") == "normalization_complete"
    ]
    if not completed:
        raise OpenPIPipelineError("preparation telemetry is not complete")
    completed_attempt = completed[-1].get("normalization_attempt", 0)
    progress = [
        record
        for record in records
        if record.get("record_type")
        in {"normalization_progress", "normalization_complete"}
        and record.get("normalization_attempt", 0) == completed_attempt
    ]
    dataset_records = [
        record for record in records if record.get("record_type") == "dataset_verified"
    ]
    if not dataset_records:
        raise OpenPIPipelineError(
            "preparation telemetry lacks a verified dataset inventory"
        )
    dataset_record = dataset_records[-1]
    journal_payload = journal_path.read_bytes()
    journal_sha256 = hashlib.sha256(journal_payload).hexdigest()
    dataset = result.get("dataset")
    normalization = result.get("normalization")
    if not isinstance(dataset, Mapping) or not isinstance(normalization, Mapping):
        raise OpenPIPipelineError("preparation result lacks dataset or normalization")
    remote_count = int(dataset_record["remote_object_count"])
    local_count = int(dataset_record["local_file_count"])
    remote_bytes = int(dataset_record["remote_size_bytes"])
    local_bytes = int(dataset_record["local_size_bytes"])
    if (
        remote_count != int(dataset["object_count"])
        or local_count != int(dataset["file_count"])
        or remote_bytes != int(dataset["remote_total_size_bytes"])
        or local_bytes != int(dataset["local_total_size_bytes"])
        or dataset_record.get("listing_sha256") != dataset.get("listing_sha256")
    ):
        raise OpenPIPipelineError(
            "preparation journal and result dataset coverage differ"
        )
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(origin="dataset", name="Dataset verification"),
            rrb.TimeSeriesView(origin="normalization", name="Normalization progress"),
            rrb.TimeSeriesView(origin="throughput", name="Preparation throughput"),
            rrb.TextDocumentView(origin="provenance", name="Run provenance"),
        ),
        _expanded_time_panel(rrb),
        auto_layout=False,
    )
    recording = rr.RecordingStream(RERUN_APPLICATION_ID, recording_id=run_id)
    rr.save(output_path, default_blueprint=blueprint, recording=recording)
    rr.log(
        "provenance/run",
        rr.TextDocument(
            "# pi0.5 full-DROID preparation\n\n"
            f"- run id: `{run_id}`\n"
            f"- upstream source ref: `{SOURCE_REF}`\n"
            f"- dataset listing sha256: `{dataset['listing_sha256']}`\n"
            f"- normalization sha256: `{normalization['sha256']}`\n"
            "- normalization-only shuffle buffer: "
            f"`{normalization['normalization_shuffle_buffer_size']}` frames\n"
            "- interrupted normalization policy: full restart from batch zero\n"
            f"- runtime image digest: `{_runtime_image_digest(runtime_image)}`\n"
            f"- source telemetry sha256: `{journal_sha256}`\n"
            "- inputs remain private; this recording contains aggregate facts only."
        ),
        static=True,
        recording=recording,
    )
    rr.log(
        "provenance/source_telemetry_sha256",
        rr.TextLog(journal_sha256),
        static=True,
        recording=recording,
    )
    _set_rerun_time(rr, recording, PREPARATION_TIMELINE, 0)
    for entity, value in {
        "dataset/remote_object_count": remote_count,
        "dataset/local_file_count": local_count,
        "dataset/remote_size_bytes": remote_bytes,
        "dataset/local_size_bytes": local_bytes,
        "dataset/object_coverage_ratio": local_count / remote_count,
        "dataset/byte_coverage_ratio": local_bytes / remote_bytes,
        "throughput/checksum_verification_bytes_per_second": float(
            dataset_record["checksum_verification_bytes_per_second"]
        ),
    }.items():
        rr.log(entity, rr.Scalars(float(value)), recording=recording)
    for record in progress:
        _set_rerun_time(
            rr, recording, PREPARATION_TIMELINE, int(record["normalization_batch"])
        )
        rr.log(
            "normalization/frames_processed",
            rr.Scalars(float(record["frames_processed"])),
            recording=recording,
        )
        rr.log(
            "normalization/frames_per_second",
            rr.Scalars(float(record["frames_per_second"])),
            recording=recording,
        )
        rr.log(
            "normalization/progress_ratio",
            rr.Scalars(
                float(record["frames_processed"])
                / float(normalization["frames_processed"])
            ),
            recording=recording,
        )
        rr.log(
            "timing/normalization_elapsed_seconds",
            rr.Scalars(float(record["elapsed_seconds"])),
            recording=recording,
        )
        rr.log(
            "normalization/shuffle_buffer_size",
            rr.Scalars(float(record["normalization_shuffle_buffer_size"])),
            recording=recording,
        )
        rr.log(
            "resources/normalization_peak_rss_bytes",
            rr.Scalars(float(record["peak_rss_bytes"])),
            recording=recording,
        )
        rr.log(
            "resources/normalization_peak_memory_bytes",
            rr.Scalars(float(record["peak_memory_bytes"])),
            recording=recording,
        )
        if record.get("record_type") == "normalization_complete":
            rr.log(
                "normalization/statistics_materialized",
                rr.Scalars(1.0),
                recording=recording,
            )
    try:
        recording.flush()
    finally:
        recording.disconnect()
    return _inspect_preparation_rrd(
        output_path, run_id=run_id, source_telemetry_sha256=journal_sha256
    )


def _set_rerun_time(
    rr: object, recording: object, timeline: str, sequence: int
) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, sequence, recording=recording)
    else:
        rr.set_time(timeline, sequence=sequence, recording=recording)


def _expanded_time_panel(rrb: object) -> object:
    """Build a panel accepted by both the vendor and isolated Rerun SDKs.

    Older Rerun SDKs do not accept ``timeline`` in the ``TimePanel``
    constructor.  The recording timeline is factual data set on every log
    operation, so selecting it in the blueprint is optional presentation state.
    """

    return rrb.TimePanel(state=rrb.PanelState.Expanded)


def _build_training_rrd_direct(
    journal_path: Path,
    output_path: Path,
    *,
    run_id: str,
    config: object,
    prepared: Mapping[str, object],
    runtime_image: str,
    hardware: Mapping[str, object],
    topology: Sequence[Mapping[str, object]],
    through_step: int | None = None,
    milestone: str = "final",
    require_checkpoint: bool = True,
) -> dict[str, object]:
    import rerun as rr
    import rerun.blueprint as rrb

    records = _load_telemetry_records(journal_path, run_id=run_id)
    source_telemetry_sha256 = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    metric_records = {
        int(record["optimizer_step"]): record
        for record in records
        if record.get("record_type") == "metrics"
    }
    final_step = int(config.num_train_steps) - 1
    if through_step is None:
        through_step = final_step
    if through_step < 0 or through_step > final_step:
        raise OpenPIPipelineError("RRD milestone is outside the optimizer run")
    expected_steps = list(range(0, through_step + 1, int(config.log_interval)))
    if sorted(metric_records) != expected_steps:
        raise OpenPIPipelineError(
            "telemetry journal does not cover every upstream logging step"
        )
    checkpoint_events = {
        (int(record["optimizer_step"]), str(record.get("event")))
        for record in records
        if record.get("record_type") == "checkpoint"
    }
    expected_requested = {
        *range(int(config.save_interval), through_step + 1, int(config.save_interval)),
    }
    if through_step == final_step:
        expected_requested.add(final_step)
    missing_requested = sorted(
        step
        for step in expected_requested
        if (step, "save_requested") not in checkpoint_events
    )
    if missing_requested:
        raise OpenPIPipelineError(
            "telemetry lacks configured checkpoint save requests"
        )
    if require_checkpoint:
        for event in ("save_requested", "materialized"):
            if (through_step, event) not in checkpoint_events:
                raise OpenPIPipelineError(
                    f"telemetry lacks milestone checkpoint {event} event"
                )
    if through_step == final_step:
        for event in ("save_requested", "materialized"):
            if (final_step, event) not in checkpoint_events:
                raise OpenPIPipelineError(
                    f"telemetry lacks final checkpoint {event} event"
                )
    if sum(bool(record.get("interval")) for record in metric_records.values()) < 1:
        raise OpenPIPipelineError("telemetry lacks factual interval throughput")

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(origin="metrics", name="Loss and learning rate"),
            rrb.TimeSeriesView(
                origin="health", name="Gradient and distributed health"
            ),
            rrb.TimeSeriesView(
                origin="throughput", name="Interval training throughput"
            ),
            rrb.TimeSeriesView(origin="checkpoint", name="Checkpoint events"),
            rrb.TextDocumentView(origin="provenance", name="Run provenance"),
        ),
        _expanded_time_panel(rrb),
        auto_layout=False,
    )
    recording = rr.RecordingStream(RERUN_APPLICATION_ID, recording_id=run_id)
    rr.save(output_path, default_blueprint=blueprint, recording=recording)
    dataset = prepared.get("dataset") or {}
    filter_dictionary = prepared.get("filter_dictionary") or {}
    normalization = prepared.get("normalization") or {}
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(filter_dictionary, Mapping)
        or not isinstance(normalization, Mapping)
    ):
        raise OpenPIPipelineError("preparation lineage is malformed")
    dataset_sha256 = str(dataset.get("listing_sha256", ""))
    filter_dictionary_sha256 = str(filter_dictionary.get("sha256", ""))
    normalization_sha256 = str(normalization.get("sha256", ""))
    if (
        not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256)
        or filter_dictionary_sha256 != FILTER_DICTIONARY_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", normalization_sha256)
    ):
        raise OpenPIPipelineError("preparation lineage lacks SHA-256 identities")
    provenance = (
        "# pi0.5 full-DROID fine-tuning\n\n"
        f"- run id: `{run_id}`\n"
        f"- producer: `npa.workbench.openpi.full_droid_finetune`\n"
        f"- upstream source ref: `{SOURCE_REF}`\n"
        f"- recipe: `{CONFIG_NAME}`\n"
        f"- upstream recipe optimizer updates: {EXPECTED_STEPS}\n"
        f"- execution target updates: {int(config.num_train_steps)}\n"
        f"- milestone: `{milestone}`\n"
        f"- factual optimizer coverage through: `{through_step}`\n"
        f"- global batch size: {int(config.batch_size)}\n"
        f"- dataset: `DROID 1.0.1`\n"
        f"- dataset listing sha256: `{dataset_sha256}`\n"
        f"- DROID filter dictionary sha256: `{filter_dictionary_sha256}`\n"
        f"- normalization sha256: `{normalization_sha256}`\n"
        f"- runtime image digest: `{_runtime_image_digest(runtime_image)}`\n"
        f"- source telemetry sha256: `{source_telemetry_sha256}`\n"
        "- learning rate: configured optimizer schedule value evaluated at the "
        "upstream optimizer step.\n"
        "- held-out/before-after policy trajectory: not produced by this "
        "offline training run; no stock or fabricated trajectory is included."
        + (
            "\n- limitation: `operator_requested_pause`; this recording is a "
            "resumable intermediate milestone, not full-recipe convergence."
            if int(config.num_train_steps) != EXPECTED_STEPS
            else ""
        )
    )
    rr.log(
        "provenance/run",
        rr.TextDocument(provenance),
        static=True,
        recording=recording,
    )
    rr.log(
        "provenance/source_telemetry_sha256",
        rr.TextLog(source_telemetry_sha256),
        static=True,
        recording=recording,
    )
    for step in expected_steps:
        record = metric_records[step]
        metrics = record.get("metrics") or {}
        health = record.get("health") or {}
        if not isinstance(metrics, Mapping) or not isinstance(health, Mapping):
            raise OpenPIPipelineError("telemetry metric payload is malformed")
        _set_rerun_step(rr, recording, step)
        rr.log("metrics/loss", rr.Scalars(float(metrics["loss"])), recording=recording)
        rr.log(
            "metrics/learning_rate",
            rr.Scalars(float(metrics["learning_rate"])),
            recording=recording,
        )
        rr.log(
            "health/gradient_norm",
            rr.Scalars(float(metrics["grad_norm"])),
            recording=recording,
        )
        rr.log(
            "health/param_norm",
            rr.Scalars(float(metrics["param_norm"])),
            recording=recording,
        )
        rr.log(
            "health/gradient_to_parameter_ratio",
            rr.Scalars(float(health["gradient_to_parameter_ratio"])),
            recording=recording,
        )
        rr.log(
            "health/nonfinite",
            rr.Scalars(0.0 if health.get("all_finite") is True else 1.0),
            recording=recording,
        )
        interval = record.get("interval")
        if isinstance(interval, Mapping):
            rr.log(
                "timing/interval_seconds",
                rr.Scalars(float(interval["seconds"])),
                recording=recording,
            )
            rr.log(
                "throughput/optimizer_steps_per_second",
                rr.Scalars(float(interval["optimizer_steps_per_second"])),
                recording=recording,
            )
            rr.log(
                "throughput/global_samples_per_second",
                rr.Scalars(float(interval["global_samples_per_second"])),
                recording=recording,
            )
    for step, event in sorted(checkpoint_events):
        _set_rerun_step(rr, recording, step)
        rr.log(f"checkpoint/{event}", rr.Scalars(1.0), recording=recording)
    distributed = {
        "health/distributed/process_count": int(hardware["process_count"]),
        "health/distributed/global_devices": int(hardware["global_gpu_count"]),
        "health/distributed/local_devices_per_process": int(
            hardware["local_devices_per_process"]
        ),
        "health/distributed/distinct_nodes": len(topology),
        "health/device/sm120_ranks": sum(
            "cc=12.0" in str(record.get("sm120_probe", "")) for record in topology
        ),
    }
    for step in (expected_steps[0], expected_steps[-1]):
        _set_rerun_step(rr, recording, step)
        for entity, value in distributed.items():
            rr.log(entity, rr.Scalars(float(value)), recording=recording)
    try:
        recording.flush()
    finally:
        recording.disconnect()
    return _inspect_training_rrd(
        output_path,
        run_id=run_id,
        source_telemetry_sha256=source_telemetry_sha256,
        require_checkpoint=require_checkpoint,
        expected_metric_steps=expected_steps,
    )


def _rrd_worker_python() -> Path | None:
    value = os.environ.get("NPA_OPENPI_RERUN_PYTHON", "").strip()
    explicitly_configured = bool(value)
    path = Path(value) if explicitly_configured else DEFAULT_RERUN_WORKER_PYTHON
    if not path.is_file() or not os.access(path, os.X_OK):
        if explicitly_configured:
            raise OpenPIPipelineError(
                "the isolated Rerun worker interpreter is unavailable"
            )
        return None
    # Distinct virtual environments commonly symlink their Python executable
    # to the same base interpreter.  Resolving those symlinks would collapse
    # the vendor and isolated Rerun environments and incorrectly force the
    # direct path even though their site-packages differ.
    if path.absolute() == Path(sys.executable).absolute():
        return None
    return path


def _run_rrd_worker(
    operation: str,
    request: Mapping[str, object],
    *,
    output_path: Path,
) -> dict[str, object]:
    worker_python = _rrd_worker_python()
    if worker_python is None:
        raise OpenPIPipelineError("the isolated Rerun worker is not configured")
    with tempfile.TemporaryDirectory(prefix="npa-openpi-rrd-worker-") as tmp:
        request_path = Path(tmp) / "request.json"
        result_path = Path(tmp) / "result.json"
        payload = {
            "schema": "npa.workbench.openpi.rrd-worker-request.v1",
            "operation": operation,
            "output_path": str(output_path),
            **dict(request),
        }
        _write_private_json(request_path, payload)
        completed = subprocess.run(
            [
                str(worker_python),
                "-m",
                "npa.workflows.byof.openpi_full_droid",
                "rrd-worker",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise OpenPIPipelineError(
                "isolated Rerun worker failed"
                + (f": {diagnostic[-1000:]}" if diagnostic else "")
            )
        if not result_path.is_file() or result_path.stat().st_size == 0:
            raise OpenPIPipelineError("isolated Rerun worker returned no result")
        if result_path.stat().st_mode & 0o077:
            raise OpenPIPipelineError("isolated Rerun worker result is not private")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, dict)
            or result.get("schema")
            != "npa.workbench.openpi.rrd-worker-result.v1"
            or not isinstance(result.get("inspection"), dict)
        ):
            raise OpenPIPipelineError("isolated Rerun worker result is malformed")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OpenPIPipelineError("isolated Rerun worker produced no RRD")
        return dict(result["inspection"])


def _build_preparation_rrd(
    journal_path: Path,
    output_path: Path,
    *,
    run_id: str,
    result: Mapping[str, object],
    runtime_image: str,
) -> dict[str, object]:
    if _rrd_worker_python() is None:
        return _build_preparation_rrd_direct(
            journal_path,
            output_path,
            run_id=run_id,
            result=result,
            runtime_image=runtime_image,
        )
    return _run_rrd_worker(
        "preparation",
        {
            "journal_path": str(journal_path),
            "run_id": run_id,
            "result": dict(result),
            "runtime_image": runtime_image,
        },
        output_path=output_path,
    )


def _build_training_rrd(
    journal_path: Path,
    output_path: Path,
    *,
    run_id: str,
    config: object,
    prepared: Mapping[str, object],
    runtime_image: str,
    hardware: Mapping[str, object],
    topology: Sequence[Mapping[str, object]],
    through_step: int | None = None,
    milestone: str = "final",
    require_checkpoint: bool = True,
) -> dict[str, object]:
    if _rrd_worker_python() is None:
        return _build_training_rrd_direct(
            journal_path,
            output_path,
            run_id=run_id,
            config=config,
            prepared=prepared,
            runtime_image=runtime_image,
            hardware=hardware,
            topology=topology,
            through_step=through_step,
            milestone=milestone,
            require_checkpoint=require_checkpoint,
        )
    config_contract = {
        key: int(getattr(config, key))
        for key in ("num_train_steps", "log_interval", "save_interval", "batch_size")
    }
    return _run_rrd_worker(
        "training",
        {
            "journal_path": str(journal_path),
            "run_id": run_id,
            "config": config_contract,
            "prepared": dict(prepared),
            "runtime_image": runtime_image,
            "hardware": dict(hardware),
            "topology": [dict(record) for record in topology],
            "through_step": through_step,
            "milestone": milestone,
            "require_checkpoint": require_checkpoint,
        },
        output_path=output_path,
    )


def _rrd_worker(args: argparse.Namespace) -> int:
    request_path = Path(args.request)
    result_path = Path(args.result)
    if request_path.stat().st_mode & 0o077:
        raise OpenPIPipelineError("Rerun worker request is not private")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (
        not isinstance(request, dict)
        or request.get("schema")
        != "npa.workbench.openpi.rrd-worker-request.v1"
    ):
        raise OpenPIPipelineError("Rerun worker request is malformed")
    output_path = Path(str(request["output_path"]))
    operation = request.get("operation")
    if operation == "preparation":
        inspection = _build_preparation_rrd_direct(
            Path(str(request["journal_path"])),
            output_path,
            run_id=str(request["run_id"]),
            result=request["result"],
            runtime_image=str(request["runtime_image"]),
        )
    elif operation == "training":
        config = argparse.Namespace(**request["config"])
        inspection = _build_training_rrd_direct(
            Path(str(request["journal_path"])),
            output_path,
            run_id=str(request["run_id"]),
            config=config,
            prepared=request["prepared"],
            runtime_image=str(request["runtime_image"]),
            hardware=request["hardware"],
            topology=request["topology"],
            through_step=request.get("through_step"),
            milestone=str(request.get("milestone", "final")),
            require_checkpoint=bool(request.get("require_checkpoint", True)),
        )
    else:
        raise OpenPIPipelineError("Rerun worker operation is unsupported")
    _write_private_json(
        result_path,
        {
            "schema": "npa.workbench.openpi.rrd-worker-result.v1",
            "inspection": inspection,
        },
    )
    return 0


def _write_private_json(path: Path, value: Mapping[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _runtime_image_digest(runtime_image: str) -> str:
    match = re.search(r"@(?P<digest>sha256:[0-9a-f]{64})$", runtime_image)
    if match is None:
        raise OpenPIPipelineError("runtime image must be pinned by SHA-256 digest")
    return match.group("digest")


def _write_once_or_verify(uri: str, payload: bytes, *, content_type: str) -> None:
    if _uri_exists(uri):
        if _read_bytes_uri(uri) != payload:
            raise OpenPIPipelineError("immutable artifact differs from this run")
        return
    try:
        _write_bytes_uri(uri, payload, content_type=content_type)
    except Exception:
        # S3 writes use If-None-Match. A racing retry is valid only when it
        # published the byte-identical artifact for this run.
        if not _uri_exists(uri) or _read_bytes_uri(uri) != payload:
            raise
    if _read_bytes_uri(uri) != payload:
        raise OpenPIPipelineError("artifact read-after-write verification failed")


def _manifest_payload(
    *,
    run_id: str,
    stage: str,
    milestone: str,
    rrd_uri: str,
    inspection: Mapping[str, object],
    source_telemetry_sha256: str,
    source_coverage: Mapping[str, object],
) -> bytes:
    value: dict[str, object] = {
        "schema": MILESTONE_MANIFEST_SCHEMA,
        "run_id": run_id,
        "stage": stage,
        "milestone": milestone,
        "rrd": {
            "uri": rrd_uri,
            "schema": RERUN_SCHEMA,
            "bytes": inspection["bytes"],
            "sha256": inspection["sha256"],
            "inspection": dict(inspection),
        },
        "source_telemetry_sha256": source_telemetry_sha256,
        "source_coverage": dict(source_coverage),
        "content_sha256": "",
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _publish_preparation_rrd(
    journal_path: Path,
    *,
    rrd_uri: str,
    manifest_uri: str,
    run_id: str,
    result: Mapping[str, object],
    runtime_image: str,
) -> dict[str, object]:
    source_sha256 = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="npa-openpi-preparation-rerun-") as tmp:
        local = Path(tmp) / "preparation.rrd"
        if _uri_exists(rrd_uri):
            local.write_bytes(_read_bytes_uri(rrd_uri))
            producer = _inspect_preparation_rrd(
                local,
                run_id=run_id,
                source_telemetry_sha256=source_sha256,
            )
        else:
            producer = _build_preparation_rrd(
                journal_path,
                local,
                run_id=run_id,
                result=result,
                runtime_image=runtime_image,
            )
            _write_once_or_verify(
                rrd_uri, local.read_bytes(), content_type=RERUN_SCHEMA
            )
        readback = Path(tmp) / "readback.rrd"
        readback.write_bytes(_read_bytes_uri(rrd_uri))
        inspection = _inspect_preparation_rrd(
            readback,
            run_id=run_id,
            source_telemetry_sha256=source_sha256,
        )
        normalization = result["normalization"]
        manifest = _manifest_payload(
            run_id=run_id,
            stage="preparation",
            milestone="normalization-complete",
            rrd_uri=rrd_uri,
            inspection=inspection,
            source_telemetry_sha256=source_sha256,
            source_coverage={
                "normalization_batches": normalization["batches_processed"],
                "normalization_frames": normalization["frames_processed"],
            },
        )
        _write_once_or_verify(
            manifest_uri, manifest, content_type="application/json"
        )
    return {
        "uri": rrd_uri,
        "schema": RERUN_SCHEMA,
        "manifest_uri": manifest_uri,
        "inspection": inspection,
        "producer_inspection": producer,
    }


def _telemetry_prefix_payload(
    journal_path: Path, *, run_id: str, through_step: int
) -> bytes:
    lines: list[bytes] = []
    for number, line in enumerate(journal_path.read_bytes().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OpenPIPipelineError(
                f"telemetry journal line {number} is invalid JSON"
            ) from exc
        if (
            not isinstance(record, dict)
            or record.get("run_id") != run_id
            or record.get("schema") != TELEMETRY_SCHEMA
        ):
            raise OpenPIPipelineError("telemetry prefix has incompatible provenance")
        if int(record["optimizer_step"]) <= through_step:
            lines.append(line)
    if not lines:
        raise OpenPIPipelineError("telemetry prefix is empty")
    return b"\n".join(lines) + b"\n"


class _TrainingMilestonePublisher:
    def __init__(
        self,
        *,
        journal_path: Path,
        run_id: str,
        kind: str,
        config: object,
        prepared: Mapping[str, object],
        runtime_image: str,
        hardware: Mapping[str, object],
        topology: Sequence[Mapping[str, object]],
        rrd_root_uri: str,
        pause_after_updates: int = 0,
    ) -> None:
        if kind not in {"qualification", "full"}:
            raise OpenPIPipelineError("unknown training milestone kind")
        self.journal_path = journal_path
        self.run_id = run_id
        self.kind = kind
        self.config = config
        self.prepared = prepared
        self.runtime_image = runtime_image
        self.hardware = hardware
        self.topology = topology
        self.rrd_root_uri = rrd_root_uri.rstrip("/")
        self.pause_after_updates = pause_after_updates
        self.published: dict[int, dict[str, object]] = {}
        if kind == "qualification":
            self.milestones = {QUALIFICATION_STEPS: QUALIFICATION_STEPS - 1}
            self.pause_checkpoint_milestone = False
        elif pause_after_updates:
            self.milestones = {
                pause_after_updates: pause_after_updates - 1,
            }
            self.pause_checkpoint_milestone = True
        else:
            self.milestones = {
                1_000: 1_000,
                10_000: 10_000,
                25_000: 25_000,
                50_000: 50_000,
                75_000: 75_000,
                100_000: 99_999,
            }
            self.pause_checkpoint_milestone = False
            existing_pause_manifest = (
                f"{self.rrd_root_uri}/progress-step-001000.manifest.json"
            )
            if _uri_exists(existing_pause_manifest):
                value = _read_json_uri(existing_pause_manifest)
                coverage = value.get("source_coverage")
                if (
                    value.get("schema") != MILESTONE_MANIFEST_SCHEMA
                    or value.get("run_id") != run_id
                    or value.get("milestone") != "progress-step-001000"
                    or not isinstance(coverage, Mapping)
                    or coverage.get("through_optimizer_step") != 999
                    or coverage.get("checkpoint_materialized") is not True
                ):
                    raise OpenPIPipelineError(
                        "existing pause milestone has incompatible provenance"
                    )
                self.milestones[1_000] = 999
                self.pause_checkpoint_milestone = True

    def _slug(self, milestone: int) -> str:
        prefix = "qualification" if self.kind == "qualification" else "progress"
        return f"{prefix}-step-{milestone:06d}"

    def is_log_only(self, optimizer_step: int) -> bool:
        return (
            self.kind == "full"
            and optimizer_step == self.milestones.get(1_000)
            and not self.pause_checkpoint_milestone
        )

    def requires_checkpoint(self, optimizer_step: int) -> bool:
        return any(
            actual == optimizer_step and not self.is_log_only(actual)
            for actual in self.milestones.values()
        )

    def reconcile_available(self) -> None:
        if not self.journal_path.is_file():
            return
        records = _load_telemetry_records(self.journal_path, run_id=self.run_id)
        metric_steps = {
            int(record["optimizer_step"])
            for record in records
            if record.get("record_type") == "metrics"
        }
        checkpoint_events = {
            (int(record["optimizer_step"]), str(record.get("event")))
            for record in records
            if record.get("record_type") == "checkpoint"
        }
        for _, actual in self.milestones.items():
            expected_metric_steps = set(
                range(0, actual + 1, int(self.config.log_interval))
            )
            available = (
                expected_metric_steps <= metric_steps
                if self.is_log_only(actual)
                else expected_metric_steps <= metric_steps
                and {
                    (actual, "save_requested"),
                    (actual, "materialized"),
                }
                <= checkpoint_events
            )
            if available:
                self.publish_for_optimizer_step(actual)

    def publish_for_optimizer_step(self, optimizer_step: int) -> dict[str, object] | None:
        matches = [
            semantic
            for semantic, actual in self.milestones.items()
            if actual == optimizer_step
        ]
        if not matches:
            return None
        semantic = matches[0]
        require_checkpoint = not self.is_log_only(optimizer_step)
        prefix_payload = _telemetry_prefix_payload(
            self.journal_path,
            run_id=self.run_id,
            through_step=optimizer_step,
        )
        source_sha256 = hashlib.sha256(prefix_payload).hexdigest()
        slug = self._slug(semantic)
        rrd_uri = f"{self.rrd_root_uri}/{slug}.rrd"
        manifest_uri = f"{self.rrd_root_uri}/{slug}.manifest.json"
        with tempfile.TemporaryDirectory(prefix="npa-openpi-milestone-") as tmp:
            prefix_path = Path(tmp) / "telemetry-prefix.jsonl"
            prefix_path.write_bytes(prefix_payload)
            metric_steps = [
                int(record["optimizer_step"])
                for record in _load_telemetry_records(
                    prefix_path, run_id=self.run_id
                )
                if record.get("record_type") == "metrics"
            ]
            local = Path(tmp) / f"{slug}.rrd"
            if _uri_exists(rrd_uri):
                local.write_bytes(_read_bytes_uri(rrd_uri))
                producer = _inspect_training_rrd(
                    local,
                    run_id=self.run_id,
                    source_telemetry_sha256=source_sha256,
                    require_checkpoint=require_checkpoint,
                    expected_metric_steps=metric_steps,
                )
            else:
                producer = _build_training_rrd(
                    prefix_path,
                    local,
                    run_id=self.run_id,
                    config=self.config,
                    prepared=self.prepared,
                    runtime_image=self.runtime_image,
                    hardware=self.hardware,
                    topology=self.topology,
                    through_step=optimizer_step,
                    milestone=slug,
                    require_checkpoint=require_checkpoint,
                )
                _write_once_or_verify(
                    rrd_uri, local.read_bytes(), content_type=RERUN_SCHEMA
                )
            readback_path = Path(tmp) / "readback.rrd"
            readback = _read_bytes_uri(rrd_uri)
            if not readback:
                raise OpenPIPipelineError("milestone RRD readback is empty")
            readback_path.write_bytes(readback)
            inspection = _inspect_training_rrd(
                readback_path,
                run_id=self.run_id,
                source_telemetry_sha256=source_sha256,
                require_checkpoint=require_checkpoint,
                expected_metric_steps=metric_steps,
            )
            manifest = _manifest_payload(
                run_id=self.run_id,
                stage=self.kind,
                milestone=slug,
                rrd_uri=rrd_uri,
                inspection=inspection,
                source_telemetry_sha256=source_sha256,
                source_coverage={
                    "through_optimizer_step": optimizer_step,
                    "metric_record_count": len(metric_steps),
                    "first_metric_step": min(metric_steps),
                    "last_metric_step": max(metric_steps),
                    "checkpoint_materialized": require_checkpoint,
                },
            )
            _write_once_or_verify(
                manifest_uri, manifest, content_type="application/json"
            )
        result = {
            "milestone": semantic,
            "through_optimizer_step": optimizer_step,
            "rrd_uri": rrd_uri,
            "manifest_uri": manifest_uri,
            "inspection": inspection,
            "producer_inspection": producer,
        }
        self.published[semantic] = result
        return result


def _local_hardware_evidence() -> dict[str, object]:
    import jax
    from jax.extend import backend as jax_backend

    local_devices = jax.local_devices()
    global_devices = jax.devices()
    if (
        len(local_devices) != EXPECTED_LOCAL_DEVICES
        or len(global_devices) != EXPECTED_DEVICES
    ):
        raise OpenPIPipelineError(
            "JAX device counts changed after topology initialization"
        )
    global_kinds = [str(device.device_kind) for device in global_devices]
    if any("RTX PRO 6000" not in kind.upper() for kind in global_kinds):
        raise OpenPIPipelineError(
            f"expected RTX PRO 6000 devices, got {global_kinds!r}"
        )
    nvidia_smi = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(nvidia_smi) != EXPECTED_LOCAL_DEVICES:
        raise OpenPIPipelineError("nvidia-smi must expose exactly one GPU per process")
    if "RTX PRO 6000" not in nvidia_smi[0].upper() or not re.search(
        r"(?:^|,)\s*12\.0\s*(?:,|$)", nvidia_smi[0]
    ):
        raise OpenPIPipelineError(
            f"expected an RTX PRO 6000 SM120 GPU, got {nvidia_smi!r}"
        )
    return {
        "global_gpu_count": len(global_devices),
        "local_gpu_count": len(local_devices),
        "global_device_kinds": global_kinds,
        "local_nvidia_smi": nvidia_smi,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "xla_platform_version": str(jax_backend.get_backend().platform_version),
    }


def _write_rank_evidence(
    work_root: Path, rank: int, hardware: dict[str, object], probe: str
) -> dict[str, object]:
    import jax

    local_devices = jax.local_devices()
    evidence: dict[str, object] = {
        "rank": rank,
        "hostname_sha256": hashlib.sha256(socket.gethostname().encode()).hexdigest(),
        "local_device_count": len(local_devices),
        "global_device_count": jax.device_count(),
        "device_kinds": sorted({str(device.device_kind) for device in local_devices}),
        "local_nvidia_smi": hardware["local_nvidia_smi"],
        "sm120_probe": probe,
    }
    root = work_root / "topology"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"rank-{rank}.json").write_text(
        json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
    )
    return evidence


def _read_topology(work_root: Path) -> list[dict[str, object]]:
    records = [
        json.loads(
            (work_root / "topology" / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(EXPECTED_PROCESSES)
    ]
    if [record.get("rank") for record in records] != list(range(EXPECTED_PROCESSES)):
        raise OpenPIPipelineError("rank evidence is incomplete")
    if len({record.get("hostname_sha256") for record in records}) != EXPECTED_PROCESSES:
        raise OpenPIPipelineError("rank evidence does not prove eight distinct nodes")
    if any(record.get("local_device_count") != 1 for record in records):
        raise OpenPIPipelineError("rank evidence does not prove one GPU per node")
    if any(
        "RTX PRO 6000"
        not in " ".join(str(value) for value in record.get("device_kinds", [])).upper()
        for record in records
    ):
        raise OpenPIPipelineError("rank evidence contains a non-RTX-PRO-6000 device")
    if any("cc=12.0" not in str(record.get("sm120_probe", "")) for record in records):
        raise OpenPIPipelineError("rank evidence does not prove SM120 on every node")
    return records


def _validate_operator_pause(
    *, checkpoint_root: Path, telemetry_path: Path, run_id: str, updates: int
) -> dict[str, object]:
    if updates != OPERATOR_PAUSE_UPDATES:
        raise OpenPIPipelineError("unsupported operator pause boundary")
    if not checkpoint_root.is_dir():
        raise OpenPIPipelineError("operator pause checkpoint root is absent")
    final_step = updates - 1
    records = _load_telemetry_records(telemetry_path, run_id=run_id)
    metric_steps = [
        int(record["optimizer_step"])
        for record in records
        if record.get("record_type") == "metrics"
    ]
    if metric_steps != list(range(updates)):
        raise OpenPIPipelineError(
            "operator pause telemetry does not prove updates 0 through 999"
        )
    events = {
        (int(record["optimizer_step"]), str(record.get("event")))
        for record in records
        if record.get("record_type") == "checkpoint"
    }
    if {
        (final_step, "save_requested"),
        (final_step, "materialized"),
    } - events:
        raise OpenPIPipelineError(
            "operator pause checkpoint events are incomplete"
        )
    if not _checkpoint_completion_is_valid(
        checkpoint_root, step=final_step, run_id=run_id
    ):
        raise OpenPIPipelineError(
            "operator pause checkpoint completion marker is invalid"
        )
    later = sorted(
        int(path.name)
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.isdigit() and int(path.name) > final_step
    )
    if later:
        raise OpenPIPipelineError(
            "operator pause checkpoint root contains later optimizer state"
        )
    return {
        "completed_updates": updates,
        "first_optimizer_step": 0,
        "last_optimizer_step": final_step,
        "metric_record_count": len(metric_steps),
        "telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
        "checkpoint_completion_marker": True,
    }


def _checkpoint_file_records(local_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(item for item in local_root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": path.relative_to(local_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records


def _upload_or_verify_checkpoint(
    local_root: Path, output_uri: str
) -> dict[str, object]:
    manifest_uri = output_uri.rstrip("/") + "/manifest.json"
    if not _uri_exists(manifest_uri):
        manifest = _upload_checkpoint(local_root, output_uri)
    else:
        manifest = _read_json_uri(manifest_uri)
    with tempfile.TemporaryDirectory(prefix="npa-openpi-checkpoint-readback-") as tmp:
        verified = _download_checkpoint(output_uri, Path(tmp) / "checkpoint")
    if verified != manifest:
        raise OpenPIPipelineError("checkpoint readback manifest differs")
    local_records = _checkpoint_file_records(local_root)
    canonical = json.dumps(
        local_records, sort_keys=True, separators=(",", ":")
    ).encode()
    if (
        manifest.get("schema")
        != "npa.workbench.openpi.checkpoint-manifest.v1"
        or manifest.get("files") != local_records
        or manifest.get("file_count") != len(local_records)
        or manifest.get("total_size_bytes")
        != sum(int(record["size"]) for record in local_records)
        or manifest.get("content_manifest_sha256")
        != hashlib.sha256(canonical).hexdigest()
    ):
        raise OpenPIPipelineError(
            "checkpoint upload does not match durable local optimizer state"
        )
    return manifest


def _fine_tune(args: argparse.Namespace) -> int:
    _require_terms()
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    _configure_openpi_cache(work_root)
    repo_root = Path(args.repo_root)
    build = _validate_source(repo_root, args.runtime_image)
    prepared = _read_json_uri(args.prepare_uri)
    if (
        prepared.get("schema") != "npa.workbench.openpi.pi05-full-droid-prepare.v1"
        or prepared.get("status") != "passed"
    ):
        raise OpenPIPipelineError(
            "full-DROID preparation artifact is absent or invalid"
        )
    filter_dictionary = prepared.get("filter_dictionary")
    if (
        not isinstance(filter_dictionary, Mapping)
        or filter_dictionary.get("source_uri") != FILTER_DICTIONARY_URI
        or filter_dictionary.get("sha256") != FILTER_DICTIONARY_SHA256
        or filter_dictionary.get("size_bytes") != FILTER_DICTIONARY_BYTES
    ):
        raise OpenPIPipelineError(
            "full-DROID preparation lacks the pinned filter dictionary lineage"
        )

    rank, node_ips = _multihost_environment()
    multihost_utils = _initialize_multihost(rank, node_ips)
    _install_distributed_rlds_adapter(rank)

    import jax
    from openpi.training import sharding

    hardware = _local_hardware_evidence()
    mesh = sharding.make_mesh(EXPECTED_FSDP_DEVICES)
    if tuple(mesh.devices.shape) != (1, EXPECTED_FSDP_DEVICES):
        raise OpenPIPipelineError(f"unexpected FSDP mesh shape {mesh.devices.shape}")
    probe = subprocess.run(
        ["/usr/local/bin/npa-openpi-sm120-probe"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    started = time.perf_counter()
    qualification = args.training_kind == "qualification"
    pause_after_updates = int(getattr(args, "pause_after_updates", 0))
    config = _configured_upstream(
        Path(args.data_root),
        work_root,
        args.experiment,
        qualification=qualification,
        pause_after_updates=pause_after_updates,
    )
    _write_rank_evidence(work_root, rank, hardware, probe)
    multihost_utils.sync_global_devices("npa-openpi-topology-ready")
    topology: list[dict[str, object]] = []
    milestone_publisher = None
    hardware_summary = {
        **hardware,
        "process_count": jax.process_count(),
        "local_devices_per_process": EXPECTED_LOCAL_DEVICES,
        "distinct_nodes": EXPECTED_PROCESSES,
    }
    if rank == 0:
        topology = _read_topology(work_root)
        milestone_publisher = _TrainingMilestonePublisher(
            journal_path=Path(config.checkpoint_base_dir)
            / config.name
            / config.exp_name
            / "npa-training-telemetry.jsonl",
            run_id=args.run_id,
            kind=args.training_kind,
            config=config,
            prepared=prepared,
            runtime_image=args.runtime_image,
            hardware=hardware_summary,
            topology=topology,
            rrd_root_uri=args.rrd_root_uri,
            pause_after_updates=pause_after_updates,
        )
    checkpoint_root, resumed, telemetry_path = _run_training(
        config,
        repo_root,
        rank=rank,
        run_id=args.run_id,
        multihost_utils=multihost_utils,
        milestone_publisher=milestone_publisher,
    )
    multihost_utils.sync_global_devices("npa-openpi-training-finished")

    if rank != 0:
        multihost_utils.sync_global_devices("npa-openpi-artifacts-published")
        print(
            json.dumps({"status": "passed", "rank": rank}, sort_keys=True), flush=True
        )
        return 0

    if telemetry_path is None:
        raise OpenPIPipelineError("rank zero telemetry journal is absent")
    pause_evidence = None
    if pause_after_updates:
        pause_evidence = _validate_operator_pause(
            checkpoint_root=checkpoint_root,
            telemetry_path=telemetry_path,
            run_id=args.run_id,
            updates=pause_after_updates,
        )
    checkpoint = _upload_or_verify_checkpoint(checkpoint_root, args.checkpoint_uri)
    hardware_summary["distinct_nodes"] = len(topology)
    _write_once_or_verify(
        args.telemetry_uri,
        telemetry_path.read_bytes(),
        content_type="application/x-ndjson",
    )
    final_milestone = (
        QUALIFICATION_STEPS
        if qualification
        else pause_after_updates or EXPECTED_STEPS
    )
    if milestone_publisher is None or final_milestone not in milestone_publisher.published:
        raise OpenPIPipelineError("final mandatory RRD milestone was not published")
    rerun = milestone_publisher.published[final_milestone]
    schema = (
        "npa.workbench.openpi.pi05-full-droid-qualification.v1"
        if qualification
        else (
            "npa.workbench.openpi.pi05-full-droid-finetune-paused.v1"
            if pause_after_updates
            else "npa.workbench.openpi.pi05-full-droid-finetune.v1"
        )
    )
    result: dict[str, object] = {
        "schema": schema,
        "status": "paused" if pause_after_updates else "passed",
        "source": {
            "repository": "https://github.com/Physical-Intelligence/openpi",
            "ref": SOURCE_REF,
            "license": "Apache-2.0",
            **build,
        },
        "runtime_image": args.runtime_image,
        "redistribution": _redistribution_evidence(trained_checkpoint=True),
        "recipe": {
            "config_name": CONFIG_NAME,
            "dataset": prepared["dataset"],
            "normalization": prepared["normalization"],
            "global_batch_size": EXPECTED_BATCH_SIZE,
            "batch_per_process": EXPECTED_BATCH_SIZE // EXPECTED_PROCESSES,
            "optimizer_steps": EXPECTED_STEPS,
            "seed": int(config.seed),
            "fsdp_devices": EXPECTED_FSDP_DEVICES,
            "mesh_shape": [1, EXPECTED_FSDP_DEVICES],
            "upstream_entrypoint": "scripts/train.py:main",
            "upstream_recipe_hyperparameters_unmodified": (
                not qualification and not pause_after_updates
            ),
            "qualification_only_step_and_log_cadence_override": qualification,
            "operator_pause_only_step_log_and_save_cadence_override": bool(
                pause_after_updates
            ),
            "distributed_rlds_adapter": "pre_shuffle_process_shard_and_local_batch",
            "distributed_shuffle_seed": "upstream_seed_plus_process_index",
            "checkpoint_coordination": "orbax_checkpoint_manager_primary_host_and_global_barriers",
        },
        "checkpoint": {
            "uri": args.checkpoint_uri.rstrip("/") + "/",
            "manifest_uri": args.checkpoint_uri.rstrip("/") + "/manifest.json",
            "content_manifest_sha256": checkpoint["content_manifest_sha256"],
            "file_count": checkpoint["file_count"],
            "total_size_bytes": checkpoint["total_size_bytes"],
            "final_step": int(config.num_train_steps),
            "upstream_checkpoint_directory": int(config.num_train_steps) - 1,
            "resumed_from_durable_checkpoint": resumed,
        },
        "hardware": {
            **hardware_summary,
            "rank_evidence": topology,
            "sm120_probe": probe,
        },
        "rerun": rerun,
        "rerun_milestones": [
            milestone_publisher.published[key]
            for key in sorted(milestone_publisher.published)
        ],
        "terms": {"forwarded": True, "persisted": False},
        "timings_seconds": {"total": round(time.perf_counter() - started, 3)},
        "limitations": [
            "offline_training_does_not_prove_physical_robot_success",
            *(
                [
                    "operator_requested_pause",
                    "full_recipe_convergence_not_completed",
                    "remaining_99000_updates_intentionally_outstanding",
                ]
                if pause_after_updates
                else []
            ),
        ],
    }
    if pause_after_updates:
        if pause_evidence is None:
            raise OpenPIPipelineError("operator pause evidence is absent")
        result["pause"] = {
            **pause_evidence,
            "reason": "operator_requested_pause",
            "upstream_recipe_updates": EXPECTED_STEPS,
            "remaining_updates": EXPECTED_STEPS - pause_after_updates,
            "resumable": True,
            "resume_next_optimizer_step": pause_after_updates,
            "checkpoint_content_manifest_sha256": checkpoint[
                "content_manifest_sha256"
            ],
            "checkpoint_total_size_bytes": checkpoint["total_size_bytes"],
            "checkpoint_file_count": checkpoint["file_count"],
            "rrd_sha256": rerun["inspection"]["sha256"],
            "rrd_bytes": rerun["inspection"]["bytes"],
            "milestone_manifest_uri": rerun["manifest_uri"],
        }
        result["content_sha256"] = ""
        canonical = json.dumps(
            result, sort_keys=True, separators=(",", ":")
        ).encode()
        result["content_sha256"] = hashlib.sha256(canonical).hexdigest()
        payload = (json.dumps(result, sort_keys=True) + "\n").encode()
        _write_once_or_verify(
            args.output_uri, payload, content_type="application/json"
        )
    else:
        _write_json_uri(args.output_uri, result)
    multihost_utils.sync_global_devices("npa-openpi-artifacts-published")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runtime-image", required=True)
    common.add_argument("--repo-root", default="/opt/byof")
    common.add_argument("--work-root", default="/workspace/openpi-full-droid")
    common.add_argument("--data-root", default="/workspace/openpi-full-droid/dataset")
    common.add_argument("--experiment", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", parents=[common])
    prepare.add_argument("--output-uri", required=True)
    prepare.add_argument("--rrd-uri", required=True)
    prepare.add_argument("--milestone-manifest-uri", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--gsutil", default="/opt/gsutil-venv/bin/gsutil")
    prepare.set_defaults(func=_prepare)

    for command, kind in (("qualify", "qualification"), ("train", "full")):
        train = subparsers.add_parser(command, parents=[common])
        train.add_argument("--prepare-uri", required=True)
        train.add_argument("--output-uri", required=True)
        train.add_argument("--checkpoint-uri", required=True)
        train.add_argument("--telemetry-uri", required=True)
        train.add_argument("--rrd-root-uri", required=True)
        train.add_argument("--run-id", required=True)
        if command == "train":
            train.add_argument("--pause-after-updates", type=int, default=0)
        train.set_defaults(func=_fine_tune, training_kind=kind)
    worker = subparsers.add_parser("rrd-worker")
    worker.add_argument("--request", required=True)
    worker.add_argument("--result", required=True)
    worker.set_defaults(func=_rrd_worker)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
