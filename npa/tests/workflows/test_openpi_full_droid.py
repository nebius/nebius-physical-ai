from __future__ import annotations

from pathlib import Path
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workflows.byof import openpi_full_droid as full_droid


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "openpi-pi05-full-droid-finetune.yaml"
)


def test_full_droid_recipe_matches_pinned_upstream_contract() -> None:
    assert full_droid.SOURCE_REF == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert full_droid.CONFIG_NAME == "pi05_full_droid_finetune"
    assert full_droid.DATASET_URI == "gs://gresearch/robotics/droid/1.0.1"
    assert full_droid.EXPECTED_STEPS == 100_000
    assert full_droid.EXPECTED_BATCH_SIZE == 256
    assert full_droid.EXPECTED_DEVICES == full_droid.EXPECTED_FSDP_DEVICES == 8
    assert full_droid.EXPECTED_BATCH_SIZE % full_droid.EXPECTED_DEVICES == 0
    assert full_droid.NORM_MAX_FRAMES == 10_000_000
    assert full_droid.PINNED_UPSTREAM_NORM_SHUFFLE_BUFFER_SIZE == 250_000
    assert full_droid.NORM_SHUFFLE_BUFFER_SIZE == 50_000
    assert full_droid.FILTER_DICTIONARY_URI.endswith(
        "/droid_sample_ranges_v1_0_1.json"
    )
    assert len(full_droid.FILTER_DICTIONARY_SHA256) == 64


def test_writable_actions_transform_isolates_read_only_numpy_view() -> None:
    np = pytest.importorskip("numpy")
    source = np.arange(32, dtype=np.float32).reshape(2, 16)
    source.setflags(write=False)

    transformed = full_droid._WritableActionsTransform()(
        {"actions": source, "prompt": "move"}
    )
    actions = transformed["actions"]

    assert actions.flags.writeable
    assert actions.shape == source.shape
    assert actions.dtype == source.dtype
    assert np.array_equal(actions, source)
    assert not np.shares_memory(actions, source)
    actions[0, 0] = -1
    assert source[0, 0] == 0


def test_writable_actions_transform_fails_closed_without_actions() -> None:
    with pytest.raises(full_droid.OpenPIPipelineError, match="missing actions"):
        full_droid._WritableActionsTransform()({"prompt": "move"})


def test_read_only_safe_factory_injects_copy_before_delta_actions() -> None:
    @dataclasses.dataclass(frozen=True)
    class Group:
        inputs: tuple[object, ...]
        outputs: tuple[object, ...] = ()

    @dataclasses.dataclass(frozen=True)
    class DataConfig:
        data_transforms: Group

    class DroidInputs:
        pass

    class DeltaActions:
        pass

    class Factory:
        def create(self, assets_dirs, model_config):
            del assets_dirs, model_config
            return DataConfig(Group((DroidInputs(), DeltaActions())))

    configured = full_droid._ReadOnlySafeDroidDataFactory(Factory()).create(
        Path("/assets"), object()
    )

    assert [type(item).__name__ for item in configured.data_transforms.inputs] == [
        "DroidInputs",
        "_WritableActionsTransform",
        "DeltaActions",
    ]


def test_read_only_safe_factory_rejects_upstream_transform_drift() -> None:
    @dataclasses.dataclass(frozen=True)
    class Group:
        inputs: tuple[object, ...]

    @dataclasses.dataclass(frozen=True)
    class DataConfig:
        data_transforms: Group

    class Factory:
        def create(self, assets_dirs, model_config):
            del assets_dirs, model_config
            return DataConfig(Group((object(),)))

    with pytest.raises(full_droid.OpenPIPipelineError, match="sequence drifted"):
        full_droid._ReadOnlySafeDroidDataFactory(Factory()).create(
            Path("/assets"), object()
        )


def test_full_droid_spec_is_exactly_eight_one_gpu_nodes() -> None:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    config = spec["config"]
    profile = spec["resources"]["rtxpro8"]
    assert config["gpu_type"] == "RTXPRO6000"
    assert config["gpu_count"] == "1"
    assert config["gpu_num_nodes"] == "8"
    assert config["multi_host_enabled"] == "true"
    assert config["pause_after_updates"] == "0"
    assert profile["accelerators"] == "{{config.gpu_type}}:{{config.gpu_count}}"
    assert profile["num_nodes"] == "{{config.gpu_num_nodes}}"
    assert "persistentVolumeClaim" in str(profile)
    prepare = spec["states"]["prepare_full_droid"]
    assert prepare["toolRef"] == "workbench.openpi.full_droid_prepare"
    assert prepare["next"] == "qualify_full_droid"
    qualification = spec["states"]["qualify_full_droid"]
    assert qualification["toolRef"] == "workbench.openpi.full_droid_qualification"
    assert qualification["next"] == "full_droid_finetune"
    state = spec["states"]["full_droid_finetune"]
    assert state["toolRef"] == "workbench.openpi.full_droid_finetune"
    assert state["terminal"] is True
    outputs = {item["uri"]: item["schema"] for item in state["outputs"]}
    for milestone in full_droid.FULL_PROGRESS_MILESTONES:
        slug = f"progress-step-{milestone:06d}"
        assert outputs[f"{{{{config.rrd_root_uri}}}}/{slug}.rrd"] == (
            "application/vnd.rerun.rrd"
        )
        assert outputs[f"{{{{config.rrd_root_uri}}}}/{slug}.manifest.json"] == (
            full_droid.MILESTONE_MANIFEST_SCHEMA
        )
    assert outputs["{{config.telemetry_uri}}"] == full_droid.TELEMETRY_SCHEMA
    prepare_outputs = {item["uri"]: item["schema"] for item in prepare["outputs"]}
    assert prepare_outputs["{{config.prepare_rrd_uri}}"] == full_droid.RERUN_SCHEMA
    assert "{{run.id}}" in spec["config"]["prefix"]


def test_full_droid_toolref_has_no_tunable_recipe_shortcuts() -> None:
    entry = TOOL_CATALOG["workbench.openpi.full_droid_finetune"]
    argv = entry.argv_template
    assert argv[:4] == [
        "/opt/venv/bin/python",
        "-m",
        "npa.workflows.byof.openpi_full_droid",
        "train",
    ]
    assert "--train-steps" not in argv
    assert "--batch-size" not in argv
    assert "--fsdp-devices" not in argv
    assert "--checkpoint-uri" in argv
    assert "--telemetry-uri" in argv
    assert "--rrd-root-uri" in argv
    assert "--run-id" in argv
    assert argv[argv.index("--pause-after-updates") + 1] == (
        "{{config.pause_after_updates}}"
    )
    assert entry.multi_node_mode == "sharded"
    assert entry.shard_activation_config == "multi_host_enabled"
    assert entry.shard_output_config == "trained_checkpoint_uri"


def test_prepare_toolref_has_no_gpu_training_flags() -> None:
    argv = TOOL_CATALOG["workbench.openpi.full_droid_prepare"].argv_template
    assert argv[:4] == [
        "/opt/venv/bin/python",
        "-m",
        "npa.workflows.byof.openpi_full_droid",
        "prepare",
    ]
    assert "--checkpoint-uri" not in argv
    assert "--output-uri" in argv
    assert "--rrd-uri" in argv
    assert "--milestone-manifest-uri" in argv
    assert "--run-id" in argv


def test_qualification_toolref_is_fixed_and_distributed() -> None:
    entry = TOOL_CATALOG["workbench.openpi.full_droid_qualification"]
    assert entry.argv_template[:4] == [
        "/opt/venv/bin/python",
        "-m",
        "npa.workflows.byof.openpi_full_droid",
        "qualify",
    ]
    assert "--train-steps" not in entry.argv_template
    assert "--rrd-root-uri" in entry.argv_template
    assert entry.multi_node_mode == "sharded"


def test_distributed_rlds_adapter_shards_before_shuffle_and_uses_local_batch(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeDataset:
        def shard(self, process_count, rank):
            calls["shard"] = (process_count, rank)
            return self

    class FakeDLataset:
        @staticmethod
        def sample_from_datasets(*args, **kwargs):
            calls["sample"] = (args, kwargs)
            return FakeDataset()

    class FakeRLDSDataLoader:
        pass

    def original_create(data_config, action_horizon, batch_size, *, shuffle=False):
        calls["create"] = (data_config, action_horizon, batch_size, shuffle)
        return FakeDataset()

    fake_loader = types.ModuleType("openpi.training.data_loader")
    fake_loader.create_rlds_dataset = original_create
    fake_loader.RLDSDataLoader = FakeRLDSDataLoader
    fake_training = types.ModuleType("openpi.training")
    fake_training.data_loader = fake_loader
    fake_openpi = types.ModuleType("openpi")
    fake_openpi.training = fake_training
    fake_tensorflow = types.ModuleType("tensorflow")
    fake_tensorflow.random = types.SimpleNamespace(
        set_seed=lambda value: calls.__setitem__("seed", value)
    )
    fake_jax = types.ModuleType("jax")
    fake_jax.sharding = types.SimpleNamespace()
    fake_dlimp = types.ModuleType("dlimp")
    fake_dlimp.DLataset = FakeDLataset
    for name, module in {
        "dlimp": fake_dlimp,
        "jax": fake_jax,
        "tensorflow": fake_tensorflow,
        "openpi": fake_openpi,
        "openpi.training": fake_training,
        "openpi.training.data_loader": fake_loader,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    full_droid._install_distributed_rlds_adapter(rank=3)
    fake_dlimp.DLataset.sample_from_datasets([FakeDataset()], weights=[1.0])
    fake_loader.create_rlds_dataset("data", 16, 256, shuffle=True)

    assert calls["seed"] == 45
    assert calls["shard"] == (8, 3)
    assert calls["create"] == ("data", 16, 32, True)


def test_remote_inventory_is_content_addressed(monkeypatch, tmp_path: Path) -> None:
    listing = (
        "       12  2026-01-01T00:00:00Z  gs://gresearch/robotics/droid/1.0.1/a\n"
        "       34  2026-01-01T00:00:01Z  gs://gresearch/robotics/droid/1.0.1/b\n"
        "TOTAL: 2 objects, 46 bytes (46 B)\n"
    )

    def fake_run(command, *, cwd=None, stdout=None):
        del command, cwd
        stdout.write(listing)

    monkeypatch.setattr(full_droid, "_run", fake_run)
    result = full_droid._remote_inventory("gsutil", tmp_path / "listing.txt")
    assert result["object_count"] == 2
    assert result["total_size_bytes"] == 46
    assert len(str(result["listing_sha256"])) == 64


def test_filter_dictionary_stages_exact_single_object_and_reuses_cache(
    monkeypatch, tmp_path: Path
) -> None:
    payload = json.dumps({"episode": [[0, 1]]}).encode()
    monkeypatch.setattr(full_droid, "FILTER_DICTIONARY_BYTES", len(payload))
    monkeypatch.setattr(
        full_droid,
        "FILTER_DICTIONARY_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    commands: list[list[str]] = []

    def fake_run(command, *, cwd=None, stdout=None):
        del cwd, stdout
        commands.append(list(command))
        Path(command[-1]).write_bytes(payload)

    monkeypatch.setattr(full_droid, "_run", fake_run)
    cache = full_droid._configure_openpi_cache(tmp_path)
    first = full_droid._stage_filter_dictionary("gsutil", cache)
    second = full_droid._stage_filter_dictionary("gsutil", cache)

    assert commands == [
        [
            "gsutil",
            "cp",
            "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
            commands[0][-1],
        ]
    ]
    assert all("*" not in item and "?" not in item for item in commands[0])
    assert first["cache_reused"] is False
    assert second["cache_reused"] is True
    assert first["entry_count"] == 1
    assert Path(os.environ["OPENPI_DATA_HOME"]) == cache


def test_filter_dictionary_rejects_malformed_download(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"not-json"
    monkeypatch.setattr(full_droid, "FILTER_DICTIONARY_BYTES", len(payload))
    monkeypatch.setattr(
        full_droid,
        "FILTER_DICTIONARY_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    def fake_run(command, *, cwd=None, stdout=None):
        del cwd, stdout
        Path(command[-1]).write_bytes(payload)

    monkeypatch.setattr(full_droid, "_run", fake_run)
    cache = full_droid._configure_openpi_cache(tmp_path)
    with pytest.raises(full_droid.OpenPIPipelineError, match="invalid JSON"):
        full_droid._stage_filter_dictionary("gsutil", cache)
    assert not list(cache.rglob("*.partial"))


def test_verified_dataset_resume_revalidates_durable_identity(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    data_root = tmp_path / "data"
    dataset_root = data_root / "droid" / "1.0.1"
    dataset_root.mkdir(parents=True)
    (dataset_root / "part").write_bytes(b"dataset")
    listing = b"8 2026-01-01T00:00:00Z gs://example/part\n"
    work_root.mkdir(parents=True)
    (work_root / "droid-1.0.1-gcs-listing.txt").write_bytes(listing)
    journal = work_root / "telemetry" / "preparation.jsonl"
    full_droid._append_jsonl(
        journal,
        {
            "schema": full_droid.PREPARATION_TELEMETRY_SCHEMA,
            "run_id": "run",
            "record_type": "dataset_verified",
            "normalization_batch": 0,
            "remote_object_count": 1,
            "local_file_count": 1,
            "remote_size_bytes": 7,
            "local_size_bytes": 7,
            "listing_sha256": hashlib.sha256(listing).hexdigest(),
            "checksum_verification_seconds": 2.0,
            "checksum_verification_bytes_per_second": 3.5,
        },
    )

    reused = full_droid._reuse_verified_dataset(
        journal, run_id="run", data_root=data_root, work_root=work_root
    )

    assert reused is not None
    assert reused["verification_reused"] is True
    assert reused["file_count"] == reused["object_count"] == 1
    assert reused["local_total_size_bytes"] == reused["remote_total_size_bytes"] == 7


def test_prepare_report_preserves_filter_dictionary_lineage(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    monkeypatch.setattr(full_droid, "_validate_source", lambda *args: {})
    monkeypatch.setattr(
        full_droid,
        "_stage_filter_dictionary",
        lambda *args: {
            "source_uri": full_droid.FILTER_DICTIONARY_URI,
            "sha256": full_droid.FILTER_DICTIONARY_SHA256,
            "size_bytes": full_droid.FILTER_DICTIONARY_BYTES,
        },
    )
    monkeypatch.setattr(
        full_droid,
        "_stage_dataset",
        lambda *args: {
            "listing_sha256": "a" * 64,
            "object_count": 2,
            "file_count": 2,
            "remote_total_size_bytes": 46,
            "local_total_size_bytes": 46,
            "timings_seconds": {"checksum_sync": 2.0},
            "checksum_verification_bytes_per_second": 23.0,
        },
    )
    monkeypatch.setattr(full_droid, "_configured_upstream", lambda *args: object())
    def fake_compute_norm_stats(*args, **kwargs):
        del args, kwargs
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        return {"sha256": "b" * 64}

    monkeypatch.setattr(full_droid, "_compute_norm_stats", fake_compute_norm_stats)
    monkeypatch.setattr(
        full_droid,
        "_publish_preparation_rrd",
        lambda *args, **kwargs: {"uri": kwargs["rrd_uri"]},
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(
        full_droid,
        "_write_json_uri",
        lambda uri, value: written.update({"uri": uri, "value": value}),
    )

    result = full_droid._prepare(
        SimpleNamespace(
            repo_root=str(tmp_path),
            runtime_image="ghcr.io/example/openpi@sha256:" + "c" * 64,
            work_root=str(tmp_path / "work"),
            data_root=str(tmp_path / "data"),
            experiment="lineage-test",
            gsutil="gsutil",
            output_uri="s3://example.invalid/private/prepare.json",
            rrd_uri="s3://example.invalid/private/preparation.rrd",
            milestone_manifest_uri=(
                "s3://example.invalid/private/preparation.manifest.json"
            ),
            run_id="prepare-lineage-test",
        )
    )

    assert result == 0
    assert written["value"]["filter_dictionary"]["sha256"] == (
        full_droid.FILTER_DICTIONARY_SHA256
    )


def test_normalization_batch_tracker_excludes_unwrapped_progress() -> None:
    observed: list[int] = []
    tracker = full_droid._FactualNormalizationBatchTracker(  # noqa: SLF001
        3, observed.append
    )

    assert list(range(95_658))[-1] == 95_657
    assert tracker.processed_batches == 0
    assert list(tracker.wrap(["batch-1", "batch-2", "batch-3"])) == [
        "batch-1",
        "batch-2",
        "batch-3",
    ]
    tracker.assert_complete()
    assert observed == [1, 2, 3]


def test_normalization_batch_tracker_fails_closed_on_partial_or_reuse() -> None:
    tracker = full_droid._FactualNormalizationBatchTracker(  # noqa: SLF001
        2, lambda _batch: None
    )
    iterator = tracker.wrap(["only-batch"])
    assert list(iterator) == ["only-batch"]
    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="pinned number of factual batches",
    ):
        tracker.assert_complete()
    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="constructed more than once",
    ):
        list(tracker.wrap(["unexpected-retry"]))


def test_normalization_memory_override_is_scoped_and_restored() -> None:
    observed: list[int] = []

    class FakeDroidRldsDataset:
        def __init__(self, *, shuffle_buffer_size: int = 250_000) -> None:
            observed.append(shuffle_buffer_size)

    module = SimpleNamespace(DroidRldsDataset=FakeDroidRldsDataset)
    with full_droid._normalization_dataset_memory_override(module):
        module.DroidRldsDataset(shuffle_buffer_size=250_000)
        assert module.DroidRldsDataset is not FakeDroidRldsDataset

    assert observed == [50_000]
    assert module.DroidRldsDataset is FakeDroidRldsDataset
    module.DroidRldsDataset()
    assert observed == [50_000, 250_000]


def test_normalization_memory_override_restores_after_error() -> None:
    class FakeDroidRldsDataset:
        def __init__(self, *, shuffle_buffer_size: int = 250_000) -> None:
            del shuffle_buffer_size

    module = SimpleNamespace(DroidRldsDataset=FakeDroidRldsDataset)
    with pytest.raises(RuntimeError, match="normalization failed"):
        with full_droid._normalization_dataset_memory_override(module):
            raise RuntimeError("normalization failed")
    assert module.DroidRldsDataset is FakeDroidRldsDataset


def test_normalization_memory_override_fails_on_contract_drift() -> None:
    class DriftedDroidRldsDataset:
        def __init__(self, *, shuffle_buffer_size: int = 42) -> None:
            del shuffle_buffer_size

    module = SimpleNamespace(DroidRldsDataset=DriftedDroidRldsDataset)
    with pytest.raises(
        full_droid.OpenPIPipelineError, match="shuffle-buffer contract changed"
    ):
        with full_droid._normalization_dataset_memory_override(module):
            pass
    assert module.DroidRldsDataset is DriftedDroidRldsDataset


def _telemetry_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_train_steps=2,
        batch_size=256,
        log_interval=1,
        save_interval=2,
    )


def test_wandb_telemetry_log_is_reinstalled_after_init_replaces_it() -> None:
    calls: list[tuple[str, object]] = []

    def pre_init_log(value):
        calls.append(("pre", value))

    def initialized_log(value):
        calls.append(("initialized", value))

    def telemetry_log(value):
        active_log[0](value)
        calls.append(("telemetry", value))

    wandb = SimpleNamespace(log=pre_init_log)

    def init_wandb():
        wandb.log = initialized_log
        return "initialized"

    upstream = SimpleNamespace(wandb=wandb, init_wandb=init_wandb)
    active_log: list[object] = [pre_init_log]
    original, wrapped = full_droid._wrap_wandb_initializer(
        upstream,
        telemetry_log=telemetry_log,
        active_log=active_log,
    )

    assert original is init_wandb
    upstream.wandb.log = telemetry_log
    assert wrapped() == "initialized"
    assert upstream.wandb.log is telemetry_log
    upstream.wandb.log("step-0")
    assert calls == [("initialized", "step-0"), ("telemetry", "step-0")]


def test_wandb_telemetry_init_does_not_make_wrapper_recursive() -> None:
    calls: list[object] = []

    def base_log(value):
        calls.append(value)

    def telemetry_log(value):
        active_log[0](value)

    wandb = SimpleNamespace(log=base_log)
    upstream = SimpleNamespace(wandb=wandb, init_wandb=lambda: None)
    active_log: list[object] = [base_log]
    upstream.wandb.log = telemetry_log
    _, wrapped = full_droid._wrap_wandb_initializer(
        upstream,
        telemetry_log=telemetry_log,
        active_log=active_log,
    )

    wrapped()
    assert active_log[0] is base_log
    upstream.wandb.log("step-0")
    assert calls == ["step-0"]


def test_checkpoint_only_resume_records_only_new_factual_metrics(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    config = _telemetry_config()
    interrupted = full_droid._TrainingTelemetryJournal(
        path, run_id="checkpoint-only-resume", config=config
    )
    interrupted.record_checkpoint(step=1, event="save_requested")
    interrupted.record_checkpoint(step=1, event="materialized")
    interrupted.close()

    resumed = full_droid._TrainingTelemetryJournal(
        path, run_id="checkpoint-only-resume", config=config
    )
    for step in range(2):
        resumed.record_metrics(
            step=step,
            values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 10.0},
            learning_rate=1e-5,
        )
    resumed.record_checkpoint(step=1, event="materialized")
    resumed.close()

    records = full_droid._load_telemetry_records(
        path, run_id="checkpoint-only-resume"
    )
    assert [
        record["optimizer_step"]
        for record in records
        if record["record_type"] == "metrics"
    ] == [0, 1]
    assert sum(
        record["record_type"] == "checkpoint"
        and record["event"] == "materialized"
        for record in records
    ) == 1


def test_telemetry_journal_is_durable_deduplicated_and_run_scoped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    config = _telemetry_config()
    journal = full_droid._TrainingTelemetryJournal(
        path, run_id="rrd-unit-run", config=config
    )
    journal.record_metrics(
        step=0,
        values={"loss": 1.2, "grad_norm": 0.4, "param_norm": 20.0},
        learning_rate=1e-6,
    )
    journal.record_metrics(
        step=1,
        values={"loss": 1.0, "grad_norm": 0.3, "param_norm": 20.1},
        learning_rate=2e-6,
    )
    journal.record_metrics(
        step=1,
        values={"loss": 99.0, "grad_norm": 99.0, "param_norm": 99.0},
        learning_rate=99.0,
    )
    journal.record_checkpoint(step=1, event="save_requested")
    journal.record_checkpoint(step=1, event="materialized")
    journal.close()

    records = full_droid._load_telemetry_records(path, run_id="rrd-unit-run")
    metrics = [record for record in records if record["record_type"] == "metrics"]
    assert [record["optimizer_step"] for record in metrics] == [0, 1]
    assert metrics[1]["metrics"]["loss"] == 1.0
    assert metrics[1]["interval"]["optimizer_steps"] == 1
    assert metrics[1]["interval"]["global_samples_per_second"] > 0
    assert "hostname" not in path.read_text(encoding="utf-8")
    assert "s3://" not in path.read_text(encoding="utf-8")


def test_operator_pause_requires_exact_zero_based_coverage_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    step_root = checkpoint_root / "999"
    step_root.mkdir(parents=True)
    (step_root / "optimizer-state").write_bytes(b"factual-state")
    path = checkpoint_root / "npa-training-telemetry.jsonl"
    config = SimpleNamespace(
        num_train_steps=1_000,
        batch_size=256,
        log_interval=1,
        save_interval=1_000,
    )
    journal = full_droid._TrainingTelemetryJournal(
        path, run_id="operator-pause", config=config
    )
    for step in range(1_000):
        journal.record_metrics(
            step=step,
            values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 10.0},
            learning_rate=1e-5,
        )
    journal.record_checkpoint(step=999, event="save_requested")
    full_droid._write_checkpoint_completion_marker(
        checkpoint_root, step=999, run_id="operator-pause"
    )
    journal.record_checkpoint(step=999, event="materialized")
    journal.close()

    evidence = full_droid._validate_operator_pause(
        checkpoint_root=checkpoint_root,
        telemetry_path=path,
        run_id="operator-pause",
        updates=1_000,
    )

    assert evidence["completed_updates"] == 1_000
    assert evidence["first_optimizer_step"] == 0
    assert evidence["last_optimizer_step"] == 999
    assert evidence["metric_record_count"] == 1_000
    assert evidence["checkpoint_completion_marker"] is True


def test_operator_pause_fails_closed_when_update_coverage_has_a_gap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    config = SimpleNamespace(
        num_train_steps=1_000,
        batch_size=256,
        log_interval=1,
        save_interval=1_000,
    )
    journal = full_droid._TrainingTelemetryJournal(
        path, run_id="operator-pause-gap", config=config
    )
    journal.record_metrics(
        step=999,
        values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 10.0},
        learning_rate=1e-5,
    )
    journal.close()

    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="does not prove updates 0 through 999",
    ):
        full_droid._validate_operator_pause(
            checkpoint_root=tmp_path,
            telemetry_path=path,
            run_id="operator-pause-gap",
            updates=1_000,
        )


def test_checkpoint_retry_verifies_existing_manifest_without_reupload(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "checkpoint"
    (root / "999").mkdir(parents=True)
    (root / "999" / "optimizer-state").write_bytes(b"optimizer")
    records = full_droid._checkpoint_file_records(root)
    manifest = {
        "schema": "npa.workbench.openpi.checkpoint-manifest.v1",
        "root_uri": "s3://example.invalid/private/checkpoint/",
        "content_manifest_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "file_count": len(records),
        "total_size_bytes": sum(int(record["size"]) for record in records),
        "files": records,
    }
    monkeypatch.setattr(full_droid, "_uri_exists", lambda _uri: True)
    monkeypatch.setattr(full_droid, "_read_json_uri", lambda _uri: manifest)
    monkeypatch.setattr(
        full_droid,
        "_download_checkpoint",
        lambda _uri, _path: manifest,
    )

    def unexpected_upload(*_args, **_kwargs):
        raise AssertionError("an immutable checkpoint must not be re-uploaded")

    monkeypatch.setattr(full_droid, "_upload_checkpoint", unexpected_upload)

    assert (
        full_droid._upload_or_verify_checkpoint(
            root, "s3://example.invalid/private/checkpoint"
        )
        == manifest
    )


def test_telemetry_resume_deduplicates_without_inventing_cross_segment_timing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    config = SimpleNamespace(
        num_train_steps=3,
        batch_size=256,
        log_interval=1,
        save_interval=3,
    )
    first = full_droid._TrainingTelemetryJournal(
        path, run_id="rrd-resume", config=config
    )
    first.record_metrics(
        step=0,
        values={"loss": 1.2, "grad_norm": 0.4, "param_norm": 20.0},
        learning_rate=1e-6,
    )
    first.close()

    resumed = full_droid._TrainingTelemetryJournal(
        path, run_id="rrd-resume", config=config
    )
    resumed.record_metrics(
        step=0,
        values={"loss": 99.0, "grad_norm": 99.0, "param_norm": 99.0},
        learning_rate=99.0,
    )
    resumed.record_metrics(
        step=1,
        values={"loss": 1.1, "grad_norm": 0.3, "param_norm": 20.0},
        learning_rate=2e-6,
    )
    resumed.record_metrics(
        step=2,
        values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 20.0},
        learning_rate=3e-6,
    )
    resumed.close()

    metrics = [
        record
        for record in full_droid._load_telemetry_records(
            path, run_id="rrd-resume"
        )
        if record["record_type"] == "metrics"
    ]
    assert [record["optimizer_step"] for record in metrics] == [0, 1, 2]
    assert [record["segment"] for record in metrics] == [1, 2, 2]
    assert metrics[1]["interval"] is None
    assert metrics[2]["interval"]["optimizer_steps"] == 1


def test_telemetry_loader_rejects_non_integer_optimizer_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        '{"schema":"'
        + full_droid.TELEMETRY_SCHEMA
        + '","run_id":"bad-step","segment":1,'
        '"record_type":"metrics","optimizer_step":"1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        full_droid.OpenPIPipelineError, match="invalid step or segment"
    ):
        full_droid._load_telemetry_records(path, run_id="bad-step")


def test_real_rrd_contains_run_identity_timeline_and_review_entities(
    tmp_path: Path,
) -> None:
    run_id = "rrd-unit-run"
    config = _telemetry_config()
    journal_path = tmp_path / "telemetry.jsonl"
    journal = full_droid._TrainingTelemetryJournal(
        journal_path, run_id=run_id, config=config
    )
    for step, loss in enumerate((1.2, 1.0)):
        journal.record_metrics(
            step=step,
            values={"loss": loss, "grad_norm": 0.4, "param_norm": 20.0},
            learning_rate=(step + 1) * 1e-6,
        )
    journal.record_checkpoint(step=1, event="save_requested")
    journal.record_checkpoint(step=1, event="materialized")
    journal.close()

    output = tmp_path / "training.rrd"
    inspection = full_droid._build_training_rrd(
        journal_path,
        output,
        run_id=run_id,
        config=config,
        prepared={
            "dataset": {"listing_sha256": "a" * 64},
            "filter_dictionary": {
                "sha256": full_droid.FILTER_DICTIONARY_SHA256
            },
            "normalization": {"sha256": "b" * 64},
        },
        runtime_image="ghcr.io/example/openpi@sha256:" + "c" * 64,
        hardware={
            "process_count": 8,
            "global_gpu_count": 8,
            "local_devices_per_process": 1,
        },
        topology=[{"sm120_probe": "devices=1 cc=12.0"} for _ in range(8)],
    )

    assert output.stat().st_size > 0
    assert inspection["parseable"] is True
    assert inspection["recording_id"] == run_id
    assert inspection["timelines"] == ["optimizer_step"]
    assert set(full_droid.REQUIRED_RRD_ENTITIES) <= set(inspection["entities"])
    assert inspection["source_telemetry_sha256"] == hashlib.sha256(
        journal_path.read_bytes()
    ).hexdigest()
    decoded_loss = subprocess.run(
        [
            full_droid._rerun_executable(),
            "rrd",
            "print",
            "-vvv",
            "--entity",
            "metrics/loss",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "optimizer_step" in decoded_loss
    assert "[1.2]" in decoded_loss
    assert "[1.0]" in decoded_loss


def test_preparation_rrd_contains_factual_progress_and_lineage(
    tmp_path: Path,
) -> None:
    run_id = "preparation-rrd-unit"
    journal = tmp_path / "preparation.jsonl"
    full_droid._append_jsonl(
        journal,
        {
            "schema": full_droid.PREPARATION_TELEMETRY_SCHEMA,
            "run_id": run_id,
            "record_type": "dataset_verified",
            "normalization_batch": 0,
            "remote_object_count": 2,
            "local_file_count": 2,
            "remote_size_bytes": 46,
            "local_size_bytes": 46,
            "listing_sha256": "a" * 64,
            "checksum_verification_seconds": 2.0,
            "checksum_verification_bytes_per_second": 23.0,
        },
    )
    for record_type, batch in (
        ("normalization_progress", 1_000),
        ("normalization_complete", 2_000),
    ):
        full_droid._append_jsonl(
            journal,
            {
                "schema": full_droid.PREPARATION_TELEMETRY_SCHEMA,
                "run_id": run_id,
                "record_type": record_type,
                "normalization_batch": batch,
                "frames_processed": batch * 256,
                "elapsed_seconds": float(batch),
                "frames_per_second": 256.0,
                "normalization_shuffle_buffer_size": 50_000,
                "peak_rss_bytes": 4_294_967_296,
                "cgroup_peak_memory_bytes": 4_831_838_208,
                "peak_memory_bytes": 4_831_838_208,
                "normalization_resume_mode": "full_restart",
            },
        )
    result = {
        "dataset": {
            "object_count": 2,
            "file_count": 2,
            "total_size_bytes": 46,
            "remote_total_size_bytes": 46,
            "local_total_size_bytes": 46,
            "listing_sha256": "a" * 64,
            "checksum_verification_bytes_per_second": 23.0,
        },
        "normalization": {
            "sha256": "b" * 64,
            "frames_processed": 2_000 * 256,
            "normalization_shuffle_buffer_size": 50_000,
            "peak_rss_bytes": 4_294_967_296,
            "cgroup_peak_memory_bytes": 4_831_838_208,
            "peak_memory_bytes": 4_831_838_208,
            "resume_mode": "full_restart",
        },
    }
    output = tmp_path / "preparation.rrd"
    inspection = full_droid._build_preparation_rrd(
        journal,
        output,
        run_id=run_id,
        result=result,
        runtime_image="ghcr.io/example/openpi@sha256:" + "c" * 64,
    )
    assert inspection["parseable"] is True
    assert inspection["timelines"] == ["normalization_batch"]
    assert set(full_droid.PREPARATION_RRD_ENTITIES) <= set(
        inspection["entities"]
    )


def test_preparation_rrd_uses_isolated_worker_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "preparation-worker-unit"
    journal = tmp_path / "preparation.jsonl"
    full_droid._append_jsonl(
        journal,
        {
            "schema": full_droid.PREPARATION_TELEMETRY_SCHEMA,
            "run_id": run_id,
            "record_type": "dataset_verified",
            "normalization_batch": 0,
            "remote_object_count": 1,
            "local_file_count": 1,
            "remote_size_bytes": 23,
            "local_size_bytes": 23,
            "listing_sha256": "a" * 64,
            "checksum_verification_seconds": 1.0,
            "checksum_verification_bytes_per_second": 23.0,
        },
    )
    full_droid._append_jsonl(
        journal,
        {
            "schema": full_droid.PREPARATION_TELEMETRY_SCHEMA,
            "run_id": run_id,
            "record_type": "normalization_complete",
            "normalization_batch": 1,
            "frames_processed": 256,
            "elapsed_seconds": 1.0,
            "frames_per_second": 256.0,
            "normalization_shuffle_buffer_size": 50_000,
            "peak_rss_bytes": 4_294_967_296,
            "cgroup_peak_memory_bytes": 4_831_838_208,
            "peak_memory_bytes": 4_831_838_208,
            "normalization_resume_mode": "full_restart",
        },
    )
    output = tmp_path / "worker.rrd"
    monkeypatch.setattr(
        full_droid, "_rrd_worker_python", lambda: Path(sys.executable)
    )
    inspection = full_droid._build_preparation_rrd(
        journal,
        output,
        run_id=run_id,
        result={
            "dataset": {
                "object_count": 1,
                "file_count": 1,
                "remote_total_size_bytes": 23,
                "local_total_size_bytes": 23,
                "listing_sha256": "a" * 64,
            },
            "normalization": {
                "sha256": "b" * 64,
                "frames_processed": 256,
                "normalization_shuffle_buffer_size": 50_000,
                "peak_rss_bytes": 4_294_967_296,
                "cgroup_peak_memory_bytes": 4_831_838_208,
                "peak_memory_bytes": 4_831_838_208,
                "resume_mode": "full_restart",
            },
        },
        runtime_image="ghcr.io/example/openpi@sha256:" + "c" * 64,
    )
    assert inspection["recording_id"] == run_id
    assert output.stat().st_size > 0


def test_rrd_worker_python_fails_closed_when_configured_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_OPENPI_RERUN_PYTHON", "/missing/rerun-python")
    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="isolated Rerun worker interpreter is unavailable",
    ):
        full_droid._rrd_worker_python()


def test_rrd_worker_python_discovers_default_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "python"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o700)
    monkeypatch.delenv("NPA_OPENPI_RERUN_PYTHON", raising=False)
    monkeypatch.setattr(full_droid, "DEFAULT_RERUN_WORKER_PYTHON", worker)

    assert full_droid._rrd_worker_python() == worker


def test_rrd_worker_python_allows_direct_fallback_without_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NPA_OPENPI_RERUN_PYTHON", raising=False)
    monkeypatch.setattr(
        full_droid, "DEFAULT_RERUN_WORKER_PYTHON", tmp_path / "missing"
    )

    assert full_droid._rrd_worker_python() is None


def test_rrd_worker_python_avoids_current_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_OPENPI_RERUN_PYTHON", sys.executable)

    assert full_droid._rrd_worker_python() is None


def test_rrd_worker_python_keeps_distinct_virtualenv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "python"
    worker.symlink_to(sys.executable)
    monkeypatch.setenv("NPA_OPENPI_RERUN_PYTHON", str(worker))

    assert worker.resolve() == Path(sys.executable).resolve()
    assert full_droid._rrd_worker_python() == worker


def test_rerun_executable_prefers_isolated_worker_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "python"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o700)
    cli = tmp_path / "rerun"
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    cli.chmod(0o700)
    monkeypatch.setenv("NPA_OPENPI_RERUN_PYTHON", str(worker))

    assert full_droid._rerun_executable() == str(cli)


def test_rerun_executable_fails_closed_without_worker_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "python"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o700)
    monkeypatch.setenv("NPA_OPENPI_RERUN_PYTHON", str(worker))

    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="isolated Rerun worker CLI is unavailable",
    ):
        full_droid._rerun_executable()


def test_expanded_time_panel_supports_legacy_constructor() -> None:
    calls: list[object] = []

    class FakePanelState:
        Expanded = object()

    class FakeBlueprint:
        PanelState = FakePanelState

        @staticmethod
        def TimePanel(*, state: object) -> dict[str, object]:
            calls.append(state)
            return {"state": state}

    panel = full_droid._expanded_time_panel(FakeBlueprint)

    assert panel == {"state": FakePanelState.Expanded}
    assert calls == [FakePanelState.Expanded]


def test_dataset_verification_retry_keeps_preparation_journal_immutable(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "preparation.jsonl"
    dataset = {
        "object_count": 2,
        "file_count": 2,
        "remote_total_size_bytes": 46,
        "local_total_size_bytes": 46,
        "listing_sha256": "a" * 64,
        "timings_seconds": {"checksum_sync": 2.0},
        "checksum_verification_bytes_per_second": 23.0,
    }
    full_droid._record_dataset_verification_once(
        journal, run_id="preparation-retry", dataset=dataset
    )
    first = journal.read_bytes()
    retried = {
        **dataset,
        "timings_seconds": {"checksum_sync": 1.0},
        "checksum_verification_bytes_per_second": 46.0,
    }
    full_droid._record_dataset_verification_once(
        journal, run_id="preparation-retry", dataset=retried
    )
    assert journal.read_bytes() == first


def test_checkpoint_completion_marker_is_atomic_and_run_scoped(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    step_path = checkpoint_root / "10000"
    step_path.mkdir(parents=True)
    (step_path / "orbax-metadata").write_text("complete", encoding="utf-8")

    assert not full_droid._checkpoint_completion_is_valid(
        checkpoint_root, step=10_000, run_id="marker-unit"
    )
    full_droid._write_checkpoint_completion_marker(
        checkpoint_root, step=10_000, run_id="marker-unit"
    )
    assert full_droid._checkpoint_completion_is_valid(
        checkpoint_root, step=10_000, run_id="marker-unit"
    )
    assert not full_droid._checkpoint_completion_is_valid(
        checkpoint_root, step=10_000, run_id="different-run"
    )
    assert not list(step_path.glob("*.tmp"))


def test_milestone_reconcile_requires_factual_metric_coverage(tmp_path: Path) -> None:
    run_id = "coverage-reconcile"
    config = SimpleNamespace(
        num_train_steps=full_droid.QUALIFICATION_STEPS,
        batch_size=256,
        log_interval=1,
        save_interval=full_droid.QUALIFICATION_STEPS,
    )
    journal = full_droid._TrainingTelemetryJournal(
        tmp_path / "npa-training-telemetry.jsonl", run_id=run_id, config=config
    )
    final_step = full_droid.QUALIFICATION_STEPS - 1
    journal.record_checkpoint(step=final_step, event="save_requested")
    journal.record_checkpoint(step=final_step, event="materialized")
    journal.close()
    publisher = full_droid._TrainingMilestonePublisher(
        journal_path=journal.path,
        run_id=run_id,
        kind="qualification",
        config=config,
        prepared={},
        runtime_image="ghcr.io/example/image@sha256:" + "0" * 64,
        hardware={},
        topology=[],
        rrd_root_uri="s3://example.invalid/private/run",
    )
    called: list[int] = []
    publisher.publish_for_optimizer_step = called.append

    publisher.reconcile_available()

    assert called == []


def test_rank_zero_prepares_fresh_shared_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "attempt-scoped-telemetry.jsonl").write_text(
        "incomplete\n", encoding="utf-8"
    )
    barriers: list[str] = []

    class Multihost:
        @staticmethod
        def broadcast_one_to_all(value: object, *, is_source: bool) -> object:
            assert is_source is True
            return value

        @staticmethod
        def sync_global_devices(name: str) -> None:
            barriers.append(name)

    assert (
        full_droid._prepare_distributed_checkpoint_root(
            root, rank=0, multihost_utils=Multihost
        )
        is False
    )
    assert root.is_dir()
    assert not any(root.iterdir())
    assert barriers == [
        "npa-openpi-checkpoint-root-cleaned",
        "npa-openpi-checkpoint-root-ready",
    ]


def test_nonzero_rank_never_cleans_shared_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    stale = root / "attempt-scoped-telemetry.jsonl"
    stale.write_text("incomplete\n", encoding="utf-8")
    barriers: list[str] = []

    class Multihost:
        @staticmethod
        def broadcast_one_to_all(value: object, *, is_source: bool) -> object:
            assert is_source is False
            return value

        @staticmethod
        def sync_global_devices(name: str) -> None:
            barriers.append(name)

    assert (
        full_droid._prepare_distributed_checkpoint_root(
            root, rank=3, multihost_utils=Multihost
        )
        is False
    )
    assert stale.is_file()
    assert barriers == [
        "npa-openpi-checkpoint-root-cleaned",
        "npa-openpi-checkpoint-root-ready",
    ]


def test_nonzero_rank_idempotently_materializes_visible_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    barriers: list[str] = []

    class Multihost:
        @staticmethod
        def broadcast_one_to_all(value: object, *, is_source: bool) -> object:
            assert is_source is False
            return value

        @staticmethod
        def sync_global_devices(name: str) -> None:
            barriers.append(name)

    assert (
        full_droid._prepare_distributed_checkpoint_root(
            root, rank=5, multihost_utils=Multihost
        )
        is False
    )
    assert root.is_dir()
    assert barriers == [
        "npa-openpi-checkpoint-root-cleaned",
        "npa-openpi-checkpoint-root-ready",
    ]


def test_distributed_checkpoint_resume_preserves_numeric_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoint"
    step = root / "999"
    step.mkdir(parents=True)
    state = step / "orbax-metadata"
    state.write_text("complete", encoding="utf-8")
    barriers: list[str] = []

    class Multihost:
        @staticmethod
        def broadcast_one_to_all(value: object, *, is_source: bool) -> object:
            assert is_source is True
            return value

        @staticmethod
        def sync_global_devices(name: str) -> None:
            barriers.append(name)

    assert (
        full_droid._prepare_distributed_checkpoint_root(
            root, rank=0, multihost_utils=Multihost
        )
        is True
    )
    assert state.read_text(encoding="utf-8") == "complete"
    assert barriers == [
        "npa-openpi-checkpoint-root-cleaned",
        "npa-openpi-checkpoint-root-ready",
    ]


def test_distributed_checkpoint_config_never_overwrites() -> None:
    @dataclasses.dataclass(frozen=True)
    class Config:
        resume: bool = False
        overwrite: bool = True

    config = Config()
    configured = full_droid._non_destructive_checkpoint_config(config)

    assert configured.resume is True
    assert configured.overwrite is False


def test_telemetry_prefix_is_immutable_milestone_source(tmp_path: Path) -> None:
    run_id = "prefix-unit"
    path = tmp_path / "telemetry.jsonl"
    journal = full_droid._TrainingTelemetryJournal(
        path,
        run_id=run_id,
        config=SimpleNamespace(
            num_train_steps=4,
            batch_size=256,
            log_interval=1,
            save_interval=2,
        ),
    )
    for step in range(4):
        journal.record_metrics(
            step=step,
            values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 20.0},
            learning_rate=1e-6,
        )
    journal.close()
    first = full_droid._telemetry_prefix_payload(
        path, run_id=run_id, through_step=1
    )
    assert b'"optimizer_step": 2' not in first
    assert b'"optimizer_step": 1' in first
    assert first == full_droid._telemetry_prefix_payload(
        path, run_id=run_id, through_step=1
    )


def test_completed_optimizer_updates_maps_zero_based_source_steps() -> None:
    assert full_droid._completed_optimizer_updates(499) == 500
    assert full_droid._completed_optimizer_updates(999) == 1_000
    with pytest.raises(full_droid.OpenPIPipelineError, match="cannot be negative"):
        full_droid._completed_optimizer_updates(-1)


def test_milestone_manifest_hashes_rrd_and_journal() -> None:
    payload = full_droid._manifest_payload(
        run_id="manifest-unit",
        stage="full",
        milestone="progress-step-001000",
        rrd_uri="s3://private.invalid/run/reports/rrd/progress-step-001000.rrd",
        inspection={"bytes": 12, "sha256": "a" * 64},
        source_telemetry_sha256="b" * 64,
        source_coverage={"through_optimizer_step": 1_000},
    )
    value = json.loads(payload)
    content_sha256 = value.pop("content_sha256")
    value["content_sha256"] = ""
    assert content_sha256 == hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert value["rrd"]["sha256"] == "a" * 64
    assert value["source_telemetry_sha256"] == "b" * 64


def test_operator_pause_milestone_is_checkpointed_at_optimizer_step_999(
    tmp_path: Path,
) -> None:
    publisher = full_droid._TrainingMilestonePublisher(
        journal_path=tmp_path / "telemetry.jsonl",
        run_id="pause-mapping",
        kind="full",
        config=SimpleNamespace(),
        prepared={},
        runtime_image="ghcr.io/example/openpi@sha256:" + "a" * 64,
        hardware={},
        topology=[],
        rrd_root_uri="s3://example.invalid/private/rrd",
        pause_after_updates=1_000,
    )

    assert publisher.milestones == {500: 499, 1_000: 999}
    assert publisher.is_log_only(499) is True
    assert publisher.requires_checkpoint(499) is False
    assert publisher.is_log_only(999) is False
    assert publisher.requires_checkpoint(999) is True


def test_full_run_500_update_milestone_maps_to_source_step_499(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(full_droid, "_uri_exists", lambda _uri: False)
    publisher = full_droid._TrainingMilestonePublisher(
        journal_path=tmp_path / "telemetry.jsonl",
        run_id="full-mapping",
        kind="full",
        config=SimpleNamespace(),
        prepared={},
        runtime_image="ghcr.io/example/openpi@sha256:" + "a" * 64,
        hardware={},
        topology=[],
        rrd_root_uri="s3://example.invalid/private/rrd",
    )

    assert publisher.milestones[500] == 499
    assert publisher.is_log_only(499) is True
    assert publisher.requires_checkpoint(499) is False


def test_pause_run_reconciles_500_update_log_only_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "telemetry.jsonl"
    journal.write_text("present\n", encoding="utf-8")
    monkeypatch.setattr(
        full_droid,
        "_load_telemetry_records",
        lambda _path, *, run_id: [
            {"record_type": "metrics", "optimizer_step": step}
            for step in range(500)
        ],
    )
    publisher = full_droid._TrainingMilestonePublisher(
        journal_path=journal,
        run_id="pause-reconcile",
        kind="full",
        config=SimpleNamespace(log_interval=1),
        prepared={},
        runtime_image="ghcr.io/example/openpi@sha256:" + "a" * 64,
        hardware={},
        topology=[],
        rrd_root_uri="s3://example.invalid/private/rrd",
        pause_after_updates=1_000,
    )
    published: list[int] = []
    publisher.publish_for_optimizer_step = published.append

    publisher.reconcile_available()

    assert published == [499]


def test_full_resume_preserves_500_log_only_and_paused_1000_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pause_manifest = {
        "schema": full_droid.MILESTONE_MANIFEST_SCHEMA,
        "run_id": "resume-mapping",
        "milestone": "progress-step-001000",
        "source_coverage": {
            "through_optimizer_step": 999,
            "checkpoint_materialized": True,
        },
    }
    monkeypatch.setattr(full_droid, "_uri_exists", lambda _uri: True)
    monkeypatch.setattr(full_droid, "_read_json_uri", lambda _uri: pause_manifest)
    publisher = full_droid._TrainingMilestonePublisher(
        journal_path=tmp_path / "telemetry.jsonl",
        run_id="resume-mapping",
        kind="full",
        config=SimpleNamespace(),
        prepared={},
        runtime_image="ghcr.io/example/openpi@sha256:" + "a" * 64,
        hardware={},
        topology=[],
        rrd_root_uri="s3://example.invalid/private/rrd",
    )

    assert publisher.milestones[500] == 499
    assert publisher.is_log_only(499) is True
    assert publisher.milestones[1_000] == 999
    assert publisher.is_log_only(999) is False
    assert publisher.requires_checkpoint(999) is True


def test_train_parser_exposes_only_the_explicit_pause_boundary() -> None:
    parser = full_droid.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--runtime-image",
            "ghcr.io/example/openpi@sha256:" + "a" * 64,
            "--experiment",
            "pause-parser",
            "--prepare-uri",
            "s3://example.invalid/private/prepare.json",
            "--output-uri",
            "s3://example.invalid/private/paused.json",
            "--checkpoint-uri",
            "s3://example.invalid/private/checkpoint",
            "--telemetry-uri",
            "s3://example.invalid/private/telemetry.jsonl",
            "--rrd-root-uri",
            "s3://example.invalid/private/rrd",
            "--run-id",
            "pause-parser",
            "--pause-after-updates",
            "1000",
        ]
    )
    assert args.pause_after_updates == 1_000


def test_training_rrd_rebuild_has_deterministic_decoded_contract(
    tmp_path: Path,
) -> None:
    run_id = "deterministic-rebuild"
    config = _telemetry_config()
    journal_path = tmp_path / "telemetry.jsonl"
    journal = full_droid._TrainingTelemetryJournal(
        journal_path, run_id=run_id, config=config
    )
    for step in range(2):
        journal.record_metrics(
            step=step,
            values={"loss": 1.0, "grad_norm": 0.2, "param_norm": 20.0},
            learning_rate=1e-6,
        )
    journal.record_checkpoint(step=1, event="save_requested")
    journal.record_checkpoint(step=1, event="materialized")
    journal.close()
    kwargs = {
        "run_id": run_id,
        "config": config,
        "prepared": {
            "dataset": {"listing_sha256": "a" * 64},
            "filter_dictionary": {
                "sha256": full_droid.FILTER_DICTIONARY_SHA256
            },
            "normalization": {"sha256": "b" * 64},
        },
        "runtime_image": "ghcr.io/example/openpi@sha256:" + "c" * 64,
        "hardware": {
            "process_count": 8,
            "global_gpu_count": 8,
            "local_devices_per_process": 1,
        },
        "topology": [
            {"sm120_probe": "devices=1 cc=12.0"} for _ in range(8)
        ],
    }
    first = tmp_path / "first.rrd"
    second = tmp_path / "second.rrd"
    first_inspection = full_droid._build_training_rrd(
        journal_path, first, **kwargs
    )
    second_inspection = full_droid._build_training_rrd(
        journal_path, second, **kwargs
    )
    for key in (
        "application_id",
        "recording_id",
        "timelines",
        "entities",
        "source_telemetry_sha256",
    ):
        assert first_inspection[key] == second_inspection[key]


def test_runtime_image_provenance_requires_only_a_digest() -> None:
    digest = "sha256:" + "c" * 64
    assert (
        full_droid._runtime_image_digest("ghcr.io/example/openpi@" + digest)
        == digest
    )
    with pytest.raises(
        full_droid.OpenPIPipelineError, match="must be pinned by SHA-256"
    ):
        full_droid._runtime_image_digest("private.registry.invalid/openpi:latest")


def test_write_once_is_idempotent_and_rejects_conflicting_bytes(
    tmp_path: Path,
) -> None:
    target = str(tmp_path / "artifact.rrd")
    full_droid._write_once_or_verify(
        target, b"first", content_type=full_droid.RERUN_SCHEMA
    )
    full_droid._write_once_or_verify(
        target, b"first", content_type=full_droid.RERUN_SCHEMA
    )
    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="immutable artifact differs from this run",
    ):
        full_droid._write_once_or_verify(
            target, b"different", content_type=full_droid.RERUN_SCHEMA
        )


def test_rrd_refuses_incomplete_actual_metric_history(tmp_path: Path) -> None:
    run_id = "rrd-incomplete"
    config = _telemetry_config()
    journal_path = tmp_path / "telemetry.jsonl"
    journal = full_droid._TrainingTelemetryJournal(
        journal_path, run_id=run_id, config=config
    )
    journal.record_metrics(
        step=0,
        values={"loss": 1.0, "grad_norm": 0.4, "param_norm": 20.0},
        learning_rate=1e-6,
    )
    journal.record_checkpoint(step=1, event="save_requested")
    journal.record_checkpoint(step=1, event="materialized")
    journal.close()

    with pytest.raises(
        full_droid.OpenPIPipelineError,
        match="does not cover every upstream logging step",
    ):
        full_droid._build_training_rrd(
            journal_path,
            tmp_path / "incomplete.rrd",
            run_id=run_id,
            config=config,
            prepared={
                "dataset": {},
                "filter_dictionary": {},
                "normalization": {},
            },
            runtime_image="ghcr.io/example/openpi@sha256:" + "d" * 64,
            hardware={
                "process_count": 8,
                "global_gpu_count": 8,
                "local_devices_per_process": 1,
            },
            topology=[{"sm120_probe": "devices=1 cc=12.0"} for _ in range(8)],
        )
