"""Contract tests for the native five-stage LeRobot FastWAM workflow."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

from npa.clients.storage import StorageError
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows import fastwam_policy as fastwam
from npa.workflows.lerobot_dataset import LeRobotDatasetSummary


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "fastwam-policy-qualification.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
FASTWAM_DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "lerobot" / "Dockerfile"
FASTWAM_NOTICE = FASTWAM_DOCKERFILE.parent / "notices" / "NOTICE-FASTWAM"


def _runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        fastwam_base_revision="a" * 40,
        wan_revision="b" * 40,
        wan_diffusers_revision="c" * 40,
        umt5_revision="d" * 40,
        device="cuda",
        train_steps=300_000,
        batch_size=8,
        checkpoint_save_freq=20_000,
        environment="libero",
        environment_task="libero_10",
        episode_length=200,
        observation_height=224,
        observation_width=224,
        eval_batch_size=1,
        policy_dtype="float32",
        n_action_steps=10,
        episodes=50,
        seed=42,
        compile_action_infer=True,
    )


def test_workflow_has_five_connected_substantive_native_stages() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="fastwam-contract")

    assert [step.state for step in plan.steps] == [
        "prepare",
        "train",
        "rollout",
        "evaluate",
        "report",
    ]
    assert [step.tool_ref for step in plan.steps] == [
        "workbench.lerobot.fastwam_prepare",
        "workbench.lerobot.fastwam_train",
        "workbench.lerobot.fastwam_rollout",
        "workbench.lerobot.fastwam_evaluate",
        "workbench.lerobot.fastwam_report",
    ]
    assert all(step.argv or step.shell for step in plan.steps)
    assert spec.states["train"].inputs[0].uri == "{{config.prepared_uri}}recipe.json"
    assert spec.states["rollout"].inputs[1].uri == "{{config.training_uri}}checkpoint/"
    assert (
        spec.states["evaluate"].inputs[-1].uri == "{{config.rollouts_uri}}rollout.json"
    )
    assert spec.states["report"].outputs[1].uri == "{{config.report_uri}}fastwam.rrd"
    assert (
        spec.resources["gpu"]["accelerators"]
        == "{{config.gpu_type}}:{{config.gpu_count}}"
    )
    train = next(step for step in plan.steps if step.state == "train")
    assert "--checkpoint-save-freq" in train.argv
    assert "20000" in train.argv
    assert (
        spec.states["train"].outputs[-1].uri
        == "{{config.training_uri}}recovery/latest.json"
    )


def test_adjacent_readiness_record_hash_binds_the_saved_workflow() -> None:
    readiness = json.loads(READINESS.read_text())

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["source_image"]["status"] == "verified"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"


def test_fastwam_image_pins_the_stable_le_robot_release_and_feature_extra() -> None:
    dockerfile = FASTWAM_DOCKERFILE.read_text()
    notice = FASTWAM_NOTICE.read_text()

    assert '"${LEROBOT_VERSION}" = "0.6.1"' in dockerfile
    assert (
        "lerobot[training,evaluation,pusht,libero,diffusion,smolvla,fastwam]"
        in dockerfile
    )
    assert "LeRobot 0.6.1 distribution" in notice
    assert "7e241bd630a3719a56157a497ce5d08f244784f1" in notice
    assert fastwam.LEROBOT_VERSION == "0.6.1"
    assert fastwam.LEROBOT_RELEASE_COMMIT == "7e241bd630a3719a56157a497ce5d08f244784f1"


def test_prepare_seals_disjoint_split_and_upstream_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text('{"features": {}}\n')
    summary = LeRobotDatasetSummary(
        source_uri="unused",
        local_path=str(source),
        repo_id="operator/robot-data",
        revision="f" * 40,
        license="CC-BY-4.0",
        total_episodes=4,
        total_frames=40,
        fps=30,
        episode_indices=[2, 3, 7, 11],
        feature_keys=["action", "observation.images.top", "observation.state"],
        camera_keys=["observation.images.top"],
        state_keys=["observation.state"],
        action_keys=["action"],
        loaded_with_lerobot_dataset=True,
    )
    monkeypatch.setattr(fastwam, "_materialize_dataset", lambda *_args: source)
    monkeypatch.setattr(
        fastwam, "summarize_lerobot_dataset", lambda *_args, **_kwargs: summary
    )
    monkeypatch.setattr(
        fastwam,
        "_validate_native_fastwam_dataset_contract",
        lambda *_args: {"has_task_or_precomputed_context": True},
    )
    args = argparse.Namespace(
        input_path="s3://input/",
        dataset_repo_id="operator/robot-data",
        dataset_revision="f" * 40,
        dataset_license="CC-BY-4.0",
        train_fraction=0.75,
        seed=42,
    )

    fastwam._prepare(args, tmp_path / "work", tmp_path / "prepared")

    recipe = json.loads((tmp_path / "prepared" / "recipe.json").read_text())
    assert set(recipe["train_episode_indices"]).isdisjoint(
        recipe["heldout_episode_indices"]
    )
    assert sorted(
        recipe["train_episode_indices"] + recipe["heldout_episode_indices"]
    ) == [2, 3, 7, 11]
    assert recipe["policy"] == "fastwam"
    assert (
        recipe["native_fastwam_feature_contract"]["has_task_or_precomputed_context"]
        is True
    )
    assert recipe["physical_robot_tested"] is False
    assert "Cosmos3" in recipe["upstream"]["distinction"]


def test_native_commands_preserve_split_and_direct_action_contract(tmp_path) -> None:
    recipe = {
        "dataset": {"repo_id": "operator/robot-data", "revision": "f" * 40},
        "train_episode_indices": [1, 4, 8],
    }
    models = {
        key: tmp_path / key for key in ("fastwam_base", "wan", "wan_diffusers", "umt5")
    }
    train = fastwam._fastwam_train_command(
        _runtime_args(), recipe, tmp_path / "data", tmp_path / "out", models
    )
    rollout = fastwam._fastwam_eval_command(
        _runtime_args(), tmp_path / "checkpoint", tmp_path / "eval", models
    )

    assert train[0] == "lerobot-train"
    assert "--policy.type=fastwam" in train
    assert "--policy.device=cuda" in train
    assert "--dataset.episodes=[1, 4, 8]" in train
    assert "--env_eval_freq=0" in train
    assert "--save_checkpoint=true" in train
    assert "--save_freq=20000" in train
    assert "--policy.push_to_hub=false" in train
    assert rollout[0] == "lerobot-eval"
    assert "--policy.compile_action_infer=true" in rollout
    assert "--env.task=libero_10" in rollout
    assert "--env.observation_height=224" in rollout
    assert "--policy.dtype=float32" in rollout
    assert "--policy.n_action_steps=10" in rollout
    assert all("generate_video" not in item for item in rollout)


def _write_resumable_checkpoint(root: Path, step: int = 20_000) -> Path:
    checkpoint = root / "checkpoints" / f"{step:06d}"
    pretrained = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    state.mkdir()
    (pretrained / "config.json").write_text("{}\n")
    (pretrained / "model.safetensors").write_bytes(b"policy")
    (pretrained / "train_config.json").write_text(
        json.dumps({"scheduler": {"type": "cosine"}})
    )
    (state / "training_step.json").write_text(json.dumps({"step": step}))
    (state / "rng_state.safetensors").write_bytes(b"rng")
    (state / "optimizer_state.safetensors").write_bytes(b"optimizer")
    (state / "optimizer_param_groups.json").write_text("[]\n")
    (state / "scheduler_state.json").write_text("{}\n")
    (checkpoint.parent / "last").symlink_to(checkpoint.name)
    return checkpoint


class _RecoveryStorage:
    """Minimal in-memory StorageClient contract for recovery publication tests."""

    def __init__(self) -> None:
        self.directories: dict[str, Path] = {}
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, str]] = []

    def read_bytes_with_etag(self, uri: str):
        return self.objects.get(uri)

    def upload_directory(
        self, source: str, uri: str, *, require_empty: bool = False
    ) -> str:
        assert require_empty is True
        assert uri not in self.directories
        self.directories[uri] = Path(source)
        self.calls.append(("directory", uri))
        return uri

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        assert content_type == "application/json"
        if if_none_match:
            assert uri not in self.objects
        elif if_match:
            assert self.objects[uri][1] == if_match
        else:  # pragma: no cover - documents the StorageClient contract
            raise AssertionError("conditional write guard is required")
        etag = f"etag-{len(self.objects) + 1}"
        self.objects[uri] = (payload, etag)
        self.calls.append(("manifest", uri))
        return etag

    def download_directory(self, uri: str, local_dir: str) -> str:
        shutil.copytree(self.directories[uri], local_dir, dirs_exist_ok=True)
        return local_dir


def test_recovery_mirror_publishes_only_complete_native_resume_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "training"
    checkpoint = _write_resumable_checkpoint(training)
    storage = _RecoveryStorage()
    published: dict[int, dict[str, object]] = {}
    destination = "s3://task-owned/training/"

    fastwam._sync_recoverable_checkpoints(  # type: ignore[arg-type]
        storage, training, destination, published
    )

    assert list(published) == [20_000]
    assert storage.calls[0][0] == "directory"
    assert storage.calls[-1] == (
        "manifest",
        "s3://task-owned/training/recovery/latest.json",
    )
    manifest = json.loads(storage.objects[storage.calls[-1][1]][0])
    assert manifest["schema"] == fastwam.RECOVERY_MANIFEST_SCHEMA
    assert manifest["step"] == 20_000
    assert manifest["checkpoint_sha256"] == fastwam._tree_digest(checkpoint)

    monkeypatch.setattr(fastwam.StorageClient, "from_environment", lambda: storage)
    restored = fastwam._restore_recoverable_checkpoint(
        destination, tmp_path / "restored"
    )
    assert (
        restored
        == tmp_path / "restored" / "checkpoints" / "020000" / "pretrained_model"
    )
    assert (restored / "model.safetensors").read_bytes() == b"policy"
    assert (restored.parent.parent / "last").resolve() == restored.parent.resolve()


@pytest.mark.parametrize(
    "missing_state", ["optimizer_state.safetensors", "optimizer_param_groups.json"]
)
def test_recovery_mirror_rejects_incomplete_optimizer_resume_state(
    tmp_path, missing_state: str
) -> None:
    checkpoint = _write_resumable_checkpoint(tmp_path / "training")
    (checkpoint / "training_state" / missing_state).unlink()

    assert fastwam._complete_recoverable_checkpoints(tmp_path / "training") == []
    with pytest.raises(fastwam.FastWAMPolicyError, match="not yet resumable") as error:
        fastwam._recoverable_checkpoint_record(checkpoint)
    assert missing_state in str(error.value)


def test_recovery_mirror_requires_the_upstream_last_pointer(tmp_path) -> None:
    checkpoint = _write_resumable_checkpoint(tmp_path / "training")
    (checkpoint.parent / "last").unlink()

    assert fastwam._complete_recoverable_checkpoints(tmp_path / "training") == []


def test_recovery_mirror_never_advances_manifest_after_interrupted_upload(
    tmp_path,
) -> None:
    class InterruptedStorage(_RecoveryStorage):
        def upload_directory(
            self, source: str, uri: str, *, require_empty: bool = False
        ) -> str:
            super().upload_directory(source, uri, require_empty=require_empty)
            raise RuntimeError("simulated interrupted upload")

    storage = InterruptedStorage()
    _write_resumable_checkpoint(tmp_path / "training")

    with pytest.raises(RuntimeError, match="interrupted upload"):
        fastwam._sync_recoverable_checkpoints(  # type: ignore[arg-type]
            storage, tmp_path / "training", "s3://task-owned/training/", {}
        )
    assert storage.objects == {}


def test_recovery_mirror_rejects_corrupt_uploaded_checkpoint(tmp_path) -> None:
    class CorruptReadbackStorage(_RecoveryStorage):
        def download_directory(self, uri: str, local_dir: str) -> str:
            result = super().download_directory(uri, local_dir)
            (Path(local_dir) / "pretrained_model" / "model.safetensors").write_bytes(
                b"corrupt"
            )
            return result

    storage = CorruptReadbackStorage()
    _write_resumable_checkpoint(tmp_path / "training")

    with pytest.raises(fastwam.FastWAMPolicyError, match="does not match"):
        fastwam._sync_recoverable_checkpoints(  # type: ignore[arg-type]
            storage, tmp_path / "training", "s3://task-owned/training/", {}
        )
    assert storage.objects == {}


def test_recovery_mirror_uses_the_actual_storage_client_contract() -> None:
    """Keep recovery calls bound to the concrete StorageClient API, not its double."""

    upload = inspect.signature(fastwam.StorageClient.upload_directory).parameters
    conditional_put = inspect.signature(
        fastwam.StorageClient.put_bytes_conditional
    ).parameters
    assert upload["require_empty"].default is False
    assert {"if_match", "if_none_match", "content_type"} <= set(conditional_put)
    assert (
        "bucket_uri"
        in inspect.signature(fastwam.StorageClient.read_bytes_with_etag).parameters
    )
    assert (
        "local_dir"
        in inspect.signature(fastwam.StorageClient.download_directory).parameters
    )


def test_recovery_mirror_storage_preconditions_use_concrete_client_semantics(
    tmp_path,
) -> None:
    """Exercise the actual CAS and empty-prefix behavior, not only a test double."""

    class StorageApi:
        def __init__(self) -> None:
            self.puts: list[dict[str, object]] = []

        def put_object(self, **kwargs: object) -> dict[str, str]:
            self.puts.append(kwargs)
            return {"ETag": f"etag-{len(self.puts)}"}

        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [{"Key": "recovery/already-there"}]}

    api = StorageApi()
    client = object.__new__(fastwam.StorageClient)
    client._s3 = api  # type: ignore[attr-defined]

    assert (
        client.put_bytes_conditional(
            b"manifest", "s3://task-owned/recovery/latest.json", if_none_match=True
        )
        == "etag-1"
    )
    assert api.puts[-1]["IfNoneMatch"] == "*"
    assert (
        client.put_bytes_conditional(
            b"replacement",
            "s3://task-owned/recovery/latest.json",
            if_match="etag-1",
        )
        == "etag-2"
    )
    assert api.puts[-1]["IfMatch"] == "etag-1"

    with pytest.raises(StorageError, match="must be empty"):
        client.upload_directory(
            str(tmp_path), "s3://task-owned/recovery", require_empty=True
        )


def test_recovery_mirror_retries_a_transient_failure_while_native_child_runs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient upload failure must not stop later mirror/readback attempts."""

    attempts: list[int] = []

    def transient_sync(
        _storage: object,
        _training: Path,
        _destination: str,
        published: dict[int, dict[str, object]],
    ) -> None:
        attempts.append(len(attempts) + 1)
        if len(attempts) == 2:
            raise RuntimeError("transient object-storage timeout")
        if len(attempts) >= 3:
            published[20_000] = {"checkpoint_sha256": "a" * 64}

    monkeypatch.setattr(fastwam.StorageClient, "from_environment", lambda: object())
    monkeypatch.setattr(fastwam, "_sync_recoverable_checkpoints", transient_sync)
    monkeypatch.setattr(fastwam, "_RECOVERY_MIRROR_POLL_SECONDS", 0.001)

    receipt = fastwam._run_training_with_checkpoint_mirror(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        tmp_path / "finite-native-child.log",
        tmp_path / "training",
        "s3://task-owned/training/",
    )

    assert len(attempts) >= 3
    assert receipt["final_mirror_verified"] is True
    assert receipt["mirror_failures"] == [
        {"type": "RuntimeError", "message": "transient object-storage timeout"}
    ]
    assert receipt["mirrored_checkpoints"] == [
        {"step": 20_000, "checkpoint_sha256": "a" * 64}
    ]


def test_train_keeps_successful_local_checkpoint_when_final_mirror_is_unverified(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mirror-only outage must not discard a completed native checkpoint."""

    args = _runtime_args()
    args.input_path = "prepared"
    args.output_path = "s3://task-owned/training/"
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    recipe = {
        "dataset": {"source_uri": "s3://task-owned/dataset/"},
        "train_episode_indices": [1],
    }
    (prepared / "recipe.json").write_text(json.dumps(recipe))
    output = tmp_path / "output"

    monkeypatch.setattr(
        fastwam, "_materialize_directory", lambda _source, _destination: prepared
    )
    monkeypatch.setattr(fastwam, "_read_recipe", lambda _prepared: recipe)
    monkeypatch.setattr(
        fastwam, "_materialize_dataset", lambda *_args, **_kwargs: tmp_path / "dataset"
    )
    monkeypatch.setattr(fastwam, "_assert_dataset_receipt", lambda *_args: None)
    monkeypatch.setattr(
        fastwam,
        "_fetch_runtime_models",
        lambda _args: {
            name: tmp_path for name in ("fastwam_base", "wan", "wan_diffusers", "umt5")
        },
    )
    monkeypatch.setattr(fastwam, "_restore_recoverable_checkpoint", lambda *_args: None)
    monkeypatch.setattr(
        fastwam, "_fastwam_train_command", lambda *_args, **_kwargs: ["native"]
    )

    def completed_training(
        _command: list[str], _log: Path, training: Path, _destination: str
    ) -> dict[str, object]:
        _write_resumable_checkpoint(training, step=args.train_steps)
        return {
            "final_mirror_verified": False,
            "mirror_attempts": 2,
            "mirror_successes": 1,
            "mirror_failures": [
                {"type": "RuntimeError", "message": "final mirror unavailable"}
            ],
            "mirrored_checkpoints": [],
        }

    monkeypatch.setattr(
        fastwam, "_run_training_with_checkpoint_mirror", completed_training
    )
    monkeypatch.setattr(fastwam, "_runtime_provenance", lambda: {})
    monkeypatch.setattr(fastwam, "_runtime_model_receipt", lambda *_args: {})

    fastwam._train(args, tmp_path / "work", output)

    assert (output / "checkpoint" / "model.safetensors").read_bytes() == b"policy"
    training = json.loads((output / "training.json").read_text())
    assert training["checkpoint_recovery"]["final_mirror_verified"] is False
    assert training["checkpoint_recovery"]["mirror_failures"] == [
        {"type": "RuntimeError", "message": "final mirror unavailable"}
    ]


def test_train_command_uses_native_resume_only_with_a_valid_recovery_path(
    tmp_path,
) -> None:
    recipe = {
        "dataset": {"repo_id": "operator/robot-data", "revision": "f" * 40},
        "train_episode_indices": [1, 4, 8],
    }
    models = {
        key: tmp_path / key for key in ("fastwam_base", "wan", "wan_diffusers", "umt5")
    }
    resumed = fastwam._fastwam_train_command(
        _runtime_args(),
        recipe,
        tmp_path / "data",
        tmp_path / "out",
        models,
        resume_checkpoint=tmp_path / "checkpoint" / "pretrained_model",
    )

    assert "--save_freq=20000" in resumed
    assert "--resume=true" in resumed
    assert f"--config_path={tmp_path / 'checkpoint' / 'pretrained_model'}" in resumed


def test_runtime_fetch_requires_distinct_immutable_component_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, revision: str, cache_dir: str | None) -> str:
        calls.append((repo_id, revision))
        return f"/cache/{repo_id.replace('/', '__')}/{revision}"

    class Hub:
        snapshot_download = staticmethod(fake_download)

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", Hub)
    resolved = fastwam._fetch_runtime_models(_runtime_args())

    assert set(resolved) == {"fastwam_base", "wan", "wan_diffusers", "umt5"}
    assert calls[1] == (fastwam.WAN_REPOSITORY, "b" * 40)
    assert calls[2] == (fastwam.WAN_DIFFUSERS_REPOSITORY, "c" * 40)


def test_latency_stage_loads_the_exact_checkpoint_not_a_fresh_base_model() -> None:
    source = inspect.getsource(fastwam._measure_direct_action_latency)
    assert "FastWAMPolicy.from_pretrained(" in source
    assert "checkpoint," in source


def test_recipe_refuses_missing_or_overlapping_episode_contract(tmp_path) -> None:
    (tmp_path / "recipe.json").write_text(
        json.dumps(
            {
                "schema": "npa.fastwam.recipe.v1",
                "policy": "fastwam",
                "train_episode_indices": [1],
                "heldout_episode_indices": [],
            }
        )
    )
    with pytest.raises(fastwam.FastWAMPolicyError, match="episode-disjoint"):
        fastwam._read_recipe(tmp_path)
