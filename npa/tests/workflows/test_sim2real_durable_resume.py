"""Content-addressed controller journal and Stage 8→9 restart tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import npa.workflows.sim2real.engine as engine
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.resume_state import ControllerIdentity, DurableStateStore


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local_file: str, destination: str) -> str:
        self.objects[destination] = Path(local_file).read_bytes()
        return destination

    def download_file(self, source: str, destination: str) -> str:
        if source not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        Path(destination).write_bytes(self.objects[source])
        return destination


def _config(tmp_path: Path) -> Sim2RealLoopConfig:
    return Sim2RealLoopConfig(
        run_id="durable-run",
        output_dir=tmp_path,
        upload_artifacts=True,
        s3_bucket="durable-bucket",
        s3_prefix="sim2real-b",
        s3_endpoint="https://storage.example.invalid",
        inner_iterations=1,
        rollout_count=1,
    )


def _identity() -> ControllerIdentity:
    return ControllerIdentity(
        run_id="durable-run",
        source_sha="1" * 40,
        runtime_image="registry/controller@sha256:" + "a" * 64,
        spec_digest="b" * 64,
    )


def test_journal_hydrates_latest_only_after_immutable_checkpoint(
    tmp_path: Path,
) -> None:
    storage = MemoryStorage()
    first = DurableStateStore(
        _config(tmp_path / "first"),
        tmp_path / "first",
        client=storage,
        identity=_identity(),
    )
    payload = {
        "run_id": "durable-run",
        "status": "running",
        "components": [{"name": "stage_08", "tier": "WORKS"}],
        "stage_records": [{"stage": 8, "sha256": "evidence"}],
    }
    checkpoint_uri = first.persist_workflow_checkpoint(payload)
    latest_uri = f"{first.root}/latest.json"
    assert checkpoint_uri in storage.objects
    assert latest_uri in storage.objects
    pointer = json.loads(storage.objects[latest_uri])
    assert pointer["checkpoint_uri"] == checkpoint_uri

    restarted_dir = tmp_path / "restarted"
    restarted = DurableStateStore(
        _config(restarted_dir),
        restarted_dir,
        client=storage,
        identity=_identity(),
    )
    hydrated = restarted.hydrate_workflow_state()
    assert hydrated is not None
    assert hydrated["components"] == payload["components"]
    assert hydrated["local_artifact_dir"] == str(restarted_dir)


def test_journal_rejects_tampering_and_exact_sha_mismatch(tmp_path: Path) -> None:
    storage = MemoryStorage()
    store = DurableStateStore(
        _config(tmp_path), tmp_path, client=storage, identity=_identity()
    )
    store.commit_unit("stage-08", {"input": 1}, {"result": "real"})
    uri = next(key for key in storage.objects if "/units/stage-08/" in key)
    envelope = json.loads(storage.objects[uri])
    envelope["payload"]["result"] = "fabricated"
    storage.objects[uri] = json.dumps(envelope).encode()
    with pytest.raises(Sim2RealLoopError, match="payload digest"):
        store.load_unit("stage-08", {"input": 1})

    different_identity = ControllerIdentity(
        run_id=_identity().run_id,
        source_sha="2" * 40,
        runtime_image=_identity().runtime_image,
        spec_digest=_identity().spec_digest,
    )
    mismatched = DurableStateStore(
        _config(tmp_path / "other"),
        tmp_path / "other",
        client=storage,
        identity=different_identity,
    )
    with pytest.raises(Sim2RealLoopError, match="identity"):
        mismatched.load_unit("stage-08", {"input": 1})


def test_driver_restart_reuses_stage8_and_continues_at_stage9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash after the Stage 8 commit; a fresh local driver must not rerun VLM."""

    storage = MemoryStorage()
    from npa.clients import storage as storage_module

    monkeypatch.setattr(
        storage_module.StorageClient,
        "from_environment",
        classmethod(lambda cls, **_kwargs: storage),
    )
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", _identity().source_sha)
    monkeypatch.setenv("NPA_SIM2REAL_RUNTIME_IMAGE", _identity().runtime_image)
    monkeypatch.setenv("NPA_SIM2REAL_CONTROLLER_SPEC_DIGEST", _identity().spec_digest)

    import npa.workflows.sim2real_stages as stages

    rollout_calls: list[str] = []
    evaluation_calls: list[str] = []
    signal_calls: list[str] = []
    monkeypatch.setattr(
        stages,
        "run_policy_rollouts",
        lambda *_args, **_kwargs: (
            rollout_calls.append("rollout")
            or [{"rollout_id": "rollout-0000", "frames_dir": str(tmp_path)}]
        ),
    )
    monkeypatch.setattr(
        engine,
        "evaluate_rollout_with_vlm",
        lambda rollout, **_kwargs: (
            evaluation_calls.append(rollout["rollout_id"])
            or {"rollout_id": rollout["rollout_id"], "score": 0.5}
        ),
    )

    def signal(evaluation: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        signal_calls.append(evaluation["rollout_id"])
        if len(signal_calls) == 1:
            raise RuntimeError("deliberate driver loss between Stage 8 and Stage 9")
        return {
            "schema": "npa.sim2real.rl_signal.v1",
            "rollout_id": evaluation["rollout_id"],
            "mean_reward": 0.5,
            "score": 0.5,
            "advantages": [0.2],
            "per_step": [{"reward": 0.4}, {"reward": 0.6}],
            "calibration": {},
        }

    monkeypatch.setattr(engine, "_convert_eval_to_signal", signal)

    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    def update(*_args: Any, control: bool = False, **_kwargs: Any):
        return VlmSignalUpdateResult.from_dict(
            {
                "schema": "npa.lerobot.vlm_signal_adapter.v1",
                "status": "success",
                "reward_head_before": 0.0,
                "reward_head_after": 0.0 if control else 0.1,
                "policy_output_before": [0.0],
                "policy_output_after": [0.0 if control else 0.2],
                "policy_delta_l2": 0.0 if control else 0.3,
                "checkpoint_path": "",
            }
        )

    monkeypatch.setattr(
        engine, "_signal_training_imports", lambda: (lambda batch: batch, update)
    )

    first_dir = tmp_path / "driver-one"
    with pytest.raises(RuntimeError, match="deliberate driver loss"):
        engine.run_inner_loop(
            _config(first_dir),
            local_dir=first_dir,
            initial_quality=0.4,
        )
    assert rollout_calls == ["rollout"]
    assert evaluation_calls == ["rollout-0000"]

    second_dir = tmp_path / "driver-two"
    evidence = engine.run_inner_loop(
        _config(second_dir),
        local_dir=second_dir,
        initial_quality=0.4,
    )
    assert rollout_calls == ["rollout"]
    assert evaluation_calls == ["rollout-0000"]
    assert signal_calls == ["rollout-0000", "rollout-0000"]
    assert evidence["iterations"][0]["sample_vlm_eval"]["score"] == 0.5
    assert any("stage09-signal" in key for key in storage.objects)
