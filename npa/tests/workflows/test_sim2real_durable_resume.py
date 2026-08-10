"""Content-addressed controller journal and Stage 8→9 restart tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import npa.workflows.sim2real.engine as engine
from npa.workflows.sim2real.component_records import (
    _persisted_loop_component_records,
)
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


def _completed_loop_records(config: Sim2RealLoopConfig) -> list[dict[str, Any]]:
    gpu = {
        "selected_gpu_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        "selected_gpu_node": "node-1",
        "image_digests": [config.isaac_image],
    }
    return [
        {
            "name": "stage_07_actions_train",
            "tier": "WORKS",
            "evidence": "real rollout",
            "artifacts": {
                **gpu,
                "job_name": "rollout-job",
                "image": config.isaac_image,
                "prefix": "s3://bucket/run/actions/train/outer-01/",
                "applied_scenario_count": 1,
                "applied_scenario_config_digests": ["scenario-1"],
            },
        },
        {
            "name": "stage_08_vlm_eval_train",
            "tier": "WORKS",
            "evidence": "real reason jobs",
            "artifacts": {
                **gpu,
                "job_name": "reason-jobs",
                "image": config.vlm_image,
                "image_digests": [config.vlm_image],
                "signal_calibration": {"step_count": 32},
            },
        },
        {
            "name": "stage_09_training_signal",
            "tier": "WORKS",
            "evidence": "real PPO",
            "artifacts": {
                **gpu,
                "job_name": "trainer-job",
                "image": config.isaac_image,
                "checkpoint": "s3://bucket/run/model_500.pt",
                "ppo_telemetry": "s3://bucket/run/ppo-telemetry.json",
                "applied_scenario_proof": {"coverage_rate": 1.0},
            },
        },
        {
            "name": "stage_10_eval_heldout",
            "tier": "WORKS",
            "evidence": "sealed gold evaluation",
            "artifacts": {
                **gpu,
                "job_name": "eval-job",
                "image": config.isaac_image,
                "report": "s3://bucket/run/eval/gold-heldout/report.json",
                "evaluation_split": "gold_heldout",
                "checkpoint_sha256": "c" * 64,
                "applied_scenario_proof": {"exact_digest_match": True},
            },
        },
        {
            "name": "stage_11_outer_loop",
            "tier": "WORKS",
            "evidence": "durable decision",
            "artifacts": {
                "job_name": config.run_id,
                "decision": "s3://bucket/run/outer_loop/decision.json",
            },
        },
    ]


def test_finalization_adopts_content_addressed_loop_records_without_pod_files(
    tmp_path: Path,
) -> None:
    """A finalization restart must not reconstruct WORKS tiers from lost files."""

    config = Sim2RealLoopConfig(
        run_id="durable-finalize",
        output_dir=tmp_path,
        isaac_image="registry/isaac@sha256:" + "a" * 64,
        vlm_image="registry/reason@sha256:" + "b" * 64,
    )
    records = _persisted_loop_component_records(config, _completed_loop_records(config))
    assert records is not None
    assert [record.name for record in records] == [
        "stage_07_actions_train",
        "stage_08_vlm_eval_train",
        "stage_09_training_signal",
        "stage_10_eval_heldout",
        "stage_11_outer_loop",
    ]
    assert {record.tier for record in records} == {"WORKS"}
    assert not (tmp_path / "actions").exists()


def test_finalization_rejects_partial_or_unproven_durable_loop_records(
    tmp_path: Path,
) -> None:
    config = Sim2RealLoopConfig(
        run_id="durable-finalize-invalid",
        output_dir=tmp_path,
        isaac_image="registry/isaac@sha256:" + "a" * 64,
        vlm_image="registry/reason@sha256:" + "b" * 64,
    )
    records = _completed_loop_records(config)
    with pytest.raises(Sim2RealLoopError, match="incomplete"):
        _persisted_loop_component_records(config, records[:-1])
    records[0]["artifacts"]["image_digests"] = []
    with pytest.raises(Sim2RealLoopError, match="GPU/image proof"):
        _persisted_loop_component_records(config, records)


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


def test_mid_stage8_restart_reuses_component_uri_and_job_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-entering an incomplete Stage 8 addresses the already-created Job/output."""

    config = _config(tmp_path)
    rollout = tmp_path / "rollout-0000"
    rollout.mkdir(parents=True)
    manifest = {
        "schema": "npa.sim2real.action_rollout.v1",
        "rollout_id": "rollout-0000",
        "task_description": "place cube",
        "actions": [{"step": 0, "action": [0.0, 0.0, 0.0]}],
        "camera_observations": ["camera-000.ppm"],
    }
    manifest_path = rollout / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (rollout / "camera-000.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")

    uploaded_attempts: list[str] = []
    output_uris: list[str] = []
    job_names: list[str] = []

    def fake_upload(
        _config: Sim2RealLoopConfig,
        _directory: Path,
        *,
        component: str,
        attempt_id: str,
        name: str,
    ) -> str:
        uploaded_attempts.append(attempt_id)
        return f"s3://durable-bucket/input/{component}/{attempt_id}/{name}/"

    def fake_run(
        _image: str,
        *,
        component: str,
        env: dict[str, str],
        output_json: Path,
        output_uri: str,
        config: Sim2RealLoopConfig,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        output_uris.append(output_uri)
        job_names.append(
            engine._k8s_job_name(config.run_id, component, identity=output_uri)
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.vlm_eval.v1",
                    "rollout_id": "rollout-0000",
                    "success": False,
                    "score": 0.4,
                    "summary": "real component result",
                    "model": env["NPA_SIM2REAL_VLM_MODEL"],
                    "per_step": [
                        {
                            "step": 0,
                            "critique_text": "placement is incomplete",
                            "error_tags": ["placement"],
                            "action": [0.0, 0.0, 0.0],
                            "camera_observation": "camera-000.ppm",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {
            "mode": "kubernetes_job",
            "component": component,
            "job_name": job_names[-1],
            "output_uri": output_uri,
        }

    monkeypatch.setattr(engine, "_upload_component_directory", fake_upload)
    monkeypatch.setattr(engine, "_run_image_component", fake_run)

    for replacement in ("driver-one", "driver-two"):
        result, invocation = engine._evaluate_reason_rollout_k8s(
            rollout,
            manifest=manifest,
            manifest_path=manifest_path,
            rollout_id="rollout-0000",
            config=config,
            model="reason-model",
            image="registry/reason@sha256:" + "a" * 64,
            component="vlm_eval_reason2",
            output_dir=tmp_path / replacement,
        )
        assert result["score"] == 0.4
        assert invocation["mode"] == "kubernetes_job"

    assert uploaded_attempts[0] == uploaded_attempts[1]
    assert output_uris[0] == output_uris[1]
    assert job_names[0] == job_names[1]
    assert len(job_names[0]) <= 63


def test_validation_and_gold_commands_receive_distinct_durable_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []

    def fake_command(
        _command: str,
        *,
        cwd: Path,
        env: dict[str, str],
        component: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        seen.append(
            (
                env["NPA_SIM2REAL_EVALUATION_SPLIT"],
                env["NPA_SIM2REAL_HELDOUT_ENVS_URI"],
            )
        )
        output = Path(env["NPA_SIM2REAL_OUTPUT_JSON"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.heldout_eval.v1",
                    "policy_checkpoint": "s3://bucket/checkpoints/model.pt",
                    "per_env": [
                        {
                            "env_id": "scenario-0",
                            "score": 0.0,
                            "success": False,
                            "details": {},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return {"mode": "command", "component": component, "returncode": 0}

    monkeypatch.setattr(engine, "_run_component_command", fake_command)
    config = Sim2RealLoopConfig(
        run_id="durable-split-run",
        output_dir=tmp_path,
        validation_envs_uri="s3://bucket/run/envs/validation/",
        gold_heldout_envs_uri="s3://bucket/run/envs/gold-heldout/",
        validation_env_count=1,
        heldout_env_count=1,
        byo_eval_command="durable-eval",
    )
    evidence = {"selected_checkpoint_uri": "s3://bucket/checkpoints/model.pt"}
    validation = engine.run_heldout_eval(
        config,
        local_dir=tmp_path,
        inner_evidence=evidence,
        outer_iteration=1,
        evaluation_split="validation",
    )
    gold = engine.run_heldout_eval(
        config,
        local_dir=tmp_path,
        inner_evidence=evidence,
        outer_iteration=1,
        evaluation_split="gold_heldout",
    )

    assert seen == [
        ("validation", "s3://bucket/run/envs/validation/envs.jsonl"),
        ("gold_heldout", "s3://bucket/run/envs/gold-heldout/envs.jsonl"),
    ]
    assert validation["scenario_records_uri"] != gold["scenario_records_uri"]
    assert validation["gold_heldout_untouched"] is True
    assert gold["gold_heldout_untouched"] is False
