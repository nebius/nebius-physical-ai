from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import npa.workflows.sim2real_loop as loop_module
from npa.sdk.workbench import cosmos2, cosmos3, sim2real
from npa.workbench.lerobot.policy_container import (
    parse_vlm_signal_batch,
    run_vlm_signal_training_step,
)
from npa.workflows.sim2real_loop import (
    SCHEMA_HELDOUT_REPORT,
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
    Sim2RealLoopConfig,
    artifact_uris,
    convert_vlm_eval_to_rl_signal,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_full_loop,
    run_heldout_eval,
)


ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = ROOT / "npa" / "workflows" / "workbench" / "sim2real" / "runbook.yaml"
COSMOS2_TRANSFER = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "cosmos2-transfer.yaml"
)
COSMOS3_REASON = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "cosmos3-reason.yaml"
)


def _component_command(tmp_path: Path) -> str:
    script = tmp_path / "component_contract.py"
    script.write_text(
        """
import json
import os
from pathlib import Path

component = os.environ["NPA_SIM2REAL_COMPONENT"]
out = Path(os.environ["NPA_SIM2REAL_OUTPUT_JSON"])
out.parent.mkdir(parents=True, exist_ok=True)
marker = Path(os.environ.get("NPA_SIM2REAL_COMPONENT_MARKER", out.parent / "component-marker.log"))

if component == "vlm_eval":
    rollout_dir = Path(os.environ["NPA_SIM2REAL_ROLLOUT_DIR"])
    manifest = json.loads((rollout_dir / "manifest.json").read_text())
    per_step = []
    for item in manifest["actions"]:
        step = int(item["step"])
        frame = rollout_dir / f"camera-{step:03d}.ppm"
        payload = frame.read_bytes()
        signal = sum(payload[-12:]) % 17
        tag = "minor_alignment"
        per_step.append({
            "step": step,
            "critique_text": f"Frame {frame.name} has content signal {signal}; adjust {tag}.",
            "error_tags": [tag],
            "action": item["action"],
            "camera_observation": frame.name,
        })
    score = 0.62 + ((sum(Path(os.environ["NPA_SIM2REAL_ROLLOUT_MANIFEST"]).read_bytes()) % 20) / 100.0)
    result = {
        "schema": "npa.sim2real.vlm_eval.v1",
        "rollout_id": manifest["rollout_id"],
        "success": score >= float(os.environ["NPA_SIM2REAL_THRESHOLD"]),
        "score": round(score, 6),
        "per_step": per_step,
        "summary": "component-derived frame judgment",
        "model": os.environ.get("NPA_SIM2REAL_VLM_MODEL", "test-vlm"),
    }
elif component == "heldout_eval":
    count = int(os.environ["NPA_SIM2REAL_HELDOUT_ENV_COUNT"])
    threshold = float(os.environ["NPA_SIM2REAL_THRESHOLD"])
    per_env = []
    for index in range(count):
        score = 0.56 + (index % 5) * 0.05
        per_env.append({
            "env_id": f"heldout-{index:04d}",
            "score": round(score, 6),
            "success": score >= threshold,
            "details": {"source": "component-contract", "index_mod": index % 5},
        })
    result = {"schema": "npa.sim2real.heldout_eval.v1", "per_env": per_env}
else:
    raise SystemExit(f"unknown component {component}")

out.write_text(json.dumps(result) + "\\n")
with marker.open("a", encoding="utf-8") as handle:
    handle.write(component + "\\n")
print(json.dumps({"component": component, "output": str(out)}))
""",
        encoding="utf-8",
    )
    return f"{sys.executable} {script}"



def test_vlm_eval_signal_converter_and_trainer_update_close_loop(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "component-marker.log"
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-unit",
        output_dir=tmp_path,
        threshold=0.75,
        rollout_count=1,
        steps_per_rollout=3,
        byo_vlm_command=f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {command}",
    )
    rollout = generate_action_rollouts(
        tmp_path / "actions",
        count=1,
        steps_per_rollout=3,
        seed=7,
        quality=0.4,
    )[0]

    evaluation = evaluate_rollout_with_vlm(
        rollout, output_dir=tmp_path / "vlm_eval", config=config
    )
    signal = convert_vlm_eval_to_rl_signal(evaluation)
    parsed = parse_vlm_signal_batch(signal)
    update = run_vlm_signal_training_step(parsed, output_dir=tmp_path / "update")
    control = run_vlm_signal_training_step(
        parsed, output_dir=tmp_path / "control", control=True
    )

    assert evaluation["schema"] == SCHEMA_VLM_EVAL
    assert evaluation["component_invocation"]["mode"] == "command"
    assert "vlm_eval" in marker.read_text(encoding="utf-8")
    assert signal["schema"] == SCHEMA_RL_SIGNAL
    assert signal["per_step"][0]["target"]["nl_correction"]
    assert update.policy_delta_l2 > control.policy_delta_l2
    assert Path(update.checkpoint_path).exists()


def test_full_loop_writes_stage_artifacts_and_candidate(tmp_path: Path) -> None:
    marker = tmp_path / "component-marker.log"
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-full-unit",
        output_dir=tmp_path,
        trigger_dataset_uri="s3://bucket/sim2real-triggers/lerobot-pusht/",
        threshold=0.45,
        inner_iterations=2,
        outer_iterations=1,
        rollout_count=2,
        steps_per_rollout=3,
        heldout_env_count=4,
        byo_vlm_command=f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {command}",
        byo_eval_command=f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {command}",
    )

    report = run_full_loop(config)
    decision = report["outer_loop"]["latest_decision"]
    reward_trend = report["inner_loop"]["reward_trend"]

    assert report["schema"] == "npa.sim2real.e2e_report.v1"
    assert reward_trend[-1] >= reward_trend[0]
    assert report["s3_artifacts"] == {}
    assert (
        report["byo_seams"]["trigger_dataset_uri"]
        == "s3://bucket/sim2real-triggers/lerobot-pusht/"
    )
    assert decision["decision"] == "promote_checkpoint"
    trigger = json.loads((tmp_path / "stage_01_trigger" / "trigger.json").read_text())
    retrigger = json.loads(
        (tmp_path / "stage_13_retrigger" / "retrigger.json").read_text()
    )
    assert (
        trigger["trigger_dataset_uri"] == "s3://bucket/sim2real-triggers/lerobot-pusht/"
    )
    assert trigger["start_condition"] == "dataset_landed_in_trigger_path"
    assert retrigger["target_stage"] == 1
    assert (
        retrigger["trigger_dataset_uri"]
        == "s3://bucket/sim2real-triggers/lerobot-pusht/"
    )
    assert (tmp_path / "vlm_eval" / "train").exists()
    assert (tmp_path / "training_signal" / "train").exists()
    assert (tmp_path / "inner_loop" / "outer-01" / "evidence.json").exists()
    assert (
        json.loads((tmp_path / "eval" / "heldout" / "report.json").read_text())[
            "success_rate"
        ]
        >= 0.45
    )
    assert (tmp_path / "checkpoints" / "candidate" / "candidate.json").exists()
    assert (tmp_path / "reports" / "sim2real-report.json").exists()
    marker_text = marker.read_text(encoding="utf-8")
    assert marker_text.count("vlm_eval") == 4
    assert "heldout_eval" in marker_text
    raw_envs = json.loads((tmp_path / "envs" / "raw" / "manifest.json").read_text())
    train_envs = json.loads((tmp_path / "envs" / "train" / "manifest.json").read_text())
    heldout_envs = json.loads((tmp_path / "envs" / "heldout" / "manifest.json").read_text())
    assert len(raw_envs["envs"]) == 6
    assert len(train_envs["envs"]) == 2
    assert len(heldout_envs["envs"]) == 4


def test_threshold_failure_loops_back_to_inner_loop(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-loopback-unit",
        output_dir=tmp_path,
        threshold=0.98,
        inner_iterations=1,
        outer_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
        byo_vlm_command=command,
        byo_eval_command=command,
    )

    report = run_full_loop(config)
    decision = report["outer_loop"]["latest_decision"]
    loopback = json.loads((tmp_path / "outer_loop" / "loopback.json").read_text())

    assert decision["decision"] == "loop_back_to_inner_loop"
    assert loopback["to_stage"] == 7


def test_empty_s3_prefix_writes_under_run_id() -> None:
    config = Sim2RealLoopConfig(
        run_id="pusht-demo",
        s3_bucket="bucket",
        s3_prefix="",
        trigger_dataset_uri="s3://bucket/sim2real-triggers/pusht-demo/lerobot-pusht/",
    )

    assert artifact_uris(config)["root"] == "s3://bucket/pusht-demo/"


class _FakeComponentStorage:
    def __init__(self, downloads: dict[str, dict]) -> None:
        self.downloads = downloads
        self.uploaded_directories: list[tuple[str, str]] = []
        self.uploaded_files: list[tuple[str, str]] = []

    def upload_directory(self, local_dir: str, bucket_uri: str, *, remote_prefix: str = "") -> str:
        self.uploaded_directories.append((local_dir, bucket_uri))
        return bucket_uri

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploaded_files.append((local_file, bucket_uri))
        return bucket_uri

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        payload = self.downloads.get(bucket_uri)
        if payload is None and "/vlm-eval/" in bucket_uri:
            payload = self.downloads.get("vlm_eval")
        if payload is None and "/heldout-eval/" in bucket_uri:
            payload = self.downloads.get("heldout_eval")
        if payload is None:
            raise KeyError(bucket_uri)
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return str(target)


def _patch_component_storage(monkeypatch, storage: _FakeComponentStorage) -> None:
    monkeypatch.setattr(
        loop_module.StorageClient,
        "from_environment",
        classmethod(lambda cls, endpoint_url="": storage),
    )


def _patch_kubectl(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        if "apply" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "job.batch/sibling created\n", "")
        if "wait" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "job.batch/sibling condition met\n", "")
        if "get" in cmd and "pods" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "sibling-pod"},
                                "spec": {
                                    "nodeName": "sm120-node",
                                    "containers": [
                                        {
                                            "resources": {
                                                "requests": {"nvidia.com/gpu": 1},
                                                "limits": {"nvidia.com/gpu": 1},
                                            }
                                        }
                                    ],
                                },
                                "status": {
                                    "phase": "Succeeded",
                                    "containerStatuses": [
                                        {
                                            "name": "component",
                                            "ready": False,
                                            "restartCount": 0,
                                            "state": {"terminated": {"exitCode": 0}},
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ),
                "",
            )
        if "logs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, '{"component":"ok"}\n', "")
        if "delete" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "job.batch deleted\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(loop_module.subprocess, "run", fake_run)
    return calls


def test_image_vlm_eval_launches_sibling_job_and_parses_output(monkeypatch, tmp_path: Path) -> None:
    output_payload = {
        "schema": SCHEMA_VLM_EVAL,
        "rollout_id": "rollout-0000",
        "success": False,
        "score": 0.512345,
        "per_step": [
            {
                "step": 0,
                "critique_text": "Parsed component output from the sibling stage.",
                "error_tags": ["collision", "minor_alignment"],
                "action": [0.1, 0.0, -0.1],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "sibling output",
        "model": "job-vlm",
    }
    storage = _FakeComponentStorage({"vlm_eval": output_payload})
    _patch_component_storage(monkeypatch, storage)
    calls = _patch_kubectl(monkeypatch)
    config = Sim2RealLoopConfig(
        run_id="sibling-vlm-unit",
        s3_bucket="bucket",
        s3_prefix="neutral-prefix",
        s3_endpoint="https://storage.example",
        threshold=0.75,
        k8s_namespace="default",
        source_repo="https://example.invalid/repo.git",
        source_ref="dev-branch",
    )
    rollout = generate_action_rollouts(
        tmp_path / "actions",
        count=1,
        steps_per_rollout=1,
        seed=7,
        quality=0.4,
    )[0]

    evaluation = evaluate_rollout_with_vlm(
        rollout,
        output_dir=tmp_path / "vlm_eval",
        config=config,
    )

    apply_call = next(call for call in calls if "apply" in call["cmd"])
    manifest = json.loads(apply_call["input"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]

    assert evaluation["score"] == 0.512345
    assert evaluation["component_invocation"]["mode"] == "kubernetes_job"
    assert evaluation["component_invocation"]["pod"]["node_name"] == "sm120-node"
    assert convert_vlm_eval_to_rl_signal(evaluation)["score"] == 0.512345
    assert storage.uploaded_directories
    assert manifest["spec"]["template"]["spec"]["serviceAccountName"] == "agent-sa"
    assert {"name": "agent-sa"} in manifest["spec"]["template"]["spec"]["imagePullSecrets"]
    assert {"secretRef": {"name": "hf-ngc-tokens", "optional": True}} in container["envFrom"]
    assert {"secretRef": {"name": "npa-storage-credentials", "optional": True}} in container["envFrom"]
    assert container["resources"]["requests"]["nvidia.com/gpu"] == 1
    assert (
        manifest["spec"]["template"]["spec"]["nodeSelector"]["nvidia.com/gpu.compute.major"]
        == "12"
    )
    assert "component-vlm-eval" in container["args"][0]


def test_heldout_eval_launches_sibling_job_and_parses_per_env_output(
    monkeypatch, tmp_path: Path
) -> None:
    output_payload = {
        "schema": SCHEMA_HELDOUT_REPORT,
        "per_env": [
            {"env_id": "env-a", "score": 0.81, "success": True, "details": {}},
            {"env_id": "env-b", "score": 0.52, "success": False, "details": {}},
        ],
    }
    storage = _FakeComponentStorage({"heldout_eval": output_payload})
    _patch_component_storage(monkeypatch, storage)
    calls = _patch_kubectl(monkeypatch)
    config = Sim2RealLoopConfig(
        run_id="sibling-heldout-unit",
        s3_bucket="bucket",
        s3_prefix="neutral-prefix",
        s3_endpoint="https://storage.example",
        heldout_envs_uri="s3://bucket/neutral-prefix/run/envs/heldout/",
        heldout_eval_limit=2,
        threshold=0.75,
        k8s_namespace="default",
        source_ref="dev-branch",
    )
    inner_evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "reward_trend": [0.1, 0.2],
    }

    report = run_heldout_eval(
        config,
        local_dir=tmp_path,
        inner_evidence=inner_evidence,
        outer_iteration=1,
    )
    apply_call = next(call for call in calls if "apply" in call["cmd"])
    manifest = json.loads(apply_call["input"])
    args = manifest["spec"]["template"]["spec"]["containers"][0]["args"][0]

    assert report["success_rate"] == 0.5
    assert report["per_env"][0]["env_id"] == "env-a"
    assert report["component_invocation"]["mode"] == "kubernetes_job"
    assert report["component_invocation"]["gpu_request"]["count"] == 1
    assert storage.uploaded_files
    assert "component-heldout-eval" in args
    assert "--limit" in args


def test_component_vlm_payload_uses_cosmos_reason_model_and_frames(
    monkeypatch, tmp_path: Path
) -> None:
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    frame = rollout_dir / "camera-000.ppm"
    frame.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
    manifest = {
        "schema": "npa.sim2real.action_rollout.v1",
        "rollout_id": "rollout-0000",
        "task_description": "Move the object to the target.",
        "actions": [{"step": 0, "action": [0.1, 0.0, -0.1]}],
        "camera_observations": [frame.name],
    }
    (rollout_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_cosmos_reason(**kwargs):
        captured.update(kwargs)
        return {
            "schema": SCHEMA_VLM_EVAL,
            "rollout_id": kwargs["rollout_id"],
            "success": False,
            "score": 0.41,
            "per_step": [
                {
                    "step": 0,
                    "critique_text": "The object remains offset from the target after contact.",
                    "error_tags": ["missed_target"],
                    "action": [0.1, 0.0, -0.1],
                    "camera_observation": frame.name,
                }
            ],
            "summary": "real model judgment",
        }

    monkeypatch.setattr(loop_module, "_run_cosmos_reason_vlm", fake_cosmos_reason)

    payload = loop_module._component_vlm_payload(
        manifest,
        rollout_root=rollout_dir,
        model="npa-cosmos3-reason",
        threshold=0.75,
        rollout_id="rollout-0000",
    )

    assert captured["model_id"] == "nvidia/Cosmos-Reason1-7B"
    assert captured["image_paths"] == [frame]
    assert captured["task_description"] == "Move the object to the target."
    assert payload["component_source"] == "cosmos_reason_vlm"
    assert payload["model"] == "nvidia/Cosmos-Reason1-7B"
    assert payload["frame_count"] == 1
    assert "synthetic_signature" not in payload


def test_component_heldout_payload_uses_genesis_rollout_backend(monkeypatch) -> None:
    envs = [
        {"env_id": "heldout-0000", "target_pose": [0.0, 0.1, 0.0]},
        {"env_id": "heldout-0001", "target_pose": [0.1, 0.0, 0.0]},
    ]
    captured = {}

    def fake_genesis_rollouts(env_payload, *, inner_evidence, threshold, scene=None):
        captured["envs"] = env_payload
        captured["scene"] = scene
        captured["inner_evidence"] = inner_evidence
        captured["threshold"] = threshold
        return [
            {
                "env_id": "heldout-0000",
                "score": 0.82,
                "success": True,
                "details": {"backend": "genesis"},
            },
            {
                "env_id": "heldout-0001",
                "score": 0.44,
                "success": False,
                "details": {"backend": "genesis"},
            },
        ]

    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_genesis_rollouts)
    inner_evidence = {"reward_trend": [0.2, 0.6], "policy_delta_l2": 0.12}

    payload = loop_module._component_heldout_payload(
        envs,
        inner_evidence=inner_evidence,
        threshold=0.75,
        sim_backend="genesis",
    )

    assert captured["envs"] == envs
    assert captured["inner_evidence"] == inner_evidence
    assert captured["threshold"] == 0.75
    assert payload["component_source"] == "genesis_rollout"
    assert payload["rollout_backend"] == "npa.genesis.env_pick_place.FrankaPickPlaceEnv"
    assert payload["policy_source"] == "inner_evidence_adapter"
    assert "synthetic_signature" not in payload


class _FakeMeshClient:
    """Fake StorageClient that writes JSON/mesh bytes for download_path."""

    def __init__(self, *, spec_doc: dict | None = None, mesh: bytes = b"MESH") -> None:
        self.spec_doc = spec_doc
        self.mesh = mesh
        self.downloads: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, str]] = []

    def download_path(self, uri: str, local_path: str) -> str:
        self.downloads.append((uri, local_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if uri.endswith(".json"):
            Path(local_path).write_text(json.dumps(self.spec_doc or {}))
        else:
            Path(local_path).write_bytes(self.mesh)
        return local_path

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploads.append((local_file, bucket_uri))
        return bucket_uri


def test_resolve_heldout_scene_byo_mesh_records_provenance(tmp_path: Path) -> None:
    from npa.genesis import scene_assets as sa

    client = _FakeMeshClient(mesh=b"OBJ-BYTES")
    scene = loop_module._resolve_heldout_scene(
        scene_spec_uri="",
        assets_uri="s3://bucket/run/object.obj",
        byo_mesh_uri="",
        dest_dir=tmp_path,
        client=client,
    )
    assert scene is not None
    obj = scene.manipuland()
    assert obj.asset_source == sa.ASSET_SOURCE_BYO_MESH
    assert obj.sha256 == sa.sha256_file(obj.local_path)
    assert scene.provenance_block()["asset_fallback_used"] is False


def test_resolve_heldout_scene_none_without_uris(tmp_path: Path) -> None:
    scene = loop_module._resolve_heldout_scene(
        scene_spec_uri="",
        assets_uri="",
        byo_mesh_uri="",
        dest_dir=tmp_path,
        client=_FakeMeshClient(),
    )
    assert scene is None


def test_resolve_heldout_scene_from_scene_spec_json(tmp_path: Path) -> None:
    doc = {
        "objects": [
            {
                "name": "widget",
                "asset_source": "byo_mesh",
                "uri": "s3://bucket/run/widget.glb",
            }
        ],
        "goal_pos": [0.5, 0.3, 0.04],
    }
    client = _FakeMeshClient(spec_doc=doc)
    scene = loop_module._resolve_heldout_scene(
        scene_spec_uri="s3://bucket/run/scene-spec.json",
        assets_uri="",
        byo_mesh_uri="",
        dest_dir=tmp_path,
        client=client,
    )
    assert scene is not None
    assert scene.manipuland().uri.endswith("widget.glb")
    assert scene.manipuland().sha256


def test_component_heldout_payload_with_scene_attaches_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.genesis import scene_assets as sa

    client = _FakeMeshClient(mesh=b"OBJ-BYTES")
    scene = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/object.obj")
    sa.resolve_scene_assets(scene, dest_dir=tmp_path, client=client)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None):
        # Simulate the env building the mesh and marking it loaded.
        if scene is not None:
            for obj in scene.objects:
                obj.loaded = True
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_rollouts)
    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
        scene=scene,
        sim_backend="genesis",
    )
    assert payload["asset_fallback_used"] is False
    prov = payload["asset_provenance"]
    assert prov["objects"][0]["asset_source"] == "byo_mesh"
    assert prov["objects"][0]["loaded"] is True
    assert prov["objects"][0]["sha256"] == scene.manipuland().sha256


def test_component_heldout_payload_raises_when_mesh_not_loaded(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.genesis import scene_assets as sa

    client = _FakeMeshClient(mesh=b"OBJ-BYTES")
    scene = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/object.obj")
    sa.resolve_scene_assets(scene, dest_dir=tmp_path, client=client)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None):
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_rollouts)
    with pytest.raises(loop_module.Sim2RealLoopError):
        loop_module._component_heldout_payload(
            [{"env_id": "heldout-0000", "seed": 1}],
            inner_evidence={},
            threshold=0.75,
            scene=scene,
            sim_backend="genesis",
        )


def test_normalize_heldout_report_propagates_provenance() -> None:
    payload = {
        "per_env": [{"env_id": "h-0", "score": 0.9, "success": True}],
        "asset_provenance": {
            "schema": "npa.sim2real.asset_provenance.v1",
            "asset_fallback_used": False,
            "objects": [{"name": "widget", "asset_source": "byo_mesh", "loaded": True}],
        },
        "asset_fallback_used": False,
    }
    config = Sim2RealLoopConfig(run_id="r", threshold=0.5)
    report = loop_module._normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=1,
        inner_evidence_uri="inner.json",
        invocation={"component": "heldout_eval"},
    )
    assert report["asset_fallback_used"] is False
    assert report["asset_provenance"]["objects"][0]["asset_source"] == "byo_mesh"


def test_run_heldout_eval_component_from_s3_writes_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    client = _FakeMeshClient(mesh=b"OBJ-BYTES")
    monkeypatch.setattr(
        loop_module.StorageClient, "from_environment", staticmethod(lambda **kw: client)
    )

    # Seed the env records the component downloads.
    def download_path(uri, local_path):
        client.downloads.append((uri, local_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if uri.endswith("heldout/"):
            Path(local_path).mkdir(parents=True, exist_ok=True)
            (Path(local_path) / "envs.jsonl").write_text(
                json.dumps({"env_id": "heldout-0000", "seed": 7}) + "\n"
            )
        elif uri.endswith(".json"):
            Path(local_path).write_text(json.dumps({"reward_trend": [0.2, 0.6]}))
        else:
            Path(local_path).write_bytes(b"OBJ-BYTES")
        return local_path

    monkeypatch.setattr(client, "download_path", download_path)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None):
        if scene is not None:
            for obj in scene.objects:
                obj.loaded = True
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_rollouts)

    payload = loop_module.run_heldout_eval_component_from_s3(
        heldout_envs_uri="s3://bucket/run/heldout/",
        inner_evidence_uri="s3://bucket/run/inner-evidence.json",
        output_uri="s3://bucket/run/output/report.json",
        threshold=0.75,
        assets_uri="s3://bucket/run/object.obj",
        sim_backend="genesis",
    )
    assert payload["asset_fallback_used"] is False
    assert payload["asset_provenance"]["objects"][0]["asset_source"] == "byo_mesh"
    # A consumed scene spec is uploaded alongside the report.
    uploaded = [u[1] for u in client.uploads]
    assert any(u.endswith("consumed-scene-spec.json") for u in uploaded)


def test_sdk_exposes_sim2real_run(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    report = sim2real.run(
        run_id="sim2real-sdk-unit",
        output_dir=tmp_path,
        trigger_dataset_uri="s3://bucket/triggers/pusht/",
        threshold=0.45,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
        byo_vlm_command=command,
        byo_eval_command=command,
    )

    assert report["run_id"] == "sim2real-sdk-unit"
    assert "vlm_image" in report["byo_seams"]
    assert report["byo_seams"]["trigger_dataset_uri"] == "s3://bucket/triggers/pusht/"


def test_raw_runbook_invokes_full_loop_and_exposes_byo_envs() -> None:
    docs = [
        doc
        for doc in yaml.safe_load_all(RUNBOOK.read_text(encoding="utf-8"))
        if doc is not None
    ]

    assert len(docs) == 1
    task = docs[0]
    assert task["name"] == "sim2real-full-loop"

    # SkyPilot 0.12.2 does not interpolate ${VAR} inside `envs` or `image_id`.
    # The raw runbook must therefore carry materialized literals and expand env
    # vars only at container runtime in the `run` block.
    env_values = "\n".join(str(value) for value in task["envs"].values())
    assert "${" not in env_values
    assert "${" not in str(task["resources"]["image_id"])
    assert task["resources"]["image_id"].startswith("docker:example.invalid/")

    # The BYO seam env names are still declared and consumed by the run block.
    for env_name in (
        "NPA_SIM2REAL_TRIGGER_DATASET_URI",
        "NPA_SIM2REAL_TRIGGER_DATASET_ID",
        "VLM_IMAGE",
        "TRAINER_IMAGE",
        "EVAL_IMAGE",
    ):
        assert env_name in task["envs"]
        assert env_name in task["run"]

    assert "npa.workflows.sim2real_loop full-loop" in task["run"]
    assert "--trigger-dataset-uri" in task["run"]
    assert "--byo-signal-converter" in task["run"]
    assert "--k8s-service-account" in task["run"]
    assert "--heldout-eval-limit" in task["run"]
    assert "nebius.cloud" not in RUNBOOK.read_text(encoding="utf-8")


def test_sim_backend_defaults_to_isaac_and_validates() -> None:
    config = Sim2RealLoopConfig(run_id="backend-default")
    assert config.sim_backend == "isaac"
    assert config.heldout_backend_image() == config.isaac_image
    config.validate()

    genesis_config = Sim2RealLoopConfig(run_id="backend-genesis", sim_backend="genesis")
    assert genesis_config.heldout_backend_image() == genesis_config.eval_image

    with pytest.raises(loop_module.Sim2RealLoopError):
        Sim2RealLoopConfig(run_id="backend-bad", sim_backend="mujoco").validate()


def test_build_config_from_env_reads_sim_backend(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SIM_BACKEND", "GENESIS")
    config = loop_module.build_config_from_env(run_id="env-backend")
    assert config.sim_backend == "genesis"
    override = loop_module.build_config_from_env(run_id="ov", sim_backend="isaac")
    assert override.sim_backend == "isaac"


def test_component_heldout_payload_dispatches_isaac_backend(monkeypatch) -> None:
    envs = [{"env_id": "heldout-0000", "seed": 1}]
    captured = {}

    def fake_isaac(env_payload, *, inner_evidence, threshold, scene=None, isaac_task):
        captured["isaac_called"] = True
        captured["isaac_task"] = isaac_task
        return [{"env_id": "heldout-0000", "score": 0.7, "success": False, "details": {}}]

    def fake_genesis(*args, **kwargs):
        raise AssertionError("genesis rollout must not run for sim_backend=isaac")

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_genesis)

    payload = loop_module._component_heldout_payload(
        envs,
        inner_evidence={"reward_trend": [0.2, 0.5]},
        threshold=0.75,
        sim_backend="isaac",
        isaac_task="Isaac-Lift-Cube-Franka-v0",
    )
    assert captured["isaac_called"] is True
    assert captured["isaac_task"] == "Isaac-Lift-Cube-Franka-v0"
    assert payload["sim_backend"] == "isaac"
    assert payload["component_source"] == "isaac_rollout"
    assert payload["rollout_backend"] == "isaaclab:Isaac-Lift-Cube-Franka-v0"
    assert payload["schema"] == SCHEMA_HELDOUT_REPORT


def test_component_heldout_payload_genesis_backend_unchanged(monkeypatch) -> None:
    def fake_genesis(env_payload, *, inner_evidence, threshold, scene=None):
        return [{"env_id": "heldout-0000", "score": 0.8, "success": True, "details": {}}]

    def fake_isaac(*args, **kwargs):
        raise AssertionError("isaac rollout must not run for sim_backend=genesis")

    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_genesis)
    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)

    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
        sim_backend="genesis",
    )
    assert payload["sim_backend"] == "genesis"
    assert payload["component_source"] == "genesis_rollout"
    assert payload["rollout_backend"] == "npa.genesis.env_pick_place.FrankaPickPlaceEnv"


def test_backends_emit_schema_compatible_reports(monkeypatch) -> None:
    """Both backends must produce the identical per-env report schema."""

    rows = [{"env_id": "heldout-0000", "score": 0.8, "success": True, "details": {"x": 1}}]
    monkeypatch.setattr(
        loop_module, "_run_genesis_heldout_rollouts", lambda *a, **k: rows
    )
    monkeypatch.setattr(
        loop_module, "_run_isaac_heldout_rollouts", lambda *a, **k: rows
    )
    common = dict(inner_evidence={"reward_trend": [0.2]}, threshold=0.75)
    genesis = loop_module._component_heldout_payload(rows, sim_backend="genesis", **common)
    isaac = loop_module._component_heldout_payload(rows, sim_backend="isaac", **common)

    assert genesis["schema"] == isaac["schema"] == SCHEMA_HELDOUT_REPORT
    for payload in (genesis, isaac):
        assert set(payload["per_env"][0]) >= {"env_id", "score", "success", "details"}
        assert payload["policy_source"] == "inner_evidence_adapter"
        assert "sim_backend" in payload
    assert genesis["per_env"] == isaac["per_env"]


def test_resolve_isaac_scene_stock_without_uris(tmp_path: Path) -> None:
    from npa.genesis import scene_assets as sa

    scene = loop_module._resolve_isaac_scene(
        scene_spec_uri="",
        assets_uri="",
        byo_mesh_uri="",
        dest_dir=tmp_path,
        client=_FakeMeshClient(),
    )
    assert scene is not None
    manip = scene.manipuland()
    assert manip.asset_source == sa.ASSET_SOURCE_ISAAC_STOCK
    assert manip.sha256 == ""
    assert scene.provenance_block()["asset_fallback_used"] is False


def test_resolve_isaac_scene_byo_mesh_records_provenance(tmp_path: Path) -> None:
    from npa.genesis import scene_assets as sa

    client = _FakeMeshClient(mesh=b"NAIL-MESH-BYTES")
    scene = loop_module._resolve_isaac_scene(
        scene_spec_uri="",
        assets_uri="s3://bucket/run/nail.obj",
        byo_mesh_uri="",
        dest_dir=tmp_path,
        client=client,
    )
    manip = scene.manipuland()
    assert manip.asset_source == sa.ASSET_SOURCE_BYO_MESH
    assert manip.sha256 == sa.sha256_file(manip.local_path)


def test_isaac_payload_stock_scene_provenance(monkeypatch) -> None:
    from npa.genesis import scene_assets as sa

    scene = sa.default_isaac_stock_scene_spec()

    def fake_isaac(envs, *, inner_evidence, threshold, scene=None, isaac_task):
        # Stock manipuland is materialized by the task env (marks loaded).
        if scene is not None:
            scene.manipuland().loaded = True
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
        scene=scene,
        sim_backend="isaac",
    )
    prov = payload["asset_provenance"]
    assert prov["objects"][0]["asset_source"] == "isaac_stock"
    assert payload["asset_fallback_used"] is False


def test_isaac_payload_byo_mesh_not_loaded_raises(monkeypatch, tmp_path: Path) -> None:
    from npa.genesis import scene_assets as sa

    client = _FakeMeshClient(mesh=b"NAIL-MESH")
    scene = sa.synthesize_scene_spec(byo_mesh_uri="s3://bucket/run/nail.obj")
    sa.resolve_scene_assets(scene, dest_dir=tmp_path, client=client)

    # Isaac rollout that "forgets" to import the mesh must trip the no-fallback gate.
    monkeypatch.setattr(
        loop_module,
        "_run_isaac_heldout_rollouts",
        lambda *a, **k: [{"env_id": "heldout-0000", "score": 0.9, "success": True}],
    )
    with pytest.raises(loop_module.Sim2RealLoopError):
        loop_module._component_heldout_payload(
            [{"env_id": "heldout-0000", "seed": 1}],
            inner_evidence={},
            threshold=0.75,
            scene=scene,
            sim_backend="isaac",
        )


class _FakeMeshConverter:
    """Fake Isaac Lab converter: writes a USD next to a tracked usd_path."""

    instances: list[str] = []

    def __init__(self, cfg) -> None:
        out = Path(cfg.usd_dir) / cfg.usd_file_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("#usda 1.0\n")
        self.usd_path = str(out)
        type(self).instances.append(cfg.asset_path)


def _install_fake_isaac_converters(monkeypatch, *, kind: dict) -> None:
    import types

    # Fake isaaclab.sim providing the spawn/property cfg classes the mesh
    # converter path references (real module needs the Isaac Sim runtime / pxr).
    root_mod = types.ModuleType("isaaclab")
    monkeypatch.setitem(sys.modules, "isaaclab", root_mod)
    sim_mod = types.ModuleType("isaaclab.sim")

    class _Cfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sim_mod.MassPropertiesCfg = _Cfg
    sim_mod.RigidBodyPropertiesCfg = _Cfg
    sim_mod.CollisionPropertiesCfg = _Cfg
    sim_mod.UsdFileCfg = _Cfg
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_mod)

    mod = types.ModuleType("isaaclab.sim.converters")

    class MeshConverterCfg:
        def __init__(self, *, asset_path, usd_dir, usd_file_name, force_usd_conversion=True, **kwargs):
            self.asset_path = asset_path
            self.usd_dir = usd_dir
            self.usd_file_name = usd_file_name
            self.__dict__.update(kwargs)

    class UrdfConverterCfg(MeshConverterCfg):
        pass

    class MeshConverter(_FakeMeshConverter):
        pass

    class UrdfConverter(_FakeMeshConverter):
        def __init__(self, cfg) -> None:
            super().__init__(cfg)
            kind["urdf"] = True

    mod.MeshConverter = MeshConverter
    mod.MeshConverterCfg = MeshConverterCfg
    mod.UrdfConverter = UrdfConverter
    mod.UrdfConverterCfg = UrdfConverterCfg
    monkeypatch.setitem(sys.modules, "isaaclab.sim.converters", mod)


def test_isaac_import_mesh_to_usd_dispatches_mesh_converter(
    monkeypatch, tmp_path: Path
) -> None:
    _FakeMeshConverter.instances = []
    kind: dict = {}
    _install_fake_isaac_converters(monkeypatch, kind=kind)
    src = tmp_path / "nail.obj"
    src.write_text("v 0 0 0\n")
    usd = loop_module._isaac_import_mesh_to_usd(str(src), work_dir=tmp_path / "usd")
    assert usd.endswith("nail.usd")
    assert Path(usd).is_file()
    assert _FakeMeshConverter.instances == [str(src)]
    assert "urdf" not in kind


def test_isaac_import_urdf_to_usd_dispatches_urdf_converter(
    monkeypatch, tmp_path: Path
) -> None:
    _FakeMeshConverter.instances = []
    kind: dict = {}
    _install_fake_isaac_converters(monkeypatch, kind=kind)
    src = tmp_path / "robot.urdf"
    src.write_text("<robot/>\n")
    usd = loop_module._isaac_import_mesh_to_usd(str(src), work_dir=tmp_path / "usd")
    assert usd.endswith("robot.usd")
    assert kind.get("urdf") is True


def test_isaac_import_mesh_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(loop_module.Sim2RealLoopError):
        loop_module._isaac_import_mesh_to_usd(
            str(tmp_path / "absent.obj"), work_dir=tmp_path / "usd"
        )


def test_isaac_heldout_eval_launches_isaac_image_job(monkeypatch, tmp_path: Path) -> None:
    output_payload = {
        "schema": SCHEMA_HELDOUT_REPORT,
        "sim_backend": "isaac",
        "per_env": [
            {"env_id": "env-a", "score": 0.81, "success": True, "details": {}},
        ],
    }
    storage = _FakeComponentStorage({"heldout_eval": output_payload})
    _patch_component_storage(monkeypatch, storage)
    calls = _patch_kubectl(monkeypatch)
    config = Sim2RealLoopConfig(
        run_id="isaac-image-job",
        s3_bucket="bucket",
        s3_prefix="neutral-prefix",
        s3_endpoint="https://storage.example",
        heldout_envs_uri="s3://bucket/neutral-prefix/run/envs/heldout/",
        threshold=0.75,
        k8s_namespace="default",
        sim_backend="isaac",
        isaac_image="npa-isaac-lab:2.3.2.post1",
        source_ref="dev-branch",
        source_repo="https://example.invalid/repo.git",
    )
    run_heldout_eval(
        config,
        local_dir=tmp_path,
        inner_evidence={"schema": "npa.sim2real.inner_loop_evidence.v1", "reward_trend": [0.1]},
        outer_iteration=1,
    )
    apply_call = next(call for call in calls if "apply" in call["cmd"])
    manifest = json.loads(apply_call["input"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "npa-isaac-lab:2.3.2.post1"
    script = container["args"][0]
    assert "/isaac-sim/python.sh" in script
    assert "--sim-backend" in script
    env_names = {item["name"] for item in container["env"]}
    assert "NPA_SIM2REAL_SIM_BACKEND" in env_names


def test_cosmos_split_sdk_and_raw_yaml_contracts() -> None:
    transfer = cosmos2.transfer(
        input_uri="s3://bucket/input/", output_uri="s3://bucket/augment/"
    )
    reason = cosmos3.reason(
        input_uri="s3://bucket/rollouts/", output_uri="s3://bucket/vlm_eval/"
    )

    assert transfer["schema"] == "npa.cosmos2.transfer.v1"
    assert reason["schema"] == "npa.cosmos3.reason.v1"
    assert "cosmos2-transfer" in COSMOS2_TRANSFER.read_text(encoding="utf-8")
    assert "cosmos3-reason" in COSMOS3_REASON.read_text(encoding="utf-8")
