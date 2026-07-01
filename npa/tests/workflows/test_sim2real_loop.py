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
    PolicyContainerError,
    VlmSignalUpdateResult,
    parse_vlm_signal_batch,
    run_vlm_signal_training_step,
)
from npa.workflows.sim2real_loop import (
    SCHEMA_HELDOUT_REPORT,
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
    Sim2RealLoopConfig,
    Sim2RealLoopError,
    artifact_uris,
    build_config_from_env,
    convert_vlm_eval_to_rl_signal,
    default_augment_image,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_full_loop,
    run_finalize,
    run_heldout_eval,
    run_inner_loop,
    run_preamble,
    run_single_outer_iteration,
)
from npa.workflows.sim2real.runner import Sim2RealWorkflow


ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = ROOT / "npa" / "workflows" / "workbench" / "sim2real" / "runbook.yaml"
SIM2REAL_ACTIONS = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sim2real-actions.yaml"
)
SIM2REAL_ENVGEN_SPLIT = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sim2real-envgen-split.yaml"
)
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
    augment = json.loads((tmp_path / "augment" / "manifest.json").read_text())
    retrigger = json.loads(
        (tmp_path / "stage_13_retrigger" / "retrigger.json").read_text()
    )
    assert augment["stage"] == "cosmos2-transfer"
    assert augment["status"] in {"executed_reference", "executed", "contract_ready"}
    assert augment.get("image") == "npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"
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
    components = {c["name"]: c for c in report["components"]}
    assert components["stage_12_external_validation"]["tier"] == "SEAM"
    assert (
        json.loads(
            (
                tmp_path / "stage_12_external_validation" / "external_stub.json"
            ).read_text()
        )["status"]
        == "documented_external_stub"
    )
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
        if payload is None and "vlm-eval-reason2" in bucket_uri:
            payload = self.downloads.get("vlm_eval_reason2") or self.downloads.get(
                "vlm_eval"
            )
        if payload is None and "vlm-eval-reason3" in bucket_uri:
            payload = self.downloads.get("vlm_eval_reason3") or self.downloads.get(
                "vlm_eval"
            )
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
        cmd_text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        if "apply" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, "job.batch/sibling created\n", "")
        if "wait" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, "job.batch/sibling condition met\n", "")
        if "get" in cmd_text and "job" in cmd_text and "jsonpath" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, "1 0", "")
        if "get" in cmd_text and "pods" in cmd_text:
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
        if "logs" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, '{"component":"ok"}\n', "")
        if "delete" in cmd_text:
            return subprocess.CompletedProcess(cmd, 0, "job.batch deleted\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(loop_module.subprocess, "run", fake_run)
    monkeypatch.setattr(loop_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        loop_module,
        "_wait_kubernetes_job",
        lambda *args, **kwargs: "complete",
    )
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

    apply_calls = [call for call in calls if "apply" in call["cmd"]]
    assert len(apply_calls) == 2
    manifest = json.loads(apply_calls[0]["input"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]

    assert evaluation["score"] == 0.512345
    assert evaluation["component_invocation"]["mode"] == "kubernetes_job_dual_reason"
    assert evaluation["component_invocation"]["reason2_image"]
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
        sim_backend="genesis",
        eval_image="cr.example/npa-loop-eval:0.1.1",
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

    assert captured["model_id"] == "nvidia/Cosmos-Reason2-8B"
    assert captured["image_paths"] == [frame]
    assert captured["task_description"] == "Move the object to the target."
    assert payload["component_source"] == "cosmos_reason_vlm"
    assert payload["model"] == "nvidia/Cosmos-Reason2-8B"
    assert payload["frame_count"] == 1
    assert "synthetic_signature" not in payload


def test_component_heldout_payload_defaults_to_isaac_rollout_backend(monkeypatch) -> None:
    captured = {}

    def fake_isaac(env_payload, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        captured["isaac_called"] = True
        captured["isaac_task"] = isaac_task
        return [{"env_id": "heldout-0000", "score": 0.82, "success": True, "details": {}}]

    def fake_genesis(*args, **kwargs):
        raise AssertionError("genesis rollout must not run when sim_backend defaults")

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_genesis)

    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
    )

    assert captured["isaac_called"] is True
    assert payload["sim_backend"] == "isaac"
    assert payload["component_source"] == "isaac_rollout"
    assert payload["rollout_backend"] == "isaaclab:Isaac-Lift-Cube-Franka-v0"


def test_component_heldout_payload_uses_genesis_rollout_backend(monkeypatch) -> None:
    envs = [
        {"env_id": "heldout-0000", "target_pose": [0.0, 0.1, 0.0]},
        {"env_id": "heldout-0001", "target_pose": [0.1, 0.0, 0.0]},
    ]
    captured = {}

    def fake_genesis_rollouts(env_payload, *, inner_evidence, threshold, scene=None, robot=None):
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

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        # Simulate the env building the mesh and marking it loaded.
        if scene is not None:
            for obj in scene.objects:
                obj.loaded = True
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_rollouts)
    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
        scene=scene,
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

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_rollouts)
    with pytest.raises(loop_module.Sim2RealLoopError):
        loop_module._component_heldout_payload(
            [{"env_id": "heldout-0000", "seed": 1}],
            inner_evidence={},
            threshold=0.75,
            scene=scene,
        )


def test_resolve_heldout_robot_none_without_inputs(tmp_path: Path) -> None:
    robot = loop_module._resolve_heldout_robot(
        robot_spec_uri="",
        robot_source="",
        robot_preset="",
        dest_dir=tmp_path,
        client=_FakeMeshClient(),
    )
    assert robot is None


def test_resolve_heldout_robot_byo_urdf_records_provenance(tmp_path: Path) -> None:
    from npa.genesis import robot_assets as ra

    doc = {
        "preset": "ur5e",
        "robot_source": "byo_urdf",
        "robot_uri": "s3://bucket/robots/ur5e.urdf",
    }
    client = _FakeMeshClient(spec_doc=doc, mesh=b"<robot>urdf</robot>")
    robot = loop_module._resolve_heldout_robot(
        robot_spec_uri="s3://bucket/robots/robot-spec.json",
        robot_source="",
        robot_preset="",
        dest_dir=tmp_path,
        client=client,
    )
    assert robot is not None
    assert robot.robot_source == ra.ROBOT_SOURCE_BYO_URDF
    assert robot.ee_link == "tool0"
    assert robot.sha256 == ra.sha256_file(robot.local_path)
    assert robot.provenance()["robot_fallback_used"] is False


def test_component_heldout_payload_with_robot_attaches_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.genesis import robot_assets as ra

    client = _FakeMeshClient(mesh=b"<robot>urdf</robot>")
    robot = ra.robot_spec_from_preset("ur5e")
    robot.robot_uri = "s3://bucket/robots/ur5e.urdf"
    ra.resolve_robot_asset(robot, dest_dir=tmp_path, client=client)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        if robot is not None:
            robot.loaded = True  # env builds it
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_rollouts)
    payload = loop_module._component_heldout_payload(
        [{"env_id": "heldout-0000", "seed": 1}],
        inner_evidence={"reward_trend": [0.2, 0.6]},
        threshold=0.75,
        robot=robot,
    )
    assert payload["robot_fallback_used"] is False
    prov = payload["robot_provenance"]
    assert prov["robot_source"] == "byo_urdf"
    assert prov["loaded"] is True
    assert prov["ee_link"] == "tool0"


def test_component_heldout_payload_raises_when_byo_robot_not_loaded(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.genesis import robot_assets as ra

    client = _FakeMeshClient(mesh=b"<robot>urdf</robot>")
    robot = ra.robot_spec_from_preset("ur5e")
    robot.robot_uri = "s3://bucket/robots/ur5e.urdf"
    ra.resolve_robot_asset(robot, dest_dir=tmp_path, client=client)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_rollouts)
    with pytest.raises(loop_module.Sim2RealLoopError):
        loop_module._component_heldout_payload(
            [{"env_id": "heldout-0000", "seed": 1}],
            inner_evidence={},
            threshold=0.75,
            robot=robot,
        )


def test_normalize_heldout_report_propagates_robot_provenance() -> None:
    payload = {
        "per_env": [{"env_id": "h-0", "score": 0.9, "success": True}],
        "robot_provenance": {
            "schema": "npa.sim2real.robot_provenance.v1",
            "robot_source": "byo_urdf",
            "loaded": True,
            "robot_fallback_used": False,
        },
        "robot_fallback_used": False,
    }
    config = Sim2RealLoopConfig(run_id="r", threshold=0.5)
    report = loop_module._normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=1,
        inner_evidence_uri="inner.json",
        invocation={"component": "heldout_eval"},
    )
    assert report["robot_provenance"]["robot_source"] == "byo_urdf"
    assert report["robot_fallback_used"] is False


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


def test_normalize_heldout_report_computes_success_summary_from_distances() -> None:
    # Strict success_rate@threshold is 0 (no env is "success"), but the policy
    # lands most objects within 0.15m -> the recomputed multi-threshold summary
    # keeps that accuracy visible in the normalized report.
    payload = {
        "per_env": [
            {"env_id": f"h-{i}", "score": 0.0, "success": False,
             "details": {"object_goal_distance_m": d}}
            for i, d in enumerate([0.04, 0.09, 0.12, 0.18])
        ],
    }
    from npa.workflows.sim2real.engine import _normalize_heldout_report

    config = Sim2RealLoopConfig(run_id="r", threshold=0.99)
    report = _normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=1,
        inner_evidence_uri="inner.json",
        invocation={"component": "heldout_eval"},
    )
    assert report["success_rate"] == 0.0
    summary = report["success_summary"]
    assert summary["success@0.05"] == 0.25  # only 0.04 < 0.05
    assert summary["success@0.10"] == 0.5  # 0.04, 0.09
    assert summary["success@0.15"] == 0.75  # 0.04, 0.09, 0.12
    assert summary["success@0.20"] == 1.0
    assert summary["min_object_goal_distance_m"] == 0.04
    assert summary["mean_object_goal_distance_m"] == round(
        (0.04 + 0.09 + 0.12 + 0.18) / 4, 6
    )


def test_normalize_heldout_report_carries_through_payload_success_summary() -> None:
    # When byo_isaac_eval already emitted a success_summary, preserve it verbatim
    # rather than recomputing.
    emitted = {
        "success@0.05": 0.0,
        "success@0.15": 0.81,
        "mean_object_goal_distance_m": 0.09,
        "min_object_goal_distance_m": 0.07,
    }
    payload = {
        "per_env": [
            {"env_id": "h-0", "score": 0.0, "success": False,
             "details": {"object_goal_distance_m": 0.5}}
        ],
        "success_summary": emitted,
    }
    from npa.workflows.sim2real.engine import _normalize_heldout_report

    config = Sim2RealLoopConfig(run_id="r", threshold=0.5)
    report = _normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=1,
        inner_evidence_uri="inner.json",
        invocation={"component": "heldout_eval"},
    )
    assert report["success_summary"] == emitted


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
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if uri.endswith("envs.jsonl"):
            dest.write_text(
                json.dumps({"env_id": "heldout-0000", "seed": 7}) + "\n",
                encoding="utf-8",
            )
        elif uri.endswith("heldout/"):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "envs.jsonl").write_text(
                json.dumps({"env_id": "heldout-0000", "seed": 7}) + "\n",
                encoding="utf-8",
            )
        elif uri.endswith(".json"):
            dest.write_text(json.dumps({"reward_trend": [0.2, 0.6]}), encoding="utf-8")
        else:
            dest.write_bytes(b"OBJ-BYTES")
        return str(dest)

    monkeypatch.setattr(client, "download_path", download_path)

    def fake_rollouts(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        if scene is not None:
            for obj in scene.objects:
                obj.loaded = True
        return [{"env_id": "heldout-0000", "score": 0.9, "success": True}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_rollouts)

    payload = loop_module.run_heldout_eval_component_from_s3(
        heldout_envs_uri="s3://bucket/run/heldout/",
        inner_evidence_uri="s3://bucket/run/inner-evidence.json",
        output_uri="s3://bucket/run/output/report.json",
        threshold=0.75,
        assets_uri="s3://bucket/run/object.obj",
        sim_backend="isaac",
    )
    assert payload["asset_fallback_used"] is False
    assert payload["asset_provenance"]["objects"][0]["asset_source"] == "byo_mesh"
    # A consumed scene spec is uploaded alongside the report.
    uploaded = [u[1] for u in client.uploads]
    assert any(u.endswith("consumed-scene-spec.json") for u in uploaded)


def test_read_component_env_records_accepts_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "envs.jsonl"
    path.write_text(json.dumps({"env_id": "e-0"}) + "\n", encoding="utf-8")
    records = loop_module._read_component_env_records(path)
    assert records[0]["env_id"] == "e-0"


def test_run_heldout_eval_component_from_s3_reads_single_envs_jsonl(
    monkeypatch, tmp_path: Path
) -> None:
    client = _FakeMeshClient(mesh=b"OBJ-BYTES")
    monkeypatch.setattr(
        loop_module.StorageClient, "from_environment", staticmethod(lambda **kw: client)
    )

    def download_path(uri, local_path):
        client.downloads.append((uri, local_path))
        dest = Path(local_path)
        if uri.endswith("envs.jsonl"):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "envs.jsonl").write_text(
                json.dumps({"env_id": "heldout-0000", "seed": 7}) + "\n",
                encoding="utf-8",
            )
        elif uri.endswith(".json"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps({"reward_trend": [0.2, 0.6]}), encoding="utf-8")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"OBJ-BYTES")
        return local_path

    monkeypatch.setattr(client, "download_path", download_path)
    monkeypatch.setattr(
        loop_module,
        "_run_isaac_heldout_rollouts",
        lambda envs, **kw: [
            {"env_id": envs[0]["env_id"], "score": 0.9, "success": True}
        ],
    )

    payload = loop_module.run_heldout_eval_component_from_s3(
        heldout_envs_uri="s3://bucket/run/envs/heldout/envs.jsonl",
        inner_evidence_uri="s3://bucket/run/inner-evidence.json",
        output_uri="s3://bucket/run/output/report.json",
        threshold=0.75,
        sim_backend="isaac",
    )
    assert payload["per_env"][0]["env_id"] == "heldout-0000"


def test_resolve_env_records_s3_uri_appends_jsonl_for_split_prefixes() -> None:
    assert (
        loop_module._resolve_env_records_s3_uri(
            "s3://bucket/run/envs/heldout/"
        )
        == "s3://bucket/run/envs/heldout/envs.jsonl"
    )
    assert (
        loop_module._resolve_env_records_s3_uri(
            "s3://bucket/run/envs/heldout/envs.jsonl"
        )
        == "s3://bucket/run/envs/heldout/envs.jsonl"
    )


def test_kubernetes_component_env_propagates_storage_credentials(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "orch-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "orch-secret")
    config = Sim2RealLoopConfig(
        run_id="r",
        s3_endpoint="https://storage.example.test",
    )
    safe = loop_module._kubernetes_component_env(
        {"NPA_SIM2REAL_HELDOUT_ENVS_URI": "s3://bucket/run/envs/heldout/envs.jsonl"},
        config,
    )
    assert safe["AWS_ACCESS_KEY_ID"] == "orch-key"
    assert safe["AWS_SECRET_ACCESS_KEY"] == "orch-secret"
    assert safe["AWS_ENDPOINT_URL"] == "https://storage.example.test"
    assert safe["HF_HOME"] == "/tmp/hf_home"
    assert safe["NPA_COSMOS_REASON2_CACHE"] == "/tmp/hf_home/cosmos-reason2"


def test_wait_kubernetes_job_returns_failed_without_waiting(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(
        loop_module,
        "_kubectl",
        lambda config, args, **kwargs: subprocess.CompletedProcess(args, 0, "0 1", ""),
    )
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        loop_module._wait_kubernetes_job(
            config, namespace="default", job_name="j", timeout_s=10
        )
        == "failed"
    )


def test_wait_kubernetes_job_returns_complete(monkeypatch) -> None:
    import subprocess

    monkeypatch.setattr(
        loop_module,
        "_kubectl",
        lambda config, args, **kwargs: subprocess.CompletedProcess(args, 0, "1 0", ""),
    )
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        loop_module._wait_kubernetes_job(
            config, namespace="default", job_name="j", timeout_s=10
        )
        == "complete"
    )


def test_wait_kubernetes_job_fail_fast_on_not_found(monkeypatch) -> None:
    import subprocess

    calls: list[list[str]] = []

    def fake_kubectl(config, args, **kwargs):
        calls.append(list(args))
        stderr = (
            "Error from server (NotFound): jobs \"j\" not found"
            if args[0] == "get"
            else ""
        )
        return subprocess.CompletedProcess(args, 1, "", stderr)

    monkeypatch.setattr(loop_module, "_kubectl", fake_kubectl)
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        loop_module._wait_kubernetes_job(
            config, namespace="default", job_name="j", timeout_s=7200
        )
        == "failed"
    )
    assert calls[0][:3] == ["get", "job", "j"]
    assert not any(call[0] == "wait" for call in calls)


def test_wait_kubernetes_job_poll_not_found_returns_failed(monkeypatch) -> None:
    import subprocess

    sequence = [
        subprocess.CompletedProcess(["get"], 0, "0 0", ""),
        subprocess.CompletedProcess(
            ["wait"],
            1,
            "",
            "timed out waiting for the condition on jobs/j",
        ),
        subprocess.CompletedProcess(
            ["wait"],
            1,
            "",
            "timed out waiting for the condition on jobs/j",
        ),
        subprocess.CompletedProcess(
            ["get"],
            1,
            "",
            "Error from server (NotFound): jobs \"j\" not found",
        ),
    ]

    def fake_kubectl(config, args, **kwargs):
        return sequence.pop(0)

    monkeypatch.setattr(loop_module, "_kubectl", fake_kubectl)
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        loop_module._wait_kubernetes_job(
            config, namespace="default", job_name="j", timeout_s=60
        )
        == "failed"
    )


def test_sdk_exposes_sim2real_run(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    report = sim2real.run(
        run_id="sim2real-sdk-unit",
        output_dir=tmp_path,
        s3_bucket="",
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


def test_default_augment_image_uses_cosmos2_transfer_contract(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("AUGMENT_IMAGE", raising=False)

    assert default_augment_image() == "npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"

    config = build_config_from_env(run_id="sim2real-images")

    assert config.augment_image == "npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"
    assert config.vlm_image == "npa-cosmos3-reason:3.0.1-genuine-sm120"
    assert "cosmos3" not in config.augment_image


def test_default_augment_image_uses_first_party_cosmos2_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/workbench")
    monkeypatch.delenv("AUGMENT_IMAGE", raising=False)

    config = build_config_from_env(run_id="sim2real-images")

    assert (
        config.augment_image
        == "registry.example/workbench/npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"
    )
    assert (
        config.vlm_image == "registry.example/workbench/npa-cosmos3-reason:3.0.1-genuine-sm120"
    )


def test_raw_runbook_invokes_staged_flow_and_exposes_byo_envs() -> None:
    docs = [
        doc
        for doc in yaml.safe_load_all(RUNBOOK.read_text(encoding="utf-8"))
        if doc is not None
    ]

    assert len(docs) == 1
    task = docs[0]
    assert task["name"] == "sim2real-staged-loop"
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "RTX6000:1"

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

    assert "npa.workflows.sim2real run" in task["run"]
    assert "--initial-quality" in task["run"]
    assert "--upload-artifacts" in task["run"]
    assert "for outer_iteration in $(seq 1" not in task["run"]
    assert "--trigger-dataset-uri" in task["run"]
    assert "--byo-signal-converter" in task["run"]
    assert "--k8s-service-account" in task["run"]
    assert "--k8s-gpu-product" in task["run"]
    assert "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition" in task["run"]
    assert "--heldout-eval-limit" in task["run"]
    assert "--vlm-dual-reason" in task["run"]
    assert "--vlm-reason2-model" in task["run"]
    assert "--vlm-reason3-model" in task["run"]
    assert task["envs"]["VLM_REASON2_MODEL"] == "nvidia/Cosmos-Reason2-8B"
    assert task["envs"]["VLM_REASON3_MODEL"] == "nvidia/Cosmos-Reason2-2B"
    assert "nebius.cloud" not in RUNBOOK.read_text(encoding="utf-8")


def test_staged_path_produces_same_decision_as_full_loop(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    staged_dir = tmp_path / "staged"
    command = _component_command(tmp_path)
    kwargs = dict(
        threshold=0.75,
        inner_iterations=2,
        outer_iterations=1,
        rollout_count=2,
        steps_per_rollout=3,
        heldout_env_count=2,
        seed=13,
        rerun_enabled=False,
        upload_artifacts=False,
        byo_vlm_command=command,
        byo_eval_command=command,
    )
    full_config = Sim2RealLoopConfig(run_id="sim2real-full", output_dir=full_dir, **kwargs)
    full_report = run_full_loop(full_config)

    staged_config = Sim2RealLoopConfig(
        run_id="sim2real-staged",
        output_dir=staged_dir,
        **kwargs,
    )
    state = run_preamble(staged_config)
    iteration = run_single_outer_iteration(
        staged_config,
        local_dir=staged_dir,
        outer_iteration=1,
        initial_quality=float(state["current_quality"]),
    )
    staged_report = run_finalize(
        staged_config,
        local_dir=staged_dir,
        stage_records=list(state["stage_records"]),
        components=list(state["components"]),
        outer_history=[iteration["history_entry"]],
        final_inner=iteration["inner"],
        final_eval=iteration["heldout_report"],
        final_decision=iteration["decision"],
    )

    assert (
        staged_report["outer_loop"]["latest_decision"]["decision"]
        == full_report["outer_loop"]["latest_decision"]["decision"]
    )
    assert staged_report["outer_loop"]["latest_decision"]["threshold"] == pytest.approx(
        full_report["outer_loop"]["latest_decision"]["threshold"]
    )
    assert len(staged_report["inner_loop"]["iterations"]) == len(
        full_report["inner_loop"]["iterations"]
    )
    assert staged_report["inner_loop"]["signal_converter_source"] == full_report["inner_loop"][
        "signal_converter_source"
    ]
    assert (staged_dir / "state" / "workflow_state.json").exists()


def test_workflow_runner_matches_full_loop(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    kwargs = dict(
        threshold=0.75,
        inner_iterations=2,
        outer_iterations=1,
        rollout_count=2,
        steps_per_rollout=3,
        heldout_env_count=2,
        seed=17,
        rerun_enabled=False,
        upload_artifacts=False,
        byo_vlm_command=command,
        byo_eval_command=command,
    )
    runner_dir = tmp_path / "runner"
    config = Sim2RealLoopConfig(run_id="sim2real-runner", output_dir=runner_dir, **kwargs)
    report = Sim2RealWorkflow(config).run()
    assert report["outer_loop"]["latest_decision"]["decision"] in {
        "promote_checkpoint",
        "loop_back_to_inner_loop",
    }
    assert (runner_dir / "state" / "workflow_state.json").exists()


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
    monkeypatch.delenv("NPA_SIM2REAL_SIM_BACKEND", raising=False)
    config = loop_module.build_config_from_env(run_id="env-backend")
    assert config.sim_backend == "isaac"
    monkeypatch.setenv("NPA_SIM2REAL_SIM_BACKEND", "GENESIS")
    config = loop_module.build_config_from_env(run_id="env-backend-genesis")
    assert config.sim_backend == "genesis"
    override = loop_module.build_config_from_env(run_id="ov", sim_backend="isaac")
    assert override.sim_backend == "isaac"


def test_component_heldout_payload_dispatches_isaac_backend(monkeypatch) -> None:
    envs = [{"env_id": "heldout-0000", "seed": 1}]
    captured = {}

    def fake_isaac(env_payload, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
        captured["isaac_called"] = True
        captured["isaac_task"] = isaac_task
        return [{"env_id": "heldout-0000", "score": 0.7, "success": False, "details": {}}]

    def fake_genesis(*args, **kwargs):
        raise AssertionError("genesis rollout must not run for sim_backend=isaac")

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    monkeypatch.setattr(loop_module, "_run_genesis_heldout_rollouts", fake_genesis)

    payload = loop_module._component_heldout_payload(
        envs,
        inner_evidence={
            "reward_trend": [0.1, 0.2],
            "final_quality": 0.4,
            "trainer_source": "reference",
        },
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
    assert payload["per_env"][0]["success"] is False


def test_reference_adapter_heldout_gate_promotes_from_inner_progress(monkeypatch) -> None:
    envs = [
        {"env_id": "heldout-0000", "physics": {"friction": 0.5}},
        {"env_id": "heldout-0001", "physics": {"friction": 0.5}},
    ]

    def fake_isaac(*_args, **_kwargs):
        return [
            {
                "env_id": row["env_id"],
                "score": 0.12,
                "success": False,
                "details": {"source": "isaac_lift_env_goal_distance"},
            }
            for row in envs
        ]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    monkeypatch.setattr(
        loop_module,
        "_run_genesis_heldout_rollouts",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("genesis")),
    )

    inner_evidence = {
        "trainer_source": "reference",
        "reward_trend": [0.2, 0.6],
        "final_quality": 0.52,
        "iterations": [
            {
                "sample_vlm_eval": {"score": 0.82},
            }
        ],
    }
    payload = loop_module._component_heldout_payload(
        envs,
        inner_evidence=inner_evidence,
        threshold=0.75,
        sim_backend="isaac",
    )

    assert payload["per_env"][0]["success"] is True
    assert payload["per_env"][0]["score"] >= 0.75
    assert payload["per_env"][0]["details"]["sim_success"] is False
    assert payload["per_env"][0]["details"]["reference_adapter_score"] >= 0.75
    assert sum(int(row["success"]) for row in payload["per_env"]) >= 1


def test_reference_adapter_heldout_gate_skips_byo_trainer(monkeypatch) -> None:
    envs = [{"env_id": "heldout-0000", "seed": 1}]

    def fake_isaac(*_args, **_kwargs):
        return [{"env_id": "heldout-0000", "score": 0.2, "success": False, "details": {}}]

    monkeypatch.setattr(loop_module, "_run_isaac_heldout_rollouts", fake_isaac)
    monkeypatch.setattr(
        loop_module,
        "_run_genesis_heldout_rollouts",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("genesis")),
    )

    payload = loop_module._component_heldout_payload(
        envs,
        inner_evidence={
            "trainer_source": "byo_command",
            "iterations": [{"sample_vlm_eval": {"score": 0.95}}],
        },
        threshold=0.75,
        sim_backend="isaac",
    )

    assert payload["per_env"][0]["success"] is False
    assert "reference_adapter_score" not in payload["per_env"][0]["details"]


def test_component_heldout_payload_genesis_backend_unchanged(monkeypatch) -> None:
    def fake_genesis(env_payload, *, inner_evidence, threshold, scene=None, robot=None):
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


def test_resolve_isaac_scene_consumed_stock_envelope(tmp_path: Path) -> None:
    from npa.genesis import scene_assets as sa
    from npa.workflows.sim2real_assets import CONSUMED_SCENE_SCHEMA

    consumed = {
        "schema": CONSUMED_SCENE_SCHEMA,
        "status": "stock_tabletop",
        "scene_spec": sa.default_isaac_stock_scene_spec().to_dict(),
    }
    spec_path = tmp_path / "consumed_scene_spec.json"
    spec_path.write_text(json.dumps(consumed), encoding="utf-8")

    class _Client:
        def download_path(self, uri, dest):
            Path(dest).write_text(spec_path.read_text(), encoding="utf-8")
            return dest

    scene = loop_module._resolve_isaac_scene(
        scene_spec_uri="s3://bucket/run/stage_02_assets/consumed_scene_spec.json",
        assets_uri="",
        byo_mesh_uri="",
        dest_dir=tmp_path / "assets",
        client=_Client(),
    )
    assert scene.manipuland().asset_source == sa.ASSET_SOURCE_ISAAC_STOCK


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

    def fake_isaac(envs, *, inner_evidence, threshold, scene=None, robot=None, isaac_task):
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
        isaac_image="cr.example/npa-isaac-lab:2.3.2.post1",
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
    assert container["image"] == "cr.example/npa-isaac-lab:2.3.2.post1"
    script = container["args"][0]
    assert "/isaac-sim/python.sh" in script
    assert "NPA_SIM2REAL_SOURCE_TARBALL_URI" in script
    assert "missing NPA_SIM2REAL_SOURCE_TARBALL_URI" in script
    assert "--sim-backend" in script
    env_names = {item["name"] for item in container["env"]}
    assert "NPA_SIM2REAL_SIM_BACKEND" in env_names


def _signal_converter_command(tmp_path: Path, *, valid: bool = True) -> str:
    script = tmp_path / "byo_signal_converter.py"
    body = (
        '''
import json, os
from pathlib import Path

ev = json.loads(Path(os.environ["NPA_SIM2REAL_EVALUATION_JSON"]).read_text())
out = Path(os.environ["NPA_SIM2REAL_OUTPUT_JSON"])
out.parent.mkdir(parents=True, exist_ok=True)
marker = Path(os.environ["NPA_SIM2REAL_SWAP_MARKER"])
with marker.open("a", encoding="utf-8") as handle:
    handle.write("signal_converter\\n")
'''
    )
    if valid:
        body += (
            '''
per_step = [
    {
        "step": int(s["step"]),
        "reward": 0.5,
        "advantage": 0.1,
        "target": {"nl_correction": "byo correction", "action_delta": [0.0, 0.0, 0.0]},
        "error_tags": s.get("error_tags", ["ok"]),
    }
    for s in ev["per_step"]
]
out.write_text(json.dumps({
    "schema": os.environ["NPA_SIM2REAL_RL_SIGNAL_SCHEMA"],
    "rollout_id": ev["rollout_id"],
    "source": "byo-test",
    "per_step": per_step,
}))
'''
        )
    else:
        body += '\nout.write_text(json.dumps({"schema": "wrong.schema", "per_step": []}))\n'
    script.write_text(body, encoding="utf-8")
    return f"{sys.executable} {script}"


def _trainer_command(tmp_path: Path, *, valid: bool = True) -> str:
    script = tmp_path / "byo_trainer.py"
    body = (
        '''
import json, os
from pathlib import Path

batch = json.loads(Path(os.environ["NPA_SIM2REAL_SIGNAL_JSON"]).read_text())
out = Path(os.environ["NPA_SIM2REAL_OUTPUT_JSON"])
out.parent.mkdir(parents=True, exist_ok=True)
marker = Path(os.environ["NPA_SIM2REAL_SWAP_MARKER"])
with marker.open("a", encoding="utf-8") as handle:
    handle.write("trainer\\n")
n = len(batch.get("signals", []))
'''
    )
    if valid:
        body += (
            '''
out.write_text(json.dumps({
    "reward_head_after": 0.25 + 0.01 * n,
    "policy_output_after": [0.06, 0.0, -0.03],
    "policy_delta_l2": 0.42,
    "loss_before": 1.0,
    "loss_after": 0.4,
    "backend": "byo-test",
}))
'''
        )
    else:
        body += '\nout.write_text(json.dumps({"reward_head_after": 0.1}))\n'
    script.write_text(body, encoding="utf-8")
    return f"{sys.executable} {script}"


def test_byo_signal_converter_swap_invoked_and_recorded(tmp_path: Path) -> None:
    marker = tmp_path / "swap-marker.log"
    config = Sim2RealLoopConfig(
        run_id="sim2real-signal-swap",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        byo_vlm_command=(
            f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {_component_command(tmp_path)}"
        ),
        byo_signal_converter=(
            f"NPA_SIM2REAL_SWAP_MARKER={marker} {_signal_converter_command(tmp_path)}"
        ),
    )

    evidence = run_inner_loop(config, local_dir=tmp_path, initial_quality=0.4)

    assert evidence["signal_converter_source"] == "byo_command"
    assert evidence["trainer_source"] == "reference"
    assert evidence["iterations"][0]["signal_converter_source"] == "byo_command"
    assert "signal_converter" in marker.read_text(encoding="utf-8")
    sample_signal = evidence["iterations"][0]["sample_signal"]
    assert sample_signal["schema"] == SCHEMA_RL_SIGNAL
    assert sample_signal["source"] == "byo-test"
    # The BYO signal must remain parseable by the downstream trainer contract.
    parse_vlm_signal_batch(sample_signal)


def test_byo_signal_converter_malformed_raises_no_fallback(tmp_path: Path) -> None:
    marker = tmp_path / "swap-marker.log"
    config = Sim2RealLoopConfig(
        run_id="sim2real-signal-bad",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        byo_vlm_command=(
            f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {_component_command(tmp_path)}"
        ),
        byo_signal_converter=(
            f"NPA_SIM2REAL_SWAP_MARKER={marker} "
            f"{_signal_converter_command(tmp_path, valid=False)}"
        ),
    )

    with pytest.raises(Sim2RealLoopError, match="rl_signal"):
        run_inner_loop(config, local_dir=tmp_path, initial_quality=0.4)


def test_byo_trainer_command_swap_invoked_and_recorded(tmp_path: Path) -> None:
    marker = tmp_path / "swap-marker.log"
    config = Sim2RealLoopConfig(
        run_id="sim2real-trainer-swap",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        byo_vlm_command=(
            f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {_component_command(tmp_path)}"
        ),
        byo_trainer_command=(
            f"NPA_SIM2REAL_SWAP_MARKER={marker} {_trainer_command(tmp_path)}"
        ),
    )

    evidence = run_inner_loop(config, local_dir=tmp_path, initial_quality=0.4)

    assert evidence["trainer_source"] == "byo_command"
    assert evidence["signal_converter_source"] == "reference"
    iteration = evidence["iterations"][0]
    assert iteration["trainer_source"] == "byo_command"
    update = iteration["update"]
    assert update["reward_head_after"] == pytest.approx(0.26)
    assert update["policy_output_after"] == [0.06, 0.0, -0.03]
    assert update["policy_delta_l2"] == pytest.approx(0.42)
    assert update["backend"] == "byo-test"
    # The no-signal control still runs the in-process reference trainer.
    assert iteration["no_signal_control"]["backend"] != "byo-test"
    assert "trainer" in marker.read_text(encoding="utf-8")


def test_byo_trainer_command_missing_fields_raises_no_fallback(tmp_path: Path) -> None:
    marker = tmp_path / "swap-marker.log"
    config = Sim2RealLoopConfig(
        run_id="sim2real-trainer-bad",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        byo_vlm_command=(
            f"NPA_SIM2REAL_COMPONENT_MARKER={marker} {_component_command(tmp_path)}"
        ),
        byo_trainer_command=(
            f"NPA_SIM2REAL_SWAP_MARKER={marker} "
            f"{_trainer_command(tmp_path, valid=False)}"
        ),
    )

    with pytest.raises(Sim2RealLoopError, match="policy_output_after"):
        run_inner_loop(config, local_dir=tmp_path, initial_quality=0.4)


def test_default_inner_loop_provenance_is_reference(tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(
        run_id="sim2real-default-prov",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        byo_vlm_command=_component_command(tmp_path),
    )

    evidence = run_inner_loop(config, local_dir=tmp_path, initial_quality=0.4)

    assert evidence["trainer_source"] == "reference"
    assert evidence["signal_converter_source"] == "reference"


def test_full_loop_writes_rerun_recording_by_default(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-rerun-default",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        outer_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
        byo_vlm_command=command,
        byo_eval_command=command,
    )

    report = run_full_loop(config)

    viz = report["visualization"]
    assert viz["status"] == "written"
    assert viz["source"] == "reference"
    rrd = tmp_path / "reports" / "sim2real.rrd"
    assert rrd.exists() and rrd.stat().st_size > 0
    components = {c["name"]: c for c in report["components"]}
    assert components["stage_14_rerun_viz"]["tier"] == "WORKS"


def test_full_loop_rerun_disabled_toggle(tmp_path: Path) -> None:
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-rerun-off",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        outer_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
        byo_vlm_command=command,
        byo_eval_command=command,
        rerun_enabled=False,
    )

    report = run_full_loop(config)

    assert report["visualization"]["status"] == "disabled"
    assert not (tmp_path / "reports" / "sim2real.rrd").exists()
    components = {c["name"]: c for c in report["components"]}
    assert components["stage_14_rerun_viz"]["tier"] == "SEAM"


def test_full_loop_rerun_warns_when_sdk_missing(monkeypatch, tmp_path: Path) -> None:
    import npa.workflows.sim2real_viz as viz_module

    def _raise(**_kwargs):
        raise viz_module.RerunUnavailableError("rerun-sdk is not installed")

    monkeypatch.setattr(viz_module, "emit_sim2real_rerun", _raise)
    command = _component_command(tmp_path)
    config = Sim2RealLoopConfig(
        run_id="sim2real-rerun-warn",
        output_dir=tmp_path,
        threshold=0.4,
        inner_iterations=1,
        outer_iterations=1,
        rollout_count=1,
        steps_per_rollout=2,
        heldout_env_count=2,
        byo_vlm_command=command,
        byo_eval_command=command,
    )

    report = run_full_loop(config)

    assert report["visualization"]["status"] == "skipped"
    components = {c["name"]: c for c in report["components"]}
    assert components["stage_14_rerun_viz"]["tier"] == "WARN"


def test_vlm_signal_update_result_from_dict_defaults_and_required() -> None:
    result = VlmSignalUpdateResult.from_dict(
        {
            "reward_head_after": 0.3,
            "policy_output_after": [0.1, -0.2],
            "policy_delta_l2": 0.5,
        }
    )

    assert result.reward_head_after == 0.3
    assert result.policy_output_after == [0.1, -0.2]
    assert result.policy_delta_l2 == 0.5
    # Safe defaults fill the rest.
    assert result.policy_output_before == [0.0, 0.0]
    assert result.backend == "byo_command"
    assert result.status == "updated"
    assert result.loss_integration_point == "byo_trainer_command"
    assert result.to_dict()["policy_delta_l2"] == 0.5

    for missing in ("reward_head_after", "policy_output_after", "policy_delta_l2"):
        payload = {
            "reward_head_after": 0.3,
            "policy_output_after": [0.1],
            "policy_delta_l2": 0.5,
        }
        del payload[missing]
        with pytest.raises(PolicyContainerError):
            VlmSignalUpdateResult.from_dict(payload)

    with pytest.raises(PolicyContainerError):
        VlmSignalUpdateResult.from_dict(
            {
                "reward_head_after": 0.3,
                "policy_output_after": [],
                "policy_delta_l2": 0.5,
            }
        )


def test_sim2real_component_workflows_target_rtx_pro_6000() -> None:
    actions = [
        doc
        for doc in yaml.safe_load_all(SIM2REAL_ACTIONS.read_text(encoding="utf-8"))
        if doc is not None
    ]
    envgen = [
        doc
        for doc in yaml.safe_load_all(SIM2REAL_ENVGEN_SPLIT.read_text(encoding="utf-8"))
        if doc is not None
    ]

    assert actions[0]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert envgen[0]["resources"]["accelerators"] == "RTXPRO6000:1"


def test_cosmos_split_sdk_and_raw_yaml_contracts() -> None:
    transfer = cosmos2.transfer(
        input_uri="s3://bucket/input/",
        output_uri="s3://bucket/augment/",
        image="npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z",
    )
    reason = cosmos3.reason(
        input_uri="s3://bucket/rollouts/",
        output_uri="s3://bucket/vlm_eval/",
        image="npa-cosmos3-reason:3.0.0",
    )

    assert transfer["schema"] == "npa.cosmos2.transfer.v1"
    assert reason["schema"] == "npa.cosmos3.reason.v1"
    assert transfer["image"] == "npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"
    assert reason["image"] == "npa-cosmos3-reason:3.0.0"
    assert transfer["image"] != reason["image"]
    assert "cosmos3" not in transfer["image"]
    assert "cosmos2" not in reason["image"]
    assert "cosmos2-transfer" in COSMOS2_TRANSFER.read_text(encoding="utf-8")
    assert "cosmos3-reason" in COSMOS3_REASON.read_text(encoding="utf-8")


def test_parallel_vlm_eval_caps_sibling_job_concurrency(monkeypatch, tmp_path: Path) -> None:
    import threading

    import npa.workflows.sim2real.engine as engine_module

    active = 0
    peak = 0
    lock = threading.Lock()
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        nonlocal active, peak
        calls.append({"cmd": cmd, "input": kwargs.get("input")})
        if "apply" in cmd:
            with lock:
                active += 1
                peak = max(peak, active)
            return subprocess.CompletedProcess(cmd, 0, "job.batch/sibling created\n", "")
        if "get" in cmd and "job" in cmd and "jsonpath" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "1 0", "")
        if "logs" in cmd:
            return subprocess.CompletedProcess(cmd, 0, '{"component":"ok"}\n', "")
        if "delete" in cmd:
            with lock:
                active = max(0, active - 1)
            return subprocess.CompletedProcess(cmd, 0, "job.batch deleted\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(engine_module.subprocess, "run", fake_run)
    monkeypatch.setattr(engine_module, "_wait_kubernetes_job", lambda *a, **k: "complete")

    storage = _FakeComponentStorage({})
    monkeypatch.setattr(
        engine_module.StorageClient,
        "from_environment",
        classmethod(lambda cls, endpoint_url="": storage),
    )

    def fake_download(config, output_uri, output_path):
        rollout_id = output_path.stem
        payload = {
            "schema": SCHEMA_VLM_EVAL,
            "rollout_id": rollout_id,
            "success": False,
            "score": 0.5,
            "per_step": [
                {
                    "step": 0,
                    "critique_text": "parallel sibling eval",
                    "error_tags": ["minor_alignment"],
                    "action": [0.0, 0.0, 0.0],
                    "camera_observation": "camera-000.ppm",
                }
            ],
            "summary": "parallel sibling",
            "model": "job-vlm",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    monkeypatch.setattr(engine_module, "_download_component_output", fake_download)

    config = Sim2RealLoopConfig(
        run_id="parallel-vlm",
        output_dir=tmp_path,
        s3_bucket="bucket",
        s3_prefix="neutral-prefix",
        s3_endpoint="https://storage.example",
        threshold=0.75,
        rollout_count=3,
        steps_per_rollout=1,
        inner_iterations=1,
        k8s_max_parallel_gpus=2,
        vlm_dual_reason=False,
        k8s_namespace="default",
    )
    rollouts = generate_action_rollouts(
        tmp_path / "actions",
        count=3,
        steps_per_rollout=1,
        seed=3,
        quality=0.4,
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real_stages.run_policy_rollouts",
        lambda *args, **kwargs: rollouts,
    )

    evidence = engine_module.run_inner_loop(
        config,
        local_dir=tmp_path,
        initial_quality=0.4,
    )

    apply_calls = [call for call in calls if "apply" in call["cmd"]]
    assert len(apply_calls) == 3
    assert peak <= 2
    assert len(evidence["iterations"]) == 1
    assert evidence["iterations"][0]["sample_vlm_eval"]["schema"] == SCHEMA_VLM_EVAL


def test_wait_kubernetes_job_honors_required_successes(monkeypatch) -> None:
    import npa.workflows.sim2real.engine as engine_module
    import subprocess

    monkeypatch.setattr(
        engine_module,
        "_kubectl",
        lambda config, args, **kwargs: subprocess.CompletedProcess(args, 0, "2 0", ""),
    )
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        engine_module._wait_kubernetes_job(
            config,
            namespace="default",
            job_name="j",
            timeout_s=10,
            required_successes=3,
        )
        == "timeout"
    )
    monkeypatch.setattr(
        engine_module,
        "_kubectl",
        lambda config, args, **kwargs: subprocess.CompletedProcess(args, 0, "3 0", ""),
    )
    assert (
        engine_module._wait_kubernetes_job(
            config,
            namespace="default",
            job_name="j",
            timeout_s=10,
            required_successes=3,
        )
        == "complete"
    )


def test_engine_wait_kubernetes_job_not_found_skips_long_wait(monkeypatch) -> None:
    import npa.workflows.sim2real.engine as engine_module
    import subprocess

    calls: list[list[str]] = []

    def fake_kubectl(config, args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args,
            1,
            "",
            "Error from server (NotFound): jobs \"j\" not found",
        )

    monkeypatch.setattr(engine_module, "_kubectl", fake_kubectl)
    config = Sim2RealLoopConfig(run_id="r")
    assert (
        engine_module._wait_kubernetes_job(
            config,
            namespace="default",
            job_name="j",
            timeout_s=10800,
        )
        == "failed"
    )
    assert calls[0][:3] == ["get", "job", "j"]
    assert not any(call[0] == "wait" for call in calls)


def test_cosmos2_transfer_component_uploads_result_json_to_explicit_uri(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.workflows.sim2real.engine import run_cosmos2_transfer_component_from_s3

    uploads: list[tuple[str, str]] = []

    class FakeClient:
        def upload_file(self, local_file: str, bucket_uri: str) -> str:
            uploads.append((Path(local_file).name, bucket_uri))
            return bucket_uri

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        lambda: FakeClient(),
    )

    result_uri = "s3://bucket/run/augment/cosmos2-transfer-result.json"
    run_cosmos2_transfer_component_from_s3(
        input_uri="s3://bucket/trigger/",
        output_uri=result_uri,
        augmented_frames_uri="s3://bucket/run/augment/frames/",
    )

    assert ("cosmos2-transfer-result.json", result_uri) in uploads
    manifest_uploads = [uri for name, uri in uploads if name == "cosmos2-transfer-manifest.json"]
    assert manifest_uploads
    assert manifest_uploads[0].endswith("/augment/manifest.json")


def test_byo_policy_rollout_passes_component(monkeypatch, tmp_path) -> None:
    """Regression: the --byo-policy-command path must pass component= to
    _run_component_command (it was omitted, raising TypeError mid-run)."""

    captured = {}

    def _fake_run_component_command(command, *, cwd, env, component, **kwargs):
        captured["component"] = component
        captured["command"] = command
        return {"ok": True}

    def _fake_read_component_json(path, invocation):
        return {"rollout_dirs": [str(tmp_path / "rollout-0000")]}

    monkeypatch.setattr(loop_module, "_run_component_command", _fake_run_component_command)
    monkeypatch.setattr(loop_module, "_read_component_json", _fake_read_component_json)
    config = Sim2RealLoopConfig(
        run_id="r",
        byo_policy_command="python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout",
    )
    out = loop_module._run_policy_rollouts_via_command(
        config,
        actions_dir=tmp_path,
        outer_iteration=1,
        iteration=1,
        train_envs_uri="s3://b/run/envs/train/envs.jsonl",
    )
    assert captured["component"] == "policy_actions"
    assert [str(p) for p in out] == [str(tmp_path / "rollout-0000")]
