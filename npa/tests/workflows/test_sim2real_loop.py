from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from npa.sdk.workbench import cosmos2, cosmos3, sim2real
from npa.workbench.lerobot.policy_container import (
    parse_vlm_signal_batch,
    run_vlm_signal_training_step,
)
from npa.workflows.sim2real_loop import (
    SCHEMA_RL_SIGNAL,
    SCHEMA_VLM_EVAL,
    Sim2RealLoopConfig,
    artifact_uris,
    convert_vlm_eval_to_rl_signal,
    evaluate_rollout_with_vlm,
    generate_action_rollouts,
    run_full_loop,
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
    assert (
        task["envs"]["NPA_SIM2REAL_TRIGGER_DATASET_URI"]
        == "${NPA_SIM2REAL_TRIGGER_DATASET_URI}"
    )
    assert (
        task["envs"]["NPA_SIM2REAL_TRIGGER_DATASET_ID"]
        == "${NPA_SIM2REAL_TRIGGER_DATASET_ID}"
    )
    assert task["envs"]["VLM_IMAGE"] == "${VLM_IMAGE}"
    assert task["envs"]["TRAINER_IMAGE"] == "${TRAINER_IMAGE}"
    assert task["envs"]["EVAL_IMAGE"] == "${EVAL_IMAGE}"
    assert "npa.workflows.sim2real_loop full-loop" in task["run"]
    assert "--trigger-dataset-uri" in task["run"]
    assert "--byo-signal-converter" in task["run"]


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
