"""Tests for the BYO Isaac held-out eval (rolls the TRAINED policy)."""

from __future__ import annotations

import json

from npa.workflows.sim2real import byo_isaac_eval as ev


def test_extract_checkpoint_uri_from_inner_evidence():
    evidence = {
        "iterations": [
            {"update": {"checkpoint_path": "s3://b/run/it0/model_latest.pt"}},
            {"update": {"checkpoint_path": "s3://b/run/it1/model_latest.pt"}},
        ]
    }
    assert ev.extract_checkpoint_uri(evidence) == "s3://b/run/it1/model_latest.pt"


def test_extract_checkpoint_uri_absent_returns_empty():
    assert ev.extract_checkpoint_uri({"iterations": [{"update": {}}]}) == ""
    assert ev.extract_checkpoint_uri({}) == ""


def test_per_env_from_distances_scoring():
    rows = ev.per_env_from_distances([0.0, 0.05, 0.2], success_dist_m=0.05)
    assert rows[0]["success"] is True and rows[0]["score"] == 1.0
    assert rows[1]["success"] is False  # 0.05 is not < 0.05
    assert rows[2]["success"] is False and rows[2]["score"] == 0.0
    assert rows[0]["details"]["object_goal_distance_m"] == 0.0


def test_build_isaac_eval_job_manifest_shape():
    m = ev.build_isaac_eval_job_manifest(
        job_name="s2r-byo-isaac-eval-run1", run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4, checkpoint_uri="s3://b/run1/model_latest.pt",
        per_env_s3_uri="s3://b/sim2real-b/run1/byo-eval/job/per_env_distances.json",
        s3_endpoint="https://s3.example", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["image"].endswith("npa-isaac-lab:2.3.2.post1")
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    args = c["args"][0]
    assert "Isaac-Lift-Cube-Franka-v0" in args
    assert "s3://b/run1/model_latest.pt" in args      # downloads the checkpoint
    assert "eval_rollout.py" in args                  # runs the policy rollout
    assert "per_env_distances.json" in args           # uploads measured distances


def test_dryrun_main_writes_normalizable_report(tmp_path, monkeypatch):
    """Dry-run output must flow through the engine's _normalize_heldout_report."""

    from npa.workflows.sim2real.engine import _normalize_heldout_report
    from npa.workflows.sim2real.config import build_config_from_env

    ev_json = tmp_path / "inner.json"
    ev_json.write_text(json.dumps(
        {"iterations": [{"update": {"checkpoint_path": "s3://b/run/model_latest.pt"}}]}))
    out = tmp_path / "report.json"
    monkeypatch.setenv("NPA_BYO_ISAAC_DRYRUN", "1")
    monkeypatch.setenv("NPA_SIM2REAL_INNER_EVIDENCE_JSON", str(ev_json))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out))
    monkeypatch.setenv("NPA_SIM2REAL_HELDOUT_ENV_COUNT", "4")
    rc = ev.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["policy_checkpoint"] == "s3://b/run/model_latest.pt"
    assert payload["deployable_policy_eval"] is True
    assert len(payload["per_env"]) == 4
    # The engine normalizer computes success_rate from per_env (2 of 4 < 0.05m).
    cfg = build_config_from_env(threshold=0.45, s3_bucket="", run_id="t")
    report = _normalize_heldout_report(
        payload, config=cfg, outer_iteration=1, inner_evidence_uri="x", invocation={})
    assert 0.0 <= report["success_rate"] <= 1.0
    assert report["success_rate"] == 0.5  # distances 0.02,0.04 pass; 0.08,0.12 fail


def test_dryrun_refuses_without_checkpoint(tmp_path, monkeypatch):
    """No trained checkpoint => must NOT fabricate success (returns nonzero)."""

    ev_json = tmp_path / "inner.json"
    ev_json.write_text(json.dumps({"iterations": [{"update": {}}]}))
    monkeypatch.delenv("NPA_BYO_ISAAC_DRYRUN", raising=False)
    monkeypatch.setenv("NPA_SIM2REAL_INNER_EVIDENCE_JSON", str(ev_json))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(tmp_path / "r.json"))
    assert ev.main() == 3


def test_read_generated_envs(tmp_path):
    d = tmp_path / "heldout"
    d.mkdir()
    (d / "envs.jsonl").write_text(
        '{"env_id":"env-00000","seed":111,"scene":{"simready_asset":"a"}}\n'
        '{"env_id":"env-00001","seed":222}\n', encoding="utf-8")
    envs = ev.read_generated_envs(str(d))
    assert [e["env_id"] for e in envs] == ["env-00000", "env-00001"]
    assert envs[0]["seed"] == 111 and envs[1]["seed"] == 222
    assert ev.read_generated_envs(str(tmp_path / "missing")) == []


def test_per_env_labelled_by_generated_env_id_and_seed():
    rows = ev.per_env_from_distances(
        [0.03, 0.2], success_dist_m=0.05,
        env_ids=["env-00007", "env-00008"], seeds=[111, 222])
    assert rows[0]["env_id"] == "env-00007"
    assert rows[0]["details"]["generated_env_seed"] == 111
    assert rows[1]["env_id"] == "env-00008" and rows[1]["success"] is False


def test_eval_manifest_embeds_generated_seed():
    m = ev.build_isaac_eval_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=2, checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=1744247227)
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'EVAL_SEED="1744247227"' in args


def test_eval_script_uses_oblique_workspace_camera_for_renders():
    script = ev.ISAAC_EVAL_SCRIPT
    assert "pos=(-2.0, 0.0, 1.0)" in script
    assert "rot=(0.9945, 0.0, 0.1045, 0.0)" in script
    assert 'convention="world"' in script
    assert "width=256" in script and "height=256" in script
    assert "clipping_range=(0.05, 20.0)" in script


def test_eval_manifest_embeds_custom_object_usd():
    m = ev.build_isaac_eval_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=2, checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="http://assets/custom.usd")
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert 'EVAL_OBJECT_USD="http://assets/custom.usd"' in args


def _byo_manifest_args(**kw):
    m = ev.build_isaac_eval_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=2, checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        robot_spec={"name": "lite6", "robot_source": "byo_usd",
                    "usd_path": "/tmp/npa_robot/robot.usd"},
        **kw)
    return m["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_eval_manifest_forwards_task_config_object_scale():
    # A BYO-robot eval given a task config injects NPA_BYO_TASK_CONFIG_JSON so the
    # eval sibling's register() sizes the manipuland to the SAME scale as training
    # (else a shrunk-object policy is scored on the stock cube and reports a false 0).
    args = _byo_manifest_args(task_config={"object_scale": 0.2, "gripper_open": 0.0089})
    # Assert the actual env export line (the module source, cat'd into the sibling,
    # also mentions the constant name, so match the `export ...=` form specifically).
    assert "export NPA_BYO_TASK_CONFIG_JSON=" in args
    assert '"object_scale": 0.2' in args  # json.dumps(sort_keys=True)


def test_eval_manifest_no_task_config_no_injection():
    # BYO robot but no task config -> the env var is not exported (stock placement);
    # and the Franka/no-robot path never exports it at all.
    assert "export NPA_BYO_TASK_CONFIG_JSON=" not in _byo_manifest_args(task_config=None)
    m = ev.build_isaac_eval_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=2, checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        task_config={"object_scale": 0.2})  # no robot_spec -> Franka path
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "export NPA_BYO_TASK_CONFIG_JSON=" not in args


def test_normalize_heldout_preserves_render_manifest_and_provenance():
    """BYO-eval render_manifest + provenance must survive engine normalization."""
    from npa.workflows.sim2real.engine import _normalize_heldout_report
    from npa.workflows.sim2real.config import build_config_from_env
    payload = {
        "per_env": [{"env_id": "env-00000", "success": True, "score": 0.9}],
        "render_manifest": {"schema": "npa.sim2real.heldout_renders.v1",
                            "episodes": [{"env_id": "env-00000", "frames": ["camera-0000.png"]}]},
        "policy_checkpoint": "s3://b/run/model_latest.pt",
        "deployable_policy_eval": True,
        "generated_envs_tested": 1,
        "generated_env_ids": ["env-00000"],
    }
    cfg = build_config_from_env(threshold=0.45, s3_bucket="", run_id="t")
    report = _normalize_heldout_report(payload, config=cfg, outer_iteration=1,
                                       inner_evidence_uri="x", invocation={})
    assert report["render_manifest"]["episodes"][0]["env_id"] == "env-00000"
    assert report["policy_checkpoint"].endswith("model_latest.pt")
    assert report["deployable_policy_eval"] is True
    assert report["generated_envs_tested"] == 1


def test_build_heldout_report_multi_threshold_success_summary():
    from npa.workflows.sim2real import byo_isaac_eval as ev

    per_env = ev.per_env_from_distances(
        [0.03, 0.08, 0.12, 0.40], success_dist_m=0.05,
        env_ids=["e0", "e1", "e2", "e3"])
    report = ev.build_heldout_report(
        per_env, isaac_task="Isaac-Lift-Cube-Franka-v0",
        checkpoint_uri="s3://b/model_latest.pt", source="byo_isaac_eval")
    s = report["success_summary"]
    assert s["success@0.05"] == 0.25   # only 0.03 < 0.05
    assert s["success@0.10"] == 0.50   # 0.03, 0.08
    assert s["success@0.15"] == 0.75   # 0.03, 0.08, 0.12
    assert s["min_object_goal_distance_m"] == 0.03
    assert report["per_env"][0]["success"] is True
