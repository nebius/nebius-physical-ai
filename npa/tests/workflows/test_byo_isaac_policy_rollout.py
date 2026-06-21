"""Tests for the BYO Isaac policy rollout (closes the loop — rolls the policy)."""

from __future__ import annotations

import json

from npa.workflows.sim2real import byo_isaac_policy_rollout as pr


def test_build_rollout_manifest_matches_action_rollout_schema():
    m = pr.build_rollout_manifest(
        rollout_id="rollout-0001",
        frames=["camera-000.png", "camera-001.png"],
        actions=[{"step": 0, "action": [0.1, 0.2]}, {"step": 1, "action": [0.0, 0.0]}],
        checkpoint_uri="s3://b/run/byo-trainer/job/model_latest.pt",
        is_trained=True,
    )
    assert m["schema"] == "npa.sim2real.action_rollout.v1"
    assert m["rollout_id"] == "rollout-0001"
    assert m["steps"] == 2
    assert m["camera_observations"] == ["camera-000.png", "camera-001.png"]
    assert len(m["actions"]) == 2
    # Provenance: real Isaac policy rollout, not the synthetic stub.
    assert m["source"] == "byo_isaac_policy_rollout"
    assert m["policy_trained"] is True
    assert m["policy_checkpoint"].endswith("model_latest.pt")


def test_latest_checkpoint_uri_empty_inputs():
    assert pr.latest_checkpoint_uri("", "run") == ""
    assert pr.latest_checkpoint_uri("bucket", "") == ""


def test_write_dryrun_rollouts_layout(tmp_path):
    dirs = pr.write_dryrun_rollouts(
        tmp_path, count=3, steps_per_rollout=4, checkpoint_uri="")
    assert len(dirs) == 3
    for d in dirs:
        from pathlib import Path

        rdir = Path(d)
        assert (rdir / "manifest.json").is_file()
        man = json.loads((rdir / "manifest.json").read_text())
        assert man["schema"] == "npa.sim2real.action_rollout.v1"
        assert man["steps"] == 4
        # untrained (no checkpoint) -> policy_trained False
        assert man["policy_trained"] is False
        assert len(man["camera_observations"]) == 4
        # frames physically written
        for name in man["camera_observations"]:
            assert (rdir / name).is_file()


def test_dryrun_main_writes_rollout_dirs_json(tmp_path, monkeypatch):
    out_json = tmp_path / "byo-policy-rollouts.json"
    out_dir = tmp_path / "actions"
    monkeypatch.setenv("NPA_BYO_ISAAC_DRYRUN", "1")
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out_json))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("NPA_SIM2REAL_ROLLOUT_COUNT", "2")
    monkeypatch.setenv("NPA_SIM2REAL_STEPS_PER_ROLLOUT", "3")
    monkeypatch.delenv("NPA_SIM2REAL_BUCKET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    rc = pr.main()
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["schema"] == "npa.sim2real.policy_rollouts.v1"
    assert len(payload["rollout_dirs"]) == 2
    # The engine consumes rollout_dirs as real dirs with manifests.
    from pathlib import Path

    for d in payload["rollout_dirs"]:
        assert (Path(d) / "manifest.json").is_file()


def test_build_isaac_rollout_job_manifest_shape():
    m = pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-run1-iter0", run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1", task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=4, steps_per_rollout=8,
        checkpoint_uri="s3://b/run1/byo-trainer/j/model_latest.pt",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/iter0",
        s3_endpoint="https://s3.example", namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="https://example/multi_color_cube_instanceable.usd",
    )
    assert m["kind"] == "Job"
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["image"] == "reg/npa-isaac-lab:2.3.2.post1"
    script = c["args"][0]
    # downloads the checkpoint, applies the custom object, runs the rollout script.
    assert "DOWNLOADED_CKPT" in script
    assert "ROLLOUT_OBJECT_USD" in script
    assert "rollout.py" in script
    assert m["spec"]["backoffLimit"] == 0


def test_untrained_job_manifest_skips_download():
    m = pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-run1-iter0", run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1", task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=2, steps_per_rollout=4, checkpoint_uri="",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/iter0",
        s3_endpoint="", namespace="default", service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    script = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "DOWNLOADED_CKPT" not in script  # no checkpoint -> untrained policy
    assert 'ROLLOUT_CKPT_LOCAL=""' in script
