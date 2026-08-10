"""Tests for the BYO Isaac held-out eval (rolls the TRAINED policy)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from npa.workflows.sim2real import byo_isaac_eval as ev
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args


def _manifest_script(manifest):
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return decode_compressed_bash_args(container["args"])


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


def test_policy_inference_provenance_requires_learned_actor_only() -> None:
    provenance = ev.policy_inference_provenance(
        checkpoint_uri="s3://bucket/model.pt",
        checkpoint={"sha256": "a" * 64, "size_bytes": 123},
    )
    assert provenance["actor_is_learned"] is True
    assert provenance["policy_composition"] == "learned_actor_only"
    assert provenance["scripted_post_actor_controller"] is False
    assert provenance["post_actor_controller"] is None


def test_per_env_from_distances_scoring():
    rows = ev.per_env_from_distances([0.0, 0.05, 0.2], success_dist_m=0.05)
    assert rows[0]["success"] is True and rows[0]["score"] == 1.0
    assert rows[1]["success"] is False  # 0.05 is not < 0.05
    assert rows[2]["success"] is False and rows[2]["score"] == 0.0
    assert rows[0]["details"]["object_goal_distance_m"] == 0.0


def test_build_isaac_eval_job_manifest_shape():
    m = ev.build_isaac_eval_job_manifest(
        job_name="s2r-byo-isaac-eval-run1",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4,
        checkpoint_uri="s3://b/run1/model_latest.pt",
        per_env_s3_uri="s3://b/sim2real-b/run1/byo-eval/job/per_env_distances.json",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["image"].endswith("npa-isaac-lab:2.3.2.post1")
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "1"
    args = decode_compressed_bash_args(c["args"])
    assert max(map(len, c["args"])) < 128 * 1024
    assert "Isaac-Lift-Cube-Franka-v0" in args
    assert "s3://b/run1/model_latest.pt" in args  # downloads the checkpoint
    assert "/opt/npa/isaac-runtime/isaac_eval.py" in args
    assert "npa.workflows.sim2real.runtime_attestation" in args
    assert "npa.workflows.sim2real.isaac_job_io download" in args
    assert "npa.workflows.sim2real.isaac_job_io upload" in args
    assert "pip install" not in args
    assert "<<" not in args
    assert "per_env_distances.json" in args  # uploads measured distances
    assert "--portable-root /tmp/npa-isaac-kit" in ev.ISAAC_EVAL_SCRIPT
    assert "kit_args=os.environ.get(" in ev.ISAAC_EVAL_SCRIPT
    assert "min_speed_in_strict_basin_mps" in ev.ISAAC_EVAL_SCRIPT
    assert "max_consecutive_strict_stable_steps" in ev.ISAAC_EVAL_SCRIPT
    assert "PLACEMENT_POST_SUCCESS_HOLD_LATCHED" not in ev.ISAAC_EVAL_SCRIPT
    assert "settle_hold_trigger(" not in ev.ISAAC_EVAL_SCRIPT
    assert "joint_position_hold_action(" not in ev.ISAAC_EVAL_SCRIPT


def test_first_episode_masks_seal_auto_reset_state():
    completed = np.array([False, False, True], dtype=bool)
    done = np.array([True, False, False], dtype=bool)

    active, newly_terminal, next_completed = ev.first_episode_masks(completed, done)

    assert active.tolist() == [True, True, False]
    assert newly_terminal.tolist() == [True, False, False]
    assert next_completed.tolist() == [True, False, True]


def test_eval_runtime_freezes_terminal_metrics_and_renders():
    script = ev.ISAAC_EVAL_SCRIPT
    assert '"final_dist": final_dist.copy()' in script
    assert "active, newly_terminal, completed = first_episode_masks" in script
    assert 'final_dist[newly_terminal] = prior["final_dist"]' in script
    assert "capture(_step, active & ~newly_terminal)" in script
    assert "capture(STEPS, ~completed)" in script
    assert '"terminal_snapshot": "first_episode_last_pre_reset"' in script
    assert "settle_trigger" not in script
    assert "settle_hold_actions" not in script


def test_eval_manifest_uses_sha_pinned_s3_scenario_transport():
    digest = "a" * 64
    uri = f"s3://b/run/scenario-input/{digest}.jsonl"
    manifest = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab@sha256:" + "b" * 64,
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64,
        checkpoint_uri="s3://b/model.pt",
        per_env_s3_uri="s3://b/out.json",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        scenarios_uri=uri,
        scenarios_sha256=digest,
    )
    script = _manifest_script(manifest)
    assert f"--uri {uri}" in script
    assert f"--sha256 {digest}" in script
    assert "--destination /tmp/evalwork/scenarios.jsonl" in script
    assert "NPA_SIM2REAL_SCENARIOS_JSONL=/tmp/evalwork/scenarios.jsonl" in script
    assert "write-base64" not in script
    assert manifest["metadata"]["annotations"] == {
        "sim2real.npa.dev/scenarios-uri": uri,
        "sim2real.npa.dev/scenarios-sha256": digest,
    }


def test_eval_manifest_rejects_oversized_embedded_scenarios():
    with pytest.raises(ValueError, match="require scenarios_uri"):
        ev.build_isaac_eval_job_manifest(
            job_name="j",
            run_id="r",
            image="reg/npa-isaac-lab:tag",
            task="Isaac-Lift-Cube-Franka-v0",
            num_envs=64,
            checkpoint_uri="s3://b/model.pt",
            per_env_s3_uri="s3://b/out.json",
            s3_endpoint="https://s3.example",
            namespace="default",
            service_account="agent-sa",
            gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            scenarios_jsonl="x" * (ev._MAX_EMBEDDED_SCENARIOS_BYTES + 1),
        )


def test_publish_eval_scenarios_is_content_addressed():
    rows = [
        {
            "env_id": "validation-easy",
            "seed": 7,
            "difficulty": "easy",
            "scenario_config_digest": "cfg-easy",
        }
    ]
    uploaded: dict[str, object] = {}

    class FakeStorage:
        def upload_file(self, source: str, uri: str) -> str:
            uploaded["bytes"] = Path(source).read_bytes()
            uploaded["uri"] = uri
            return uri

    provenance = ev.publish_eval_scenarios(
        rows,
        destination_prefix="s3://bucket/run/scenario-input",
        storage=FakeStorage(),  # type: ignore[arg-type]
    )
    expected = uploaded["bytes"]
    assert isinstance(expected, bytes)
    digest = hashlib.sha256(expected).hexdigest()
    assert provenance == {
        "uri": f"s3://bucket/run/scenario-input/{digest}.jsonl",
        "sha256": digest,
        "size_bytes": len(expected),
        "scenario_count": 1,
        "transport": "s3_sha256",
        "content_addressed": True,
    }
    assert uploaded["uri"] == provenance["uri"]


def test_run_isaac_eval_job_uses_outer_iteration_artifact_tag(monkeypatch):
    captured: dict[str, str] = {}

    def fake_build(**kwargs):
        captured["job_name"] = kwargs["job_name"]
        captured["per_env_s3_uri"] = kwargs["per_env_s3_uri"]
        captured["renders_s3_prefix"] = kwargs["renders_s3_prefix"]
        captured["scenarios_uri"] = kwargs["scenarios_uri"]
        captured["scenarios_sha256"] = kwargs["scenarios_sha256"]
        return {"kind": "Job"}

    monkeypatch.setattr(ev, "build_isaac_eval_job_manifest", fake_build)
    monkeypatch.setattr(
        ev,
        "publish_eval_scenarios",
        lambda *args, **kwargs: {
            "uri": "s3://bkt/scenario-input/" + "c" * 64 + ".jsonl",
            "sha256": "c" * 64,
            "size_bytes": 123,
            "scenario_count": 1,
            "transport": "s3_sha256",
            "content_addressed": True,
        },
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.gpu_fallback.run_gpu_job_with_fallback",
        lambda **kwargs: {
            "job_name": kwargs["base_job_name"],
            "job_uid": "uid",
            "selected_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            "image_digests": ["reg/runtime@sha256:" + "a" * 64],
        },
    )
    monkeypatch.setattr(
        ev,
        "_download_json",
        lambda _uri: {
            "note": "rollout_ok",
            "object_goal_distances": [0.01],
            "per_env_metrics": [{"place": True, "placement_stable": True}],
            "policy_checkpoint": {
                "uri": "s3://bkt/run/model_latest.pt",
                "sha256": "a" * 64,
                "size_bytes": 4096,
            },
            "applied_scenarios": {
                "records": [{"scenario_config_digest": "cfg-1", "applied_count": 1}]
            },
        },
    )
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_IMAGE", "reg/npa-isaac-lab:2.3.2.post1")
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "bkt")
    monkeypatch.setenv("NPA_SIM2REAL_EVAL_TAG", "outer-02")
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")

    rows = ev.run_isaac_eval_job(
        "myrun",
        checkpoint_uri="s3://bkt/run/model_latest.pt",
        num_envs=1,
        generated_envs=[
            {
                "env_id": "gold-1",
                "seed": 7,
                "difficulty": "easy",
                "scenario_config_digest": "cfg-1",
            }
        ],
    )

    assert rows[0]["success"] is True
    assert captured["job_name"].endswith("outer-02")
    assert captured["per_env_s3_uri"].endswith(
        "/byo-eval/s2r-byo-isaac-eval-myrun-outer-02/per_env_distances.json"
    )
    assert captured["renders_s3_prefix"].endswith(
        "/byo-eval/s2r-byo-isaac-eval-myrun-outer-02/renders"
    )
    assert captured["scenarios_uri"].endswith("/" + "c" * 64 + ".jsonl")
    assert captured["scenarios_sha256"] == "c" * 64
    assert ev._APPLIED_SCENARIO_AUDIT["scenario_input_provenance"] == (
        ev._SCENARIO_INPUT_PROVENANCE
    )


def test_dryrun_main_writes_normalizable_report(tmp_path, monkeypatch):
    """Dry-run output must flow through the engine's _normalize_heldout_report."""

    from npa.workflows.sim2real.engine import _normalize_heldout_report
    from npa.workflows.sim2real.config import build_config_from_env

    ev_json = tmp_path / "inner.json"
    ev_json.write_text(
        json.dumps(
            {
                "iterations": [
                    {"update": {"checkpoint_path": "s3://b/run/model_latest.pt"}}
                ]
            }
        )
    )
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
    assert [item["name"] for item in payload["camera_metadata"]] == [
        "primary",
        "side",
        "overhead",
    ]
    assert payload["camera_metadata"][0]["width"] == 640
    assert payload["camera_metadata"][0]["height"] == 480
    # The engine normalizer computes success_rate from per_env (2 of 4 < 0.05m).
    cfg = build_config_from_env(threshold=0.45, s3_bucket="", run_id="t")
    report = _normalize_heldout_report(
        payload, config=cfg, outer_iteration=1, inner_evidence_uri="x", invocation={}
    )
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
        '{"env_id":"env-00001","seed":222}\n',
        encoding="utf-8",
    )
    envs = ev.read_generated_envs(str(d))
    assert [e["env_id"] for e in envs] == ["env-00000", "env-00001"]
    assert envs[0]["seed"] == 111 and envs[1]["seed"] == 222
    assert ev.read_generated_envs(str(tmp_path / "missing")) == []


def test_read_durable_generated_envs_downloads_exact_object(tmp_path, monkeypatch):
    class FakeStorage:
        def download_file(self, source: str, destination: str) -> str:
            assert source == "s3://bucket/run/envs/validation/envs.jsonl"
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                '{"env_id":"validation-000","seed":17}\n', encoding="utf-8"
            )
            return str(target)

    monkeypatch.setattr(
        ev.StorageClient,
        "from_environment",
        classmethod(lambda cls, **_kwargs: FakeStorage()),
    )
    rows = ev.read_durable_generated_envs(
        "s3://bucket/run/envs/validation/envs.jsonl",
        cache_dir=tmp_path / "cache",
    )
    assert rows == [{"env_id": "validation-000", "seed": 17}]

    with pytest.raises(ValueError, match="exact s3:// object"):
        ev.read_durable_generated_envs(
            "s3://bucket/run/envs/validation/",
            cache_dir=tmp_path / "prefix",
        )


def test_main_prefers_durable_s3_scenarios_after_local_cache_loss(
    tmp_path, monkeypatch
):
    durable_rows = [
        {
            "env_id": f"validation-{difficulty}",
            "seed": index + 1,
            "difficulty": difficulty,
            "scenario_config_digest": f"digest-{difficulty}",
        }
        for index, difficulty in enumerate(("easy", "medium", "hard"))
    ]

    class FakeStorage:
        def download_file(self, source: str, destination: str) -> str:
            assert source == "s3://bucket/run/envs/validation/envs.jsonl"
            target = Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(json.dumps(row) + "\n" for row in durable_rows),
                encoding="utf-8",
            )
            return str(target)

    monkeypatch.setattr(
        ev.StorageClient,
        "from_environment",
        classmethod(lambda cls, **_kwargs: FakeStorage()),
    )
    output = tmp_path / "report.json"
    missing_local = tmp_path / "deleted-controller-cache"
    evidence = tmp_path / "inner.json"
    evidence.write_text(
        json.dumps({"selected_checkpoint_uri": "s3://bucket/run/model.pt"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NPA_BYO_ISAAC_DRYRUN", "1")
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(output))
    monkeypatch.setenv("NPA_SIM2REAL_INNER_EVIDENCE_JSON", str(evidence))
    monkeypatch.setenv("NPA_SIM2REAL_HELDOUT_ENVS_DIR", str(missing_local))
    monkeypatch.setenv(
        "NPA_SIM2REAL_HELDOUT_ENVS_URI",
        "s3://bucket/run/envs/validation/envs.jsonl",
    )
    monkeypatch.setenv("NPA_SIM2REAL_EVALUATION_SPLIT", "validation")
    monkeypatch.setenv("NPA_SIM2REAL_HELDOUT_ENV_COUNT", "3")

    assert ev.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scenario_records_source"] == "durable_s3_object"
    assert report["scenario_records_uri"].endswith("/validation/envs.jsonl")
    assert set(report["generated_env_ids"]) == {
        "validation-easy",
        "validation-medium",
        "validation-hard",
    }


def test_per_env_labelled_by_generated_env_id_and_seed():
    rows = ev.per_env_from_distances(
        [0.03, 0.2],
        success_dist_m=0.05,
        env_ids=["env-00007", "env-00008"],
        seeds=[111, 222],
    )
    assert rows[0]["env_id"] == "env-00007"
    assert rows[0]["details"]["generated_env_seed"] == 111
    assert rows[1]["env_id"] == "env-00008" and rows[1]["success"] is False


def test_eval_manifest_embeds_generated_seed():
    m = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=2,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=1744247227,
    )
    args = _manifest_script(m)
    assert 'EVAL_SEED="1744247227"' in args


def test_eval_script_applies_zero_seed_unconditionally():
    seed_block = ev.ISAAC_EVAL_SCRIPT.split(
        "# Drive randomization from the GENERATED env seed", 1
    )[1].split("# Capture synchronized primary", 1)[0]
    assert "if SEED:" not in seed_block
    assert "env_cfg.seed = SEED" in seed_block
    assert "torch.manual_seed(SEED)" in seed_block
    assert "np.random.seed(SEED % (2**32))" in seed_block
    assert 'print("EVAL_SEED_APPLIED", SEED' in seed_block


def test_eval_script_uses_oblique_workspace_camera_for_renders():
    script = ev.ISAAC_EVAL_SCRIPT
    assert "for view in CAMERA_VIEWS" in script
    assert '"heldout_cam" if name == "primary"' in script
    assert '"heldout_cam_" + name' in script
    assert 'convention="world"' in script
    assert 'EVAL_CAPTURE_WIDTH", "640"' in script
    assert 'EVAL_CAPTURE_HEIGHT", "480"' in script
    assert "width=CAPTURE_WIDTH" in script and "height=CAPTURE_HEIGHT" in script
    assert "clipping_range=(0.05, 20.0)" in script


def test_eval_manifest_enables_default_multi_camera_views():
    m = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=2,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    args = _manifest_script(m)
    assert "EVAL_CAMERA_VIEWS_JSON=" in args
    assert '"name":"primary"' in args
    assert '"name":"side"' in args
    assert '"name":"overhead"' in args


def test_eval_manifest_embeds_custom_object_usd():
    m = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=2,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="http://assets/custom.usd",
    )
    args = _manifest_script(m)
    assert 'EVAL_OBJECT_USD="http://assets/custom.usd"' in args


def _byo_manifest_args(**kw):
    m = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=2,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        robot_spec={
            "name": "lite6",
            "robot_source": "byo_usd",
            "usd_path": "/tmp/npa_robot/robot.usd",
        },
        **kw,
    )
    return _manifest_script(m)


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
    assert "export NPA_BYO_TASK_CONFIG_JSON=" not in _byo_manifest_args(
        task_config=None
    )
    m = ev.build_isaac_eval_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=2,
        checkpoint_uri="s3://b/m.pt",
        per_env_s3_uri="s3://b/o/d.json",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        task_config={"object_scale": 0.2},
    )  # no robot_spec -> Franka path
    args = _manifest_script(m)
    assert "export NPA_BYO_TASK_CONFIG_JSON=" not in args


def test_normalize_heldout_preserves_render_manifest_and_provenance():
    """BYO-eval render_manifest + provenance must survive engine normalization."""
    from npa.workflows.sim2real.engine import _normalize_heldout_report
    from npa.workflows.sim2real.config import build_config_from_env

    payload = {
        "per_env": [{"env_id": "env-00000", "success": True, "score": 0.9}],
        "render_manifest": {
            "schema": "npa.sim2real.heldout_renders.v1",
            "episodes": [{"env_id": "env-00000", "frames": ["camera-0000.png"]}],
        },
        "policy_checkpoint": "s3://b/run/model_latest.pt",
        "deployable_policy_eval": True,
        "generated_envs_tested": 1,
        "generated_env_ids": ["env-00000"],
    }
    cfg = build_config_from_env(threshold=0.45, s3_bucket="", run_id="t")
    report = _normalize_heldout_report(
        payload, config=cfg, outer_iteration=1, inner_evidence_uri="x", invocation={}
    )
    assert report["render_manifest"]["episodes"][0]["env_id"] == "env-00000"
    assert report["policy_checkpoint"].endswith("model_latest.pt")
    assert report["deployable_policy_eval"] is True
    assert report["generated_envs_tested"] == 1


def test_build_heldout_report_multi_threshold_success_summary():
    from npa.workflows.sim2real import byo_isaac_eval as ev

    per_env = ev.per_env_from_distances(
        [0.03, 0.08, 0.12, 0.40], success_dist_m=0.05, env_ids=["e0", "e1", "e2", "e3"]
    )
    report = ev.build_heldout_report(
        per_env,
        isaac_task="Isaac-Lift-Cube-Franka-v0",
        checkpoint_uri="s3://b/model_latest.pt",
        source="byo_isaac_eval",
    )
    s = report["success_summary"]
    assert s["success@0.05"] == 0.25  # only 0.03 < 0.05
    assert s["success@0.10"] == 0.50  # 0.03, 0.08
    assert s["success@0.15"] == 0.75  # 0.03, 0.08, 0.12
    assert s["min_object_goal_distance_m"] == 0.03
    assert report["success_rate"] == 0.25
    assert report["strict_success"]["rate"] == report["success_rate"]
    assert report["per_env"][0]["success"] is True
