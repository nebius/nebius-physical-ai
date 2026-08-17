"""Tests for the BYO Isaac policy rollout (closes the loop — rolls the policy)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from npa.workflows.sim2real import byo_isaac_policy_rollout as pr
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args


def _manifest_script(manifest):
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return decode_compressed_bash_args(container["args"])


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
    assert m["camera_views"] == {"primary": ["camera-000.png", "camera-001.png"]}
    assert len(m["actions"]) == 2
    # Provenance: real Isaac policy rollout, not the synthetic stub.
    assert m["source"] == "byo_isaac_policy_rollout"
    assert m["policy_trained"] is True
    assert m["policy_checkpoint"].endswith("model_latest.pt")


def test_build_rollout_manifest_keeps_primary_compatibility_and_named_views():
    views = {
        "primary": ["camera-000.png"],
        "side": ["camera-side-000.png"],
        "overhead": ["camera-overhead-000.png"],
    }
    manifest = pr.build_rollout_manifest(
        rollout_id="rollout-0001",
        frames=views["primary"],
        camera_views=views,
        actions=[{"step": 0, "action": [0.1]}],
        checkpoint_uri="s3://bucket/model_latest.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_size_bytes=12345,
        simulation_device="cpu",
        is_trained=True,
        capture={"width": 640, "height": 480, "fps": 10.0},
        camera_metadata_items=[
            {
                "name": "primary",
                "position": [-2.0, 0.0, 1.0],
                "width": 640,
                "height": 480,
            }
        ],
        frame_metadata={
            "primary": [
                {
                    "view_name": "primary",
                    "frame_index": 0,
                    "sim_step": 0,
                    "timestamp_s": 0.0,
                    "checkpoint_uri": "s3://bucket/model_latest.pt",
                }
            ]
        },
    )
    assert manifest["camera_observations"] == views["primary"]
    assert manifest["camera_views"] == views
    assert manifest["capture"]["width"] == 640
    assert manifest["camera_metadata"][0]["name"] == "primary"
    assert manifest["camera_frame_metadata"]["primary"][0]["checkpoint_uri"].endswith(
        "model_latest.pt"
    )
    assert manifest["policy_checkpoint_sha256"] == "b" * 64
    assert manifest["policy_checkpoint_size_bytes"] == 12345
    assert manifest["simulation_device"] == "cpu"


def test_latest_checkpoint_uri_empty_inputs():
    assert pr.latest_checkpoint_uri("", "run") == ""
    assert pr.latest_checkpoint_uri("bucket", "") == ""


def test_write_dryrun_rollouts_layout(tmp_path):
    dirs = pr.write_dryrun_rollouts(
        tmp_path, count=3, steps_per_rollout=4, checkpoint_uri=""
    )
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
        job_name="s2r-byo-isaac-roll-run1-iter0",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=4,
        steps_per_rollout=8,
        checkpoint_uri="s3://b/run1/byo-trainer/j/model_latest.pt",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/iter0",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="https://example/multi_color_cube_instanceable.usd",
    )
    assert m["kind"] == "Job"
    c = m["spec"]["template"]["spec"]["containers"][0]
    assert c["image"] == "reg/npa-isaac-lab:2.3.2.post1"
    script = decode_compressed_bash_args(c["args"])
    assert max(map(len, c["args"])) < 128 * 1024
    # downloads the checkpoint, applies the custom object, runs the rollout script.
    assert "npa.workflows.sim2real.isaac_job_io download" in script
    assert "ROLLOUT_OBJECT_USD" in script
    assert "ROLLOUT_CAMERA_VIEWS_JSON=" in script
    assert 'ROLLOUT_CAPTURE_WIDTH="640"' in script
    assert 'ROLLOUT_CAPTURE_HEIGHT="480"' in script
    assert "ROLLOUT_SIM_DEVICE=cuda:0" in script
    assert '"name":"side"' in script
    assert '"name":"overhead"' in script
    assert "/opt/npa/isaac-runtime/isaac_rollout.py" in script
    assert "npa.workflows.sim2real.runtime_attestation" in script
    assert "pip install" not in script
    assert "<<" not in script
    assert m["spec"]["backoffLimit"] == 1
    assert "--portable-root /tmp/npa-isaac-kit" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "kit_args=os.environ.get(" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "device=SIM_DEVICE" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "OnPolicyRunner(env, acfg, log_dir=None, device=SIM_DEVICE)" in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert "get_inference_policy(device=SIM_DEVICE)" in pr.ISAAC_ROLLOUT_SCRIPT
    assert 'OnPolicyRunner(env, acfg, log_dir=None, device="cuda:0")' not in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert 'CameraType = CameraCfg if SIM_DEVICE == "cpu" else TiledCameraCfg' in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert "env_cfg.sim.use_fabric = False" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "CPU physics camera fallback requires ROLLOUT_COUNT=1" in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert "def _write_rgb_png(path, rgb):" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "from PIL import" not in pr.ISAAC_ROLLOUT_SCRIPT
    assert '"/rtx/dataWindowNDC/2", 1.0' in pr.ISAAC_ROLLOUT_SCRIPT
    assert '"/rtx/dataWindow/fitOutputToDataWindow", False' in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert "if arr.ndim == 3:" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "arr = arr[None, ...]" in pr.ISAAC_ROLLOUT_SCRIPT
    assert 'get_annotator("rgb", device="cuda:0")' in pr.ISAAC_ROLLOUT_SCRIPT
    assert "annotator.attach(sensor.render_product_paths)" in pr.ISAAC_ROLLOUT_SCRIPT
    assert "arr = arr.view(np.uint8)" in pr.ISAAC_ROLLOUT_SCRIPT
    assert pr.ISAAC_ROLLOUT_SCRIPT.index("obs, _, dones, extras = env.step(actions)") < (
        pr.ISAAC_ROLLOUT_SCRIPT.index("capture(_step)")
    )
    assert "_write_rgb_png(os.path.join(d, name), arr[i])" in (
        pr.ISAAC_ROLLOUT_SCRIPT
    )
    assert '"simulation_device": SIM_DEVICE' in pr.ISAAC_ROLLOUT_SCRIPT


def test_rollout_job_can_select_cpu_physics_without_releasing_gpu(
    monkeypatch,
):
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_DEVICE", "cpu")
    manifest = pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-run1-cpu",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=1,
        steps_per_rollout=1,
        checkpoint_uri="",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/cpu",
        s3_endpoint="",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )

    script = _manifest_script(manifest)
    assert "ROLLOUT_SIM_DEVICE=cpu" in script
    resources = manifest["spec"]["template"]["spec"]["containers"][0][
        "resources"
    ]
    assert resources["requests"]["nvidia.com/gpu"] == "1"
    assert resources["limits"]["nvidia.com/gpu"] == "1"


def test_untrained_job_manifest_skips_download():
    m = pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-run1-iter0",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=2,
        steps_per_rollout=4,
        checkpoint_uri="",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/iter0",
        s3_endpoint="",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    script = _manifest_script(m)
    assert "npa.workflows.sim2real.isaac_job_io download" not in script
    assert 'ROLLOUT_CKPT_LOCAL=""' in script


def test_rollout_manifest_embeds_scenario_and_byo_robot_contract():
    scenario = {
        "task_id": "Isaac-Lift-Cube-Franka-v0",
        "task_contract_digest": "contract-123",
        "scenario_config_digest": "scenario-456",
    }
    robot_spec = {
        "name": "customer-arm",
        "robot_source": "customer_usd",
        "usd_path": "/tmp/npa-byo-robot/customer.usd",
    }
    manifest = pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-custom",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=1,
        steps_per_rollout=32,
        checkpoint_uri="s3://bucket/model_500.pt",
        out_s3_prefix="s3://bucket/run/rollouts",
        s3_endpoint="https://storage.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        scenarios_jsonl=json.dumps(scenario) + "\n",
        robot_spec=robot_spec,
        robot_usd_uri="s3://bucket/assets/customer.usd",
        task_config={"task_id": "Isaac-Lift-Cube-Franka-v0"},
    )
    script = _manifest_script(manifest)
    assert "npa.workflows.sim2real.isaac_job_io write-base64" in script
    assert "/opt/npa/isaac-runtime/isaac_rollout.py" in script
    # Scenario and robot application lives in source baked into the immutable
    # runtime image; it must never be copied into the live Job manifest.
    assert "import isaac_scenario_task as _scenarios" in pr.ISAAC_ROLLOUT_SCRIPT
    assert pr.ISAAC_ROLLOUT_SCRIPT.index(
        "import isaac_scenario_task as _scenarios"
    ) < pr.ISAAC_ROLLOUT_SCRIPT.index("applied = _scenarios.runtime_audit")
    assert "isaac_scenario_task.py" not in script
    assert "NPA_BYO_ROBOT_SPEC_JSON" in script
    assert "NPA_BYO_TASK_CONFIG_JSON" in script
    assert "NPA_EXPECTED_ROBOT_USD" in script
    assert "s3://bucket/assets/customer.usd" in script
    assert "--destination /tmp/npa-byo-robot/customer.usd" in script


def test_run_isaac_rollout_job_uses_outer_iteration_artifact_tag(tmp_path, monkeypatch):
    captured: dict[str, str] = {}

    class _FakeS3:
        def download_file(self, _bucket: str, _key: str, local: str) -> None:
            Path(local).write_text(json.dumps({"rollouts": []}), encoding="utf-8")

    class _FakeBoto3:
        def client(self, *_args, **_kwargs):
            return _FakeS3()

    def fake_build(**kwargs):
        captured["job_name"] = kwargs["job_name"]
        captured["out_s3_prefix"] = kwargs["out_s3_prefix"]
        return {"kind": "Job"}

    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3())
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
    monkeypatch.setattr(pr, "build_isaac_rollout_job_manifest", fake_build)
    monkeypatch.setattr(
        pr, "latest_checkpoint_uri", lambda *a, **k: "s3://b/run/model_latest.pt"
    )
    monkeypatch.setattr(pr, "materialize_rollout_dirs", lambda *a, **k: [])
    monkeypatch.setattr(
        "npa.workflows.sim2real.byo_isaac_trainer.read_generated_train_envs",
        lambda *a, **k: (
            [
                {
                    "difficulty": difficulty,
                    "scenario_config_digest": f"cfg-{difficulty}",
                }
                for difficulty in ("easy", "medium", "hard")
            ],
            "",
        ),
    )
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_IMAGE", "reg/npa-isaac-lab:2.3.2.post1")
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "bkt")
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")

    pr.run_isaac_rollout_job(
        tmp_path / "actions" / "train" / "outer-02" / "iter-01",
        run_id="myrun",
        rollout_count=1,
        steps_per_rollout=1,
    )

    assert captured["job_name"].endswith("outer-02-iter-01")
    assert captured["out_s3_prefix"].endswith("/byo-rollouts/outer-02-iter-01")


def test_inline_rollout_provenance_reaches_main_component_record(tmp_path, monkeypatch):
    image = "reg/runtime@sha256:" + "a" * 64
    proof = {
        "mode": "npa_workflow_skypilot_task",
        "job_name": "managed-wave-stage-07",
        "image": image,
        "gpu_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        "owner": "standard_npa_workflow_runtime",
    }

    monkeypatch.setattr(
        pr,
        "build_isaac_rollout_job_manifest",
        lambda **_kwargs: {"kind": "Job"},
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.isaac_job_payload.execute_manifest_container_inline",
        lambda _manifest: proof,
    )
    monkeypatch.setattr(pr, "latest_checkpoint_uri", lambda *a, **k: "")
    monkeypatch.setattr(pr, "_download_rollout_metadata", lambda *a, **k: {})
    monkeypatch.setattr(pr, "materialize_rollout_dirs", lambda *a, **k: [])
    monkeypatch.setattr(
        "npa.workflows.sim2real.byo_isaac_trainer.read_generated_train_envs",
        lambda *a, **k: (
            [
                {
                    "difficulty": difficulty,
                    "scenario_config_digest": f"cfg-{difficulty}",
                }
                for difficulty in ("easy", "medium", "hard")
            ],
            "",
        ),
    )
    monkeypatch.setenv("NPA_SIM2REAL_INLINE_TASK", "1")
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_IMAGE", image)
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "bucket")
    monkeypatch.setenv("NPA_SIM2REAL_PREFIX", "sim2real")
    monkeypatch.setenv("NPA_TASK_IMAGE", image)

    pr._LAST_GPU_PROVENANCE = {}
    result = pr.run_isaac_rollout_job(
        tmp_path / "actions" / "train" / "outer-01" / "iter-01",
        run_id="run",
        rollout_count=1,
        steps_per_rollout=1,
    )

    assert result == []
    assert pr._LAST_GPU_PROVENANCE == proof
