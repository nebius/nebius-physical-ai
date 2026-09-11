from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "byof-gymnasium-robotics.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
PROFILE = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-gymnasium-robotics-rtxpro-gpu.yaml"
)
DOC = ROOT / "docs" / "workbench" / "byof-gymnasium-robotics.md"
LIVE_E2E = ROOT / "npa" / "tests" / "e2e" / "test_byof_onboarding_live_e2e.py"

SOURCE_COMMIT = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
BASE_DIGEST = "sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61"
MUJOCO_WHEEL_SHA = "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0"
ENV_ID = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
ARTIFACT = "gymnasium-robotics-smoke.json"


def _workflow() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _config() -> dict[str, object]:
    config = _workflow()["config"]
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    smoke = str(_config()["smoke_command"])
    return smoke.split("/opt/venv/bin/python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _literal_assignment(source: str, name: str) -> object:
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def test_gymnasium_robotics_source_runtime_and_output_are_immutable() -> None:
    config = _config()
    build = str(config["build_command"])

    assert config["repo_url"] == (
        "https://github.com/Farama-Foundation/Gymnasium-Robotics.git"
    )
    assert config["repo_ref"] == SOURCE_COMMIT
    assert config["base_profile"] == "ubuntu"
    assert str(config["base_image"]).endswith(BASE_DIGEST)
    assert "noble-20260905" in str(config["base_image"])
    assert MUJOCO_WHEEL_SHA in build
    assert "mujoco-3.12.0-cp312-cp312" in build
    for dependency in (
        "absl-py==2.5.0",
        "boto3==1.43.91",
        "etils[epath]==1.14.0",
        "gymnasium==1.3.0",
        "numpy==2.5.3",
        "pettingzoo==1.27.0",
        "PyOpenGL==3.1.10",
    ):
        assert dependency in build
    assert "--no-build-isolation --no-deps -e ." in build
    assert "pip check" in build
    assert config["workload"] == "solution-smoke"
    assert config["solution_name"] == "gymnasium-robotics"
    assert config["capability_name"] == ENV_ID
    assert config["smoke_artifact_name"] == ARTIFACT
    assert config["resource_profile_yaml"] == (
        "byof-solution-smoke-gymnasium-robotics-rtxpro-gpu"
    )
    assert config["output_root"] == (
        "s3://{{config.bucket}}/oss-solutions/gymnasium-robotics"
    )
    assert config["summary_uri"] == (
        "{{config.output_root}}/{{run.id}}/npa_byof_summary.json"
    )
    assert config["wait_timeout"] == -1


def test_gymnasium_robotics_smoke_is_real_physics_touch_and_egl() -> None:
    smoke = str(_config()["smoke_command"])
    python_source = _smoke_python()
    compile(python_source, "<gymnasium-robotics-smoke>", "exec")

    for required in (
        "gym.make(",
        "observation, reward, terminated, truncated, _info = env.step(action)",
        "raw.data.ncon",
        "raw.data.sensordata[touch_ids]",
        "frame = np.asarray(env.render())",
        "quaternion_angle(initial_goal[3:7], achieved[3:7])",
        '"synthetic_only_fixture": False',
        '"physics_substeps"',
        '"nonzero_reading_count"',
        '"distinct_rgb_frame_sha256"',
        '"loaded_gl_egl_libraries"',
        'rglob("libmujoco.so*")',
        "expected exactly one MuJoCo shared library",
        '"pod_observed_image_digest"',
        '"exit_status": 0',
        '"media_type": "application/json"',
        '"size_bytes"',
        '"sha256_scope"',
        "(output_dir / OUTPUT_NAME).write_text",
    ):
        assert required in smoke
    assert "MUJOCO_GL=egl" in smoke
    assert "PYOPENGL_PLATFORM=egl" in smoke
    assert "ROLLOUT_STEPS = 120" in smoke
    assert '"observation": (153,)' in smoke
    assert '"achieved_goal": (7,)' in smoke
    assert '"desired_goal": (7,)' in smoke
    assert "touch_ids.shape != (92,)" in smoke
    assert "[(240, 320, 3)]" in smoke
    assert "len(distinct_frame_hashes) < 2" in smoke
    assert '"libEGL" in item["path"] and "nvidia" in item["path"].lower()' in smoke
    assert "expected exactly one GPU" in smoke
    assert '"RTX PRO 6000"' in smoke
    assert '"12.0"' in smoke


def test_gymnasium_robotics_hashes_every_loaded_xml_and_hand_asset() -> None:
    source = _smoke_python()
    xml_hashes = _literal_assignment(source, "XML_HASHES")
    asset_hashes = _literal_assignment(source, "ASSET_HASHES")

    assert isinstance(xml_hashes, dict)
    assert set(xml_hashes) == {
        "hand/manipulate_block_touch_sensors.xml",
        "hand/robot_touch_sensors_92.xml",
        "hand/shared.xml",
        "hand/shared_asset.xml",
        "hand/shared_touch_sensors_92.xml",
    }
    assert isinstance(asset_hashes, dict)
    assert len(asset_hashes) == 14
    assert {Path(name).suffix for name in asset_hashes} == {".stl", ".png"}
    assert "stls/hand/forearm_electric.stl" in asset_hashes
    assert "stls/hand/TH1_z.stl" in asset_hashes
    assert "textures/block.png" in asset_hashes
    assert "textures/block_hidden.png" in asset_hashes
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in [*xml_hashes.values(), *asset_hashes.values()]
    )


def test_gymnasium_robotics_has_no_gated_or_external_payload() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "repo_auth: none" in text
    assert "huggingface" not in lowered
    assert "hf_token" not in lowered
    assert "ngc" not in lowered
    assert "accept_terms" not in lowered
    assert "eula" not in lowered
    assert "snapshot_download" not in lowered
    assert "dataset" not in lowered
    assert "checkpoint" not in lowered
    assert "B200" not in text


def test_gymnasium_robotics_profile_is_one_strict_rtxpro_shape() -> None:
    workflow_resources = _workflow()["resources"]["gpu"]
    assert workflow_resources["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    docs = [
        value
        for value in yaml.safe_load_all(PROFILE.read_text(encoding="utf-8"))
        if value is not None
    ]
    assert len(docs) == 2
    task = docs[1]
    assert task["resources"]["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    assert "B200" not in PROFILE.read_text(encoding="utf-8")
    envs = task["envs"]
    assert envs["MUJOCO_GL"] == "egl"
    assert envs["PYOPENGL_PLATFORM"] == "egl"
    assert envs["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert envs["NPA_BYOF_POD_IMAGE_RECEIPT_TIMEOUT_SECONDS"] == "900"
    assert "NVIDIA_VISIBLE_DEVICES" not in envs
    run = str(task["run"])
    assert "npa_pod_image_receipt.json" in run
    assert "owner-side-kubernetes-status" in run
    assert "NPA_BYOF_POD_IMAGE_ID" in run
    assert "grep -Eq '@sha256:[0-9a-f]{64}$'" in run
    assert "RECEIPT_DEADLINE" in run
    assert "timed out waiting for the owner-side Pod image receipt" in run
    assert "serviceaccount" not in run
    assert "urllib.request" not in run
    assert "timeout=" not in run
    assert "RTX PRO 6000" in run
    assert "Blackwell" in run
    assert "12.0" in run
    assert '"smoke_artifact": artifact_evidence' in run


def test_gymnasium_robotics_readiness_is_bound_to_workflow_bytes() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    digest = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == digest
    assert set(readiness["planning"]) == {"validation", "task_fidelity"}
    assert set(readiness["prerequisites"]) == {
        "output_storage",
        "worker_input",
        "credentials",
        "source_image",
        "target_runtime",
    }
    for section in (
        *readiness["planning"].values(),
        *readiness["prerequisites"].values(),
    ):
        assert section["status"] in {
            "verified",
            "unverified",
            "blocked",
            "not_applicable",
        }
        assert section["reason"]
        if section["status"] == "verified":
            assert section["evidence"]


def test_gymnasium_robotics_documentation_records_license_and_scope() -> None:
    text = DOC.read_text(encoding="utf-8")

    for required in (
        SOURCE_COMMIT,
        "MIT",
        "GPL-2.0",
        "Apache-2.0",
        "MuJoCo 3.12.0",
        BASE_DIGEST,
        ENV_ID,
        ARTIFACT,
        "RTX PRO 6000 Blackwell",
        "No model, dataset, gated asset, or terms acceptance",
        "| Source |",
        "| Baked runtime |",
        "| Weights | None.",
        "| Data/assets |",
        "| Cache |",
        "| Outputs |",
        "Private delivery does not change",
        "allowed_nodes",
        "mode-0600 child-local copy",
        "ledger its exact",
        "Kubernetes UID",
        "RL training",
    ):
        assert required in text


def test_gymnasium_robotics_live_gate_requires_authorized_output_root() -> None:
    source = LIVE_E2E.read_text(encoding="utf-8")

    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_OUTPUT_ROOT" in source
    assert 'run_id = f"gymnasium-robotics-{time.time_ns()}"' in source
    assert 'spec.config["output_root"] = authorized_output_root.rstrip("/")' in source
    assert (
        "the manager-authorized output root must include a bucket and prefix" in source
    )
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_NAMESPACE" in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR" in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME" in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_UID" in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_PROVIDER_NODE_GROUP_ID" in source
    assert "_require_gymnasium_scheduling_contract(" in source
    assert "owner-side-kubernetes-status" in source
    assert '("list", "pods"), ("create", "pods/exec")' in source
    assert 'get("skypilot-cluster-name")' in source
    assert '"parent=skypilot"' in source
    assert "observed_digests == {expected_digest}" in source
    assert "assert gpu_requests == gpu_limits == 1" in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE must be the already scanned" in source
    assert "stdout=stdout_stream" in source
    assert "stderr=stderr_stream" in source
    assert "_cleanup_gymnasium_run(" in source
    assert "_cleanup_gymnasium_run_after_success(" in source
    assert 'command.extend(["--yes", run_id])' in source
    assert "exact-run cleanup also failed" in source
    assert "active exact-run cleanup also failed" in source
    assert "include_terminating=True" in source
    assert "assert_no_credential_leakage(" in source
    assert "GYMNASIUM_KUBECTL_TIMEOUT_SECONDS" in source
    assert "GYMNASIUM_RECEIPT_TIMEOUT_SECONDS" in source
    assert "GYMNASIUM_CLEANUP_TIMEOUT_SECONDS" in source
    assert "GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS" in source
    assert "GYMNASIUM_RUNNER_TERM_GRACE_SECONDS" in source
