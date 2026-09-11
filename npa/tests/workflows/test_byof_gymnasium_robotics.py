from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/byof-gymnasium-robotics.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
PROFILE = (
    ROOT
    / "npa/src/npa/workflows/byof/profiles/byof-solution-smoke-gymnasium-robotics-rtxpro-gpu.yaml"
)
IMAGE_ROOT = ROOT / "npa/docker/workbench/gymnasium-robotics"
SMOKE = IMAGE_ROOT / "capability_smoke.py"
ASSETS = IMAGE_ROOT / "asset-lock.json"
SOURCE_COMMIT = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
ENV_ID = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_phase_a_uses_only_the_unbuilt_prebuilt_candidate() -> None:
    config = _workflow()["config"]
    assert config["repo_ref"] == SOURCE_COMMIT
    assert config["repo_auth"] == "none"
    assert config["base_profile"] == "prebuilt"
    assert (
        config["base_image"]
        == "registry.example.invalid/gymnasium-robotics:phase-a-unbuilt"
    )
    assert config["build_command"] == ""
    assert config["smoke_command"].endswith(
        "exec /opt/venv/bin/python -I -B /opt/npa/gymnasium-robotics/capability_smoke.py\n"
    )
    assert config["capability_name"] == ENV_ID
    assert config["smoke_artifact_name"] == "gymnasium-robotics-smoke.json"


def test_phase_a_lock_gate_refuses_before_any_fetch() -> None:
    completed = subprocess.run(
        [str(IMAGE_ROOT / "build.sh"), "verify-locks"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Phase A packaging refusal" in completed.stderr
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert not any(
        command in text
        for command in ("curl ", "wget ", "git clone", "pip install", "apt-get")
    )


def test_source_runtime_and_reciprocal_evidence_stay_incomplete() -> None:
    source = json.loads((IMAGE_ROOT / "source-lock.json").read_text())
    apt = json.loads((IMAGE_ROOT / "apt-runtime.lock.json").read_text())
    corresponding = json.loads(
        (IMAGE_ROOT / "corresponding-source.lock.json").read_text()
    )
    assert {source["status"], apt["status"], corresponding["status"]} == {
        "phase-a-incomplete"
    }
    assert source["components"]["farama_gymnasium_robotics"]["commit"] == SOURCE_COMMIT
    assert source["components"]["mujoco"]["version"] == "3.12.0"
    assert source["components"]["shadow_sr_common"]["preferred_form_sha256"] is None
    assert apt["resolved_binary_packages"] == apt["resolved_source_packages"] == []
    assert any(
        item.get("transformation_manifest_sha256") is None
        for item in corresponding["deliveries"]
    )


def test_directly_loaded_shadow_asset_closure_is_exact() -> None:
    lock = json.loads(ASSETS.read_text(encoding="utf-8"))
    assert lock["source_commit"] == SOURCE_COMMIT
    assert len(lock["directly_loaded_xml"]) == 5
    assert len(lock["directly_loaded_mesh_texture"]) == 14
    assert set(Path(name).suffix for name in lock["directly_loaded_mesh_texture"]) == {
        ".stl",
        ".png",
    }
    assert all(
        len(value) == 64
        for group in (lock["directly_loaded_xml"], lock["directly_loaded_mesh_texture"])
        for value in group.values()
    )


def test_capability_script_keeps_the_real_hard_gate() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    compile(source, str(SMOKE), "exec")
    for token in (
        ENV_ID,
        "ROLLOUT_STEPS = 120",
        "raw.data.ncon",
        "raw.data.sensordata[touch_ids]",
        "env.step(action)",
        "env.render()",
        "libmujoco.so*",
        '"synthetic_only_fixture": False',
        '"pod_observed_image_digest"',
        '"physics_substeps"',
        '"max_reading"',
        '"distinct_rgb_frame_sha256"',
        '"exit_status": 0',
    ):
        assert token in source
    assert (
        "RTX PRO 6000" in source
        and "Blackwell" in source
        and 'capability != "12.0"' in source
    )


def test_workflow_and_profile_never_route_to_b200() -> None:
    workflow = _workflow()
    assert (
        workflow["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    profile = PROFILE.read_text(encoding="utf-8")
    assert "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in profile
    assert 'NVIDIA_DRIVER_CAPABILITIES: "graphics,utility"' in profile
    assert "export NVIDIA_DRIVER_CAPABILITIES=graphics,utility" in WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert "B200" not in WORKFLOW.read_text(encoding="utf-8")
    assert "B200" not in profile
    assert "/opt/npa/gymnasium-robotics/verify_image.py" in profile
    assert "npa_pod_image_receipt.json" in profile
    assert 'IfNoneMatch="*"' in profile


def test_readiness_is_bound_and_all_execution_evidence_is_blocked() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "unverified"
    assert readiness["prerequisites"]["source_image"]["status"] == "blocked"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"
    assert "historical" in readiness["prerequisites"]["source_image"]["reason"].lower()


def test_no_gated_payload_or_consent_proxy_is_part_of_phase_a() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (WORKFLOW, SMOKE, IMAGE_ROOT / "Dockerfile")
    )
    lowered = text.lower()
    for forbidden in (
        "hf_token",
        "snapshot_download",
        "accept_terms",
        "privacy_consent",
    ):
        assert forbidden not in lowered
    assert "nvcr.io" not in lowered
    assert "nvidia/cuda" not in lowered


def test_live_gate_still_requires_both_explicit_environment_gates() -> None:
    source = (ROOT / "npa/tests/e2e/test_byof_onboarding_live_e2e.py").read_text(
        encoding="utf-8"
    )
    assert 'NPA_INTEGRATION_E2E") != "1"' in source
    assert 'NPA_BYOF_GYMNASIUM_ROBOTICS_LIVE_GPU") != "1"' in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE" in source


def test_documentation_keeps_phase_a_and_historical_evidence_separate() -> None:
    text = (ROOT / "docs/workbench/byof-gymnasium-robotics.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "Phase A",
        "pre-registration quarantine",
        "historical",
        "corresponding source",
        "No model, dataset, gated asset, or terms acceptance",
        "RTX PRO 6000 Blackwell",
    ):
        assert token in text
