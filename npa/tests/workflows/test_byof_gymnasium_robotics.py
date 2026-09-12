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


def test_neutral_bootstrap_uses_only_the_unbuilt_prebuilt_candidate() -> None:
    config = _workflow()["config"]
    assert config["repo_ref"] == SOURCE_COMMIT
    assert config["repo_auth"] == "none"
    assert config["base_profile"] == "prebuilt"
    assert (
        config["base_image"]
        == "registry.example.invalid/gymnasium-robotics:neutral-unbuilt"
    )
    assert config["build_command"] == ""
    assert config["smoke_command"].endswith(
        "exec /usr/local/bin/npa-gymnasium-entrypoint run-smoke\n"
    )
    assert config["capability_name"] == ENV_ID
    assert config["smoke_artifact_name"] == "gymnasium-robotics-smoke.json"


def test_neutral_image_lock_gate_accepts_only_the_reviewed_content_closure() -> None:
    completed = subprocess.run(
        [str(IMAGE_ROOT / "build.sh"), "verify-bootstrap-locks"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert not any(
        command in text for command in ("curl ", "wget ", "git clone", "pip install")
    )


def test_runtime_and_neutral_baked_closures_are_exact_but_publicly_quarantined() -> None:
    source = json.loads((IMAGE_ROOT / "source-lock.json").read_text())
    apt = json.loads((IMAGE_ROOT / "apt-runtime.lock.json").read_text())
    corresponding = json.loads(
        (IMAGE_ROOT / "corresponding-source.lock.json").read_text()
    )
    assert {source["status"], apt["status"], corresponding["status"]} == {"complete"}
    assert source["source_commit"] == SOURCE_COMMIT
    assert source["mujoco_version"] == "3.12.0"
    assert source["components"]["shadow_sr_common"]["preferred_form_complete"] is False
    assert len(apt["resolved_binary_packages"]) == 142
    assert len(apt["resolved_source_packages"]) == 102
    assert sum(
        len(item["artifacts"]) for item in apt["resolved_source_packages"]
    ) == 318
    assert "python3-boto3" in {
        item["package"] for item in apt["requested_runtime_packages"]
    }
    assert corresponding["scope"] == "candidate-image-layers-only"
    assert corresponding["runtime_fetched_material_excluded"]
    assert len(source["requirements_lock_sha256"]) == 64
    assert (
        source["resolved_python_artifact_count"]
        == source["expected_python_distribution_count"]
    )
    assert source["expected_python_distribution_count"] == 19
    runtime_requirements = (IMAGE_ROOT / "requirements.in").read_text()
    assert "boto3" not in runtime_requirements
    assert "botocore" not in runtime_requirements


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
    assert "/opt/npa/gymnasium-robotics/runtime-bootstrap.py" in profile
    assert "NPA_GYMNASIUM_RUNTIME_CACHE" in profile
    assert "current/runtime/bin/python" not in profile
    assert "${NPA_GYMNASIUM_RUNTIME_CACHE}/current" not in profile
    assert "RUNTIME_PYTHON" not in profile
    assert "/usr/bin/python3 -I -B" in profile
    assert "/usr/local/bin/npa-gymnasium-entrypoint prepare-runtime" in profile
    assert "/bin/bash -lc \"${BYOF_SMOKE_COMMAND}\"" in profile
    assert "-u AWS_SECRET_ACCESS_KEY" in profile
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


def test_no_gated_payload_or_consent_proxy_is_part_of_neutral_design() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            WORKFLOW,
            SMOKE,
            IMAGE_ROOT / "Dockerfile",
            IMAGE_ROOT / "runtime-bootstrap.py",
        )
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


def test_documentation_keeps_neutral_and_historical_evidence_separate() -> None:
    text = (ROOT / "docs/workbench/byof-gymnasium-robotics.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "neutral bootstrap",
        "pre-registration quarantine",
        "historical",
        "corresponding-source",
        "No model, external dataset, gated artifact, or terms-acceptance flag",
        "RTX PRO 6000 Blackwell",
    ):
        assert token in text
