"""Unit boundaries for the EmbodiedGen operator-private runtime fetcher."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa" / "docker" / "workbench" / "embodiedgen"
SPEC = importlib.util.spec_from_file_location(
    "embodiedgen_runtime_bootstrap", IMAGE / "runtime-bootstrap.py"
)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)


def test_manifest_pins_the_selected_upstream_components() -> None:
    manifest = IMAGE / "runtime-manifest.json"
    payload = BOOTSTRAP._read_manifest(manifest)
    assert payload["solution"]["license"] == "Apache-2.0"
    assert payload["backend"]["license"] == "MIT"
    assert payload["backend"]["model_revision"] == BOOTSTRAP.MODEL_REVISION
    assert payload["runtime"]["baked"] == {
        "source": False,
        "model": False,
        "input": False,
        "cache": False,
        "output": False,
        "credentials": False,
    }


def test_manifest_refuses_a_changed_model_revision(tmp_path: Path) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["backend"]["model_revision"] = "0" * 40
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model revision"):
        BOOTSTRAP._read_manifest(path)


def test_model_receipt_rejects_changed_cached_bytes(tmp_path: Path) -> None:
    model = tmp_path / "TRELLIS-image-large"
    model.mkdir()
    payload = model / "weights.safetensors"
    payload.write_bytes(b"pinned model bytes")
    receipt = model / ".npa-model-receipt.json"
    BOOTSTRAP._write_model_receipt(model)
    BOOTSTRAP._verify_model_receipt(model, receipt)
    payload.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="do not match"):
        BOOTSTRAP._verify_model_receipt(model, receipt)


def test_trellis_only_patch_removes_unselected_sam_import(tmp_path: Path) -> None:
    inference = tmp_path / "embodied_gen" / "utils" / "inference.py"
    image_to_3d = tmp_path / "embodied_gen" / "scripts" / "imageto3d.py"
    inference.parent.mkdir(parents=True)
    image_to_3d.parent.mkdir(parents=True)
    inference.write_text(
        "from embodied_gen.models.sam3d import Sam3dInference\n"
        "def run(pipe: TrellisImageTo3DPipeline | Sam3dInference):\n"
        "    if isinstance(pipe, TrellisImageTo3DPipeline):\n        return 1\n"
        "    elif isinstance(pipe, Sam3dInference):\n        return 2\n"
        "    else:\n        raise ValueError()\n"
    )
    image_to_3d.write_text(
        "TrellisImageTo3DPipeline.from_pretrained(\n"
        '            "microsoft/TRELLIS-image-large"\n        )'
    )
    BOOTSTRAP._patch_trellis_only_import(tmp_path)
    assert "Sam3dInference" not in inference.read_text()
    assert "NPA_EMBODIEDGEN_TRELLIS_MODEL_DIR" in image_to_3d.read_text()


def test_capability_smoke_requires_real_generation_and_simulation() -> None:
    text = (IMAGE / "capability_smoke.py").read_text(encoding="utf-8")
    for token in (
        "img3d-cli",
        "--image3d_model",
        "TRELLIS",
        "collision_meshes",
        "cvt_embodiedgen_asset_to_anysim",
        "embodiedgen_urdf_to_mjcf_conversion",
        "bullet.loadURDF",
        "bullet.stepSimulation",
        "require_rigid_body_settle",
        "linear_speed_m_per_s",
        "decoded_video_frames",
        "VLM_estimated_not_calibrated_ground_truth",
        "http.client.HTTPSConnection",
        "defusedxml",
    ):
        assert token in text


def test_prebuilt_image_and_standalone_profile_keep_the_byof_contract() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    profile = (
        ROOT
        / "npa/src/npa/workflows/byof/profiles"
        / "byof-solution-smoke-embodiedgen-rtxpro-gpu.yaml"
    ).read_text(encoding="utf-8")
    build = (IMAGE / "build.sh").read_text(encoding="utf-8")
    assert "/opt/byof/npa_source_metadata.json" in dockerfile
    assert "git status --porcelain --untracked-files=all" in build
    assert "npa_byof_summary.json" in profile
    assert "client.head_object" in profile


def test_token_factory_key_is_forwarded_only_to_embodiedgen() -> None:
    runner_path = ROOT / "npa/scripts/run_byof_container_verify.py"
    spec = importlib.util.spec_from_file_location(
        "embodiedgen_byof_runner", runner_path
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    assert runner.resolve_secret_envs(
        None,
        solution_name="embodiedgen",
        environment={"NEBIUS_TOKEN_FACTORY_KEY": "value"},
    ) == ["NEBIUS_TOKEN_FACTORY_KEY"]
