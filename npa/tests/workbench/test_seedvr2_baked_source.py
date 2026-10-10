"""Baked source markers must not attest code imported from an overlay."""

from pathlib import Path

import pytest

from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.schemas import RestoreRequest


def test_checkout_cannot_claim_baked_revision(monkeypatch, tmp_path):
    marker = tmp_path / "revision"
    marker.write_text("a" * 40)
    monkeypatch.setattr(runtime, "SOURCE_REVISION_PATH", marker)
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.example/seedvr2@sha256:" + "b" * 64)
    monkeypatch.setattr(
        runtime, "_gpu_inventory", lambda: pytest.fail("must reject before GPU probe")
    )
    with pytest.raises(runtime.SeedVR2Error, match="baked NPA installation"):
        runtime._runtime_identity()


@pytest.mark.parametrize("overlay", ["1", "true", "yes", "on"])
def test_explicit_overlay_cannot_claim_baked_location(monkeypatch, overlay):
    monkeypatch.setattr(runtime, "BAKED_NPA_ROOT", Path(runtime.__file__).parents[2])
    monkeypatch.setenv("NPA_SRC_OVERLAY", overlay)
    with pytest.raises(runtime.SeedVR2Error, match="overlays are forbidden"):
        runtime._require_baked_source()


def test_verification_rejects_active_overlay_before_trusting_artifact():
    request = RestoreRequest(
        input_path="s3://example-bucket/input.mp4",
        output_path="s3://example-bucket/output/",
        run_id="baked",
    )
    with pytest.raises(runtime.SeedVR2Error, match="baked NPA installation"):
        artifacts._validate_result_runtime({}, request)


def test_baked_location_passes_but_symlink_to_overlay_fails(monkeypatch, tmp_path):
    baked = tmp_path / "npa"
    module = baked / "workbench/seedvr2/runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fixture baked code")
    monkeypatch.setattr(runtime, "BAKED_NPA_ROOT", baked)
    monkeypatch.setattr(runtime, "__file__", str(module))
    runtime._require_baked_source()
    overlay = tmp_path / "overlay.py"
    overlay.write_text("# different code")
    module.unlink()
    module.symlink_to(overlay)
    with pytest.raises(runtime.SeedVR2Error, match="baked NPA installation"):
        runtime._require_baked_source()
