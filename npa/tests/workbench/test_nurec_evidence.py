from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from npa.workbench.nurec import evidence


NRE_IMAGE = (
    "nvcr.io/nvidia/nre/nre-ga@"
    "sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284"
)


def _runtime_attestation() -> dict:
    return {
        "format": "npa_nurec_runtime_attestation_v1",
        "status": "pass",
        "source": "kubernetes_pod_status",
        "requested_image": NRE_IMAGE,
        "observed_image_digest": NRE_IMAGE.split("@", 1)[1],
        "gpu_names": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
        "gpu_count": 1,
        "resource_identity_sha256": "a" * 64,
    }


def test_runtime_attestation_requires_control_plane_digest_and_exact_gpu() -> None:
    payload = _runtime_attestation()
    assert (
        evidence.validate_runtime_attestation(payload, expected_image=NRE_IMAGE)
        == payload
    )

    payload["observed_image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(evidence.NurecEvidenceError):
        evidence.validate_runtime_attestation(payload, expected_image=NRE_IMAGE)


def _sequence(tmp_path: Path) -> Path:
    root = tmp_path / "sequence"
    root.mkdir()
    (root / "capture.zarr.itar").write_bytes(b"ncore-store")
    (root / "conversion.json").write_text('{"status":"ok"}\n')
    meta = root / "sequence.json"
    meta.write_text(
        json.dumps(
            {
                "version": "v4",
                "component_stores": [{"path": "capture.zarr.itar", "components": {}}],
            }
        )
    )
    return meta


def test_reconstruction_receipt_binds_real_input_recipe_and_outputs(
    tmp_path: Path,
) -> None:
    meta = _sequence(tmp_path)
    parsed = tmp_path / "parsed.yaml"
    parsed.write_text(
        "trainer:\n  max_epochs: 1\ndataset:\n  samples_per_epoch: 30000\n"
    )
    metrics = tmp_path / "metrics.yaml"
    metrics.write_text("test:\n  psnr: 24.5\n  ssim: 0.8\n  lpips: 0.2\n")
    usdz = tmp_path / "last.usdz"
    usdz.write_bytes(b"reopenable-usdz")
    target = tmp_path / "nre-reconstruction.json"

    receipt = evidence.write_reconstruction_receipt(
        receipt_path=target,
        ncore_json=meta,
        nre_image=NRE_IMAGE,
        config_name="configs/experimental/3dgut/3dgut_colmap.yaml",
        mode="trainval",
        max_epochs_argument=0,
        command=["/app/run", "--config-name=3dgut_colmap"],
        train_exit_code=0,
        gpu_names=["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
        parsed_config_path=parsed,
        metrics_path=metrics,
        usdz_path=usdz,
        metrics={"test/psnr": 24.5, "test/ssim": 0.8, "test/lpips": 0.2},
    )

    assert receipt["status"] == "pass"
    assert receipt["requested_nre_digest"] == NRE_IMAGE.split("@", 1)[1]
    assert receipt["gpu"] == {
        "count": 1,
        "names": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
        "all_rt_core_models": True,
    }
    assert receipt["recipe"]["resolved_epochs"] == 1
    assert receipt["recipe"]["resolved_samples_per_epoch"] == 30000
    assert len(receipt["input"]["sequence_members"]) == 3
    assert receipt["input"]["conversion_report_sha256"]
    assert json.loads(target.read_text()) == receipt


def test_reconstruction_receipt_does_not_promote_missing_metrics(
    tmp_path: Path,
) -> None:
    meta = _sequence(tmp_path)
    parsed = tmp_path / "parsed.yaml"
    parsed.write_text("trainer:\n  max_epochs: 1\n")
    usdz = tmp_path / "last.usdz"
    usdz.write_bytes(b"usdz")

    receipt = evidence.write_reconstruction_receipt(
        receipt_path=tmp_path / "failed.json",
        ncore_json=meta,
        nre_image=NRE_IMAGE,
        config_name="configs/experimental/3dgut/3dgut_colmap.yaml",
        mode="trainval",
        max_epochs_argument=0,
        command=["/app/run"],
        train_exit_code=0,
        parsed_config_path=parsed,
        usdz_path=usdz,
    )

    assert receipt["status"] == "failed"
    assert receipt["observed_metrics"]["test/psnr"] is None


def test_reconstruction_receipt_rejects_unobserved_gpu_or_mutable_image(
    tmp_path: Path,
) -> None:
    meta = _sequence(tmp_path)
    parsed = tmp_path / "parsed.yaml"
    parsed.write_text(
        "trainer:\n  max_epochs: 1\ndataset:\n  samples_per_epoch: 30000\n"
    )
    metrics = tmp_path / "metrics.yaml"
    metrics.write_text("test: {psnr: 24.5, ssim: 0.8, lpips: 0.2}\n")
    usdz = tmp_path / "last.usdz"
    usdz.write_bytes(b"usdz")

    receipt = evidence.write_reconstruction_receipt(
        receipt_path=tmp_path / "identity-failed.json",
        ncore_json=meta,
        nre_image="nvcr.io/nvidia/nre/nre-ga:26.04",
        config_name="configs/experimental/3dgut/3dgut_colmap.yaml",
        mode="trainval",
        max_epochs_argument=0,
        command=["/app/run"],
        train_exit_code=0,
        parsed_config_path=parsed,
        metrics_path=metrics,
        usdz_path=usdz,
        metrics={"test/psnr": 24.5, "test/ssim": 0.8, "test/lpips": 0.2},
    )

    assert receipt["status"] == "failed"
    assert receipt["requested_nre_digest"] == ""
    assert receipt["gpu"]["count"] == 0


def _rgb(path: Path, *, uniform: bool = False) -> None:
    image = Image.new("RGB", (8, 8), (20, 40, 60))
    if not uniform:
        image.putpixel((0, 0), (200, 100, 10))
    image.save(path)


def test_render_receipt_decodes_and_hashes_every_frame(tmp_path: Path) -> None:
    usdz = tmp_path / "last.usdz"
    usdz.write_bytes(b"trained-scene")
    output = tmp_path / "render"
    (output / "camera1").mkdir(parents=True)
    _rgb(output / "camera1" / "000000.png")
    _rgb(output / "camera1" / "000001.png")

    receipt = evidence.write_render_receipt(
        receipt_path=output / "nre-render.json",
        artifact_path=usdz,
        output_dir=output,
        nre_image=NRE_IMAGE,
        command=["/app/run", "render"],
        render_exit_code=0,
        novel_view=True,
        rig_translation_offset="0.0,0.25,0.0",
        rig_rotation_offset="0.0,0.0,0.0",
        gpu_names=["NVIDIA L40S"],
    )

    assert receipt["status"] == "pass"
    assert receipt["output"]["frame_count"] == 2
    assert receipt["output"]["all_frames_decoded"] is True
    assert receipt["output"]["finite_pixels"] is True
    assert receipt["output"]["nonuniform_frames"] is True
    assert [item["path"] for item in receipt["output"]["inventory"]] == [
        "camera1/000000.png",
        "camera1/000001.png",
    ]


def test_render_receipt_rejects_uniform_visual_control(tmp_path: Path) -> None:
    usdz = tmp_path / "last.usdz"
    usdz.write_bytes(b"trained-scene")
    output = tmp_path / "render"
    output.mkdir()
    _rgb(output / "uniform.png", uniform=True)

    receipt = evidence.write_render_receipt(
        receipt_path=output / "nre-render.json",
        artifact_path=usdz,
        output_dir=output,
        nre_image=NRE_IMAGE,
        command=["/app/run", "render"],
        render_exit_code=0,
        novel_view=True,
        rig_translation_offset="0.0,0.25,0.0",
        rig_rotation_offset="0.0,0.0,0.0",
    )

    assert receipt["status"] == "failed"
    assert receipt["output"]["nonuniform_frames"] is False
