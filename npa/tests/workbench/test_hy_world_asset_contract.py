"""Unit tests for HY-World's generated-scene evidence boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from npa.workbench.hy_world import asset_contract as contract


def _metadata() -> dict[str, object]:
    return {
        "pipeline_mode": "image_to_world",
        "stages": ["pano", "traj", "render", "expand", "gs_data", "gs_train"],
        "source_repository": contract.SOURCE_REPOSITORY,
        "source_ref": contract.SOURCE_REF,
        "hy_world_model": {
            "repository": contract.HY_WORLD_MODEL_REPOSITORY,
            "ref": contract.HY_WORLD_MODEL_REF,
        },
        "worldstereo_model": {
            "repository": contract.WORLD_STEREO_REPOSITORY,
            "ref": contract.WORLD_STEREO_REF,
        },
        "wan_i2v_model": {
            "repository": contract.WAN_I2V_REPOSITORY,
            "ref": contract.WAN_I2V_REF,
        },
        "qwen_image_model": {
            "repository": contract.QWEN_IMAGE_REPOSITORY,
            "ref": contract.QWEN_IMAGE_REF,
        },
        "qwen_vlm": {
            "repository": contract.QWEN_VLM_REPOSITORY,
            "ref": contract.QWEN_VLM_REF,
        },
        "zim_model": {"repository": contract.ZIM_REPOSITORY, "ref": contract.ZIM_REF},
        "grounding_dino_model": {
            "repository": contract.GROUNDING_DINO_REPOSITORY,
            "ref": contract.GROUNDING_DINO_REF,
        },
        "dinov2_model": {
            "repository": contract.DINOV2_REPOSITORY,
            "ref": contract.DINOV2_REF,
        },
        "sam3_model": {
            "repository": contract.SAM3_REPOSITORY,
            "ref": contract.SAM3_REF,
        },
        "moge_model": {
            "repository": contract.MOGE_REPOSITORY,
            "ref": contract.MOGE_REF,
        },
        "uni3c_model": {
            "repository": contract.UNI3C_REPOSITORY,
            "ref": contract.UNI3C_REF,
        },
        "runtime_build_dependencies": {
            "pytorch3d": {
                "repository": contract.PYTORCH3D_REPOSITORY,
                "ref": contract.PYTORCH3D_REF,
            },
            "flash_attn": {
                "package": "flash-attn",
                "version": contract.FLASH_ATTN_VERSION,
            },
            "gsplat_maskgaussian": {"source_ref": contract.SOURCE_REF},
            "navmesh": {"source_ref": contract.SOURCE_REF},
            "rtree": {"package": "rtree", "version": contract.RTREE_VERSION},
        },
        "container_image": "example.invalid/npa-hy-world@sha256:" + "0" * 64,
        "resolved_python_packages": {
            "path": "npa_resolved_inventory.txt",
            "sha256": "1" * 64,
        },
        "report_resolved_python_packages": {
            "path": "npa_report_resolved_inventory.txt",
            "sha256": "6" * 64,
        },
        "native_runtime_dependencies": {
            "path": "npa_native_dependency_identity.json",
            "sha256": "7" * 64,
        },
        "report_runtime_dependencies": {
            "path": "npa_report_dependency_identity.json",
            "sha256": "8" * 64,
        },
        "torch_auxiliary_assets": {
            "path": "npa_torch_auxiliary_assets.json",
            "sha256": "9" * 64,
        },
        "submodules": {"path": "npa_submodules.txt", "sha256": "2" * 64},
        "model_snapshot_revisions": {
            "path": "npa_model_snapshot_revisions.json",
            "sha256": "3" * 64,
        },
        "runtime_contract": {
            "path": "npa_runtime_contract.json",
            "sha256": "4" * 64,
        },
        "torch_cuda_probe": {
            "path": "npa_torch_cuda_probe.json",
            "sha256": "5" * 64,
        },
        "wan_offline_shards": {
            "path": "npa_wan_offline_shard_resolution.json",
            "sha256": "a" * 64,
        },
    }


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    scene, result = tmp_path / "scene", tmp_path / "result"
    input_image = _write(tmp_path / "input.png", b"x" * 512)
    _write(scene / "panorama.png", b"p" * 512)
    _write(
        scene / "render_results/view0/traj0/worldstereo-memory-dmd_result.mp4", b"video"
    )
    (scene / "render_results/view0/traj0/camera.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (scene / "render_results/view0/traj0/camera.json").write_text(
        json.dumps({"c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    )
    _write(scene / "navmesh/reconstruct_pairs.json", b"[]\n")
    _write(
        result / "ply/point_cloud.ply",
        b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n" + b"0 " * 200,
    )
    _write(result / "ply/point_cloud.spz", b"compressed-splat")
    _write(result / "videos/traj_1499.mp4", b"render")
    return scene, result, input_image


def _video(_: Path) -> dict[str, int]:
    return {"width": 640, "height": 480, "decoded_frames": 12}


def _image(_: Path) -> dict[str, int]:
    return {"width": 1024, "height": 512}


def test_panorama_probe_decodes_an_equirectangular_image(tmp_path: Path) -> None:
    panorama = tmp_path / "panorama.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=1024x512",
            "-frames:v",
            "1",
            str(panorama),
        ],
        check=True,
    )
    assert contract.probe_image(panorama) == {"width": 1024, "height": 512}


def test_full_pipeline_evidence_requires_generated_assets(tmp_path: Path) -> None:
    scene, result, input_image = _fixture(tmp_path)
    evidence = contract.build_evidence(
        scene_dir=scene,
        result_dir=result,
        input_image=input_image,
        runtime_metadata=_metadata(),
        image_checker=_image,
        video_checker=_video,
    )
    assert evidence["capabilities_exercised"] == [
        contract.PRIMARY_CAPABILITY,
        contract.RENDER_CAPABILITY,
        contract.REPORT_CAPABILITY,
    ]
    assert evidence["trajectory_cameras"][0]["finite_matrix_count"] == 1
    assert evidence["worldnav"]["reconstruct_pair_count"] == 0
    assert evidence["rendered_camera_dataset"]["decode"]["decoded_frames"] == 12
    assert "metric_or_calibrated_scene_scale" in evidence["not_claimed"]


def test_worldmirror_only_or_placeholder_output_does_not_pass(tmp_path: Path) -> None:
    scene, result, input_image = _fixture(tmp_path)
    (scene / "render_results/view0/traj0/worldstereo-memory-dmd_result.mp4").unlink()
    with pytest.raises(contract.HyWorldEvidenceError, match="WorldStereo"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=_metadata(),
            image_checker=_image,
            video_checker=_video,
        )


def test_regular_trajectories_do_not_imply_worldnav_success(tmp_path: Path) -> None:
    """A caught upstream NavMesh error must not pass as a navigation handoff."""

    scene, result, input_image = _fixture(tmp_path)
    (scene / "navmesh/reconstruct_pairs.json").unlink()
    with pytest.raises(contract.HyWorldEvidenceError, match="WorldNav"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=_metadata(),
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_must_match_the_pinned_components(tmp_path: Path) -> None:
    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata["source_ref"] = "main"
    with pytest.raises(contract.HyWorldEvidenceError, match="source_ref"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_rejects_missing_gated_sam3_component(
    tmp_path: Path,
) -> None:
    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata.pop("sam3_model")
    with pytest.raises(contract.HyWorldEvidenceError, match="sam3_model"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


@pytest.mark.parametrize("component", ("wan_i2v_model", "dinov2_model"))
def test_runtime_provenance_requires_reachable_video_stage_models(
    tmp_path: Path, component: str
) -> None:
    """WorldStereo and WorldMirror dependencies cannot be omitted from evidence."""

    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata.pop(component)
    with pytest.raises(contract.HyWorldEvidenceError, match=component):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_requires_the_torch_cuda_probe(tmp_path: Path) -> None:
    """Generated-scene evidence must retain the B200 runtime compatibility proof."""

    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata.pop("torch_cuda_probe")
    with pytest.raises(contract.HyWorldEvidenceError, match="torch_cuda_probe"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_requires_offline_wan_shard_resolution(
    tmp_path: Path,
) -> None:
    """The sharded Diffusers resolver remains bound to the immutable cache."""

    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata.pop("wan_offline_shards")
    with pytest.raises(contract.HyWorldEvidenceError, match="wan_offline_shards"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_requires_the_pinned_worldnav_spatial_index(
    tmp_path: Path,
) -> None:
    """WorldNav evidence binds the rtree package that upstream actually imports."""

    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    dependencies = metadata["runtime_build_dependencies"]
    assert isinstance(dependencies, dict)
    dependencies.pop("rtree")
    with pytest.raises(
        contract.HyWorldEvidenceError, match="runtime_build_dependencies"
    ):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_runtime_provenance_requires_torch_auxiliary_asset_inventory(
    tmp_path: Path,
) -> None:
    """The trainer's separately fetched VGG backbone stays traceable."""

    scene, result, input_image = _fixture(tmp_path)
    metadata = _metadata()
    metadata.pop("torch_auxiliary_assets")
    with pytest.raises(contract.HyWorldEvidenceError, match="torch_auxiliary_assets"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=metadata,
            image_checker=_image,
            video_checker=_video,
        )


def test_empty_splat_assets_do_not_pass_as_a_trained_scene(tmp_path: Path) -> None:
    scene, result, input_image = _fixture(tmp_path)
    (result / "ply/point_cloud.ply").write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n" + b"x" * 512
    )
    (result / "ply/point_cloud.spz").write_bytes(b"\x00" * 64)
    with pytest.raises(contract.HyWorldEvidenceError, match="PLY has no vertices"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=_metadata(),
            image_checker=_image,
            video_checker=_video,
        )


def test_empty_compressed_splat_does_not_pass_as_a_trained_scene(
    tmp_path: Path,
) -> None:
    scene, result, input_image = _fixture(tmp_path)
    (result / "ply/point_cloud.spz").write_bytes(b"\x00" * 64)
    with pytest.raises(contract.HyWorldEvidenceError, match="SPZ header is empty"):
        contract.build_evidence(
            scene_dir=scene,
            result_dir=result,
            input_image=input_image,
            runtime_metadata=_metadata(),
            image_checker=_image,
            video_checker=_video,
        )
