"""Contracts for the licensed LIBERO-Plus asset camera compatibility workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import yaml

from npa.workflows import libero_plus_assets as assets


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/libero-plus-licensed-assets-camera-compatibility.yaml"
DOCKERFILE = ROOT / "npa/docker/workbench/libero-plus-assets/Dockerfile.private"
BUILD = ROOT / "npa/docker/workbench/libero-plus-assets/build-private.sh"
NOTICE = ROOT / "npa/docker/workbench/libero-plus-assets/THIRD_PARTY_NOTICES.md"
ENTRYPOINT = ROOT / "npa/docker/workbench/libero-plus-assets/entrypoint.sh"


def test_workflow_has_five_connected_native_asset_stages() -> None:
    """The declared workflow is an actual connected camera-evidence pipeline."""
    payload = yaml.safe_load(SPEC.read_text())
    states = payload["states"]
    assert list(states) == [
        "acquire-mit-assets",
        "assemble-native-scene",
        "render-matched-cameras",
        "validate-camera-metrics",
        "emit-gallery-report",
    ]
    assert states["acquire-mit-assets"]["next"] == "assemble-native-scene"
    assert states["assemble-native-scene"]["next"] == "render-matched-cameras"
    assert states["render-matched-cameras"]["next"] == "validate-camera-metrics"
    assert states["validate-camera-metrics"]["next"] == "emit-gallery-report"
    assert states["emit-gallery-report"]["terminal"] is True
    assert all(
        state["run"]["argv"][:2]
        == [
            "/opt/openwam-libero/bin/python",
            "/opt/npa/src/npa/workflows/libero_plus_assets.py",
        ]
        for state in states.values()
    )
    assert (
        "{{state.acquire-mit-assets.uri}}"
        in states["assemble-native-scene"]["run"]["argv"]
    )
    assert (
        "{{state.assemble-native-scene.uri}}"
        in states["render-matched-cameras"]["run"]["argv"]
    )
    assert (
        "{{state.render-matched-cameras.uri}}"
        in states["validate-camera-metrics"]["run"]["argv"]
    )
    report_argv = states["emit-gallery-report"]["run"]["argv"]
    assert "{{state.validate-camera-metrics.uri}}" in report_argv
    assert any(
        output["schema"] == "application/vnd.rerun.rrd"
        for output in states["emit-gallery-report"]["outputs"]
    )


def test_acquire_only_emits_allowlisted_mit_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archive selection never extracts task definitions or source utilities."""
    scene = b"<mujoco model='licensed-scene'/>"
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(assets.SCENE_MEMBER, scene)
        bundle.writestr("assets/not-selected.py", "not executed")
        bundle.writestr("bddl_files/not-a-task.bddl", "not fetched")
    monkeypatch.setattr(assets, "ASSET_MEMBER_COUNT", 3)
    monkeypatch.setattr(assets, "SCENE_SHA256", hashlib.sha256(scene).hexdigest())
    monkeypatch.setattr(assets, "_verified_archive", lambda: archive)

    manifest = tmp_path / "manifest.json"
    selected = tmp_path / "scene.xml"
    assets.acquire(str(manifest), str(selected))

    observed = json.loads(manifest.read_text())
    assert selected.read_bytes() == scene
    assert observed["asset"]["selected_member"] == assets.SCENE_MEMBER
    assert observed["exclusions"]["bddl_task_definitions"] == (
        "not present in asset archive and not fetched"
    )
    assert "10,030-task" in observed["limitation"]


def _png(path: Path, pixel: tuple[int, int, int]) -> dict[str, object]:
    image = np.full((256, 256, 3), pixel, dtype=np.uint8)
    Image.fromarray(image).save(path)
    payload = path.read_bytes()
    return {
        "uri": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def test_decode_metrics_and_rrd_are_factual_artifacts(tmp_path: Path) -> None:
    """Decode emitted PNG bytes and independently verify the generated RRD."""
    agentview = _png(tmp_path / "agentview.png", (20, 40, 60))
    agentview_60 = _png(tmp_path / "agentview_60.png", (120, 140, 160))
    agentview["shape"] = [256, 256, 3]
    agentview_60["shape"] = [256, 256, 3]
    agentview["pose"] = {
        "name": "agentview",
        "id": 1,
        "position": [0.0, 0.0, 0.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    agentview_60["pose"] = {
        "name": "agentview_60",
        "id": 2,
        "position": [1.0, 0.0, 0.0],
        "quaternion": [0.0, 1.0, 0.0, 0.0],
    }
    assembly = {
        "schema": "npa.libero-plus.licensed-assets.scene-assembly.v1",
        "native_executor": {"revision": assets.NATIVE_EXECUTOR_REVISION},
    }
    assembly_path = tmp_path / "assembly.json"
    assembly_path.write_text(json.dumps(assembly))
    gallery = {
        "schema": "npa.libero-plus.licensed-assets.camera-gallery.v1",
        "assembly_sha256": hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
        "render_backend": "egl",
        "frames": {"agentview": agentview, "agentview_60": agentview_60},
    }
    gallery_path = tmp_path / "gallery.json"
    gallery_path.write_text(json.dumps(gallery))
    metrics_path = tmp_path / "metrics.json"
    assets.validate(str(gallery_path), str(metrics_path))
    metrics = json.loads(metrics_path.read_text())
    assert metrics["metrics"]["rgb_mean_absolute_difference"] > 0.0
    assert metrics["metrics"]["camera_position_l2_distance"] == 1.0

    report_path = tmp_path / "report.json"
    rrd_path = tmp_path / "report.rrd"
    assets.report(
        str(assembly_path),
        str(gallery_path),
        str(metrics_path),
        str(report_path),
        str(rrd_path),
    )
    report = json.loads(report_path.read_text())
    assert report["benchmark_equivalence"] is False
    assert report["policy_rollout"] is False
    assert report["physical_robot"] is False
    assert report["rrd"]["sha256"] == hashlib.sha256(rrd_path.read_bytes()).hexdigest()
    verified = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "verify", str(rrd_path)],
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr.decode()


def test_parser_requires_real_stage_arguments() -> None:
    """A rendered state cannot silently omit a stage's artifact handoff."""
    parsed = assets.build_parser().parse_args(
        [
            "render",
            "--assembly-uri",
            "assembly.json",
            "--output-uri",
            "gallery.json",
            "--camera-a-uri",
            "a.png",
            "--camera-b-uri",
            "b.png",
        ]
    )
    assert parsed.command == "render"
    with pytest.raises(SystemExit):
        assets.build_parser().parse_args(["render", "--assembly-uri", "assembly.json"])


def test_module_never_fetches_or_imports_the_unlicensed_source() -> None:
    """The partial profile must not become an execution path for the fork."""
    source = (ROOT / "npa/src/npa/workflows/libero_plus_assets.py").read_text()
    assert "UPSTREAM_REPOSITORY" not in source
    assert "git clone" not in source
    assert "OffScreenRenderEnv" not in source
    assert "benchmarkrandomizer" not in source
    assert (
        "bddl_files"
        not in source.split("def acquire", 1)[1].split("def _native_source_root", 1)[0]
    )
    assert assets.NATIVE_EXECUTOR_REPOSITORY in source


def test_private_camera_image_is_runtime_fetch_only_and_refuses_public_targets() -> (
    None
):
    """The private derivative has no baked asset archive or public release path."""
    dockerfile = DOCKERFILE.read_text()
    build = BUILD.read_text()
    notice = NOTICE.read_text()
    entrypoint = ENTRYPOINT.read_text()
    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert 'org.nebius.npa.libero-plus-source="absent"' in dockerfile
    assert 'org.nebius.npa.libero-plus-assets="runtime-fetch-only"' in dockerfile
    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in dockerfile
    )
    for package in ("openssh-server", "rsync", "sudo"):
        assert package in dockerfile
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    assert "PasswordAuthentication no" in dockerfile
    assert "PermitRootLogin no" in dockerfile
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in dockerfile
    assert "ssh-keygen -A" in entrypoint
    assert "sudo -n ssh-keygen -A" in entrypoint
    assert 'exec "$@"' in entrypoint
    assert "boto3==1.42.91 rerun-sdk==0.38.1" in dockerfile
    assert "import libero, mujoco" in dockerfile
    assert "NPA_BAKED_PYTHON=/opt/openwam-libero/bin/python" in dockerfile
    assert "scanner-quarantined" in dockerfile
    assert "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2" in dockerfile
    assert "chmod 0444 /opt/npa/src/npa/workflows/libero_plus_assets.py" in dockerfile
    assert "COPY src/npa/workflows/libero_plus_assets.py" in dockerfile
    assert "COPY assets.zip" not in dockerfile
    assert "ghcr.io/nebius/nebius-physical-ai/*|docker.io/*|index.docker.io/*" in build
    assert "refusing public image target" in build
    assert "scan_image_omniverse_payload.py" in build
    assert "base image is not marked operator-private" in build
    assert "EXPECTED_BASE_IMAGE_ID" in build
    assert "LOCAL_BASE_TAG EXPECTED_BASE_IMAGE_ID" in build
    assert "docker export" in build
    assert "docker import" in build
    assert "refusing to reuse an existing flattened private base" in build
    assert "docker-daemon:$image" in build
    assert build.index("scan_image_omniverse_payload.py") < build.index(
        "cleanup_flat_base\n"
    )
    assert "ssh_host_*_key" in build
    assert "built-local-not-pushed" in build
    assert "It is **not**" in notice
    assert "LIBERO-Plus benchmark image" in notice
    assert "not copied, fetched, imported, or executed" in notice
