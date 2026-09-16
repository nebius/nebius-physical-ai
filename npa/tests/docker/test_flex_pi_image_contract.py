from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_image_flex_pi_payload.py"
SPEC = importlib.util.spec_from_file_location("scan_image_flex_pi_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _image(tmp_path: Path, members: dict[str, bytes]) -> Path:
    layer = _tar(tmp_path / "layer.tar", members)
    return _tar(tmp_path / "image.tar", {
        "manifest.json": json.dumps([{"Config": "config.json", "RepoTags": ["test:latest"], "Layers": ["layer.tar"]}]).encode(),
        "config.json": b"{}", "layer.tar": layer.read_bytes(),
    })


def test_clean_source_and_manifest_pass(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(_image(tmp_path, {
        "opt/flex-pi/LICENSE": b"MIT License",
        "opt/flex-pi/npa/public_robotwin_sample.json": b'{"path":"episode.mp4"}',
    }))
    assert layers == 1
    assert findings == []


def test_runtime_weights_media_and_secrets_fail(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(_image(tmp_path, {
        "workspace/.cache/huggingface/model.safetensors": b"x",
        "opt/flex-pi/episode_000000.mp4": b"x",
        "opt/npa-src/config": b"hf_abcdefghijklmnopqrstuvwxyz",
    }))
    assert {item.kind for item in findings} == {
        "model_weight", "populated_model_cache", "robotwin_payload", "credential_content",
    }


def test_dockerfile_pins_source_base_and_nonroot() -> None:
    text = (Path(__file__).resolve().parents[2] / "docker/workbench/flex-pi/Dockerfile").read_text()
    assert "@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9" in text
    assert "20c1b2b71ea35a415d5d47c39b04443cfadad7a1" in text
    assert "USER ubuntu" in text
    assert "'sm_120' in flags" in text
    assert "NPA_SOURCE_SHA" in text
    assert "IMAGEIO_FFMPEG_EXE=/opt/conda/bin/ffmpeg" in text
    assert "rm -f /opt/conda/lib/python3.11/site-packages/imageio_ffmpeg/binaries/" in text
    assert "apt-get install -y --only-upgrade linux-libc-dev" in text
    assert "rm -rf /opt/nvidia/nsight-compute/2025.1.0" in text
    assert "security-dependencies.patch" in text
    assert "hf-snapshot-config.patch" in text
    assert "checkpoint-only-deploy.patch" in text
    assert "MODELSCOPE_DOWNLOAD_PARALLELS=16" in text

    inference = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/inference.py"
    ).read_text()
    assert 'MODELSCOPE_BRANCH = "master"' in inference
    assert 'MODELSCOPE_GIT_COMMIT = "150f75d811d51f6c7760154aa7fec371dccda529"' in inference
    assert '"git", "ls-remote", MODELSCOPE_GIT_URL' in inference
    assert "immutable ModelScope branch head verification failed" in inference

    dependency_patch = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/pins/security-dependencies.patch"
    ).read_text()
    assert '"gitpython==3.1.62"' in dependency_patch
    assert '"wandb==0.30.0"' in dependency_patch

    snapshot_patch = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/pins/hf-snapshot-config.patch"
    ).read_text()
    assert "lexical_parents" in snapshot_patch
    assert "resolved_parents" in snapshot_patch
    assert snapshot_patch.index("lexical_parents") < snapshot_patch.index(
        "resolved_parents"
    )

    deploy_patch = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/flex-pi/pins/checkpoint-only-deploy.patch"
    ).read_text()
    assert "model_cfg_copy.skip_dit_load_from_pretrain = True" in deploy_patch
    assert deploy_patch.index("skip_dit_load_from_pretrain = True") < deploy_patch.index(
        "self.model = instantiate"
    )
    assert deploy_patch.index("self.model = instantiate") < deploy_patch.index(
        "self.model.load_checkpoint"
    )
