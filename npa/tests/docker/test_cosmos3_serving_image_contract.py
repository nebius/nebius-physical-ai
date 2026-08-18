"""Static and behavioral guards for the public OSS Cosmos3 serving image."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
NPA_ROOT = REPO_ROOT / "npa"
IMAGE_DIR = NPA_ROOT / "docker/workbench/cosmos3-serving"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
ENTRYPOINT = IMAGE_DIR / "entrypoint.sh"
ACCESS_PREFLIGHT = IMAGE_DIR / "access_preflight.py"
LOCK = IMAGE_DIR / "requirements.lock"
RUNTIME_BOOTSTRAP = IMAGE_DIR / "runtime_bootstrap.sh"
SERVING_SMOKE = IMAGE_DIR / "smoke_serving.sh"
GUARDRAIL_PREP = IMAGE_DIR / "prepare_guardrail_runtime.py"
HF_SNAPSHOT_PIN = IMAGE_DIR / "hf_snapshot_pin.py"
CONTRACT = NPA_ROOT / "docker/workbench/packaging-contract.yaml"
SOURCE_REVISION = "a4ea67a21b20054dacc6e83952f9bd407e8ee4e7"
SOURCE_SHA256 = "2a4ca4d3d83417a88717767fcdfdc5cb214200c6957d26d70625f17f58954800"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _instructions() -> str:
    return "\n".join(
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_source_base_and_dependency_closure_are_immutable() -> None:
    text = _instructions()
    assert re.search(r"ARG BASE_IMAGE=python:[^\s]+@sha256:[0-9a-f]{64}", text)
    assert "ARG DEBIAN_SNAPSHOT=20260817T000000Z" in text
    assert "gcc g++ libgl1" in text
    assert "libgnutls30 libssl3 openssl" in text
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in text
    assert SOURCE_REVISION in text
    assert SOURCE_SHA256 in text
    assert "e0262be9d8f7586bc24c069a2aed2b665bdff266" in text
    assert "cf03c0395fac8c4de386c0bdab12cc4fc8d66362" in text
    bootstrap = RUNTIME_BOOTSTRAP.read_text(encoding="utf-8")
    assert "sha256sum -c" in bootstrap
    assert "--require-hashes" in bootstrap
    assert 'python -m venv "${VENV}"' in bootstrap
    assert 'mv "${work}/venv" "${VENV}"' not in bootstrap
    assert "prepare_guardrail_runtime.py" in bootstrap
    assert "sitecustomize.py" in bootstrap
    assert "vllm/vllm-omni" not in text
    assert "nvcr.io" not in text
    assert "vllm==0.26.0" in LOCK.read_text(encoding="utf-8")
    assert "torch==2.11.0" in LOCK.read_text(encoding="utf-8")
    assert "COPY --chmod=0644" in text
    assert "su -s /bin/sh -c 'test -r" in text


@pytest.mark.parametrize(
    "mutation",
    [
        "FROM vllm/vllm-omni:cosmos3",
        "FROM nvcr.io/nvidia/pytorch:latest",
        "RUN hf download nvidia/Cosmos3-Super",
        "ENV HF_TOKEN=secret",
        "ENV HF_HUB_DISABLE_XET=1",
        "ENV ACCEPT_EULA=YES",
    ],
)
def test_forbidden_vendor_payload_and_build_fetch_mutations(mutation: str) -> None:
    forbidden = re.compile(
        r"(?i)(vllm/vllm-omni|nvcr\.io|hf\s+download|HF_TOKEN=|"
        r"HF_HUB_DISABLE_XET|ACCEPT_EULA=YES)"
    )
    assert forbidden.search(mutation)
    assert not forbidden.search(_instructions())


def test_public_contract_nonroot_health_and_inventory() -> None:
    from npa.deploy.images import (
        CONTAINER_IMAGE_NAMES,
        GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS,
        RESTRICTED_PUBLICATION_TOOLS,
        UNVALIDATED_PUBLICATION_TOOLS,
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["cosmos3-serving"]
    assert entry["redistribution"] == "public"
    assert entry["tier"] == "service"
    assert CONTAINER_IMAGE_NAMES["cosmos3-serving"] == "npa-cosmos3-serving"
    assert "cosmos3-serving" not in RESTRICTED_PUBLICATION_TOOLS
    assert "cosmos3-serving" not in UNVALIDATED_PUBLICATION_TOOLS
    assert GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS["cosmos3-serving"] == (
        "sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180"
    )
    users = re.findall(r"(?im)^USER\s+(\S+)$", _instructions())
    assert users[-1] == "ubuntu"
    assert "--start-period=1800s" in _instructions()


def test_runtime_bootstrap_requires_operator_license_acceptance() -> None:
    bootstrap = RUNTIME_BOOTSTRAP.read_text(encoding="utf-8")
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE" in bootstrap
    assert "cuda-bindings" in bootstrap
    assert "exit 78" in bootstrap
    assert "ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES" not in _instructions()

    env = {"PATH": "/usr/bin:/bin"}
    refused = subprocess.run(
        ["bash", RUNTIME_BOOTSTRAP],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 78
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES" in refused.stderr


def test_access_preflight_requires_token_and_probes_both_repositories(
    monkeypatch,
) -> None:
    module = _module("cosmos3_access_preflight", ACCESS_PREFLIGHT)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert module.main() == 2

    probed: list[str] = []

    def model_info(repo_id, _token):
        probed.append(repo_id)
        return {"sha": "abc123"}

    monkeypatch.setenv("HF_TOKEN", "test-only-placeholder")
    monkeypatch.setattr(module, "_model_info", model_info)
    assert module.main() == 0
    assert probed == ["nvidia/Cosmos3-Super", "nvidia/Cosmos-1.0-Guardrail"]


def test_access_preflight_preserves_repository_path_separator(monkeypatch) -> None:
    module = _module("cosmos3_access_preflight_url", ACCESS_PREFLIGHT)
    requested: list[str] = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(request, timeout):
        requested.append(request.full_url)
        assert timeout == 30
        return Response(b'{"sha":"abc123"}')

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    assert module._model_info("nvidia/Cosmos3-Super", "placeholder") == {
        "sha": "abc123"
    }
    assert requested == ["https://huggingface.co/api/models/nvidia/Cosmos3-Super"]


def test_access_preflight_refuses_pinned_revision_drift(monkeypatch) -> None:
    module = _module("cosmos3_access_preflight_revision", ACCESS_PREFLIGHT)
    monkeypatch.setenv("HF_TOKEN", "test-only-placeholder")
    monkeypatch.setenv("NPA_COSMOS3_SERVE_MODEL_REVISION", "expected-model")
    monkeypatch.setattr(module, "_model_info", lambda *_: {"sha": "other-model"})
    assert module.main() == 5


def test_guardrail_runtime_materializes_snapshot_symlinks(tmp_path: Path) -> None:
    module = _module("cosmos3_guardrail_prep", GUARDRAIL_PREP)
    blob = tmp_path / "blob"
    blob.write_text("runtime-only data", encoding="utf-8")
    blocklist = tmp_path / "blocklist"
    blocklist.mkdir()
    link = blocklist / "data.txt"
    link.symlink_to(blob)

    assert module.materialize_symlinks(blocklist) == 1
    assert not link.is_symlink()
    assert link.read_text(encoding="utf-8") == "runtime-only data"
    assert module.materialize_symlinks(blocklist) == 0
    assert link.is_file()
    assert not link.is_symlink()
    pin = HF_SNAPSHOT_PIN.read_text(encoding="utf-8")
    assert "NPA_COSMOS3_SERVE_GUARDRAIL_REVISION" in pin
    assert "refusing unpinned guardrail revision" in pin


@pytest.fixture
def harness(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv"
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "nvidia-smi").write_text(
        "#!/bin/sh\nseq 0 $(( ${NPA_TEST_GPUS:-8} - 1 ))\n", encoding="utf-8"
    )
    (bin_dir / "vllm").write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {argv_file}\n", encoding="utf-8"
    )
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    cache = tmp_path / "cache"
    cache.mkdir()

    def run(gpus: int = 8, **overrides):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HF_HOME": str(cache),
            "HF_TOKEN": "test-only-placeholder",
            "NPA_TEST_GPUS": str(gpus),
        }
        env.update({key: str(value) for key, value in overrides.items()})
        argv_file.unlink(missing_ok=True)
        result = subprocess.run(
            [ENTRYPOINT], capture_output=True, text=True, env=env, check=False
        )
        command = argv_file.read_text().strip() if argv_file.exists() else ""
        return result, command

    return run


def test_entrypoint_enforces_eight_gpus_and_guardrails_on(harness) -> None:
    result, command = harness(8)
    assert result.returncode == 0, result.stderr
    assert command.startswith("serve nvidia/Cosmos3-Super --revision ")
    assert " --omni " in command
    assert "--cfg-parallel-size 2" in command
    assert "--ulysses-degree 4" in command
    assert "--hsdp-shard-size 8" in command
    assert "--no-guardrails" not in command

    result, command = harness(7)
    assert result.returncode != 0
    assert not command
    assert "needs 8 GPUs, found 7" in result.stderr


def test_serving_smoke_requires_real_video_and_release_topology() -> None:
    smoke = SERVING_SMOKE.read_text(encoding="utf-8")
    assert "/v1/videos/sync" in smoke
    assert "ffprobe" in smoke
    assert 'codec_name") != "h264"' in smoke
    assert 'result["guardrails"] != "on"' in smoke
    assert 'result["gpu_count"] != 8' in smoke


def _docker_save(path: Path, *, layer_paths: list[str], created_by: str = "") -> None:
    layer_data = BytesIO()
    with tarfile.open(fileobj=layer_data, mode="w") as layer:
        for name in layer_paths:
            info = tarfile.TarInfo(name)
            info.size = 0
            layer.addfile(info, BytesIO())
    config = json.dumps(
        {"config": {}, "history": [{"created_by": created_by}]}
    ).encode()
    manifest = json.dumps(
        [
            {
                "Config": "config.json",
                "RepoTags": ["test:latest"],
                "Layers": ["layer.tar"],
            }
        ]
    ).encode()
    with tarfile.open(path, mode="w") as outer:
        for name, data in (
            ("config.json", config),
            ("manifest.json", manifest),
            ("layer.tar", layer_data.getvalue()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            outer.addfile(info, BytesIO(data))


def test_built_payload_scanner_rejects_old_vendor_base_and_license(
    tmp_path: Path,
) -> None:
    scanner = _module(
        "cosmos3_payload_scanner",
        NPA_ROOT / "scripts/scan_image_cosmos3_serving_payload.py",
    )
    clean = tmp_path / "clean.tar"
    _docker_save(clean, layer_paths=["usr/local/bin/curl"])
    assert scanner.scan_tarball(clean)["verdict"] == "clean"

    dirty = tmp_path / "dirty.tar"
    _docker_save(
        dirty,
        layer_paths=["NGC-DL-CONTAINER-LICENSE"],
        created_by="FROM vllm/vllm-omni:cosmos3",
    )
    assert scanner.scan_tarball(dirty)["verdict"] == "restricted-payload-detected"

    baked_closure = tmp_path / "baked-closure.tar"
    _docker_save(baked_closure, layer_paths=["usr/local/bin/vllm"])
    assert (
        scanner.scan_tarball(baked_closure)["verdict"]
        == "restricted-payload-detected"
    )
