"""Mutation tests for the HY-World neutral-image payload scanner."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from npa.workbench.hy_world.private_delivery import (
    PrivateDeliveryError,
    bind_remote_manifest,
    private_registry,
)


def _load(name: str):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scanner = _load("scan_image_hy_world_payload")
IMAGE_ROOT = Path(__file__).resolve().parents[2] / "docker" / "workbench" / "hy-world"


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_runtime_fetch_plumbing_is_not_payload(tmp_path: Path) -> None:
    rootfs = {
        "usr/local/bin/hy-world-runtime": b"hf download tencent/HY-World-2.0\n",
        "opt/npa/hy-world/asset_contract.py": b"SOURCE_REF = 'df9988'\n",
        "usr/local/lib/python3.12/site-packages/uv/__init__.py": b"",
    }
    findings = scanner.scan(_tar(tmp_path / "clean.tar", rootfs), {"history": []})
    assert findings == [], _kinds(findings)


def test_source_and_weights_are_rejected(tmp_path: Path) -> None:
    rootfs = {
        "opt/runtime/HY-World-2.0/hyworld2/worldgen/video_gen.py": b"source",
        "workspace/model-cache/worldstereo-memory-dmd/model.safetensors": b"weights",
    }
    kinds = _kinds(
        scanner.scan(_tar(tmp_path / "payload.tar", rootfs), {"history": []})
    )
    assert "hy_world_source_tree" in kinds
    assert {"hy_world_or_worldstereo_weight", "checkpoint_or_weight"} & kinds


def test_build_time_fetch_is_rejected(tmp_path: Path) -> None:
    findings = scanner.scan(
        _tar(tmp_path / "history.tar", {"usr/local/bin/hy-world-runtime": b""}),
        {"history": [{"created_by": "RUN hy-world-runtime ensure"}]},
    )
    assert "runtime_bootstrap_at_build" in _kinds(findings)


def test_candidate_build_helper_refuses_direct_push() -> None:
    result = subprocess.run(
        [str(IMAGE_ROOT / "build.sh"), "--push"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--push requires --operator-private" in result.stderr


@pytest.mark.parametrize(
    "registry",
    (
        "team/project",  # Docker Hub shorthand is never an operator-private host.
        "docker.io/operator",
        "quay.io/operator",
        "registry.example",  # A host alone is not a registry namespace.
        "registry.example//operator",
        "https://registry.example/operator",
        "registry.example/operator?tag=unsafe",
    ),
)
def test_private_delivery_refuses_public_shorthand_and_malformed_registry(
    registry: str,
) -> None:
    with pytest.raises(PrivateDeliveryError):
        private_registry(registry)


def test_private_delivery_refuses_configured_public_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/example/public")
    with pytest.raises(PrivateDeliveryError, match="anonymous/public"):
        private_registry("ghcr.io/example/public")


def test_private_delivery_accepts_a_fully_qualified_private_namespace() -> None:
    assert private_registry("registry.example/operator-private") == (
        "registry.example/operator-private"
    )


def test_private_delivery_binds_remote_layers_to_the_scanned_local_config() -> None:
    local = "sha256:" + "a" * 64
    report = bind_remote_manifest(
        local_config_digest=local,
        remote_digest="sha256:" + "b" * 64,
        manifest={
            "schemaVersion": 2,
            "config": {"digest": local},
            "layers": [{"digest": "sha256:" + "c" * 64}],
        },
    )
    assert report == {
        "schema": "npa.hy_world.private_delivery_identity.v1",
        "status": "matched",
        "local_config_digest": local,
        "remote_manifest_digest": "sha256:" + "b" * 64,
        "remote_config_digest": local,
        "remote_layer_digests": ["sha256:" + "c" * 64],
    }


def test_private_delivery_refuses_remote_config_mismatch_before_receipt() -> None:
    local = "sha256:" + "a" * 64
    with pytest.raises(PrivateDeliveryError, match="does not match"):
        bind_remote_manifest(
            local_config_digest=local,
            remote_digest="sha256:" + "b" * 64,
            manifest={
                "schemaVersion": 2,
                "config": {"digest": "sha256:" + "c" * 64},
                "layers": [{"digest": "sha256:" + "d" * 64}],
            },
        )


def test_private_delivery_build_transaction_pushes_the_scanned_archive() -> None:
    build = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert 'private-registry "$REGISTRY"' in build
    assert 'crane push "$IMAGE_TAR" "$REMOTE_IMAGE"' in build
    assert 'crane manifest "$IMMUTABLE_IMAGE"' in build
    assert "bind-manifest" in build
    assert "trivy-policy-remote.json" in build
    assert "refusing to overwrite existing private immutable development tag" in build
    assert 'docker push "$REMOTE_IMAGE"' not in build
    assert build.index("bind-manifest") < build.index("private-delivery.json")


def test_private_delivery_private_registry_scan_uses_authenticated_host_pull() -> None:
    """Scanner containers receive bytes, never a private registry credential."""

    build = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert 'crane pull "$IMMUTABLE_IMAGE" "$REMOTE_TAR"' in build
    assert "--input /delivery/remote-image.tar" in build
    assert '--docker-save "$REMOTE_TAR"' in build
    post_pull = build.split('crane pull "$IMMUTABLE_IMAGE" "$REMOTE_TAR"', 1)[1]
    scanner_calls = post_pull.split("SOURCE_COMMIT=", 1)[0]
    assert '"$IMMUTABLE_IMAGE"' not in scanner_calls
    assert "DOCKER_CONFIG" not in build
    assert "SKYPILOT_DOCKER_PASSWORD" not in build
    assert build.index("bind-manifest") < build.index('crane pull "$IMMUTABLE_IMAGE"')
    assert build.index('crane pull "$IMMUTABLE_IMAGE"') < build.index(
        "trivy-policy-remote.json"
    )


def test_runtime_bootstrap_compiles_upstream_native_extensions_and_pins_hub_refs() -> (
    None
):
    runtime = (IMAGE_ROOT / "hy_world_runtime.sh").read_text(encoding="utf-8")
    assert "gsplat_maskgaussian" in runtime
    assert "third_party/navmesh" in runtime
    assert "PYTORCH3D_REF=" in runtime
    assert "FLASH_ATTN_VERSION=" in runtime
    assert "register_hub_main_ref" in runtime
    assert '"$refs/main"' in runtime


def test_hy_world_and_sam3_manifest_base_images_match_their_dockerfiles() -> None:
    """Safety metadata is audited provenance, not a copy-pasted description."""

    manifest = yaml.safe_load(
        (IMAGE_ROOT.parents[2] / "src/npa/smoke/golden_evals.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(manifest, dict)
    containers = manifest["containers"]
    assert isinstance(containers, dict)
    for name, dockerfile, expected in (
        (
            "sam3",
            IMAGE_ROOT.parent / "sam3" / "Dockerfile",
            "python:3.12-slim-bookworm",
        ),
        (
            "hy-world",
            IMAGE_ROOT / "Dockerfile",
            "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        ),
    ):
        first_from = (
            next(
                line
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if line.startswith("FROM ")
            )
            .split()[1]
            .split("@", 1)[0]
        )
        safety = containers[name]["safety"]
        assert first_from == expected
        assert str(safety["base_image"]).startswith(expected)
