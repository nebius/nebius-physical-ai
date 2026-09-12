"""Mutation-sensitive byte tests for the LIBERO neutral image scanner."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCANNER = ROOT / "npa" / "scripts" / "scan_image_libero_payload.py"
REVISION = "1" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location("scan_image_libero_payload_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layer(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return path


def _config(module, *, user: str = "ubuntu") -> dict[str, object]:
    return {
        "config": {
            "User": user,
            "Entrypoint": ["/usr/local/bin/npa-libero-entrypoint"],
            "Labels": {
                "org.opencontainers.image.revision": REVISION,
                "org.nebius.npa.redistribution": "public-neutral-bootstrap",
                "org.nebius.npa.validation-status": "quarantined-unvalidated",
                "org.nebius.npa.base-manifest": module.BASE_MANIFEST,
                "org.nebius.npa.base-rootfs-material": module.BASE_ROOTFS_MATERIAL,
                "org.nebius.npa.skypilot-bootstrap-contract": module.BOOTSTRAP_CONTRACT,
            },
        },
        "history": [{"created_by": "COPY neutral bootstrap files /opt/npa/libero/"}],
    }


def _metadata(module, *, provenance: object | None = None) -> dict[str, object]:
    return {
        "containerimage.config.digest": "sha256:" + "2" * 64,
        "buildx.build.provenance": provenance
        if provenance is not None
        else {
            "materials": [module.BASE_MANIFEST, module.BASE_ROOTFS_MATERIAL],
            "invocation": {"source_revision": REVISION},
        },
    }


def _base_provenance(module) -> tuple[bytes, dict[str, object]]:
    provenance = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "subject": [
            {
                "name": "pkg:docker/python@3.10-slim-bookworm?platform=linux%2Famd64",
                "digest": {"sha256": module.BASE_MANIFEST.removeprefix("sha256:")},
            }
        ],
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "predicate": {
            "builder": {"id": "https://github.com/docker-library"},
            "materials": [
                {
                    "uri": "pkg:docker/oisupport/staging-amd64",
                    "digest": {
                        "sha256": module.BASE_ROOTFS_MATERIAL.removeprefix("sha256:")
                    },
                },
                {
                    "uri": "https://github.com/docker-library/python.git",
                    "digest": {"sha1": module.BASE_SOURCE_REVISION},
                },
            ],
        },
    }
    content = (json.dumps(provenance, sort_keys=True) + "\n").encode()
    module.BASE_PROVENANCE_SHA256 = hashlib.sha256(content).hexdigest()
    return content, provenance


def _scan(module, layers, config, metadata, *, base_provenance=None):
    provenance_bytes, provenance = _base_provenance(module)
    if base_provenance is not None:
        provenance = base_provenance
        provenance_bytes = (json.dumps(provenance, sort_keys=True) + "\n").encode()
    return module.scan_tars(
        layers,
        config,
        metadata,
        provenance_bytes,
        provenance,
    )


def test_scanner_accepts_only_neutral_bytes_and_independent_lineage(tmp_path) -> None:
    module = _load_module()
    first = _layer(tmp_path / "base.tar", {"usr/bin/sh": b"neutral shell fixture\n"})
    second = _layer(
        tmp_path / "npa.tar",
        {"opt/npa/libero/runtime-manifest.json": b'{"payload":"metadata only"}\n'},
    )

    assert _scan(module, [first, second], _config(module), _metadata(module)) == []


@pytest.mark.parametrize(
    ("path", "content", "kind"),
    [
        ("usr/lib/python3/site-packages/torch/__init__.py", b"", "torch_distribution"),
        ("workspace/demo.hdf5", b"data", "model_weight_checkpoint_or_dataset"),
        ("root/.aws/credentials", b"neutral", "credential_or_private_configuration"),
        ("opt/cache/readme", b"AKIA0000000000000000", "credential_content"),
        ("usr/lib/libcudart.so.12", b"ELF", "cuda_or_nvidia_payload"),
        ("workspace/.cache/npa/libero/current/file", b"x", "populated_runtime_cache"),
        ("workspace/byof-runs/run/checkpoint.pth", b"x", "workflow_output"),
    ],
)
def test_scanner_refuses_each_forbidden_boundary(tmp_path, path, content, kind) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {path: content})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == kind for item in findings)


def test_scanner_refuses_forbidden_bytes_hidden_in_nested_archive(tmp_path) -> None:
    module = _load_module()
    nested = _layer(tmp_path / "nested.tar", {"site-packages/torch/version.py": b"x"})
    layer = _layer(tmp_path / "layer.tar", {"opt/archive.tar": nested.read_bytes()})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "torch_distribution" for item in findings)


@pytest.mark.parametrize(
    "mutation",
    ["user", "revision", "base-label", "provenance", "config-digest"],
)
def test_scanner_refuses_config_or_independent_lineage_drift(tmp_path, mutation) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/npa/libero/readme": b"neutral\n"})
    config = _config(module)
    metadata = _metadata(module)
    if mutation == "user":
        config["config"]["User"] = "root"
    elif mutation == "revision":
        config["config"]["Labels"]["org.opencontainers.image.revision"] = "main"
    elif mutation == "base-label":
        config["config"]["Labels"]["org.nebius.npa.base-manifest"] = "sha256:" + "0" * 64
    elif mutation == "provenance":
        metadata["buildx.build.provenance"] = {"materials": []}
    else:
        metadata["containerimage.config.digest"] = ""

    findings = _scan(module, [layer], config, metadata)

    assert findings
    assert any(
        item.kind in {"config_contract", "independent_build_lineage"}
        for item in findings
    )


@pytest.mark.parametrize("mutation", ["subject", "rootfs", "source", "builder", "bytes"])
def test_scanner_refuses_published_base_provenance_drift(tmp_path, mutation) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/npa/libero/readme": b"neutral\n"})
    provenance_bytes, provenance = _base_provenance(module)
    if mutation == "subject":
        provenance["subject"][0]["digest"]["sha256"] = "0" * 64
    elif mutation == "rootfs":
        provenance["predicate"]["materials"][0]["digest"]["sha256"] = "0" * 64
    elif mutation == "source":
        provenance["predicate"]["materials"][1]["digest"]["sha1"] = "0" * 40
    elif mutation == "builder":
        provenance["predicate"]["builder"]["id"] = "https://example.invalid"
    else:
        module.BASE_PROVENANCE_SHA256 = hashlib.sha256(provenance_bytes + b"x").hexdigest()
    if mutation != "bytes":
        provenance_bytes = (json.dumps(provenance, sort_keys=True) + "\n").encode()

    findings = module.scan_tars(
        [layer],
        _config(module),
        _metadata(module),
        provenance_bytes,
        provenance,
    )

    assert any(item.kind == "independent_base_provenance" for item in findings)
