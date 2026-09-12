"""Mutation-sensitive byte tests for the LIBERO neutral image scanner."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCANNER = ROOT / "npa" / "scripts" / "scan_image_libero_payload.py"
REVISION = "1" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scan_image_libero_payload_test", SCANNER
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layer(
    path: Path, members: dict[str, bytes], *, neutral_link: bool = False
) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        if neutral_link:
            link = tarfile.TarInfo("opt/byof")
            link.type = tarfile.SYMTYPE
            link.linkname = "/workspace/.cache/npa/libero/current/source"
            archive.addfile(link)
    return path


def _descriptor(content: bytes, media_type: str) -> dict[str, object]:
    return {
        "mediaType": media_type,
        "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _outer_archive(
    path: Path, members: dict[str, bytes], *, directories: tuple[str, ...] = ()
) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        for name in directories:
            member = tarfile.TarInfo(name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)
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
            "materials": [
                {
                    "uri": f"pkg:docker/python@{module.BASE_MANIFEST}",
                    "digest": {"sha256": module.BASE_MANIFEST.removeprefix("sha256:")},
                }
            ],
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
    def contains_neutral_link(layer: Path) -> bool:
        with tarfile.open(layer, "r:") as archive:
            return "opt/byof" in archive.getnames()

    if not any(contains_neutral_link(layer) for layer in layers):
        layers = [
            *layers,
            _layer(layers[-1].parent / "neutral-link.tar", {}, neutral_link=True),
        ]
    provenance_bytes, provenance = _base_provenance(module)
    if base_provenance is not None:
        provenance = base_provenance
        provenance_bytes = (json.dumps(provenance, sort_keys=True) + "\n").encode()
    _findings, inventory = module._layer_graph_findings(layers)
    return module.scan_tars(
        layers,
        config,
        metadata,
        provenance_bytes,
        provenance,
        observed_config_digest="sha256:" + "2" * 64,
        expected_image_inventory_sha256=inventory.sha256,
        expected_config_digest="sha256:" + "2" * 64,
    )


def test_scanner_accepts_only_neutral_bytes_and_independent_lineage(tmp_path) -> None:
    module = _load_module()
    first = _layer(tmp_path / "base.tar", {"usr/bin/sh": b"neutral shell fixture\n"})
    second = _layer(
        tmp_path / "npa.tar",
        {"opt/npa/libero/runtime-manifest.json": b'{"payload":"metadata only"}\n'},
    )

    assert _scan(module, [first, second], _config(module), _metadata(module)) == []


def test_scanner_accepts_neutral_bootstrap_symlink_and_system_metadata(
    tmp_path,
) -> None:
    module = _load_module()
    layer = tmp_path / "neutral.tar"
    with tarfile.open(layer, "w") as archive:
        for name in (
            "etc/security/namespace.init",
            "usr/local/lib/python3.10/site-packages/distutils-precedence.pth",
        ):
            content = b"neutral system metadata\n"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("opt/byof")
        link.type = tarfile.SYMTYPE
        link.linkname = "/workspace/.cache/npa/libero/current/source"
        archive.addfile(link)

    assert _scan(module, [layer], _config(module), _metadata(module)) == []


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (
            "USR/LOCAL/LIB/PYTHON3.10/SITE-PACKAGES/DISTUTILS-PRECEDENCE.PTH",
            "model_weight_checkpoint_or_dataset",
        ),
        ("Etc/Security/Namespace.Init", "libero_task_or_render_asset"),
    ],
)
def test_scanner_refuses_case_variant_of_exact_system_metadata_exception(
    tmp_path, path, kind
) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "case-drift.tar", {path: b"not exact metadata\n"})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == kind and item.path == path for item in findings)


def test_scanner_allows_only_exact_audited_secret_literal_bytes(tmp_path) -> None:
    module = _load_module()
    path = "usr/lib/x86_64-linux-gnu/libneutral.so.1"
    content = b"binary credential-shaped literal"
    module.SECRET_CONTENT = (re.compile(rb"credential-shaped"),)
    module.AUDITED_SECRET_LITERAL_FILE_SHA256 = {
        path: hashlib.sha256(content).hexdigest()
    }
    accepted = _layer(tmp_path / "accepted.tar", {path: content})
    drifted = _layer(tmp_path / "drifted.tar", {path: content + b" changed"})

    assert _scan(module, [accepted], _config(module), _metadata(module)) == []
    findings = _scan(module, [drifted], _config(module), _metadata(module))

    assert any(item.kind == "audited_literal_byte_drift" for item in findings)
    assert any(item.kind == "credential_content" for item in findings)


def test_scanner_accepts_only_the_reviewed_smoke_driver_at_its_exact_path(
    tmp_path,
) -> None:
    module = _load_module()
    smoke = ROOT / "npa" / "docker" / "workbench" / "libero" / "libero_smoke.py"
    content = smoke.read_bytes()
    assert (
        hashlib.sha256(content).hexdigest()
        == module.NEUTRAL_PAYLOAD_CONTENT_ALLOWLIST["opt/npa/libero/libero_smoke.py"]
    )
    layer = _layer(tmp_path / "layer.tar", {"opt/npa/libero/libero_smoke.py": content})

    assert _scan(module, [layer], _config(module), _metadata(module)) == []


@pytest.mark.parametrize(
    ("path", "content", "kind"),
    [
        ("usr/lib/python3/site-packages/torch/__init__.py", b"", "torch_distribution"),
        ("workspace/demo.hdf5", b"data", "model_weight_checkpoint_or_dataset"),
        ("tmp/policy.pth", b"weights", "model_weight_checkpoint_or_dataset"),
        ("tmp/policy.h5", b"weights", "model_weight_checkpoint_or_dataset"),
        ("tmp/policy.onnx", b"weights", "model_weight_checkpoint_or_dataset"),
        ("tmp/model-policy.pt", b"weights", "model_weight_checkpoint_or_dataset"),
        ("etc/unrelated-task.init", b"state", "libero_task_or_render_asset"),
        ("root/.aws/credentials", b"neutral", "credential_or_private_configuration"),
        (
            "etc/ssh/ssh_host_ed25519_key",
            b"private host key",
            "credential_or_private_configuration",
        ),
        (
            "etc/ssh/ssh_host_ed25519_key.pub",
            b"public host key",
            "credential_or_private_configuration",
        ),
        ("opt/cache/readme", b"AKIA0000000000000000", "credential_content"),
        ("usr/lib/libcudart.so.12", b"ELF", "cuda_or_nvidia_payload"),
        ("workspace/.cache/npa/libero/current/file", b"x", "populated_runtime_cache"),
        ("workspace/byof-runs/run/checkpoint.pth", b"x", "workflow_output"),
        (
            "opt/byof/libero/lifelong/algos/base.py",
            b"source",
            "libero_source_outside_neutral_bootstrap",
        ),
        (
            "opt/torch/lib/libtorch.so",
            b"runtime",
            "torch_runtime_outside_python_distribution",
        ),
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


def test_scanner_refuses_non_symlink_neutral_source_path(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/byof": b"hidden source"})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "neutral_bootstrap_link" for item in findings)


def test_scanner_requires_neutral_bootstrap_symlink(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "missing-link.tar", {"opt/neutral": b"neutral\n"})

    findings, _inventory = module._layer_graph_findings([layer])

    assert any(item.kind == "neutral_bootstrap_link" for item in findings)


def test_scanner_refuses_wrong_neutral_bootstrap_symlink_target(tmp_path) -> None:
    module = _load_module()
    layer = tmp_path / "wrong-link.tar"
    with tarfile.open(layer, "w") as archive:
        link = tarfile.TarInfo("opt/byof")
        link.type = tarfile.SYMTYPE
        link.linkname = "/workspace/.cache/npa/libero/other/source"
        archive.addfile(link)

    findings, _inventory = module._layer_graph_findings([layer])

    assert any(item.kind == "neutral_bootstrap_link" for item in findings)


@pytest.mark.parametrize(
    "content",
    [
        b"from libero.lifelong.algos import Sequential\n",
        b"import robomimic.utils.obs_utils\n",
        b"from torch._C import TensorBase\n",
        b"import torchvision.transforms\n",
        b"import mujoco\n",
        b"from robosuite import macros\n",
    ],
)
def test_scanner_refuses_renamed_runtime_source_by_content(tmp_path, content) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"srv/renamed/module.py": content})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "renamed_runtime_payload_content" for item in findings)


def test_scanner_refuses_payload_signature_at_allowlisted_path_when_bytes_drift(
    tmp_path,
) -> None:
    module = _load_module()
    layer = _layer(
        tmp_path / "layer.tar",
        {
            "opt/npa/libero/libero_smoke.py": (
                b"from libero.lifelong.algos import Sequential\n# substituted bytes\n"
            )
        },
    )

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "renamed_runtime_payload_content" for item in findings)


def test_scanner_refuses_exact_runtime_payload_bytes_under_neutral_name(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    content = b"exact runtime artifact fixture"
    monkeypatch.setattr(
        module,
        "_runtime_payload_hashes",
        lambda: frozenset({hashlib.sha256(content).hexdigest()}),
    )
    layer = _layer(tmp_path / "layer.tar", {"srv/neutral-name": content})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "exact_runtime_payload_bytes" for item in findings)


def test_scanner_refuses_nonzero_tar_padding(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/neutral": b"x"})
    payload = bytearray(layer.read_bytes())
    with tarfile.open(layer, "r:") as archive:
        member = archive.getmember("opt/neutral")
    payload[member.offset_data + member.size] = 1
    layer.write_bytes(payload)

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "nonzero_archive_padding" for item in findings)


def test_scanner_refuses_unaccounted_trailing_archive_bytes(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/neutral": b"x"})
    layer.write_bytes(layer.read_bytes() + b"not-accounted")

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "unaccounted_archive_bytes" for item in findings)


def test_scanner_refuses_unsupported_archive_member(tmp_path) -> None:
    module = _load_module()
    layer = tmp_path / "layer.tar"
    with tarfile.open(layer, "w") as archive:
        member = tarfile.TarInfo("opt/neutral-pipe")
        member.type = tarfile.FIFOTYPE
        archive.addfile(member)

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "unsupported_archive_member" for item in findings)


def test_scanner_validates_whiteouts_and_flattened_rootfs(tmp_path) -> None:
    module = _load_module()
    first = _layer(tmp_path / "first.tar", {"opt/neutral": b"x"}, neutral_link=True)
    second = _layer(tmp_path / "second.tar", {"opt/.wh.neutral": b""})
    exported = _layer(tmp_path / "rootfs.tar", {"usr/bin/sh": b"shell"})

    valid, _inventory = module._layer_graph_findings([first, second])
    mismatch, _inventory = module._layer_graph_findings([first, second], exported)

    assert valid == []
    assert any(item.kind == "flattened_rootfs_mismatch" for item in mismatch)


def test_scanner_accounts_export_root_and_runtime_injected_files(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"usr/bin/sh": b"shell"}, neutral_link=True)
    exported = tmp_path / "rootfs.tar"
    with tarfile.open(exported, "w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        content = b"shell"
        shell = tarfile.TarInfo("usr/bin/sh")
        shell.size = len(content)
        archive.addfile(shell, io.BytesIO(content))
        console = tarfile.TarInfo("dev/console")
        console.size = 0
        archive.addfile(console, io.BytesIO())
        mtab = tarfile.TarInfo("etc/mtab")
        mtab.type = tarfile.SYMTYPE
        mtab.linkname = "/proc/mounts"
        archive.addfile(mtab)
        byof = tarfile.TarInfo("opt/byof")
        byof.type = tarfile.SYMTYPE
        byof.linkname = "/workspace/.cache/npa/libero/current/source"
        archive.addfile(byof)

    findings, _inventory = module._layer_graph_findings([layer], exported)

    assert findings == []


def test_scanner_refuses_non_directory_archive_root(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "root-file.tar", {".": b"not a directory"})

    findings, _inventory = module._layer_graph_findings([layer])

    assert any(item.kind == "unsafe_archive_member" for item in findings)


def test_scanner_refuses_deleted_layer_bytes_outside_accepted_private_inventory(
    tmp_path,
) -> None:
    module = _load_module()
    accepted_base = _layer(
        tmp_path / "accepted-base.tar", {"opt/neutral": b"reviewed\n"}
    )
    accepted_delete = _layer(tmp_path / "accepted-delete.tar", {"srv/.wh.hidden": b""})
    _findings, inventory = module._layer_graph_findings(
        [accepted_base, accepted_delete]
    )
    candidate_base = _layer(
        tmp_path / "candidate-base.tar",
        {
            "opt/neutral": b"reviewed\n",
            "srv/hidden": b"opaque compiled payload with relative imports",
        },
    )
    candidate_delete = _layer(
        tmp_path / "candidate-delete.tar", {"srv/.wh.hidden": b""}
    )
    provenance_bytes, provenance = _base_provenance(module)

    findings = module.scan_tars(
        [candidate_base, candidate_delete],
        _config(module),
        _metadata(module),
        provenance_bytes,
        provenance,
        observed_config_digest="sha256:" + "2" * 64,
        expected_image_inventory_sha256=inventory.sha256,
        expected_config_digest="sha256:" + "2" * 64,
    )

    assert any(item.kind == "accepted_private_stage_identity" for item in findings)


def test_scanner_refuses_without_accepted_private_stage_identity(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/neutral": b"reviewed\n"})
    provenance_bytes, provenance = _base_provenance(module)

    findings = module.scan_tars(
        [layer],
        _config(module),
        _metadata(module),
        provenance_bytes,
        provenance,
        observed_config_digest="sha256:" + "2" * 64,
        expected_image_inventory_sha256="",
        expected_config_digest="",
    )

    assert sum(item.kind == "accepted_private_stage_identity" for item in findings) == 2


def test_scanner_emits_complete_canonical_image_inventory(tmp_path) -> None:
    module = _load_module()
    layer = _layer(
        tmp_path / "layer.tar",
        {"opt/neutral": b"reviewed\n", "srv/other": b"other\n"},
        neutral_link=True,
    )
    _findings, accepted = module._layer_graph_findings([layer])
    provenance_bytes, provenance = _base_provenance(module)
    inventory_path = tmp_path / "inventory.json"
    evidence = {}

    findings = module.scan_tars(
        [layer],
        _config(module),
        _metadata(module),
        provenance_bytes,
        provenance,
        observed_config_digest="sha256:" + "2" * 64,
        expected_image_inventory_sha256=accepted.sha256,
        expected_config_digest="sha256:" + "2" * 64,
        inventory_output=inventory_path,
        evidence=evidence,
    )

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert findings == []
    assert inventory_path.stat().st_mode & 0o777 == 0o600
    assert inventory["schema"] == module.IMAGE_INVENTORY_SCHEMA
    assert len(inventory["layers"]) == 1
    assert inventory["layers"][0]["position"] == 0
    assert (
        inventory["layers"][0]["sha256"]
        == hashlib.sha256(layer.read_bytes()).hexdigest()
    )
    assert inventory["layers"][0]["size_bytes"] == len(layer.read_bytes())
    assert [item["path"] for item in inventory["rootfs_entries"]] == [
        "opt/byof",
        "opt/neutral",
        "srv/other",
    ]
    assert evidence == {
        "image_inventory_sha256": accepted.sha256,
        "image_inventory_layer_count": 1,
        "image_inventory_rootfs_entry_count": 3,
        "observed_config_digest": "sha256:" + "2" * 64,
    }


def test_scanner_refuses_malformed_whiteout(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/.wh.payload": b"not-empty"})

    findings = _scan(module, [layer], _config(module), _metadata(module))

    assert any(item.kind == "malformed_whiteout" for item in findings)


def test_scanner_refuses_outer_member_not_bound_by_docker_manifest(tmp_path) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "inner.tar", {"opt/neutral": b"x"})
    manifest = json.dumps(
        [{"Config": "config.json", "Layers": ["layer.tar"]}], sort_keys=True
    ).encode()
    archive = _layer(
        tmp_path / "image.tar",
        {
            "manifest.json": manifest,
            "config.json": b"{}",
            "layer.tar": layer.read_bytes(),
            "unbound.bin": b"hidden",
        },
    )

    findings = module._docker_save_outer_findings(archive)

    assert any(item.kind == "unaccounted_outer_archive_member" for item in findings)


def test_scanner_refuses_duplicate_outer_archive_member(tmp_path) -> None:
    module = _load_module()
    archive = tmp_path / "duplicate-image.tar"
    manifest = b'[{"Config":"config.json","Layers":[]}]'
    with tarfile.open(archive, "w") as output:
        for name, content in (
            ("manifest.json", manifest),
            ("./manifest.json", manifest),
            ("config.json", b"{}"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))

    findings = module._docker_save_outer_findings(archive)

    assert any(item.kind == "duplicate_outer_archive_member" for item in findings)


def test_scanner_accepts_digest_bound_oci_index_descendants(tmp_path) -> None:
    module = _load_module()

    def descriptor(content: bytes, media_type: str) -> dict[str, object]:
        return {
            "mediaType": media_type,
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    config = b"{}"
    layer = b"neutral layer bytes"
    config_descriptor = descriptor(config, "application/vnd.oci.image.config.v1+json")
    layer_descriptor = descriptor(layer, "application/vnd.oci.image.layer.v1.tar+gzip")
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": [layer_descriptor],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = descriptor(
        image_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    layer_path = "blobs/sha256/" + str(layer_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    docker_manifest = json.dumps(
        [{"Config": config_path, "Layers": [layer_path]}],
        separators=(",", ":"),
    ).encode()
    archive = _layer(
        tmp_path / "image.tar",
        {
            "manifest.json": docker_manifest,
            "index.json": index,
            "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
            config_path: config,
            layer_path: layer,
            manifest_path: image_manifest,
        },
    )

    assert module._docker_save_outer_findings(archive) == []


def test_scanner_refuses_oci_document_media_type_drift(tmp_path) -> None:
    module = _load_module()
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "config": {},
            "layers": [],
        },
        separators=(",", ":"),
    ).encode()
    descriptor = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "digest": "sha256:" + hashlib.sha256(image_manifest).hexdigest(),
        "size": len(image_manifest),
    }
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [descriptor]},
        separators=(",", ":"),
    ).encode()
    member = "blobs/sha256/" + str(descriptor["digest"])[7:]
    archive = _layer(
        tmp_path / "media-type-drift.tar",
        {
            "manifest.json": b'[{"Config":"","Layers":[]}]',
            "index.json": index,
            member: image_manifest,
        },
    )

    with pytest.raises(RuntimeError, match="media type differs from descriptor"):
        module._docker_save_outer_findings(archive)


@pytest.mark.parametrize(
    ("attestation", "kind", "runtime_identity"),
    [
        ({"predicate": "AKIA0000000000000000"}, "credential_content", False),
        (
            {"predicate": "neutral runtime artifact"},
            "exact_runtime_payload_bytes",
            True,
        ),
    ],
)
def test_scanner_refuses_restricted_content_in_oci_attestation(
    monkeypatch, tmp_path, attestation, kind, runtime_identity
) -> None:
    module = _load_module()

    def descriptor(content: bytes, media_type: str) -> dict[str, object]:
        return {
            "mediaType": media_type,
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    config = b"{}"
    payload = json.dumps(attestation, separators=(",", ":")).encode()
    if runtime_identity:
        monkeypatch.setattr(
            module,
            "_runtime_payload_hashes",
            lambda: frozenset({hashlib.sha256(payload).hexdigest()}),
        )
    config_descriptor = descriptor(config, "application/vnd.oci.empty.v1+json")
    layer_descriptor = descriptor(payload, "application/vnd.in-toto+json")
    attestation_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": [layer_descriptor],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = descriptor(
        attestation_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    layer_path = "blobs/sha256/" + str(layer_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    archive = _layer(
        tmp_path / f"attestation-{kind}.tar",
        {
            "manifest.json": json.dumps(
                [{"Config": config_path, "Layers": []}], separators=(",", ":")
            ).encode(),
            "index.json": index,
            config_path: config,
            layer_path: payload,
            manifest_path: attestation_manifest,
        },
    )

    findings = module._docker_save_outer_findings(archive)

    assert any(item.kind == kind and item.path == layer_path for item in findings)


def test_scanner_refuses_oci_runtime_layer_absent_from_docker_manifest(
    tmp_path,
) -> None:
    module = _load_module()

    def descriptor(content: bytes, media_type: str) -> dict[str, object]:
        return {
            "mediaType": media_type,
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    config = b"{}"
    auxiliary_layer = b"unlisted runtime layer"
    config_descriptor = descriptor(config, "application/vnd.oci.image.config.v1+json")
    layer_descriptor = descriptor(
        auxiliary_layer, "application/vnd.oci.image.layer.v1.tar+gzip"
    )
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": [layer_descriptor],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = descriptor(
        image_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    layer_path = "blobs/sha256/" + str(layer_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    archive = _layer(
        tmp_path / "auxiliary-layer.tar",
        {
            "manifest.json": json.dumps(
                [{"Config": config_path, "Layers": []}], separators=(",", ":")
            ).encode(),
            "index.json": index,
            config_path: config,
            layer_path: auxiliary_layer,
            manifest_path: image_manifest,
        },
    )

    findings = module._docker_save_outer_findings(archive)

    assert any(
        item.kind == "auxiliary_oci_runtime_layer" and item.path == layer_path
        for item in findings
    )


@pytest.mark.parametrize(
    ("role", "media_type"),
    [
        ("config", "application/vnd.oci.image.config.v1+json"),
        ("layer", "application/vnd.in-toto+json"),
    ],
)
def test_scanner_refuses_nonobject_oci_json_leaf(role, media_type) -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match=f"OCI {role} JSON descriptor target"):
        module._oci_descriptor_content_findings(
            member_name="blobs/sha256/" + "0" * 64,
            blob=b"[]",
            media_type=media_type,
            role=role,
            legacy_layers=frozenset(),
            runtime_payload_hashes=frozenset(),
        )


def test_scanner_refuses_restricted_content_in_outer_metadata(tmp_path) -> None:
    module = _load_module()
    archive = _layer(
        tmp_path / "metadata-secret.tar",
        {
            "manifest.json": b'[{"Config":"config.json","Layers":[]}]',
            "config.json": b"{}",
            "repositories": b"AKIA0000000000000000",
        },
    )

    findings = module._docker_save_outer_findings(archive)

    assert any(
        item.kind == "credential_content" and item.path == "repositories"
        for item in findings
    )


def test_scanner_refuses_oci_descriptor_byte_drift(tmp_path) -> None:
    module = _load_module()
    expected = b'{"schemaVersion":2,"config":{},"layers":[]}'
    descriptor = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "digest": "sha256:" + hashlib.sha256(expected).hexdigest(),
        "size": len(expected),
    }
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [descriptor]},
        separators=(",", ":"),
    ).encode()
    member = "blobs/sha256/" + str(descriptor["digest"])[7:]
    archive = _layer(
        tmp_path / "image.tar",
        {
            "manifest.json": b'[{"Config":"","Layers":[]}]',
            "index.json": index,
            member: b"drifted!",
        },
    )

    findings = module._docker_save_outer_findings(archive)

    assert any(item.kind == "oci_descriptor_identity_mismatch" for item in findings)


@pytest.mark.parametrize("mutation", ["missing", "non-regular"])
def test_scanner_refuses_missing_or_nonregular_oci_descriptor_target(
    tmp_path, mutation
) -> None:
    module = _load_module()
    content = b'{"schemaVersion":2,"config":{},"layers":[]}'
    descriptor = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [descriptor]},
        separators=(",", ":"),
    ).encode()
    member_name = "blobs/sha256/" + str(descriptor["digest"])[7:]
    archive_path = tmp_path / f"{mutation}.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, payload in (
            ("manifest.json", b'[{"Config":"","Layers":[]}]'),
            ("index.json", index),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if mutation == "non-regular":
            member = tarfile.TarInfo(member_name)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)

    findings = module._docker_save_outer_findings(archive_path)

    expected = (
        "missing_oci_descriptor_member"
        if mutation == "missing"
        else "invalid_oci_descriptor_member"
    )
    assert any(item.kind == expected for item in findings)


@pytest.mark.parametrize("mutation", ["missing", "non-regular"])
def test_scanner_refuses_missing_or_nonregular_recursive_oci_config(
    tmp_path, mutation
) -> None:
    module = _load_module()

    def descriptor(content: bytes, media_type: str) -> dict[str, object]:
        return {
            "mediaType": media_type,
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    config = b"{}"
    config_descriptor = descriptor(config, "application/vnd.oci.image.config.v1+json")
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": [],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = descriptor(
        image_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    archive_path = tmp_path / f"recursive-{mutation}.tar"
    with tarfile.open(archive_path, "w") as archive:
        members = (
            (
                "manifest.json",
                json.dumps(
                    [{"Config": config_path, "Layers": []}], separators=(",", ":")
                ).encode(),
            ),
            ("index.json", index),
            (manifest_path, image_manifest),
        )
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if mutation == "non-regular":
            member = tarfile.TarInfo(config_path)
            member.type = tarfile.DIRTYPE
            archive.addfile(member)

    findings = module._docker_save_outer_findings(archive_path)

    expected = (
        "missing_oci_descriptor_member"
        if mutation == "missing"
        else "invalid_oci_descriptor_member"
    )
    assert any(item.kind == expected and item.path == config_path for item in findings)


@pytest.mark.parametrize("mutation", ["missing", "non-regular", "drift", "malformed"])
def test_scanner_refuses_invalid_recursive_oci_subject(tmp_path, mutation) -> None:
    module = _load_module()
    manifest_media_type = "application/vnd.oci.image.manifest.v1+json"
    config = b"{}"
    config_descriptor = _descriptor(config, "application/vnd.oci.image.config.v1+json")
    subject = (
        b"[]"
        if mutation == "malformed"
        else json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": manifest_media_type,
                "config": config_descriptor,
                "layers": [],
            },
            separators=(",", ":"),
        ).encode()
    )
    subject_descriptor = _descriptor(subject, manifest_media_type)
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": manifest_media_type,
            "config": config_descriptor,
            "layers": [],
            "subject": subject_descriptor,
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = _descriptor(image_manifest, manifest_media_type)
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    subject_path = "blobs/sha256/" + str(subject_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    members = {
        "manifest.json": json.dumps(
            [{"Config": config_path, "Layers": []}], separators=(",", ":")
        ).encode(),
        "index.json": index,
        config_path: config,
        manifest_path: image_manifest,
    }
    directories: tuple[str, ...] = ()
    if mutation == "non-regular":
        directories = (subject_path,)
    elif mutation == "drift":
        members[subject_path] = b"drifted subject"
    elif mutation != "missing":
        members[subject_path] = subject
    archive = _outer_archive(
        tmp_path / f"subject-{mutation}.tar", members, directories=directories
    )

    if mutation == "malformed":
        with pytest.raises(RuntimeError, match="OCI descriptor document is malformed"):
            module._docker_save_outer_findings(archive)
        return

    findings = module._docker_save_outer_findings(archive)
    expected = {
        "missing": "missing_oci_descriptor_member",
        "non-regular": "invalid_oci_descriptor_member",
        "drift": "oci_descriptor_identity_mismatch",
    }[mutation]
    assert any(item.kind == expected and item.path == subject_path for item in findings)


@pytest.mark.parametrize("mutation", ["missing", "non-regular", "drift", "malformed"])
def test_scanner_refuses_invalid_oci_attestation_descendant(tmp_path, mutation) -> None:
    module = _load_module()
    config = b'{"architecture":"amd64"}'
    attestation = b"[]" if mutation == "malformed" else b"{}"
    config_descriptor = _descriptor(config, "application/vnd.oci.empty.v1+json")
    layer_descriptor = _descriptor(attestation, "application/vnd.in-toto+json")
    image_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_descriptor,
            "layers": [layer_descriptor],
        },
        separators=(",", ":"),
    ).encode()
    manifest_descriptor = _descriptor(
        image_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    config_path = "blobs/sha256/" + str(config_descriptor["digest"])[7:]
    layer_path = "blobs/sha256/" + str(layer_descriptor["digest"])[7:]
    manifest_path = "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]
    members = {
        "manifest.json": json.dumps(
            [{"Config": config_path, "Layers": []}], separators=(",", ":")
        ).encode(),
        "index.json": index,
        config_path: config,
        manifest_path: image_manifest,
    }
    directories: tuple[str, ...] = ()
    if mutation == "non-regular":
        directories = (layer_path,)
    elif mutation == "drift":
        members[layer_path] = b"drifted attestation"
    elif mutation != "missing":
        members[layer_path] = attestation
    archive = _outer_archive(
        tmp_path / f"attestation-descendant-{mutation}.tar",
        members,
        directories=directories,
    )

    if mutation == "malformed":
        with pytest.raises(RuntimeError, match="OCI layer JSON descriptor target"):
            module._docker_save_outer_findings(archive)
        return

    findings = module._docker_save_outer_findings(archive)
    expected = {
        "missing": "missing_oci_descriptor_member",
        "non-regular": "invalid_oci_descriptor_member",
        "drift": "oci_descriptor_identity_mismatch",
    }[mutation]
    assert any(item.kind == expected and item.path == layer_path for item in findings)


def test_scanner_refuses_non_manifest_oci_index_root(tmp_path) -> None:
    module = _load_module()
    hidden = b"opaque hidden bytes"
    descriptor = {
        "mediaType": "application/octet-stream",
        "digest": "sha256:" + hashlib.sha256(hidden).hexdigest(),
        "size": len(hidden),
    }
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [descriptor]},
        separators=(",", ":"),
    ).encode()
    member = "blobs/sha256/" + str(descriptor["digest"])[7:]
    archive = _layer(
        tmp_path / "image.tar",
        {
            "manifest.json": b'[{"Config":"","Layers":[]}]',
            "index.json": index,
            member: hidden,
        },
    )

    with pytest.raises(RuntimeError, match="unsupported media type"):
        module._docker_save_outer_findings(archive)


@pytest.mark.parametrize("role", ["config", "layer", "subject"])
def test_scanner_refuses_unsupported_oci_leaf_or_subject_media_type(
    tmp_path, role
) -> None:
    module = _load_module()

    def descriptor(content: bytes, media_type: str) -> dict[str, object]:
        return {
            "mediaType": media_type,
            "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    config = b"{}"
    layer = b"neutral layer"
    subject = b'{"schemaVersion":2,"config":{},"layers":[]}'
    config_type = "application/vnd.oci.image.config.v1+json"
    layer_type = "application/vnd.oci.image.layer.v1.tar"
    subject_type = "application/vnd.oci.image.manifest.v1+json"
    if role == "config":
        config_type = "application/octet-stream"
    elif role == "layer":
        layer_type = "application/octet-stream"
    else:
        subject_type = "application/octet-stream"
    config_descriptor = descriptor(config, config_type)
    layer_descriptor = descriptor(layer, layer_type)
    subject_descriptor = descriptor(subject, subject_type)
    image_document = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": config_descriptor,
        "layers": [layer_descriptor],
    }
    if role == "subject":
        image_document["subject"] = subject_descriptor
    image_manifest = json.dumps(image_document, separators=(",", ":")).encode()
    manifest_descriptor = descriptor(
        image_manifest, "application/vnd.oci.image.manifest.v1+json"
    )
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest_descriptor]},
        separators=(",", ":"),
    ).encode()
    members = {
        "manifest.json": b'[{"Config":"","Layers":[]}]',
        "index.json": index,
        "blobs/sha256/" + str(config_descriptor["digest"])[7:]: config,
        "blobs/sha256/" + str(layer_descriptor["digest"])[7:]: layer,
        "blobs/sha256/" + str(subject_descriptor["digest"])[7:]: subject,
        "blobs/sha256/" + str(manifest_descriptor["digest"])[7:]: image_manifest,
    }
    archive = _layer(tmp_path / f"wrong-{role}.tar", members)

    with pytest.raises(RuntimeError, match=f"OCI {role} descriptor"):
        module._docker_save_outer_findings(archive)


@pytest.mark.parametrize(
    "mutation",
    ["user", "revision", "base-label", "provenance", "config-digest"],
)
def test_scanner_refuses_config_or_independent_lineage_drift(
    tmp_path, mutation
) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/npa/libero/readme": b"neutral\n"})
    config = _config(module)
    metadata = _metadata(module)
    if mutation == "user":
        config["config"]["User"] = "root"
    elif mutation == "revision":
        config["config"]["Labels"]["org.opencontainers.image.revision"] = "main"
    elif mutation == "base-label":
        config["config"]["Labels"]["org.nebius.npa.base-manifest"] = (
            "sha256:" + "0" * 64
        )
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


def test_scanner_rejects_base_identity_buried_in_unstructured_metadata(
    tmp_path,
) -> None:
    module = _load_module()
    layer = _layer(tmp_path / "layer.tar", {"opt/npa/libero/readme": b"neutral\n"})
    metadata = _metadata(
        module,
        provenance={"unrelated": {"note": module.BASE_MANIFEST}},
    )

    findings = _scan(module, [layer], _config(module), metadata)

    assert any(item.kind == "independent_build_lineage" for item in findings)


def test_docker_save_scans_require_an_independent_exported_rootfs() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module.main(
            [
                "--docker-save",
                "image.tar",
                "--build-metadata",
                "metadata.json",
                "--base-provenance",
                "provenance.json",
            ]
        )


@pytest.mark.parametrize(
    "mutation", ["subject", "rootfs", "source", "builder", "bytes"]
)
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
        module.BASE_PROVENANCE_SHA256 = hashlib.sha256(
            provenance_bytes + b"x"
        ).hexdigest()
    if mutation != "bytes":
        provenance_bytes = (json.dumps(provenance, sort_keys=True) + "\n").encode()

    findings = module.scan_tars(
        [layer],
        _config(module),
        _metadata(module),
        provenance_bytes,
        provenance,
        observed_config_digest="sha256:" + "2" * 64,
        expected_image_inventory_sha256=(
            module._layer_graph_findings([layer])[1].sha256
        ),
        expected_config_digest="sha256:" + "2" * 64,
    )

    assert any(item.kind == "independent_base_provenance" for item in findings)
