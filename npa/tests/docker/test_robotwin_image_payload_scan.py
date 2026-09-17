"""Mutation tests for the RoboTwin zero-vendor-payload image boundary."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("scan_image_wan_payload")
scanner = _load("scan_image_robotwin_payload")


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_source_vendor_runtime_and_cuda_are_all_refused(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/script/collect_data.py": b"official source",
            "usr/local/lib/python3.10/site-packages/curobo/__init__.py": b"",
            "usr/local/cuda/lib64/libcudart.so.12": b"authorized private runtime",
            "usr/lib/libnvidia-ml.so.1": b"vendor driver runtime",
            "opaque/extension.bin": b"\x7fELF\x00libnvjitlink.so.12\x00",
        },
    )
    config = {
        "history": [
            {"created_by": "FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"},
            {"created_by": "RUN pip install -r /tmp/robotwin-requirements.lock"},
        ]
    }
    kinds = _kinds(scanner.scan(rootfs, config))
    assert {
        "robotwin_source",
        "curobo_source_or_runtime",
        "cuda_or_cudnn_runtime",
        "cuda_library",
        "cuda_elf_dependency",
        "nvidia_or_pytorch_base",
    } <= kinds


def test_asset_archives_and_extracted_assets_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/assets/objects.zip": b"asset bytes",
            "opt/robotwin/assets/embodiments/aloha-agilex/config.yml": b"x",
        },
    )
    kinds = _kinds(scanner.scan(rootfs, {}))
    assert {"robotwin_asset_archive", "robotwin_extracted_asset"} <= kinds


def test_renamed_nested_archive_is_traversed(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/opaque/payload.bin": _zip(
                {"objects/020_hammer/base0/model_data0.json": b"asset bytes"}
            )
        },
    )
    assert "robotwin_extracted_asset" in _kinds(scanner.scan(rootfs, {}))


def test_renamed_source_metadata_and_private_evidence_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opaque/source-payload.bin": (
                b"from envs._base_task import Base_Task\n"
                b"class beat_block_hammer(Base_Task):\n    pass\n"
            ),
            "opaque/metadata/.git/config": b"repository metadata",
            "renamed/credential.bin": b"AKIAABCDEFGHIJKLMNOP",
            "renamed/evidence.bin": (
                b'{"solution":"robotwin",'
                b'"ownership_provenance":"manager-issued",'
                b'"project":"private-project-canary",'
                b'"output_root":"s3://private-bucket-canary/output"}'
            ),
            "renamed/runtime-envelope.bin": (
                b'{"context_base64":"eyJwcml2YXRlIjoiYnl0ZXMifQ==",'
                b'"schema_version":"npa.byof.robotwin.runtime-authorization.v2"}'
            ),
            "renamed/customer-entitlement.bin": (
                b'{"schema_version":"npa.byof.robotwin.'
                b'authenticated-customer-authorization.v1",'
                b'"provenance":"customer-issued","decision":"accepted"}'
            ),
        },
    )

    findings = scanner.scan(rootfs, {})
    kinds = _kinds(findings)
    assert {
        "source_control_metadata",
        "credential_content",
    } <= kinds
    credential_paths = {
        finding.path for finding in findings if finding.kind == "credential_content"
    }
    assert credential_paths == {
        "opaque/source-payload.bin",
        "renamed/credential.bin",
        "renamed/customer-entitlement.bin",
        "renamed/evidence.bin",
        "renamed/runtime-envelope.bin",
    }


def test_cache_outputs_and_build_time_fetch_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "root/.cache/huggingface/hub/datasets--TianxingChen--RoboTwin2.0/ref": b"x",
            "opt/custom-cache/datasets--TianxingChen--RoboTwin2.0/snapshots/ref": b"x",
            "workspace/byof-runs/run/robotwin-native/episode_0/episode_0000000.hdf5": b"x",
        },
    )
    config = {
        "history": [
            {
                "created_by": (
                    "RUN huggingface-cli download TianxingChen/RoboTwin2.0 objects.zip"
                )
            }
        ]
    }
    kinds = _kinds(scanner.scan(rootfs, config))
    assert {
        "robotwin_huggingface_cache",
        "robotwin_generated_output",
        "robotwin_asset_fetch_at_build",
    } <= kinds


@pytest.mark.parametrize(
    ("history", "kind"),
    [
        (
            "RUN \\\n curl https://huggingface.co/TianxingChen/RoboTwin2.0/objects.zip",
            "robotwin_asset_fetch_at_build",
        ),
        (
            "RUN \\\n python -m pip install \\\n curobo",
            "vendor_runtime_installed_at_build",
        ),
        (
            "RUN \\\n git clone https://github.com/RoboTwin-Platform/RoboTwin",
            "vendor_source_fetched_at_build",
        ),
        (
            "RUN curl -o objects.zip https://huggingface.co/TianxingChen/"
            "RoboTwin2.0/objects.zip && rm -f objects.zip",
            "robotwin_asset_fetch_at_build",
        ),
    ],
)
def test_multiline_build_history_fetches_are_refused(
    tmp_path: Path, history: str, kind: str
) -> None:
    rootfs = _tar(tmp_path / "rootfs.tar", {"neutral.txt": b"neutral"})
    assert kind in _kinds(scanner.scan(rootfs, {"history": [{"created_by": history}]}))


def test_phase_a_cli_refuses_before_any_candidate_can_pass(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {"opt/npa/robotwin/REDISTRIBUTION.md": b"NPA bootstrap notice"},
    )
    output = tmp_path / "report.json"
    assert scanner.main(["--rootfs-tar", str(rootfs), "--output", str(output)]) == 2
    assert not output.exists()


def test_complete_looking_policy_cannot_activate_unimplemented_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "npa.image-native-content-policy.v1",
                "detector_identity": {
                    "config_sha256": "a" * 64,
                    "helper_sha256": "b" * 64,
                },
                "entries": [{"untrusted": "catalog-entry"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner, "PUBLIC_POLICY_PATH", policy)
    monkeypatch.setattr(
        scanner,
        "docker_save_material",
        lambda *_args, **_kwargs: pytest.fail(
            "unimplemented exact-content policy reached candidate bytes"
        ),
    )
    image = tmp_path / "candidate.tar"
    image.write_bytes(b"not-an-image")

    assert scanner.main(["--docker-save", str(image)]) == 2


def test_private_image_stdin_reader_remains_byte_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = b"x" * (scanner.MAX_IMAGE_REFERENCE_BYTES + 1)
    monkeypatch.setattr(
        scanner.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(oversized), encoding="utf-8"),
    )
    with pytest.raises(ValueError, match="too large"):
        scanner._read_image_stdin()


def test_private_image_stdin_is_bounded_and_never_echoed_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private = "registry.example/private/npa-robotwin@sha256:" + "a" * 64
    monkeypatch.setattr(
        scanner.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(private.encode()), encoding="utf-8"),
    )
    monkeypatch.setattr(
        scanner,
        "_private_remote_material",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(private)),
    )

    assert scanner.main(["--image-stdin"]) == 2
    output = capsys.readouterr().out
    assert "private image scan failed" in output
    assert private not in output

    oversized = b"x" * (scanner.MAX_IMAGE_REFERENCE_BYTES + 1)
    monkeypatch.setattr(
        scanner.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(oversized), encoding="utf-8"),
    )
    assert scanner.main(["--image-stdin"]) == 2
    assert "private image scan failed" in capsys.readouterr().out


class _RegistryResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_cross_origin_bearer_realm_never_receives_registry_basic_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scanner,
        "_docker_credentials",
        lambda _registry: {"username": "operator", "secret": "private-secret"},
    )
    observed: list[object] = []

    class FakeOpener:
        def open(self, request, **_kwargs):
            observed.append(request)
            return _RegistryResponse(b'{"token":"scoped-token"}')

    monkeypatch.setattr(scanner, "_URL_OPENER", FakeOpener())
    client = scanner._RegistryClient("registry.example", "private/image")
    token = client._bearer_authorization(
        'Bearer realm="https://attacker.example/token",service="registry.example"'
    )
    assert token == "Bearer scoped-token"
    assert observed
    assert observed[0].get_header("Authorization") is None


def test_registry_blob_limit_removes_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"x" * 9
    digest = _digest(payload)
    client = object.__new__(scanner._RegistryClient)
    monkeypatch.setattr(scanner, "MAX_OCI_COMPRESSED_LAYER_BYTES", 8)
    monkeypatch.setattr(
        client, "open", lambda *_args, **_kwargs: _RegistryResponse(payload)
    )
    destination = tmp_path / "partial.blob"

    with pytest.raises(RuntimeError, match="byte limit"):
        client.blob(digest, destination, expected_size=8)
    assert not destination.exists()


def test_oci_descriptor_and_uncompressed_limits_refuse_before_flattening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="byte limit"):
        scanner._descriptor(
            {
                "digest": "sha256:" + "a" * 64,
                "size": 9,
                "mediaType": "application/vnd.oci.image.layer.v1.tar",
            },
            "layer",
            scanner.LAYER_MEDIA_TYPES,
            max_size=8,
        )
    layer = _tar(tmp_path / "layer.tar", {"payload": b"123456789"})
    monkeypatch.setattr(scanner, "MAX_OCI_UNCOMPRESSED_LAYER_BYTES", 8)
    with pytest.raises(RuntimeError, match="uncompressed scanner byte limit"):
        scanner._uncompressed_layer_digest(
            layer, "application/vnd.oci.image.layer.v1.tar"
        )


def test_oci_layer_count_and_aggregate_quotas_are_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layers = [
        _tar_bytes({"one": b"1"}),
        _tar_bytes({"two": b"2"}),
    ]
    layer_digests = [_digest(layer) for layer in layers]
    config = json.dumps(
        {
            "architecture": "amd64",
            "history": [],
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": layer_digests},
        },
        separators=(",", ":"),
    ).encode()
    config_digest = _digest(config)
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "digest": config_digest,
                "size": len(config),
                "mediaType": "application/vnd.oci.image.config.v1+json",
            },
            "layers": [
                {
                    "digest": digest,
                    "size": len(layer),
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                }
                for digest, layer in zip(layer_digests, layers, strict=True)
            ],
        },
        separators=(",", ":"),
    ).encode()
    payloads = {config_digest: config, **dict(zip(layer_digests, layers, strict=True))}

    class FakeClient:
        def __init__(self, *_args) -> None:
            pass

        def metadata(self, *_args, **_kwargs) -> bytes:
            return manifest

        def blob(self, digest, destination, *, expected_size) -> None:
            payload = payloads[digest]
            assert len(payload) == expected_size
            destination.write_bytes(payload)

    monkeypatch.setattr(scanner, "_RegistryClient", FakeClient)
    image = "registry.example/private/image@" + _digest(manifest)

    count = tmp_path / "count"
    count.mkdir()
    monkeypatch.setattr(scanner, "MAX_OCI_LAYER_COUNT", 1)
    with pytest.raises(RuntimeError, match="layer-count"):
        scanner._private_remote_material(image, count)

    compressed = tmp_path / "compressed"
    compressed.mkdir()
    monkeypatch.setattr(scanner, "MAX_OCI_LAYER_COUNT", 2)
    monkeypatch.setattr(
        scanner, "MAX_OCI_COMPRESSED_TOTAL_BYTES", sum(map(len, layers)) - 1
    )
    with pytest.raises(RuntimeError, match="compressed aggregate"):
        scanner._private_remote_material(image, compressed)

    uncompressed = tmp_path / "uncompressed"
    uncompressed.mkdir()
    monkeypatch.setattr(
        scanner, "MAX_OCI_COMPRESSED_TOTAL_BYTES", sum(map(len, layers))
    )
    monkeypatch.setattr(
        scanner, "MAX_OCI_UNCOMPRESSED_TOTAL_BYTES", sum(map(len, layers)) - 1
    )
    with pytest.raises(RuntimeError, match="uncompressed aggregate"):
        scanner._private_remote_material(image, uncompressed)


def test_flattening_member_and_byte_quotas_are_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layer = _tar(tmp_path / "layer.tar", {"one": b"1", "two": b"2"})
    monkeypatch.setattr(scanner, "MAX_OCI_LAYER_MEMBERS", 1)
    with pytest.raises(RuntimeError, match="member-count"):
        scanner._write_flattened_rootfs([layer], tmp_path / "members.tar", tmp_path)

    # The failed member-count attempt populated its private scratch directory;
    # use a fresh owner directory for the independent byte quota mutation.
    byte_root = tmp_path / "bytes"
    byte_root.mkdir()
    byte_layer = _tar(byte_root / "layer.tar", {"payload": b"12"})
    monkeypatch.setattr(scanner, "MAX_OCI_LAYER_MEMBERS", 10)
    monkeypatch.setattr(scanner, "MAX_OCI_FLATTENED_BYTES", 1)
    with pytest.raises(RuntimeError, match="flattened byte limit"):
        scanner._write_flattened_rootfs(
            [byte_layer], byte_root / "rootfs.tar", byte_root
        )


@pytest.mark.parametrize("whiteout_first", [False, True])
def test_same_layer_normal_whiteout_never_removes_same_layer_file(
    tmp_path: Path, whiteout_first: bool
) -> None:
    lower = _tar(tmp_path / "lower.tar", {"tree/value": b"lower"})
    upper = tmp_path / "upper.tar"
    members = (
        [("tree/.wh.value", b""), ("tree/value", b"upper")]
        if whiteout_first
        else [("tree/value", b"upper"), ("tree/.wh.value", b"")]
    )
    with tarfile.open(upper, "w") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    flattened = tmp_path / "flattened.tar"
    scanner._write_flattened_rootfs([lower, upper], flattened, tmp_path)
    with tarfile.open(flattened) as archive:
        assert archive.extractfile("tree/value").read() == b"upper"


@pytest.mark.parametrize("whiteout_first", [False, True])
def test_same_layer_opaque_whiteout_never_removes_same_layer_child(
    tmp_path: Path, whiteout_first: bool
) -> None:
    lower = _tar(tmp_path / "lower.tar", {"tree/lower": b"lower"})
    upper = tmp_path / "upper.tar"
    members = (
        [("tree/.wh..wh..opq", b""), ("tree/current", b"current")]
        if whiteout_first
        else [("tree/current", b"current"), ("tree/.wh..wh..opq", b"")]
    )
    with tarfile.open(upper, "w") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    flattened = tmp_path / "flattened.tar"
    scanner._write_flattened_rootfs([lower, upper], flattened, tmp_path)
    with tarfile.open(flattened) as archive:
        assert "tree/lower" not in archive.getnames()
        assert archive.extractfile("tree/current").read() == b"current"


def test_private_registry_transport_downloads_exact_oci_bytes_without_child_image_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layer = _tar_bytes({"opt/npa/robotwin/REDISTRIBUTION.md": b"NPA notice"})
    diff_id = _digest(layer)
    config = json.dumps(
        {
            "architecture": "amd64",
            "history": [],
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": [diff_id]},
        },
        separators=(",", ":"),
    ).encode()
    layer_digest = _digest(layer)
    config_digest = _digest(config)
    platform_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "digest": config_digest,
                "size": len(config),
                "mediaType": "application/vnd.oci.image.config.v1+json",
            },
            "layers": [
                {
                    "digest": layer_digest,
                    "size": len(layer),
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    platform_digest = _digest(platform_manifest)
    index = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": platform_digest,
                    "size": len(platform_manifest),
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    index_digest = _digest(index)
    image = f"registry.example/private/npa-robotwin@{index_digest}"
    payloads = {
        f"manifests/{index_digest}": index,
        f"manifests/{platform_digest}": platform_manifest,
        f"blobs/{config_digest}": config,
        f"blobs/{layer_digest}": layer,
    }

    class FakeOpener:
        def open(self, request, **_kwargs):
            suffix = request.full_url.split("/v2/private/npa-robotwin/", 1)[1]
            return _RegistryResponse(payloads[suffix])

    monkeypatch.setattr(scanner, "_URL_OPENER", FakeOpener())
    monkeypatch.setattr(scanner, "_docker_credentials", lambda _registry: {})

    def no_child_process(argv, **_kwargs):
        assert image not in argv
        raise AssertionError("private OCI materialization must not launch a child")

    monkeypatch.setattr(scanner.subprocess, "run", no_child_process)
    tars, parsed_config = scanner._private_remote_material(image, tmp_path)

    assert len(tars) == 2
    assert parsed_config == {
        "architecture": "amd64",
        "history": [],
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": [diff_id]},
    }
    assert scanner.scan_tars(tars, parsed_config) == []


def test_private_oci_graph_requires_sizes_media_platform_and_diff_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layer = _tar_bytes({"opt/npa/robotwin/REDISTRIBUTION.md": b"NPA notice"})
    layer_digest = _digest(layer)

    def attempt(config_record: dict, manifest_updates: dict | None = None) -> None:
        config = json.dumps(config_record, separators=(",", ":")).encode()
        config_digest = _digest(config)
        manifest: dict = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "digest": config_digest,
                "size": len(config),
                "mediaType": "application/vnd.oci.image.config.v1+json",
            },
            "layers": [
                {
                    "digest": layer_digest,
                    "size": len(layer),
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                }
            ],
        }
        manifest.update(manifest_updates or {})
        raw_manifest = json.dumps(manifest, separators=(",", ":")).encode()
        manifest_digest = _digest(raw_manifest)
        payloads = {
            f"manifests/{manifest_digest}": raw_manifest,
            f"blobs/{config_digest}": config,
            f"blobs/{layer_digest}": layer,
        }

        class FakeOpener:
            def open(self, request, **_kwargs):
                suffix = request.full_url.split("/v2/private/npa-robotwin/", 1)[1]
                return _RegistryResponse(payloads[suffix])

        monkeypatch.setattr(scanner, "_URL_OPENER", FakeOpener())
        monkeypatch.setattr(scanner, "_docker_credentials", lambda _registry: {})
        image = f"registry.example/private/npa-robotwin@{manifest_digest}"
        destination = tmp_path / manifest_digest[-8:]
        destination.mkdir()
        scanner._private_remote_material(image, destination)

    valid_config = {
        "architecture": "amd64",
        "history": [],
        "os": "linux",
        "rootfs": {"type": "layers", "diff_ids": [_digest(layer)]},
    }
    with pytest.raises(RuntimeError, match="size is invalid"):
        attempt(
            valid_config,
            {
                "config": {
                    "digest": "sha256:" + "a" * 64,
                    "mediaType": "application/vnd.oci.image.config.v1+json",
                }
            },
        )
    with pytest.raises(RuntimeError, match="not linux/amd64"):
        attempt({**valid_config, "architecture": "arm64"})
    with pytest.raises(RuntimeError, match="diff IDs do not match"):
        attempt(
            {
                **valid_config,
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "b" * 64]},
            }
        )
    with pytest.raises(RuntimeError, match="media type is invalid"):
        attempt(valid_config, {"mediaType": "application/json"})


def test_private_image_cannot_enter_positional_or_descendant_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    private = "registry.example/private/npa-robotwin@sha256:" + "a" * 64
    monkeypatch.setattr(
        scanner.walker,
        "remote_material",
        lambda *_args, **_kwargs: pytest.fail("private image reached descendant argv"),
    )

    with pytest.raises(SystemExit):
        scanner.main([private])
    assert private not in capsys.readouterr().err
    assert scanner._is_public_image_reference(
        "ghcr.io/nebius/nebius-physical-ai/npa-robotwin@sha256:" + "b" * 64
    )


def test_docker_credential_helper_receives_registry_only_on_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docker_config = tmp_path / "docker"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        json.dumps({"credHelpers": {"registry.example": "unit"}}),
        encoding="utf-8",
    )
    calls: list[tuple[list[str], str]] = []

    def helper(argv, **kwargs):
        calls.append((list(argv), kwargs["input"]))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"Username": "operator", "Secret": "credential"}),
            stderr="",
        )

    monkeypatch.setattr(scanner.subprocess, "run", helper)
    credentials = scanner._docker_credentials(
        "registry.example", {"DOCKER_CONFIG": str(docker_config), "HOME": str(tmp_path)}
    )

    assert credentials == {"username": "operator", "secret": "credential"}
    assert calls == [(["docker-credential-unit", "get"], "registry.example\n")]
