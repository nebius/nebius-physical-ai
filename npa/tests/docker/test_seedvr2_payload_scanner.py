"""Complete-layer negative controls for the SeedVR2 runtime-payload scanner."""

from __future__ import annotations

import gzip
import copy
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tarfile
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_seedvr2_payload.py"
)
SPEC = importlib.util.spec_from_file_location("scan_image_seedvr2_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def _clean_root_config():
    contract = json.loads(scanner.CLEAN_ROOT_CONTRACT.read_bytes())
    config = copy.deepcopy(contract["config"])
    substitutions = {
        "NPA_SOURCE_SHA": "1" * 40,
        "FLASH_ATTN_MAX_JOBS": "8",
        "FLASH_ATTN_NVCC_THREADS": "1",
        "APEX_MAX_JOBS": "8",
        "FLASH_ATTN_CUDA_ARCHS": "90;100",
        "TORCH_CUDA_ARCH_LIST": "9.0;10.0",
    }
    serialized = json.dumps(config)
    for key, value in substitutions.items():
        serialized = serialized.replace("${" + key + "}", value)
    return {**contract["platform"], "config": json.loads(serialized)}


def _clean_root_codes(config, layers=1, source="1" * 40):
    if isinstance(layers, int):
        layers = [{} for _ in range(layers)]
    return {
        finding.kind
        for finding in scanner._clean_root_findings(
            config, layers=layers, expected_source_sha=source
        )
    }


def test_clean_root_measured_runtime_config_is_preserved():
    assert not _clean_root_codes(_clean_root_config())


@pytest.mark.parametrize("layers", [0, 2, 32])
def test_clean_root_rejects_missing_or_retained_ancestor_layers(layers):
    assert "clean_root_ancestor_layers" in _clean_root_codes(
        _clean_root_config(), layers
    )


def test_clean_root_accepts_only_exact_optional_empty_terminal_metadata():
    contract = json.loads(scanner.CLEAN_ROOT_CONTRACT.read_bytes())
    empty = contract["optional_terminal_metadata_layer"]
    assert not _clean_root_codes(_clean_root_config(), [{}, empty])
    assert "clean_root_ancestor_layers" in _clean_root_codes(
        _clean_root_config(), [empty, {}]
    )
    assert "clean_root_ancestor_layers" in _clean_root_codes(
        _clean_root_config(), [{}, empty, empty]
    )
    for key, value in empty.items():
        changed = dict(empty)
        changed[key] = value + 1 if isinstance(value, int) else value + "changed"
        assert "clean_root_ancestor_layers" in _clean_root_codes(
            _clean_root_config(), [{}, changed]
        )


@pytest.mark.parametrize("key,value", [("os", "windows"), ("architecture", "arm64")])
def test_clean_root_rejects_wrong_platform(key, value):
    config = _clean_root_config()
    config[key] = value
    assert "clean_root_platform" in _clean_root_codes(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("User", "root"),
        ("ExposedPorts", {}),
        ("Entrypoint", ["sh"]),
        ("Cmd", ["version"]),
        ("Volumes", {}),
        ("WorkingDir", "/"),
        ("Healthcheck", {}),
        ("Shell", ["/bin/sh", "-c"]),
        ("OnBuild", ["RUN unexpected"]),
        ("ArgsEscaped", False),
    ],
)
def test_clean_root_rejects_changed_launch_config(key, value):
    config = _clean_root_config()
    config["config"][key] = value
    assert "clean_root_runtime_config" in _clean_root_codes(config)


@pytest.mark.parametrize("index", range(len(_clean_root_config()["config"]["Env"])))
def test_clean_root_rejects_each_changed_environment_value(index):
    config = _clean_root_config()
    environment = config["config"]["Env"]
    environment[index] = environment[index].split("=", 1)[0] + "=changed"
    assert "clean_root_runtime_config" in _clean_root_codes(config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "not-string"])
def test_clean_root_rejects_ambiguous_environment(mutation):
    config = _clean_root_config()
    environment = config["config"]["Env"]
    if mutation == "missing":
        environment.pop()
    elif mutation == "extra":
        environment.append("UNEXPECTED=value")
    elif mutation == "duplicate":
        environment.append(environment[0])
    else:
        environment.append(None)
    assert "clean_root_runtime_config" in _clean_root_codes(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("npa.build.apex.max-jobs", "0"),
        ("npa.build.flash-attn.max-jobs", "unbounded"),
        ("npa.build.flash-attn.nvcc-threads", "-1"),
        ("npa.build.flash-attn.cuda-archs", "90;90"),
        ("npa.build.flash-attn.cuda-archs", "90"),
        ("npa.build.torch.cuda-arch-list", "9.0+PTX"),
    ],
)
def test_clean_root_rejects_inconsistent_build_labels(key, value):
    config = _clean_root_config()
    config["config"]["Labels"][key] = value
    assert "clean_root_build_metadata" in _clean_root_codes(config)


def test_clean_root_rejects_changed_source_or_extra_label():
    config = _clean_root_config()
    assert "clean_root_runtime_config" in _clean_root_codes(config, source="2" * 40)
    config["config"]["Labels"]["unknown"] = "unexpected"
    assert "clean_root_runtime_config" in _clean_root_codes(config)


def test_clean_root_rejects_changed_contract_bytes(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    contract.write_bytes(scanner.CLEAN_ROOT_CONTRACT.read_bytes() + b" ")
    monkeypatch.setattr(scanner, "CLEAN_ROOT_CONTRACT", contract)
    with pytest.raises(RuntimeError, match="contract digest"):
        _clean_root_codes(_clean_root_config())


def test_clean_root_source_argument_fails_before_image_inspection():
    with pytest.raises(SystemExit) as error:
        scanner.main(["not-read", "--clean-root-source-sha", "main"])
    assert error.value.code == 2


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _saved_image(
    tmp_path: Path,
    layers: list[dict[str, bytes]],
    *,
    config: bytes = b"{}",
    rootfs_diff_ids: list[str] | None = None,
) -> Path:
    layer_names = []
    payloads = {
        "manifest.json": b"",
    }
    diff_ids = []
    for index, members in enumerate(layers):
        name = f"layer-{index}.tar"
        payloads[name] = _tar(tmp_path / name, members).read_bytes()
        layer_names.append(name)
        diff_ids.append("sha256:" + hashlib.sha256(payloads[name]).hexdigest())
    config_document = json.loads(config)
    config_document["rootfs"] = {
        "type": "layers",
        "diff_ids": rootfs_diff_ids if rootfs_diff_ids is not None else diff_ids,
    }
    payloads["config.json"] = json.dumps(config_document).encode()
    payloads["manifest.json"] = json.dumps(
        [
            {
                "Config": "config.json",
                "RepoTags": ["seedvr2:test"],
                "Layers": layer_names,
            }
        ]
    ).encode()
    return _tar(tmp_path / "image.tar", payloads)


@pytest.mark.parametrize("ancestor", [False, True])
def test_clean_root_cli_binds_shape_source_and_payload_scan(tmp_path, capsys, ancestor):
    layers = [{"usr/share/doc/source.txt": b"source-only runtime"}]
    if ancestor:
        layers.insert(0, {"superseded-libssl.so": b"old object"})
        layers[1][".wh.superseded-libssl.so"] = b""
    saved = _saved_image(
        tmp_path, layers, config=json.dumps(_clean_root_config()).encode()
    )
    findings, identity = scanner.inspect_saved_image(saved)
    assert not findings  # A payload-only pass is not clean-root acceptance.
    result = scanner.main(
        [
            "--tarball",
            str(saved),
            "--expected-image-id",
            identity["image_id"],
            "--clean-root-source-sha",
            "1" * 40,
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == int(ancestor)
    assert report["clean_root_qualification"] == {
        "contract_sha256": scanner.CLEAN_ROOT_CONTRACT_SHA256,
        "expected_source_sha": "1" * 40,
    }
    assert report["layers_scanned"] == len(layers)
    assert bool(report["findings"]) == ancestor


def test_clean_root_does_not_exempt_forbidden_payload(tmp_path, capsys):
    saved = _saved_image(
        tmp_path,
        [{"opt/model.pt": b"forbidden weights"}],
        config=json.dumps(_clean_root_config()).encode(),
    )
    _, identity = scanner.inspect_saved_image(saved)
    assert (
        scanner.main(
            [
                "--tarball",
                str(saved),
                "--expected-image-id",
                identity["image_id"],
                "--clean-root-source-sha",
                "1" * 40,
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert any(row["kind"] == "model_weight" for row in report["findings"])


@pytest.mark.parametrize(
    "variant", ["canonical", "other-gzip", "whiteout", "forged-diff"]
)
def test_clean_root_empty_tail_passes_full_byte_and_shape_checks(
    tmp_path, capsys, variant
):
    # Exact Docker empty-layer bytes; both stored and decoded identities matter.
    canonical = bytes.fromhex(
        "1f8b08000000000000ff621805a360148c5800080000ffff2eafb5ef00040000"
    )
    decoded = gzip.decompress(canonical)
    assert decoded == b"\0" * 1024
    assert (
        hashlib.sha256(canonical).hexdigest()
        == "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1"
    )
    if variant == "other-gzip":
        tail = gzip.compress(decoded, mtime=0)
        assert tail != canonical
    elif variant in {"whiteout", "forged-diff"}:
        decoded = _tar(tmp_path / "whiteout.tar", {".wh.old.so": b""}).read_bytes()
        tail = gzip.compress(decoded, mtime=0)
    else:
        tail = canonical
    original = _saved_image(
        tmp_path,
        [{"source.txt": b"source-only runtime"}],
        config=json.dumps(_clean_root_config()).encode(),
    )
    with tarfile.open(original) as archive:
        members = {
            row.name: archive.extractfile(row).read() for row in archive.getmembers()
        }
    config = json.loads(members["config.json"])
    diff = hashlib.sha256(
        b"\0" * 1024 if variant == "forged-diff" else decoded
    ).hexdigest()
    config["rootfs"]["diff_ids"].append("sha256:" + diff)
    members["config.json"] = json.dumps(config).encode()
    manifest = json.loads(members["manifest.json"])
    manifest[0]["Layers"].append("tail.tar.gz")
    members["manifest.json"] = json.dumps(manifest).encode()
    members["tail.tar.gz"] = tail
    saved = _tar(tmp_path / "with-tail.tar", members)
    if variant == "forged-diff":
        with pytest.raises(RuntimeError, match="rootfs diff IDs"):
            scanner.inspect_saved_image(saved)
        return
    _, identity = scanner.inspect_saved_image(saved)
    result = scanner.main(
        [
            "--tarball",
            str(saved),
            "--expected-image-id",
            identity["image_id"],
            "--clean-root-source-sha",
            "1" * 40,
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == (0 if variant == "canonical" else 1)
    assert report["layers_scanned"] == 2


def _gzip_saved_image(
    tmp_path: Path,
    members: dict[str, bytes],
    *,
    layer_tar_bytes: bytes | None = None,
    compressed_layer_bytes: bytes | None = None,
    blob_digest: str | None = None,
    rootfs_diff_id: str | None = None,
    config_size_delta: int = 0,
    layer_size_delta: int = 0,
    manifest_size_delta: int = 0,
    layout_version: str = "1.0.0",
    omit_oci_layer_descriptor: bool = False,
) -> Path:
    layer = (
        layer_tar_bytes
        if layer_tar_bytes is not None
        else _tar(tmp_path / "compressed-layer.tar", members).read_bytes()
    )
    compressed = (
        compressed_layer_bytes
        if compressed_layer_bytes is not None
        else gzip.compress(layer, mtime=0)
    )
    actual_blob_digest = hashlib.sha256(compressed).hexdigest()
    name = f"blobs/sha256/{blob_digest or actual_blob_digest}"
    config = json.dumps(
        {
            "rootfs": {
                "type": "layers",
                "diff_ids": [
                    rootfs_diff_id or "sha256:" + hashlib.sha256(layer).hexdigest()
                ],
            }
        }
    ).encode()
    config_digest = hashlib.sha256(config).hexdigest()
    config_name = f"blobs/sha256/{config_digest}"
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": f"sha256:{config_digest}",
                "size": len(config) + config_size_delta,
            },
            "layers": (
                []
                if omit_oci_layer_descriptor
                else [
                    {
                        "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
                        "digest": f"sha256:{blob_digest or actual_blob_digest}",
                        "size": len(compressed) + layer_size_delta,
                    }
                ]
            ),
        },
        separators=(",", ":"),
    ).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    return _tar(
        tmp_path / "compressed-image.tar",
        {
            name: compressed,
            config_name: config,
            f"blobs/sha256/{manifest_digest}": manifest,
            "oci-layout": json.dumps(
                {"imageLayoutVersion": layout_version}, separators=(",", ":")
            ).encode(),
            "index.json": json.dumps(
                {
                    "schemaVersion": 2,
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "manifests": [
                        {
                            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                            "digest": f"sha256:{manifest_digest}",
                            "size": len(manifest) + manifest_size_delta,
                            "annotations": {"config.digest": f"sha256:{config_digest}"},
                        }
                    ],
                },
                separators=(",", ":"),
            ).encode(),
            "manifest.json": json.dumps(
                [
                    {
                        "Config": config_name,
                        "RepoTags": ["seedvr2:test"],
                        "Layers": [name],
                    }
                ]
            ).encode(),
        },
    )


def test_clean_source_runtime_and_exact_rerun_path_file_pass(tmp_path: Path) -> None:
    saved = _saved_image(
        tmp_path,
        [
            {
                "opt/seedvr2/LICENSE": b"Apache License 2.0",
                "opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth": (
                    b"rerun_sdk\n"
                ),
            }
        ],
    )
    findings, layers = scanner.scan_saved_image(saved)
    assert layers == 1
    assert findings == []
    _, identity = scanner.inspect_saved_image(saved)
    assert identity["image_config_digest"].startswith("sha256:")
    assert identity["layers"][0]["diff_id"] == identity["rootfs_diff_ids"][0]


def test_arbitrary_python_path_file_is_still_a_model_payload(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/npa-venv/lib/python3.12/site-packages/weights.pth": (
                        b"checkpoint"
                    )
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == ["model_weight"]


def test_allowed_python_path_file_is_content_bound(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth": (
                        b"checkpoint"
                    )
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == ["model_weight"]


def test_allowed_python_path_file_must_be_regular(tmp_path: Path) -> None:
    layer = tmp_path / "path-symlink.tar"
    with tarfile.open(layer, "w") as archive:
        info = tarfile.TarInfo(
            "opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth"
        )
        info.type = tarfile.SYMTYPE
        info.linkname = "/workspace/model"
        archive.addfile(info)
    findings = scanner.scan_layer(layer, layer="test")
    assert [finding.kind for finding in findings] == ["model_weight"]


def test_allowed_python_path_directory_is_rejected(tmp_path: Path) -> None:
    layer = tmp_path / "path-directory.tar"
    with tarfile.open(layer, "w") as archive:
        info = tarfile.TarInfo(
            "opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth"
        )
        info.type = tarfile.DIRTYPE
        archive.addfile(info)
    findings = scanner.scan_layer(layer, layer="test")
    assert [finding.kind for finding in findings] == ["model_weight"]


def test_gzip_content_addressed_layer_binds_blob_and_diff_id(tmp_path: Path) -> None:
    saved = _gzip_saved_image(tmp_path, {"opt/seedvr2/LICENSE": b"Apache License 2.0"})
    findings, identity = scanner.inspect_saved_image(saved)
    layer = identity["layers"][0]
    assert findings == []
    assert layer["compression"] == "gzip"
    assert layer["path"] == "blobs/" + layer["blob_digest"].replace(":", "/")
    assert layer["diff_id"] == identity["rootfs_diff_ids"][0]
    assert layer["blob_bytes"] < layer["uncompressed_bytes"]
    assert identity["image_id"] == identity["oci"]["manifest_digest"]


def test_tarball_cli_binds_expected_oci_manifest_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = _gzip_saved_image(tmp_path, {"opt/seedvr2/LICENSE": b"Apache License 2.0"})
    _, identity = scanner.inspect_saved_image(saved)
    assert (
        scanner.main(
            [
                "--tarball",
                str(saved),
                "--expected-image-id",
                identity["image_id"],
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "npa_seedvr2_payload_scan_v2"
    assert report["source"] == {
        "kind": "docker-save-tarball",
        "filename": saved.name,
    }
    assert report["archive"] == {
        "bytes": saved.stat().st_size,
        "sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
    }
    assert report["scanner_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert report["expected_image_id"] == identity["image_id"]
    assert report["image_id"] == identity["image_id"]
    assert report["image_config_digest"] != report["image_id"]
    assert report["oci"]["manifest_digest"] == report["image_id"]


def test_tarball_cli_accepts_standard_config_image_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = _gzip_saved_image(tmp_path, {"opt/seedvr2/LICENSE": b"Apache License 2.0"})
    _, identity = scanner.inspect_saved_image(saved)
    assert (
        scanner.main(
            [
                "--tarball",
                str(saved),
                "--expected-image-id",
                identity["image_config_digest"],
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["expected_image_id"] == identity["image_config_digest"]
    assert report["image_id"] == identity["oci"]["manifest_digest"]


def test_tarball_cli_rejects_unbound_expected_image_id(tmp_path: Path) -> None:
    saved = _gzip_saved_image(tmp_path, {"opt/seedvr2/LICENSE": b"Apache License 2.0"})
    with pytest.raises(RuntimeError, match="identity differs"):
        scanner.main(
            [
                "--tarball",
                str(saved),
                "--expected-image-id",
                "sha256:" + "f" * 64,
            ]
        )


def test_archive_mutation_during_scan_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _gzip_saved_image(tmp_path, {"opt/seedvr2/LICENSE": b"Apache License 2.0"})
    inspect = scanner.inspect_saved_image

    def mutate_after_inspection(path: Path) -> tuple[list[object], dict]:
        result = inspect(path)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(scanner, "inspect_saved_image", mutate_after_inspection)
    with pytest.raises(RuntimeError, match="changed while it was being scanned"):
        scanner._inspect_stable_saved_image(saved)


def test_gzip_layer_still_scans_runtime_only_payload(tmp_path: Path) -> None:
    saved = _gzip_saved_image(
        tmp_path, {"opt/seedvr2/test_videos/customer.mp4": b"sensor"}
    )
    findings, _ = scanner.inspect_saved_image(saved)
    assert [finding.kind for finding in findings] == ["sensor_or_output_media"]


def test_content_addressed_layer_path_must_match_blob(tmp_path: Path) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {"opt/seedvr2/LICENSE": b"Apache License 2.0"},
        blob_digest="0" * 64,
    )
    with pytest.raises(RuntimeError, match="content-addressed path"):
        scanner.inspect_saved_image(saved)


@pytest.mark.parametrize(
    ("mutations", "message"),
    [
        ({"manifest_size_delta": 1}, "manifest differs"),
        ({"config_size_delta": 1}, "config identity differs"),
        ({"layer_size_delta": 1}, "layer descriptor differs"),
        ({"layout_version": "0.9.0"}, "layout version is unsupported"),
    ],
)
def test_oci_identity_mutations_fail_closed(
    tmp_path: Path, mutations: dict[str, int | str], message: str
) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {"opt/seedvr2/LICENSE": b"Apache License 2.0"},
        **mutations,
    )
    with pytest.raises(RuntimeError, match=message):
        scanner.inspect_saved_image(saved)


def test_duplicate_outer_archive_member_fails_closed(tmp_path: Path) -> None:
    saved = tmp_path / "duplicate.tar"
    with tarfile.open(saved, "w") as archive:
        for payload in (b"[]", b"{}"):
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="duplicate member"):
        scanner.inspect_saved_image(saved)


def test_outer_archive_required_member_must_be_regular(tmp_path: Path) -> None:
    saved = tmp_path / "symlink.tar"
    with tarfile.open(saved, "w") as archive:
        info = tarfile.TarInfo("manifest.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "other.json"
        archive.addfile(info)
    with pytest.raises(RuntimeError, match="not a regular file"):
        scanner.inspect_saved_image(saved)


def test_unsafe_outer_archive_member_fails_closed(tmp_path: Path) -> None:
    saved = _tar(tmp_path / "unsafe.tar", {"../manifest.json": b"[]"})
    with pytest.raises(RuntimeError, match="unsafe member path"):
        scanner.inspect_saved_image(saved)


def test_empty_outer_archive_fails_through_explicit_guard(tmp_path: Path) -> None:
    saved = _tar(tmp_path / "empty.tar", {})
    with pytest.raises(RuntimeError, match="no compatibility manifest"):
        scanner.inspect_saved_image(saved)


def test_garbage_outer_archive_fails_through_explicit_guard(tmp_path: Path) -> None:
    saved = tmp_path / "garbage.tar"
    saved.write_bytes(b"not a tar archive")
    with pytest.raises(RuntimeError, match="not a readable tar archive"):
        scanner.inspect_saved_image(saved)


def test_outer_compression_is_rejected_before_archive_parsing(tmp_path: Path) -> None:
    saved = _saved_image(
        tmp_path,
        [{"opt/seedvr2/LICENSE": b"Apache License 2.0"}],
    )
    compressed = tmp_path / "outer.tar.gz"
    compressed.write_bytes(gzip.compress(saved.read_bytes(), mtime=0))
    with pytest.raises(RuntimeError, match="outer compression is unsupported"):
        scanner.inspect_saved_image(compressed)


def test_malformed_decompressed_layer_fails_through_explicit_guard(
    tmp_path: Path,
) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {},
        layer_tar_bytes=b"valid gzip, but not a tar archive",
    )
    with pytest.raises(RuntimeError, match="layer is not a readable tar archive"):
        scanner.inspect_saved_image(saved)


def test_truncated_gzip_stream_fails_through_explicit_guard(tmp_path: Path) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {},
        compressed_layer_bytes=b"\x1f\x8b\x08truncated",
    )
    with pytest.raises(RuntimeError, match="layer gzip is invalid"):
        scanner.inspect_saved_image(saved)


def test_invalid_deflate_stream_fails_through_explicit_zlib_guard(
    tmp_path: Path,
) -> None:
    gzip_header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
    saved = _gzip_saved_image(
        tmp_path,
        {},
        compressed_layer_bytes=gzip_header + b"\xff" * 20,
    )
    with pytest.raises(RuntimeError, match="layer gzip is invalid"):
        scanner.inspect_saved_image(saved)


def test_rootfs_layer_count_mismatch_fails_through_explicit_guard(
    tmp_path: Path,
) -> None:
    saved = _saved_image(
        tmp_path,
        [{"opt/seedvr2/LICENSE": b"Apache License 2.0"}],
        rootfs_diff_ids=[],
    )
    with pytest.raises(RuntimeError, match="layer count"):
        scanner.inspect_saved_image(saved)


def test_oci_descriptor_layer_count_mismatch_fails_through_explicit_guard(
    tmp_path: Path,
) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {"opt/seedvr2/LICENSE": b"Apache License 2.0"},
        omit_oci_layer_descriptor=True,
    )
    with pytest.raises(RuntimeError, match="layer orders differ"):
        scanner.inspect_saved_image(saved)


def test_unexpected_structural_lookup_failure_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_structural_lookup(_image_tar: Path) -> tuple[list, dict]:
        raise KeyError("Layers")

    monkeypatch.setattr(scanner, "_inspect_saved_image", fail_structural_lookup)
    with pytest.raises(RuntimeError, match="malformed structural data"):
        scanner.inspect_saved_image(tmp_path / "not-read.tar")


def test_excessively_nested_manifest_json_is_normalized(tmp_path: Path) -> None:
    nested_json = b'{"nested":' * 10_000 + b"0" + b"}" * 10_000
    saved = _tar(tmp_path / "deep-json.tar", {"manifest.json": nested_json})
    with pytest.raises(
        RuntimeError, match="excessively nested structural data"
    ) as caught:
        scanner.inspect_saved_image(saved)
    assert type(caught.value) is RuntimeError


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schemaVersion":2,"schemaVersion":1}',
        b'{"schemaVersion":NaN}',
    ],
)
def test_identity_json_rejects_ambiguous_values(payload: bytes) -> None:
    with pytest.raises(ValueError):
        scanner._strict_json(payload)


def test_saved_layer_must_match_config_diff_id(tmp_path: Path) -> None:
    saved = _saved_image(
        tmp_path,
        [{"opt/seedvr2/LICENSE": b"Apache License 2.0"}],
        rootfs_diff_ids=["sha256:" + "0" * 64],
    )
    with pytest.raises(RuntimeError, match="rootfs diff IDs"):
        scanner.inspect_saved_image(saved)


def test_gzip_layer_must_match_config_diff_id(tmp_path: Path) -> None:
    saved = _gzip_saved_image(
        tmp_path,
        {"opt/seedvr2/LICENSE": b"Apache License 2.0"},
        rootfs_diff_id="sha256:" + "0" * 64,
    )
    with pytest.raises(RuntimeError, match="rootfs diff IDs"):
        scanner.inspect_saved_image(saved)


def test_cudnn_sdk_headers_and_static_archives_fail(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h": b"sdk",
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/include/unreviewed.hpp": b"sdk",
                    "opt/seedvr2-venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_static.a": b"sdk",
                    "usr/include/cudnn_version.h": b"sdk",
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == [
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
        "cudnn_sdk_payload",
    ]


def test_nested_wheel_cudnn_and_nvshmem_sdk_payloads_fail(tmp_path: Path) -> None:
    wheel = _zip(
        tmp_path / "cuda-runtime.whl",
        {
            "nvidia/cudnn/include/cudnn.h": b"sdk",
            "nvidia/nvshmem/include/device/nvshmem.cuh": b"sdk",
            "nvidia/nvshmem/lib/libnvshmem_device.a": b"sdk",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/cuda-runtime.whl": wheel.read_bytes()}],
        )
    )
    assert [finding.kind for finding in findings] == [
        "cudnn_sdk_payload",
        "nvshmem_sdk_payload",
        "nvshmem_sdk_payload",
    ]


@pytest.mark.parametrize(
    ("version", "contents", "accepted"),
    [
        ("84.0.0", scanner.SETUPTOOLS_DISTUTILS_PATH_FILE, True),
        ("84.0.0", scanner.SETUPTOOLS_DISTUTILS_PATH_FILE + b"changed", False),
        ("80.9.0", scanner.SETUPTOOLS_DISTUTILS_PATH_FILE, False),
        ("68.1.2", scanner.SETUPTOOLS_DISTUTILS_PATH_FILE, False),
    ],
)
def test_setuptools_seed_wheel_path_and_contents_are_exactly_bound(
    tmp_path: Path, version: str, contents: bytes, accepted: bool
) -> None:
    filename = f"setuptools-{version}-py3-none-any.whl"
    wheel = _zip(
        tmp_path / filename,
        {
            "distutils-precedence.pth": contents,
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "usr/share/python-wheels/" + filename: wheel.read_bytes(),
                    "opt/seedvr2-venv/lib/python3.12/site-packages/"
                    "distutils-precedence.pth": (
                        scanner.SETUPTOOLS_DISTUTILS_PATH_FILE
                    ),
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == (
        [] if accepted else ["model_weight"]
    )


def test_exact_setuptools_path_file_cannot_be_zip_symlink(tmp_path: Path) -> None:
    wheel = tmp_path / "setuptools-84.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        info = zipfile.ZipInfo("distutils-precedence.pth")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, scanner.SETUPTOOLS_DISTUTILS_PATH_FILE)
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "usr/share/python-wheels/"
                    "setuptools-84.0.0-py3-none-any.whl": wheel.read_bytes(),
                }
            ],
        )
    )
    assert [finding.kind for finding in findings] == ["model_weight"]


def test_weights_cache_and_sensor_media_fail_in_lower_layer(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/model.pth": b"x",
                    "opt/seedvr2/test_videos/customer.mp4": b"x",
                },
                {
                    "workspace/.cache/huggingface/hub/models--seedvr/.wh.model.pth": b"",
                    "opt/seedvr2/test_videos/.wh.customer.mp4": b"",
                },
            ],
        )
    )
    assert layers == 2
    kinds = {finding.kind for finding in findings}
    assert {"model_weight", "populated_hf_cache", "sensor_or_output_media"} <= kinds


def test_weight_hidden_in_nested_source_tar_fails(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "seedvr2-source.tar",
        {"SeedVR/pos_emb.pt": b"runtime-only embedding"},
    )
    findings, layers = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/seedvr2.tar.gz": source_tar.read_bytes()}],
        )
    )
    assert layers == 1
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("model_weight", "tmp/seedvr2.tar.gz!/SeedVR/pos_emb.pt")
    ]


def test_unreadable_nested_source_tar_fails_closed(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/seedvr2.tar.gz": b"not a tar archive"}],
        )
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("unreadable_nested_archive", "tmp/seedvr2.tar.gz")
    ]


def test_weight_hidden_in_nested_source_zip_fails(tmp_path: Path) -> None:
    source_zip = _zip(
        tmp_path / "source.zip",
        {"SeedVR/weights/model.safetensors": b"runtime-only weight"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.zip": source_zip.read_bytes()}])
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("model_weight", "tmp/source.zip!/SeedVR/weights/model.safetensors")
    ]


def test_workspace_media_and_nested_egg_payload_fail(tmp_path: Path) -> None:
    egg = _zip(
        tmp_path / "dependency.egg",
        {"package/hidden/model.pth": b"runtime-only weight"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "workspace/output.mp4": b"generated media",
                    "tmp/dependency.egg": egg.read_bytes(),
                }
            ],
        )
    )
    assert {finding.kind for finding in findings} == {
        "model_weight",
        "sensor_or_output_media",
    }


def test_unsupported_nested_archive_fails_closed(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"tmp/source.tar.zst": b"opaque archive bytes"}],
        )
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("unsupported_nested_archive", "tmp/source.tar.zst")
    ]


def test_nested_dotfile_path_is_not_normalized_away(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {"./.aws/credentials": b"[default]\ncredential_process = example"},
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert [(finding.kind, finding.path) for finding in findings] == [
        ("credential_file", "tmp/source.tar!/.aws/credentials")
    ]


def test_nested_parent_or_backslash_path_fails_closed(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {
            "../escaped.txt": b"x",
            "folder\\disguised.txt": b"x",
            "/absolute.txt": b"x",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert [finding.kind for finding in findings] == [
        "unsafe_archive_path",
        "unsafe_archive_path",
        "unsafe_archive_path",
    ]


def test_nested_application_media_and_secret_content_fail(tmp_path: Path) -> None:
    source_tar = _tar(
        tmp_path / "source.tar",
        {
            "opt/seedvr2/private/customer.mp4": b"video",
            "opt/npa-src/config.txt": b"hf_abcdefghijklmnopqrstuvwxyz",
        },
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, [{"tmp/source.tar": source_tar.read_bytes()}])
    )
    assert {finding.kind for finding in findings} == {
        "sensor_or_output_media",
        "credential_content",
    }


def test_secret_in_application_or_image_environment_fails(tmp_path: Path) -> None:
    token = b"hf_abcdefghijklmnopqrstuvwxyz"
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [{"opt/npa-src/config.json": token}],
            config=b'{"config":{"Env":["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"]}}',
        )
    )
    assert [finding.kind for finding in findings].count("credential_content") == 2


def test_dependency_secret_shaped_fixture_is_not_operator_payload(
    tmp_path: Path,
) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            [
                {
                    "opt/seedvr2-venv/lib/python3.12/site-packages/pkg/test.py": (
                        b"hf_abcdefghijklmnopqrstuvwxyz"
                    )
                }
            ],
        )
    )
    assert findings == []
