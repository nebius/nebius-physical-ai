"""Exercise Habitat's OCI quarantine verifier with synthetic image bytes."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tarfile

import pytest

from test_image_byte_scan import file, js, tar_data


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "npa/scripts"
sys.path.insert(0, str(SCRIPTS))
from image_byte_scan import habitat_sim_verification as H  # noqa: E402


INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
INTOTO = "application/vnd.in-toto+json"
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
CONTRACT = json.loads((PACKAGE / "runtime-payload.json").read_text())
SOURCE_REVISION = "a" * 40


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _oci(
    tmp_path: Path,
    layers: list[list[tuple]],
    *,
    user: str = "ubuntu",
    diff_ids: list[str] | None = None,
    source_revision: str = SOURCE_REVISION,
) -> tuple[Path, str, list[str]]:
    blobs: dict[str, bytes] = {}

    def blob(data: bytes, media: str) -> dict[str, object]:
        digest = "sha256:" + _digest(data)
        blobs["blobs/sha256/" + digest.removeprefix("sha256:")] = data
        return {"mediaType": media, "digest": digest, "size": len(data)}

    raw_layers = [tar_data(rows) for rows in layers]
    packed_layers = [blob(gzip.compress(raw, mtime=0), LAYER) for raw in raw_layers]
    labels = copy.deepcopy(CONTRACT["required_labels"])
    labels["org.opencontainers.image.revision"] = source_revision
    actual_diff_ids = ["sha256:" + _digest(raw) for raw in raw_layers]
    configured_diff_ids = diff_ids or actual_diff_ids
    config = blob(
        js(
            {
                "architecture": "amd64",
                "os": "linux",
                "rootfs": {
                    "type": "layers",
                    "diff_ids": configured_diff_ids,
                },
                "config": {
                    "User": user,
                    "Entrypoint": ["/usr/local/bin/npa-habitat-entrypoint"],
                    "Labels": labels,
                },
            }
        ),
        CONFIG,
    )
    runtime = blob(
        js(
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST,
                "config": config,
                "layers": packed_layers,
            }
        ),
        MANIFEST,
    )
    statement = blob(js({"predicateType": "https://spdx.dev/Document"}), INTOTO)
    attestation_config = blob(
        js(
            {
                "architecture": "unknown",
                "os": "unknown",
                "rootfs": {"type": "layers", "diff_ids": [statement["digest"]]},
            }
        ),
        CONFIG,
    )
    attestation = blob(
        js(
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST,
                "config": attestation_config,
                "layers": [statement],
            }
        ),
        MANIFEST,
    )
    runtime["platform"] = {"os": "linux", "architecture": "amd64"}
    attestation.update(
        platform={"os": "unknown", "architecture": "unknown"},
        annotations={
            "vnd.docker.reference.type": "attestation-manifest",
            "vnd.docker.reference.digest": runtime["digest"],
        },
    )
    index = js(
        {
            "schemaVersion": 2,
            "mediaType": INDEX,
            "manifests": [runtime, attestation],
        }
    )
    expected = "sha256:" + _digest(index)
    files = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index,
        **blobs,
    }
    archive = tmp_path / "habitat.oci.tar"
    archive.write_bytes(tar_data([file(name, data) for name, data in files.items()]))
    return archive, expected, configured_diff_ids


def _json(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.decode().rstrip("=")


def _elf64(
    *,
    needed: tuple[str, ...] = (),
    soname: str = "",
    rpath: str = "",
    runpath: str = "",
) -> bytes:
    strings = bytearray(b"\0")

    def add(value: str) -> int:
        offset = len(strings)
        strings.extend(value.encode() + b"\0")
        return offset

    needed_offsets = [add(value) for value in needed]
    soname_offset = add(soname) if soname else None
    rpath_offset = add(rpath) if rpath else None
    runpath_offset = add(runpath) if runpath else None
    string_offset = 0x200
    dynamic_rows = [(5, 0x400000 + string_offset), (10, len(strings))]
    dynamic_rows.extend((1, offset) for offset in needed_offsets)
    if soname_offset is not None:
        dynamic_rows.append((14, soname_offset))
    if rpath_offset is not None:
        dynamic_rows.append((15, rpath_offset))
    if runpath_offset is not None:
        dynamic_rows.append((29, runpath_offset))
    dynamic_rows.append((0, 0))
    dynamic = b"".join(struct.pack("<qQ", *row) for row in dynamic_rows)
    size = string_offset + len(strings)
    payload = bytearray(size)
    payload[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        payload,
        16,
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    struct.pack_into(
        "<IIQQQQQQ", payload, 64, 1, 5, 0, 0x400000, 0x400000, size, size, 0x1000
    )
    struct.pack_into(
        "<IIQQQQQQ",
        payload,
        120,
        2,
        6,
        0x100,
        0x400100,
        0x400100,
        len(dynamic),
        len(dynamic),
        8,
    )
    payload[0x100 : 0x100 + len(dynamic)] = dynamic
    payload[string_offset:] = strings
    return bytes(payload)


def _fixture() -> tuple[dict[str, object], list[tuple]]:
    contract = copy.deepcopy(CONTRACT)
    contract["expected_missing_python_record_count"] = 0
    source_files = {
        "LICENSE": b"source-license",
        "data/default.physics_config.json": b"{}\n",
    }
    projection = {
        "schema_version": "npa.habitat-sim.source-projection.v1",
        "file_count": len(source_files),
        "files": [
            {"path": path, "bytes": len(payload), "sha256": _digest(payload)}
            for path, payload in sorted(source_files.items())
        ],
    }
    projection_bytes = _json(projection)
    manifest = {
        "expected_projection": {
            "file_count": len(source_files),
            "inventory_sha256": _digest(projection_bytes),
        }
    }
    apt_lock = (
        "schema_version: npa.habitat-sim.apt-lock.v1\npackages:\n"
        "  - {binary: python3, version: 3.10.6-1~22.04.1, "
        "source: python3-defaults, sha256: " + "1" * 64 + "}\n"
    ).encode()
    python_lock = ("fixture==1.0 --hash=sha256:" + "2" * 64 + "\n").encode()
    native = _elf64(soname="native.so")
    controls = {
        "opt/npa-runtime/npa/workflows/habitat_sim_smoke.py": b"smoke\n",
        "usr/share/doc/npa-habitat-sim/source-manifest.json": _json(manifest),
        "usr/share/doc/npa-habitat-sim/source-projection.json": projection_bytes,
        "usr/share/doc/npa-habitat-sim/licenses.json": b"{}\n",
        "usr/share/doc/npa-habitat-sim/apt-runtime.lock": apt_lock,
        "usr/share/doc/npa-habitat-sim/requirements-runtime.lock": python_lock,
        "usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md": b"redistribution\n",
    }
    contract["required_final_file_sha256"] = {
        "/" + path: _digest(payload)
        for path, payload in controls.items()
        if not path.endswith("source-projection.json")
    }
    entries = [file(path, payload) for path, payload in controls.items()]
    entries.extend(
        file("usr/src/habitat-sim/" + path, payload)
        for path, payload in source_files.items()
    )
    entries.extend(
        [
            file(
                "var/lib/dpkg/status",
                b"Package: python3\nStatus: install ok installed\n"
                b"Version: 3.10.6-1~22.04.1\nSource: python3-defaults\n\n",
            ),
            file("var/lib/dpkg/info/python3.list", b"/usr/bin/python3\n"),
            file("usr/share/doc/python3/copyright", b"python license\n"),
            file("opt/venv/lib/python3.10/site-packages/fixture/native.so", native),
        ]
    )
    distributions = {
        "fixture": "1.0",
        "habitat_sim": "0.3.3",
        "pip": "22.0.2",
        "setuptools": "59.6.0",
    }
    for dist, version in distributions.items():
        root = f"opt/venv/lib/python3.10/site-packages/{dist}-{version}.dist-info"
        entries.append(
            file(f"{root}/METADATA", f"Name: {dist}\nVersion: {version}\n".encode())
        )
        record = (
            f"{dist}/native.so,{_record_hash(native)},{len(native)}\n"
            if dist == "fixture"
            else ""
        )
        entries.append(
            file(
                f"{root}/RECORD",
                (record + f"{dist}-{version}.dist-info/RECORD,,\n").encode(),
            )
        )
    return contract, entries


def _required_entries() -> list[tuple]:
    return _fixture()[1]


def _verify(
    tmp_path: Path,
    layers: list[list[tuple]],
    *,
    contract: dict[str, object] | None = None,
    expected_dpkg_inventory_sha256: str | None = None,
    expected_native_closure_sha256: str | None = None,
    **kwargs,
) -> dict[str, object]:
    archive, expected, diff_ids = _oci(tmp_path, layers, **kwargs)
    selected_contract = copy.deepcopy(contract or _fixture()[0])
    selected_contract["required_base_diff_ids"] = [diff_ids[0]]
    fd = os.open(archive, os.O_RDONLY)
    try:
        probe = H.verify(
            fd,
            archive.stat().st_size,
            expected,
            selected_contract,
            _digest(archive.read_bytes()),
            SOURCE_REVISION,
            "0" * 64,
            "0" * 64,
        )
        return H.verify(
            fd,
            archive.stat().st_size,
            expected,
            selected_contract,
            _digest(archive.read_bytes()),
            SOURCE_REVISION,
            expected_dpkg_inventory_sha256 or probe["dpkg_inventory_sha256"],
            expected_native_closure_sha256 or probe["native_elf_closure_sha256"],
        )
    finally:
        os.close(fd)


def _codes(report: dict[str, object]) -> set[str]:
    return {row["code"] for row in report["findings"]}


def test_valid_attested_oci_has_complete_graph_and_payload_receipt(tmp_path) -> None:
    report = _verify(tmp_path, [_required_entries()])
    assert report["valid"] is True
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 21
    assert report["installed_package_count"] == 1
    assert report["dpkg_inventory"]["python3"] == {
        "version": "3.10.6-1~22.04.1",
        "architecture": "",
        "source": "python3-defaults",
        "source_version": "3.10.6-1~22.04.1",
        "file_lists": [
            {
                "path": "var/lib/dpkg/info/python3.list",
                "sha256": _digest(b"/usr/bin/python3\n"),
            }
        ],
        "copyright_path": "usr/share/doc/python3/copyright",
        "copyright_sha256": _digest(b"python license\n"),
    }
    assert report["projected_source_file_count"] == 2
    assert report["python_distribution_count"] == 4
    assert report["python_record_files_verified"] == 1
    assert report["native_elf_count"] == 1
    assert report["dpkg_inventory_sha256"] == report["expected_dpkg_inventory_sha256"]
    assert (
        report["native_elf_closure_sha256"] == report["expected_native_closure_sha256"]
    )
    assert report["native_elf_closure"][0]["owners"] == ["python-wheel-record"]
    assert report["expected_source_revision"] == SOURCE_REVISION
    assert report["oci_graph"]["attestation_manifest_count"] == 1


def test_complete_runtime_inventory_hashes_are_required(tmp_path) -> None:
    report = _verify(
        tmp_path,
        [_required_entries()],
        expected_dpkg_inventory_sha256="0" * 64,
        expected_native_closure_sha256="0" * 64,
    )
    assert {
        "runtime_dpkg_inventory_lock_mismatch",
        "native_elf_closure_lock_mismatch",
    } <= _codes(report)


def test_graph_binding_requires_a_valid_exact_runtime_closure() -> None:
    digest = "sha256:" + "1" * 64
    closure = "2" * 64
    result = {
        "image_manifest_digest": "sha256:" + "3" * 64,
        "image_config_digest": "sha256:" + "4" * 64,
        "layers": [{"diff_id": "sha256:" + "5" * 64}],
    }
    receipt = {
        "valid": True,
        "expected_image_id": digest,
        "image_index_digest": digest,
        "image_manifest_digest": result["image_manifest_digest"],
        "image_config_digest": result["image_config_digest"],
        "layer_count": 1,
        "verified_layer_diff_ids": [result["layers"][0]["diff_id"]],
        "regular_files_read": 1,
        "content_bytes_read": 1,
        "expected_dpkg_inventory_sha256": closure,
        "dpkg_inventory_sha256": closure,
        "expected_native_closure_sha256": closure,
        "native_elf_closure_sha256": closure,
        "expected_source_revision": SOURCE_REVISION,
    }
    H.bind(result, receipt, digest)

    for key in (
        "valid",
        "dpkg_inventory_sha256",
        "native_elf_closure_sha256",
    ):
        invalid = copy.deepcopy(receipt)
        invalid[key] = False if key == "valid" else "6" * 64
        with pytest.raises(
            ValueError,
            match="habitat_oci_verifier_not_valid|habitat_oci_runtime_closure_binding",
        ):
            H.bind(result, invalid, digest)


def test_every_installed_package_requires_copyright_bytes(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file(
        "var/lib/dpkg/status",
        entries[status][1] + b"Package: transitive\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: transitive-source\n\n",
    )
    assert "runtime_package_copyright_missing" in _codes(_verify(tmp_path, [entries]))


def test_every_installed_package_list_bytes_are_bound(tmp_path) -> None:
    baseline = _verify(tmp_path, [_required_entries()])
    entries = _required_entries()
    list_index = next(
        index for index, row in enumerate(entries) if row[0].endswith("python3.list")
    )
    entries[list_index] = file(
        "var/lib/dpkg/info/python3.list", b"/usr/bin/python3.10\n"
    )
    report = _verify(
        tmp_path,
        [entries],
        expected_dpkg_inventory_sha256=baseline["dpkg_inventory_sha256"],
    )
    assert "runtime_dpkg_inventory_lock_mismatch" in _codes(report)

    missing = [
        row for row in _required_entries() if not row[0].endswith("python3.list")
    ]
    assert "runtime_package_file_list_population" in _codes(
        _verify(tmp_path, [missing])
    )

    duplicate = _required_entries()
    duplicate.append(
        file("var/lib/dpkg/info/python3:amd64.list", b"/usr/bin/python3\n")
    )
    assert "runtime_package_file_list_population" in _codes(
        _verify(tmp_path, [duplicate])
    )


def test_native_needed_resolution_and_dpkg_ownership_are_closed(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file(
        "var/lib/dpkg/status",
        entries[status][1] + b"Package: fixture-lib\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: fixture-source\n\n"
        b"Package: fixture-tool\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: fixture-source\n\n",
    )
    entries.extend(
        [
            file("usr/share/doc/fixture-lib/copyright", b"fixture license\n"),
            file("usr/share/doc/fixture-tool/copyright", b"fixture license\n"),
            file(
                "var/lib/dpkg/info/fixture-lib.list",
                b"/usr/lib/x86_64-linux-gnu/libfixture.so.1\n",
            ),
            file("var/lib/dpkg/info/fixture-tool.list", b"/usr/bin/fixture-tool\n"),
            file(
                "usr/lib/x86_64-linux-gnu/libfixture.so.1",
                _elf64(soname="libfixture.so.1"),
            ),
            file(
                "usr/bin/fixture-tool",
                _elf64(needed=("libfixture.so.1",)),
            ),
        ]
    )
    report = _verify(tmp_path, [entries])
    assert report["valid"] is True
    tool = next(
        row
        for row in report["native_elf_closure"]
        if row["path"] == "usr/bin/fixture-tool"
    )
    assert tool["owners"] == ["fixture-tool"]
    assert tool["needed"] == {
        "libfixture.so.1": "usr/lib/x86_64-linux-gnu/libfixture.so.1"
    }
    assert tool["rpath"] == [] and tool["runpath"] == []
    assert tool["effective_search_path"][:2] == [
        "lib/x86_64-linux-gnu",
        "usr/lib/x86_64-linux-gnu",
    ]
    assert tool["needed_resolution"]["libfixture.so.1"] == {
        "path": "usr/lib/x86_64-linux-gnu/libfixture.so.1",
        "lookup_path": "usr/lib/x86_64-linux-gnu/libfixture.so.1",
        "search_index": 1,
    }


def test_runpath_precedes_rpath_and_resolution_is_serialized(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file(
        "var/lib/dpkg/status",
        entries[status][1] + b"Package: fixture-tool\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: fixture-source\n\n",
    )
    entries.extend(
        [
            file("usr/share/doc/fixture-tool/copyright", b"fixture license\n"),
            file(
                "var/lib/dpkg/info/fixture-tool.list",
                b"/opt/runtime/libfixture.so.1\n/usr/bin/fixture-tool\n",
            ),
            file("opt/runtime/libfixture.so.1", _elf64(soname="libfixture.so.1")),
            file(
                "usr/bin/fixture-tool",
                _elf64(
                    needed=("libfixture.so.1",),
                    rpath="/opt/ignored",
                    runpath="/opt/runtime",
                ),
            ),
        ]
    )
    report = _verify(tmp_path, [entries])
    assert report["valid"] is True
    tool = next(
        row
        for row in report["native_elf_closure"]
        if row["path"] == "usr/bin/fixture-tool"
    )
    assert tool["rpath"] == ["/opt/ignored"]
    assert tool["runpath"] == ["/opt/runtime"]
    assert tool["effective_search_path"][0] == "opt/runtime"
    assert "opt/ignored" not in tool["effective_search_path"]


def test_multiple_compatible_native_targets_fail_as_ambiguous(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file(
        "var/lib/dpkg/status",
        entries[status][1] + b"Package: fixture-lib\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: fixture-source\n\n"
        b"Package: fixture-tool\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: fixture-source\n\n",
    )
    entries.extend(
        [
            file("usr/share/doc/fixture-lib/copyright", b"fixture license\n"),
            file("usr/share/doc/fixture-tool/copyright", b"fixture license\n"),
            file(
                "var/lib/dpkg/info/fixture-lib.list",
                b"/usr/lib/x86_64-linux-gnu/libfixture.so.1\n"
                b"/usr/local/lib/libfixture.so.1\n",
            ),
            file("var/lib/dpkg/info/fixture-tool.list", b"/usr/bin/fixture-tool\n"),
            file(
                "usr/lib/x86_64-linux-gnu/libfixture.so.1",
                _elf64(soname="libfixture.so.1"),
            ),
            file(
                "usr/local/lib/libfixture.so.1",
                _elf64(soname="libfixture.so.1"),
            ),
            file("usr/bin/fixture-tool", _elf64(needed=("libfixture.so.1",))),
        ]
    )
    assert "native_elf_dependency_ambiguous" in _codes(_verify(tmp_path, [entries]))


def test_unresolved_or_unowned_native_elf_fails_closed(tmp_path) -> None:
    entries = _required_entries()
    entries.append(file("usr/bin/unowned", _elf64(needed=("missing.so",))))
    assert {"native_elf_unowned", "native_elf_dependency_unresolved"} <= _codes(
        _verify(tmp_path, [entries])
    )


def test_native_needed_name_cannot_escape_the_library_search(tmp_path) -> None:
    entries = _required_entries()
    entries.append(file("usr/bin/unowned", _elf64(needed=("../missing.so",))))
    with pytest.raises(ValueError, match="habitat_elf_needed_name"):
        _verify(tmp_path, [entries])


def test_source_projection_rejects_undeclared_links(tmp_path) -> None:
    entries = _required_entries()
    entries.append(
        file(
            "usr/src/habitat-sim/undeclared-link",
            kind=tarfile.SYMTYPE,
            link="LICENSE",
        )
    )
    assert "source_projection_population_mismatch" in _codes(
        _verify(tmp_path, [entries])
    )


def test_every_layer_rejects_scene_paths_and_known_payload_hashes(tmp_path) -> None:
    contract = _fixture()[0]
    contract["forbidden_content_sha256"].append(_digest(b"scene payload"))
    archive, expected, diff_ids = _oci(
        tmp_path,
        [
            [
                file("opt/scene_datasets/skokloster-castle.glb", b"scene payload"),
                *_required_entries(),
            ]
        ],
    )
    contract["required_base_diff_ids"] = [diff_ids[0]]
    fd = os.open(archive, os.O_RDONLY)
    try:
        report = H.verify(
            fd,
            archive.stat().st_size,
            expected,
            contract,
            _digest(archive.read_bytes()),
            SOURCE_REVISION,
            "0" * 64,
            "0" * 64,
        )
    finally:
        os.close(fd)
    assert {"forbidden_path", "forbidden_payload_hash"} <= _codes(report)


def test_empty_base_cache_directories_are_allowed_but_cache_bytes_are_not(
    tmp_path,
) -> None:
    directories = [
        file("var/cache/apt", kind=tarfile.DIRTYPE),
        file("var/lib/apt/lists", kind=tarfile.DIRTYPE),
    ]
    assert _verify(tmp_path, [[*directories, *_required_entries()]])["valid"] is True

    cached = file("var/lib/apt/lists/archive.example_Packages", b"package metadata")
    assert "forbidden_path" in _codes(
        _verify(tmp_path, [[*directories, cached, *_required_entries()]])
    )


def test_final_runtime_refuses_builder_package_and_root_user(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file(
        "var/lib/dpkg/status",
        b"Package: cmake\nStatus: install ok installed\nVersion: 1\n\n"
        b"Package: python3\nStatus: install ok installed\n"
        b"Version: 3.10.6-1~22.04.1\nSource: python3-defaults\n\n",
    )
    report = _verify(tmp_path, [entries], user="root")
    assert {"forbidden_runtime_package", "final_user_not_ubuntu"} <= _codes(report)


def test_whiteout_and_required_symlink_cannot_fake_final_payload(tmp_path) -> None:
    required = _required_entries()
    target = "usr/src/habitat-sim/LICENSE"
    lower = required
    upper = [file("usr/src/habitat-sim/.wh.LICENSE")]
    assert "required_path_missing" in _codes(_verify(tmp_path, [lower, upper]))

    replaced = [row for row in required if row[0] != target]
    replaced.append(file(target, kind=tarfile.SYMTYPE, link="/elsewhere"))
    assert "required_path_not_file_or_directory" in _codes(
        _verify(tmp_path, [replaced])
    )


def test_wrong_uncompressed_diff_id_and_malformed_whiteout_fail(tmp_path) -> None:
    with pytest.raises(Exception, match="habitat_oci_layer_diff_id"):
        _verify(tmp_path, [_required_entries()], diff_ids=["sha256:" + "0" * 64])
    with pytest.raises(Exception, match="habitat_oci_malformed_whiteout"):
        _verify(
            tmp_path,
            [[file("opt/.wh.payload", b"nonempty"), *_required_entries()]],
        )


def test_duplicate_canonical_layer_path_fails_closed(tmp_path) -> None:
    entries = _required_entries()
    entries.append(file("./" + entries[0][0], b"duplicate"))
    with pytest.raises(Exception, match="habitat_oci_duplicate_layer_path"):
        _verify(tmp_path, [entries])


def test_lock_source_record_and_reviewed_revision_mismatches_fail(tmp_path) -> None:
    entries = _required_entries()
    metadata = next(
        index
        for index, row in enumerate(entries)
        if row[0].endswith("fixture-1.0.dist-info/METADATA")
    )
    entries[metadata] = file(entries[metadata][0], b"Name: fixture\nVersion: 2.0\n")
    report = _verify(tmp_path, [entries], source_revision="b" * 40)
    assert {
        "python_distribution_lock_mismatch",
        "source_revision_label_mismatch",
    } <= _codes(report)


def test_runtime_payload_hashes_bind_the_repository_lock_and_notice_bytes() -> None:
    expected = CONTRACT["required_final_file_sha256"]
    mappings = {
        "/opt/npa-runtime/npa/workflows/habitat_sim_smoke.py": ROOT
        / "npa/src/npa/workflows/habitat_sim_smoke.py",
        "/usr/share/doc/npa-habitat-sim/source-manifest.json": PACKAGE
        / "source-manifest.json",
        "/usr/share/doc/npa-habitat-sim/licenses.json": PACKAGE / "licenses.json",
        "/usr/share/doc/npa-habitat-sim/apt-runtime.lock": PACKAGE / "apt-runtime.lock",
        "/usr/share/doc/npa-habitat-sim/requirements-runtime.lock": PACKAGE
        / "requirements-runtime.lock",
        "/usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md": PACKAGE
        / "REDISTRIBUTION.md",
    }
    assert set(expected) == set(mappings)
    for image_path, source_path in mappings.items():
        assert expected[image_path] == _digest(source_path.read_bytes())


def test_runtime_payload_pins_the_selected_ubuntu_base_diff_id() -> None:
    assert CONTRACT["required_base_diff_ids"] == [
        "sha256:ea16cace89338c84eb6bcb91a7efdfcae6838fff359efe951858227436486c34"
    ]


def test_runtime_copyright_resolution_accepts_only_a_resolved_package_link() -> None:
    paths = {
        "usr/share/doc/python3": "symlink",
        "usr/share/doc/python3-minimal/copyright": "file",
    }
    links = {"usr/share/doc/python3": ("symlink", "python3-minimal")}
    assert (
        H._resolve_final_path("usr/share/doc/python3/copyright", paths, links)
        == "usr/share/doc/python3-minimal/copyright"
    )
    links["usr/share/doc/python3"] = ("symlink", "../../../../outside")
    assert (
        H._resolve_final_path("usr/share/doc/python3/copyright", paths, links) is None
    )
