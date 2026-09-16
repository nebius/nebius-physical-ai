"""Exercise Habitat's OCI quarantine verifier with synthetic image bytes."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import textwrap

import pytest
import yaml

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
VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "habitat_image_verifier", PACKAGE / "verify_image.py"
)
assert VERIFIER_SPEC is not None and VERIFIER_SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(VERIFIER_SPEC)
VERIFIER_SPEC.loader.exec_module(VERIFIER)
SOURCE_REVISION = "a" * 40


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _layer_tar(entries: list[tuple]) -> bytes:
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for row in entries:
            name, data, kind, link, pax, *metadata = row
            item = tarfile.TarInfo(name)
            item.type, item.linkname, item.pax_headers = kind, link, pax
            item.size = len(data) if kind in {tarfile.REGTYPE, tarfile.AREGTYPE} else 0
            if metadata:
                item.mode, item.uid, item.gid = metadata
            archive.addfile(item, io.BytesIO(data) if item.isfile() else None)
    return result.getvalue()


def _image_entry(
    name: str,
    data: bytes = b"",
    *,
    kind: bytes = tarfile.REGTYPE,
    mode: int = 0o644,
    uid: int = 0,
    gid: int = 0,
) -> tuple:
    return name, data, kind, "", {}, mode, uid, gid


def _oci(
    tmp_path: Path,
    layers: list[list[tuple]],
    *,
    user: str = "ubuntu",
    diff_ids: list[str] | None = None,
    source_revision: str = SOURCE_REVISION,
    source_manifest_sha256: str | None = None,
    command: list[str] | None = None,
    exposed_ports: dict[str, object] | None = None,
) -> tuple[Path, str, list[str]]:
    blobs: dict[str, bytes] = {}

    def blob(data: bytes, media: str) -> dict[str, object]:
        digest = "sha256:" + _digest(data)
        blobs["blobs/sha256/" + digest.removeprefix("sha256:")] = data
        return {"mediaType": media, "digest": digest, "size": len(data)}

    raw_layers = [_layer_tar(rows) for rows in layers]
    packed_layers = [blob(gzip.compress(raw, mtime=0), LAYER) for raw in raw_layers]
    labels = copy.deepcopy(CONTRACT["required_labels"])
    labels["org.opencontainers.image.revision"] = source_revision
    if source_manifest_sha256 is not None:
        labels["org.nebius.npa.source-manifest-sha256"] = source_manifest_sha256
        labels["org.nebius.npa.source-provenance-schema"] = VERIFIER.PROVENANCE_SCHEMA
        labels["org.nebius.npa.sudo-bootstrap-contract"] = (
            "habitat-sim-skypilot-0.12.2-v1"
        )
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
                    "Cmd": command
                    or [
                        "python3",
                        "-m",
                        "npa.workflows.habitat_sim_smoke",
                        "--help",
                    ],
                    "ExposedPorts": exposed_ports,
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
    projected_pbr = {
        row["path"]: f"pbr fixture:{row['path']}\n".encode()
        for row in json.loads((PACKAGE / "source-manifest.json").read_text())["source"][
            "required_projection_files"
        ]
    }
    source_files = {
        "LICENSE": b"source-license",
        "data/default.physics_config.json": b"{}\n",
        **projected_pbr,
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
        "usr/share/doc/npa-habitat-sim/apt-build.lock": b"build lock\n",
        "usr/share/doc/npa-habitat-sim/apt-runtime.lock": apt_lock,
        "usr/share/doc/npa-habitat-sim/requirements-runtime.lock": python_lock,
        "usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md": b"redistribution\n",
        "usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md": b"notices\n",
    }
    contract["required_final_file_sha256"] = {
        "/" + path: _digest(payload)
        for path, payload in controls.items()
        if not path.endswith("source-projection.json")
    }
    smoke_module = "/opt/npa-runtime/npa/workflows/habitat_sim_smoke.py"
    contract["executable_source_bindings"] = {
        smoke_module: {
            "kind": "file",
            "uid": 0,
            "gid": 0,
            "mode": 0o644,
            "sha256": _digest(controls[smoke_module.removeprefix("/")]),
        }
    }
    for path, payload in projected_pbr.items():
        contract["required_final_file_sha256"][f"/usr/src/habitat-sim/{path}"] = (
            _digest(payload)
        )
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
    expected_python_venv_inventory_sha256: str | None = None,
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
            expected_python_venv_inventory_sha256
            or probe["python_venv_inventory_sha256"],
            expected_native_closure_sha256 or probe["native_elf_closure_sha256"],
        )
    finally:
        os.close(fd)


def _cli_source_manifest() -> tuple[bytes, dict[str, bytes]]:
    inputs = {
        path: f"committed fixture:{path}\n".encode()
        for path in VERIFIER.NPA_SOURCE_PATHS
    }
    rows = [
        f"{_digest(payload)}  inputs/{path}\n".encode()
        for path, payload in inputs.items()
    ]
    return b"".join(sorted(rows, key=lambda row: row.split(b"  ", 1)[1])), inputs


def _cli_fixture(
    tmp_path: Path,
    *,
    mutation: str | None = None,
    label_manifest_sha256: str | None = None,
    image_source_revision: str = SOURCE_REVISION,
) -> dict[str, object]:
    analysis = tmp_path / "analysis"
    analysis.mkdir(mode=0o700)
    contract, entries = _fixture()
    manifest, source_inputs = _cli_source_manifest()
    manifest_sha256 = _digest(manifest)
    provenance = VERIFIER._provenance_bytes(SOURCE_REVISION, manifest_sha256)
    provenance_entries = [
        file(
            f"{VERIFIER.PROVENANCE_ROOT}/npa-source-manifest.sha256",
            manifest if mutation != "manifest" else manifest + b"hostile\n",
        ),
        file(
            f"{VERIFIER.PROVENANCE_ROOT}/npa-source-provenance.json",
            provenance if mutation != "provenance" else provenance + b"hostile\n",
        ),
    ]
    for path, payload in source_inputs.items():
        if mutation == "file" and path == VERIFIER.NPA_SOURCE_PATHS[0]:
            payload += b"hostile\n"
        provenance_entries.append(
            file(f"{VERIFIER.PROVENANCE_ROOT}/inputs/{path}", payload)
        )
    installed_entries = [
        _image_entry(
            destination.lstrip("/"),
            source_inputs[source.removeprefix("inputs/")],
            mode=mode,
        )
        for source, (
            destination,
            mode,
        ) in VERIFIER.EXECUTABLE_SOURCE_DESTINATIONS.items()
    ]
    installed_paths = {row[0] for row in installed_entries}
    entries = [row for row in entries if row[0] not in installed_paths]
    installed_entries.extend(
        [
            _image_entry("etc/passwd", b"ubuntu:x:1000:1000::/home/ubuntu:/bin/bash\n"),
            _image_entry("etc/group", b"ubuntu:x:1000:\n"),
            _image_entry(
                "home/ubuntu/.ssh", kind=tarfile.DIRTYPE, mode=0o700, uid=1000, gid=1000
            ),
            *[
                _image_entry(
                    path.lstrip("/"),
                    payload,
                    mode=0o440 if "sudoers" in path else 0o644,
                )
                for path, payload in VERIFIER.SYSTEM_FILE_BYTES.items()
            ],
        ]
    )
    if mutation == "path-set":
        provenance_entries.append(
            file(f"{VERIFIER.PROVENANCE_ROOT}/inputs/undeclared.txt", b"hostile\n")
        )
    archive, expected, diff_ids = _oci(
        analysis,
        [[*entries, *provenance_entries, *installed_entries]],
        source_revision=image_source_revision,
        source_manifest_sha256=label_manifest_sha256 or manifest_sha256,
    )
    archive.chmod(0o600)
    contract["required_base_diff_ids"] = [diff_ids[0]]
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline = _verify(baseline_root, [entries], contract=contract)
    contract_path = analysis / "runtime-payload.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_path.chmod(0o600)
    manifest_path = analysis / "expected-npa-source-manifest.sha256"
    manifest_path.write_bytes(manifest)
    manifest_path.chmod(0o600)
    return {
        "analysis": analysis,
        "archive": archive,
        "contract": contract_path,
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "expected": expected,
        "dpkg": baseline["dpkg_inventory_sha256"],
        "python": baseline["python_venv_inventory_sha256"],
        "native": baseline["native_elf_closure_sha256"],
    }


def _write_cli_wrapper(wrapper: Path) -> None:
    wrapper.write_text(
        textwrap.dedent(
            f"""
            import importlib.util
            import json
            import os
            from pathlib import Path
            import sys

            os.umask(0o077)
            spec = importlib.util.spec_from_file_location(
                "habitat_image_verifier", {str(PACKAGE / "verify_image.py")!r}
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            contract = json.loads(Path(sys.argv[1]).read_text())
            manifest = Path(sys.argv[2]).read_bytes()
            module._load_contract = lambda: contract
            module._source_manifest_from_git = lambda _revision: manifest
            raise SystemExit(module.main(sys.argv[3:]))
            """
        ),
        encoding="utf-8",
    )


def _cli_arguments(
    wrapper: Path,
    report: Path,
    fixture: dict[str, object],
    analysis_root: Path,
    trusted_root: Path,
    archive: Path,
) -> list[str]:
    return [
        sys.executable,
        str(wrapper),
        str(fixture["contract"]),
        str(fixture["manifest"]),
        "--analysis-root",
        str(analysis_root or fixture["analysis"]),
        "--trusted-root",
        str(trusted_root),
        "--oci-archive",
        str(archive or fixture["archive"]),
        "--expected-image-id",
        str(fixture["expected"]),
        "--expected-source-revision",
        SOURCE_REVISION,
        "--expected-npa-source-manifest-sha256",
        str(fixture["manifest_sha256"]),
        "--expected-dpkg-inventory-sha256",
        str(fixture["dpkg"]),
        "--expected-python-venv-inventory-sha256",
        str(fixture["python"]),
        "--expected-native-closure-sha256",
        str(fixture["native"]),
        "--json",
        str(report),
    ]


def _run_cli(
    tmp_path: Path,
    fixture: dict[str, object],
    *,
    analysis_root: Path | None = None,
    trusted_root: Path = ROOT,
    archive: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    wrapper = tmp_path / "invoke-verifier.py"
    _write_cli_wrapper(wrapper)
    report = tmp_path / "cli-report.json"
    command = _cli_arguments(
        wrapper,
        report,
        fixture,
        analysis_root or fixture["analysis"],
        trusted_root,
        archive or fixture["archive"],
    )
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result, json.loads(report.read_text())


def _codes(report: dict[str, object]) -> set[str]:
    return {row["code"] for row in report["findings"]}


def test_valid_attested_oci_has_complete_graph_and_payload_receipt(tmp_path) -> None:
    report = _verify(tmp_path, [_required_entries()])
    assert report["valid"] is True
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 36
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
    assert report["projected_source_file_count"] == 15
    assert report["python_distribution_count"] == 4
    assert report["python_record_files_verified"] == 1
    assert len(report["python_venv_inventory"]) == 9
    assert (
        report["python_venv_inventory_sha256"]
        == report["expected_python_venv_inventory_sha256"]
    )
    assert report["native_elf_count"] == 1
    assert report["dpkg_inventory_sha256"] == report["expected_dpkg_inventory_sha256"]
    assert (
        report["native_elf_closure_sha256"] == report["expected_native_closure_sha256"]
    )
    assert report["native_elf_closure"][0]["owners"] == ["python-wheel-record"]
    assert report["expected_source_revision"] == SOURCE_REVISION
    assert report["oci_graph"]["attestation_manifest_count"] == 1


def test_executable_sources_reject_extra_replacement_mode_and_whiteout(
    tmp_path,
) -> None:
    for name in ("valid", "extra", "replacement", "mode", "owner", "type", "whiteout"):
        (tmp_path / name).mkdir()
    contract, entries = _fixture()
    destination = "opt/npa-runtime/npa/workflows/habitat_sim_smoke.py"
    payload = b"smoke\n"
    contract["required_final_metadata"] = {
        "/" + destination: {"kind": "file", "uid": 0, "gid": 0, "mode": 0o644}
    }
    contract["executable_source_bindings"] = {
        "/" + destination: {
            "kind": "file",
            "uid": 0,
            "gid": 0,
            "mode": 0o644,
            "sha256": _digest(payload),
        }
    }
    assert _verify(tmp_path / "valid", [entries], contract=contract)["valid"] is True

    extra = [*entries, file("opt/npa-runtime/npa/hostile.py", b"hostile")]
    assert "executable_source_path_set_mismatch" in _codes(
        _verify(tmp_path / "extra", [extra], contract=contract)
    )
    replacement = [row for row in entries if row[0] != destination]
    replacement.append(_image_entry(destination, b"hostile replacement\n"))
    assert "executable_source_layer_policy" in _codes(
        _verify(tmp_path / "replacement", [replacement], contract=contract)
    )
    wrong_mode = [row for row in entries if row[0] != destination]
    wrong_mode.append(_image_entry(destination, payload, mode=0o777))
    assert "required_final_metadata_mismatch" in _codes(
        _verify(tmp_path / "mode", [wrong_mode], contract=contract)
    )
    wrong_owner = [row for row in entries if row[0] != destination]
    wrong_owner.append(_image_entry(destination, payload, uid=1000, gid=1000))
    assert "executable_source_layer_policy" in _codes(
        _verify(tmp_path / "owner", [wrong_owner], contract=contract)
    )
    wrong_type = [row for row in entries if row[0] != destination]
    wrong_type.append(file(destination, kind=tarfile.SYMTYPE, link="/" + "tmp/hostile"))
    assert "required_final_metadata_mismatch" in _codes(
        _verify(tmp_path / "type", [wrong_type], contract=contract)
    )
    whiteout = file("opt/npa-runtime/npa/workflows/.wh.habitat_sim_smoke.py")
    report = _verify(tmp_path / "whiteout", [entries, [whiteout]], contract=contract)
    assert "executable_source_layer_policy" in _codes(report)


def test_config_and_account_boundary_fail_closed(tmp_path) -> None:
    (tmp_path / "command").mkdir()
    (tmp_path / "ports").mkdir()
    wrong_command = _verify(
        tmp_path / "command",
        [_required_entries()],
        command=["/usr/sbin/sshd", "-D"],
    )
    assert "unexpected_default_command" in _codes(wrong_command)
    exposed = _verify(
        tmp_path / "ports",
        [_required_entries()],
        exposed_ports={"22/tcp": {}},
    )
    assert "unexpected_exposed_ports" in _codes(exposed)
    assert H._account_boundary_findings(
        {
            "etc/passwd": b"ubuntu:x:0:0::/root:/bin/bash\n",
            "etc/group": b"ubuntu:x:0:\n",
        }
    ) == [
        {"code": "runtime_user_identity_mismatch"},
        {"code": "runtime_group_identity_mismatch"},
    ]


def test_host_verifier_cli_accepts_owner_only_synthetic_oci(tmp_path) -> None:
    fixture = _cli_fixture(tmp_path)
    assert fixture["analysis"].stat().st_mode & 0o777 == 0o700
    assert fixture["archive"].stat().st_mode & 0o777 == 0o600
    result, report = _run_cli(tmp_path, fixture)
    assert result.returncode == 0, result.stderr
    assert report["valid"] is True
    assert report["npa_source_manifest_sha256"] == fixture["manifest_sha256"]
    assert report["npa_source_file_count"] == len(VERIFIER.NPA_SOURCE_PATHS)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("manifest", "required_file_hash_mismatch"),
        ("provenance", "required_file_hash_mismatch"),
        ("file", "required_file_hash_mismatch"),
        ("path-set", "source_provenance_unexpected_path"),
    ],
)
def test_host_verifier_cli_refuses_source_provenance_mutations(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    fixture = _cli_fixture(tmp_path, mutation=mutation)
    result, report = _run_cli(tmp_path, fixture)
    assert result.returncode == 1
    assert expected_code in _codes(report)


def test_host_verifier_cli_refuses_source_label_or_revision_drift(tmp_path) -> None:
    label = tmp_path / "label"
    label.mkdir()
    fixture = _cli_fixture(label, label_manifest_sha256="0" * 64)
    result, report = _run_cli(label, fixture)
    assert result.returncode == 1
    assert "required_label_mismatch" in _codes(report)

    revision = tmp_path / "revision"
    revision.mkdir()
    fixture = _cli_fixture(revision, image_source_revision="b" * 40)
    result, report = _run_cli(revision, fixture)
    assert result.returncode == 1
    assert "source_revision_label_mismatch" in _codes(report)


@pytest.mark.parametrize(
    ("root_name", "expected_code"),
    [
        ("analysis", "analysis_root_missing"),
        ("trusted", "trusted_root_missing"),
    ],
)
def test_host_verifier_cli_identifies_missing_root(
    tmp_path, root_name: str, expected_code: str
) -> None:
    fixture = _cli_fixture(tmp_path)
    roots = {"analysis": fixture["analysis"], "trusted": ROOT}
    roots[root_name] = tmp_path / f"missing-{root_name}"
    result, report = _run_cli(
        tmp_path,
        fixture,
        analysis_root=roots["analysis"],
        trusted_root=roots["trusted"],
    )
    assert result.returncode == 1
    assert _codes(report) == {expected_code}


def test_host_verifier_cli_refuses_archive_outside_analysis_root(tmp_path) -> None:
    fixture = _cli_fixture(tmp_path)
    unrelated = tmp_path / "unrelated-analysis"
    unrelated.mkdir(mode=0o700)
    result, report = _run_cli(tmp_path, fixture, analysis_root=unrelated)
    assert result.returncode == 1
    assert _codes(report) == {"input_outside_authorized_roots"}


def test_host_verifier_cli_refuses_unsafe_or_wrong_trusted_root(tmp_path) -> None:
    fixture = _cli_fixture(tmp_path)
    unsafe = tmp_path / "unsafe-analysis"
    unsafe.mkdir(mode=0o755)
    unsafe.chmod(0o755)
    result, report = _run_cli(tmp_path, fixture, analysis_root=unsafe)
    assert result.returncode == 1
    assert _codes(report) == {"root_permissions"}

    wrong_trusted = tmp_path / "wrong-trusted"
    wrong_trusted.mkdir(mode=0o755)
    result, report = _run_cli(tmp_path, fixture, trusted_root=wrong_trusted)
    assert result.returncode == 1
    assert _codes(report) == {"trusted_source_root_mismatch"}


def test_complete_runtime_inventory_hashes_are_required(tmp_path) -> None:
    report = _verify(
        tmp_path,
        [_required_entries()],
        expected_dpkg_inventory_sha256="0" * 64,
        expected_python_venv_inventory_sha256="0" * 64,
        expected_native_closure_sha256="0" * 64,
    )
    assert {
        "runtime_dpkg_inventory_lock_mismatch",
        "python_venv_inventory_lock_mismatch",
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
        "expected_python_venv_inventory_sha256": closure,
        "python_venv_inventory_sha256": closure,
        "expected_native_closure_sha256": closure,
        "native_elf_closure_sha256": closure,
        "expected_source_revision": SOURCE_REVISION,
    }
    H.bind(result, receipt, digest)

    for key in (
        "valid",
        "dpkg_inventory_sha256",
        "python_venv_inventory_sha256",
        "native_elf_closure_sha256",
    ):
        invalid = copy.deepcopy(receipt)
        invalid[key] = False if key == "valid" else "6" * 64
        with pytest.raises(
            ValueError,
            match="habitat_oci_verifier_not_valid|habitat_oci_runtime_closure_binding",
        ):
            H.bind(result, invalid, digest)


def test_python_venv_inventory_rejects_record_rewrite_and_unrecorded_bytes(
    tmp_path,
) -> None:
    original = b"VALUE = 'reviewed'\n"
    changed = b"VALUE = 'changed'\n"
    path = "opt/venv/lib/python3.10/site-packages/fixture/runtime.py"
    record_path = "opt/venv/lib/python3.10/site-packages/fixture-1.0.dist-info/RECORD"
    native = _elf64(soname="native.so")
    entries = _required_entries()
    record_index = next(
        index for index, row in enumerate(entries) if row[0] == record_path
    )
    entries[record_index] = file(
        record_path,
        (
            f"fixture/native.so,{_record_hash(native)},{len(native)}\n"
            f"fixture/runtime.py,{_record_hash(original)},{len(original)}\n"
            "fixture-1.0.dist-info/RECORD,,\n"
        ).encode(),
    )
    entries.append(file(path, original))
    baseline = _verify(tmp_path, [entries])
    assert baseline["valid"] is True

    rewritten = copy.deepcopy(entries)
    rewritten[-1] = file(path, changed)
    rewritten[record_index] = file(
        record_path,
        (
            f"fixture/native.so,{_record_hash(native)},{len(native)}\n"
            f"fixture/runtime.py,{_record_hash(changed)},{len(changed)}\n"
            "fixture-1.0.dist-info/RECORD,,\n"
        ).encode(),
    )
    rewritten_report = _verify(
        tmp_path,
        [rewritten],
        expected_python_venv_inventory_sha256=baseline["python_venv_inventory_sha256"],
    )
    assert "python_venv_inventory_lock_mismatch" in _codes(rewritten_report)

    unrecorded = _required_entries()
    unrecorded_baseline = _verify(tmp_path, [unrecorded])
    unrecorded.append(file(path, changed))
    unrecorded_report = _verify(
        tmp_path,
        [unrecorded],
        expected_python_venv_inventory_sha256=unrecorded_baseline[
            "python_venv_inventory_sha256"
        ],
    )
    assert "python_venv_inventory_lock_mismatch" in _codes(unrecorded_report)


def test_python_distribution_record_population_and_unhashed_entries_are_closed(
    tmp_path,
) -> None:
    missing_record = [
        row
        for row in _required_entries()
        if not row[0].endswith("pip-22.0.2.dist-info/RECORD")
    ]
    assert "python_record_population_invalid" in _codes(
        _verify(tmp_path, [missing_record])
    )

    record_path = "opt/venv/lib/python3.10/site-packages/fixture-1.0.dist-info/RECORD"
    native = _elf64(soname="native.so")
    unhashed = _required_entries()
    record_index = next(
        index for index, row in enumerate(unhashed) if row[0] == record_path
    )
    unhashed[record_index] = file(
        record_path,
        (
            f"fixture/native.so,{_record_hash(native)},{len(native)}\n"
            "fixture/runtime.py,,\n"
            "fixture-1.0.dist-info/RECORD,,\n"
        ).encode(),
    )
    assert "python_record_entry_unbound" in _codes(_verify(tmp_path, [unhashed]))


def test_hashless_generated_bytecode_is_bound_by_the_exact_venv_inventory(
    tmp_path,
) -> None:
    bytecode = b"\x42\x0d\x0d\x0a deterministic fixture bytecode"
    changed = b"\x42\x0d\x0d\x0a changed fixture bytecode"
    path = (
        "opt/venv/lib/python3.10/site-packages/fixture/__pycache__/"
        "runtime.cpython-310.pyc"
    )
    record_path = "opt/venv/lib/python3.10/site-packages/fixture-1.0.dist-info/RECORD"
    native = _elf64(soname="native.so")
    entries = _required_entries()
    record_index = next(
        index for index, row in enumerate(entries) if row[0] == record_path
    )
    entries[record_index] = file(
        record_path,
        (
            f"fixture/native.so,{_record_hash(native)},{len(native)}\n"
            "fixture/__pycache__/runtime.cpython-310.pyc,,\n"
            "fixture-1.0.dist-info/RECORD,,\n"
        ).encode(),
    )
    entries.append(file(path, bytecode))
    baseline = _verify(tmp_path, [entries])
    assert baseline["valid"] is True

    entries[-1] = file(path, changed)
    report = _verify(
        tmp_path,
        [entries],
        expected_python_venv_inventory_sha256=baseline["python_venv_inventory_sha256"],
    )
    assert "python_venv_inventory_lock_mismatch" in _codes(report)


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


@pytest.mark.parametrize(
    "target",
    [
        "usr/src/habitat-sim/data/pbr/PbrImages.conf",
        "usr/src/habitat-sim/data/pbr/bluts/brdflut_ldr_512x512.png",
        "usr/src/habitat-sim/data/pbr/env_maps/anniversary_lounge_1k.hdr",
        "usr/src/habitat-sim/data/pbr/license.txt",
    ],
)
def test_source_projection_refuses_missing_or_changed_required_pbr_file(
    tmp_path,
    target: str,
) -> None:
    omitted = tmp_path / "omitted"
    omitted.mkdir()
    without_pbr = [row for row in _required_entries() if row[0] != target]
    assert {
        "required_path_missing",
        "required_file_hash_mismatch",
        "source_projection_file_mismatch",
        "source_projection_population_mismatch",
    } <= _codes(_verify(omitted, [without_pbr]))

    changed = tmp_path / "changed"
    changed.mkdir()
    changed_pbr = _required_entries()
    index = next(i for i, row in enumerate(changed_pbr) if row[0] == target)
    changed_pbr[index] = file(target, b"changed pbr fixture\n")
    assert {
        "required_file_hash_mismatch",
        "source_projection_file_mismatch",
    } <= _codes(_verify(changed, [changed_pbr]))


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


@pytest.mark.parametrize(
    "path",
    [
        "root/.cache/payload.bin",
        "home/ubuntu/.cache/payload.bin",
        "opt/runtime/.cache/payload.bin",
    ],
)
def test_dot_cache_payloads_are_refused(tmp_path, path: str) -> None:
    cached = file(path, b"cached payload")
    assert "forbidden_path" in _codes(
        _verify(tmp_path, [[cached, *_required_entries()]])
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
    repository_mappings = {
        "/usr/share/doc/npa-habitat-sim/source-manifest.json": PACKAGE
        / "source-manifest.json",
        "/usr/share/doc/npa-habitat-sim/licenses.json": PACKAGE / "licenses.json",
        "/usr/share/doc/npa-habitat-sim/apt-build.lock": PACKAGE / "apt-build.lock",
        "/usr/share/doc/npa-habitat-sim/apt-runtime.lock": PACKAGE / "apt-runtime.lock",
        "/usr/share/doc/npa-habitat-sim/requirements-runtime.lock": PACKAGE
        / "requirements-runtime.lock",
        "/usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md": PACKAGE
        / "REDISTRIBUTION.md",
        "/usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md": PACKAGE
        / "THIRD_PARTY_NOTICES.md",
    }
    projected_source = {
        f"/usr/src/habitat-sim/{row['path']}": row["sha256"]
        for row in json.loads((PACKAGE / "source-manifest.json").read_text())["source"][
            "required_projection_files"
        ]
    }
    source_artifacts = {
        f"/usr/share/doc/npa-habitat-sim/ubuntu-sources/rsync/{row['filename']}": row[
            "sha256"
        ]
        for row in yaml.safe_load((PACKAGE / "apt-runtime.lock").read_text())[
            "corresponding_sources"
        ][0]["artifacts"]
    }
    assert set(expected) == (
        set(repository_mappings)
        | {"/opt/npa-runtime/npa/workflows/habitat_sim_smoke.py"}
        | set(projected_source)
        | set(source_artifacts)
    )
    for image_path, source_path in repository_mappings.items():
        assert expected[image_path] == _digest(source_path.read_bytes())
    for image_path, digest in source_artifacts.items():
        assert expected[image_path] == digest
    for image_path, digest in projected_source.items():
        assert expected[image_path] == digest
    inputs = {
        f"inputs/{path}": (ROOT / "npa" / path).read_bytes()
        for path in VERIFIER.NPA_SOURCE_PATHS
    }
    manifest = b"".join(
        sorted(
            (
                f"{_digest(payload)}  {path}\n".encode()
                for path, payload in inputs.items()
            ),
            key=lambda row: row.split(b"  ", 1)[1],
        )
    )
    bound, _source_inputs = VERIFIER._bind_source_contract(
        copy.deepcopy(CONTRACT), SOURCE_REVISION, manifest, _digest(manifest)
    )
    for source, (destination, _mode) in VERIFIER.EXECUTABLE_SOURCE_DESTINATIONS.items():
        assert bound["required_final_file_sha256"][destination] == _digest(
            inputs[source]
        )
        assert bound["executable_source_bindings"][destination]["sha256"] == _digest(
            inputs[source]
        )
    for destination, payload in VERIFIER.SYSTEM_FILE_BYTES.items():
        assert bound["required_final_file_sha256"][destination] == _digest(payload)
    assert bound["required_final_metadata"]["/home/ubuntu/.ssh"] == {
        "kind": "directory",
        "uid": 1000,
        "gid": 1000,
        "mode": 0o700,
    }


def test_host_verifier_refuses_incomplete_or_wrong_pbr_byte_contract() -> None:
    manifest = json.loads((PACKAGE / "source-manifest.json").read_text())
    VERIFIER._validate_source_contract(copy.deepcopy(CONTRACT), manifest)
    for row in manifest["source"]["required_projection_files"]:
        path = f"/usr/src/habitat-sim/{row['path']}"
        omitted = copy.deepcopy(CONTRACT)
        omitted["required_final_file_sha256"].pop(path)
        with pytest.raises(ValueError, match="PBR byte contract"):
            VERIFIER._validate_source_contract(omitted, manifest)
        changed = copy.deepcopy(CONTRACT)
        changed["required_final_file_sha256"][path] = "0" * 64
        with pytest.raises(ValueError, match="PBR byte contract"):
            VERIFIER._validate_source_contract(changed, manifest)


def test_host_verifier_refuses_unrequired_pbr_path_or_missing_notice() -> None:
    manifest = json.loads((PACKAGE / "source-manifest.json").read_text())
    extra = copy.deepcopy(CONTRACT)
    extra["required_final_file_sha256"][
        "/usr/src/habitat-sim/data/pbr/undeclared.bin"
    ] = "0" * 64
    with pytest.raises(ValueError, match="PBR byte contract"):
        VERIFIER._validate_source_contract(extra, manifest)
    missing_notice = copy.deepcopy(CONTRACT)
    missing_notice["required_final_file_sha256"].pop(
        "/usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md"
    )
    with pytest.raises(ValueError, match="PBR notice contract"):
        VERIFIER._validate_source_contract(missing_notice, manifest)


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
