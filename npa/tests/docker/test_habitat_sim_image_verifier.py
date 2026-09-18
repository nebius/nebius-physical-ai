"""Exercise Habitat's OCI quarantine verifier with synthetic image bytes."""

from __future__ import annotations

import base64
import ast
import copy
from contextlib import nullcontext
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
from types import SimpleNamespace
from unittest.mock import Mock, call

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


def _source_delivery_state() -> H._ScanState:
    state = H._ScanState()
    state.paths["usr/share/doc/base-package/copyright"] = "file"
    state.files["usr/share/doc/base-package/copyright"] = {"sha256": "b" * 64}
    state.tracked["var/lib/dpkg/status"] = (
        b"Package: base-package\nStatus: install ok installed\n"
        b"Version: 1.0\nSource: base-source (1.0-1)\n\n"
    )
    H._record_source_population(state)
    state.tracked["var/lib/dpkg/status"] = state.tracked["var/lib/dpkg/status"].replace(
        b"1.0", b"2.0"
    )
    H._record_source_population(state)
    state.files["usr/share/doc/npa-habitat-sim/fixture-source.txt"] = {
        "sha256": _digest(b"inert source fixture\n"),
        "size": len(b"inert source fixture\n"),
    }
    return state


def test_source_delivery_covers_superseded_base_and_installed_versions() -> None:
    state = _source_delivery_state()
    assert {row["version"] for row in state.source_inventory.values()} == {"1.0", "2.0"}
    assert {row["source_version"] for row in state.source_inventory.values()} == {
        "1.0-1",
        "2.0-1",
    }
    closure = _fixture_source_delivery(
        {
            "corresponding_source_inventory": state.source_inventory,
            "corresponding_source_inventory_sha256": H._source_identity(
                state.source_inventory
            ),
        }
    )
    assert (
        H._source_delivery_findings(state, {"corresponding_source_closure": closure})
        == []
    )
    # Whiteout/removal from the final rootfs cannot erase an ancestor obligation.
    state.tracked.clear()
    H._record_source_population(state)
    assert len(state.source_inventory) == 2


@pytest.mark.parametrize(
    "mutation",
    ["pending", "partial", "version", "digest", "missing", "size", "unbound"],
)
def test_source_delivery_refuses_incomplete_or_unbound_closure(mutation: str) -> None:
    state = _source_delivery_state()
    closure = _fixture_source_delivery(
        {
            "corresponding_source_inventory": copy.deepcopy(state.source_inventory),
            "corresponding_source_inventory_sha256": H._source_identity(
                state.source_inventory
            ),
        }
    )
    identity = next(iter(closure["records"]))
    record = closure["records"][identity]
    if mutation == "pending":
        closure["status"] = "pending-exact-layer-inventory-and-delivery"
    elif mutation == "partial":
        del closure["records"][identity]
    elif mutation == "version":
        record["component"]["source_version"] = "wrong"
    elif mutation == "digest":
        record["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        record["artifacts"][0]["bytes"] += 1
    elif mutation == "unbound":
        closure["inventory_sha256"] = "0" * 64
    else:
        state.files.clear()
    findings = H._source_delivery_findings(
        state, {"corresponding_source_closure": closure}
    )
    assert findings
    assert all(row["code"].startswith("corresponding_source_") for row in findings)


def test_current_image_contract_remains_source_delivery_quarantined() -> None:
    assert CONTRACT["corresponding_source_closure"] == {
        "status": "pending-exact-layer-inventory-and-delivery",
        "inventory_sha256": None,
        "records": {},
    }
    findings = H._source_delivery_findings(_source_delivery_state(), CONTRACT)
    assert {row["code"] for row in findings} >= {
        "corresponding_source_closure_pending",
        "corresponding_source_population_mismatch",
    }


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mock_archive_info():
    """Describe a synthetic held descriptor without opening any object."""
    return SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_size=12,
        st_mtime_ns=3,
        st_ctime_ns=4,
        st_mode=0o100600,
        st_uid=5,
        st_gid=6,
    )


def _mock_archive_arguments():
    """Supply inert source and inventory identities for the descriptor test."""
    return SimpleNamespace(
        analysis_root=Path("analysis"),
        trusted_root=VERIFIER.CHECKOUT_ROOT,
        oci_archive=Path("inert-archive"),
        expected_source_revision=SOURCE_REVISION,
        expected_npa_source_manifest_sha256="b" * 64,
        expected_image_id="c" * 64,
        expected_dpkg_inventory_sha256="d" * 64,
        expected_python_venv_inventory_sha256="e" * 64,
        expected_native_closure_sha256="f" * 64,
    )


def _mock_archive_verification(monkeypatch):
    """Prepare an inert descriptor; never open or parse an actual image."""
    args, info = _mock_archive_arguments(), _mock_archive_info()
    monkeypatch.setattr(VERIFIER, "_require_root", Mock())
    monkeypatch.setattr(
        VERIFIER.W,
        "authorized_roots",
        lambda *_: nullcontext((None, VERIFIER.CHECKOUT_ROOT)),
    )
    monkeypatch.setattr(VERIFIER, "_source_manifest_from_git", lambda _: b"manifest")
    monkeypatch.setattr(VERIFIER, "_load_contract", lambda: {})
    monkeypatch.setattr(
        VERIFIER, "_bind_source_contract", lambda *_: ({}, {"input": "hash"})
    )
    unopened = Mock(spec=Path)
    unopened.read_bytes.side_effect = AssertionError("pathname reopen forbidden")
    opened = Mock(return_value=(unopened, 42, info))
    digest = Mock(return_value=_digest(b"held descriptor fixture"))
    verify = Mock(return_value={"findings": [], "valid": True})
    provenance, close = Mock(return_value=[]), Mock()
    monkeypatch.setattr(VERIFIER.W, "open_private_fd", opened)
    monkeypatch.setattr(VERIFIER.W, "descriptor_digest", digest)
    monkeypatch.setattr(VERIFIER.os, "fstat", Mock(return_value=info))
    monkeypatch.setattr(VERIFIER.os, "close", close)
    monkeypatch.setattr(VERIFIER.H, "verify", verify)
    monkeypatch.setattr(VERIFIER, "_source_provenance_findings", provenance)
    return args, info, opened, digest, verify, provenance, close


def test_archive_digest_uses_only_the_held_verification_descriptor(monkeypatch) -> None:
    args, _, opened, digest, verify, provenance, close = _mock_archive_verification(
        monkeypatch
    )
    report = VERIFIER._verify_archive(args)
    opened.assert_called_once_with(args.oci_archive)
    opened.return_value[0].read_bytes.assert_not_called()
    digest.assert_called_once_with(42)
    assert verify.call_args.args[:2] == (42, 12)
    assert verify.call_args.args[4] == _digest(b"held descriptor fixture")
    assert provenance.call_args.args[:2] == (42, 12)
    assert VERIFIER.os.fstat.call_args_list == [call(42)] * 3
    close.assert_called_once_with(42)
    assert report["valid"] is True and report["npa_source_file_count"] == 1


@pytest.mark.parametrize("check", [0, 1, 2])
@pytest.mark.parametrize(
    "field",
    [
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "st_mode",
        "st_uid",
        "st_gid",
    ],
)
def test_archive_identity_mismatch_refuses_before_publication(
    monkeypatch, capsys, check, field
) -> None:
    """Inject stat observations only; no concurrent filesystem mutation occurs."""
    args, info, _, digest, verify, provenance, close = _mock_archive_verification(
        monkeypatch
    )
    changed = SimpleNamespace(**{**vars(info), field: getattr(info, field) + 1})
    monkeypatch.setattr(
        VERIFIER.os, "fstat", Mock(side_effect=[info] * check + [changed])
    )
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    monkeypatch.setattr(VERIFIER, "_report_destination", lambda _: nullcontext(41))
    publish = Mock()
    monkeypatch.setattr(VERIFIER, "_publish_report", publish)
    assert VERIFIER.main([]) == 1
    assert "archive_changed_during_verification" in capsys.readouterr().err
    publish.assert_not_called()
    assert digest.call_count == (check > 0)
    assert verify.call_count == provenance.call_count == (check == 2)
    close.assert_called_once_with(42)


def test_descriptor_digest_preserves_benign_file_position(tmp_path) -> None:
    payload = b"benign owner-created descriptor fixture\n"
    path = tmp_path / "fixture.txt"
    path.write_bytes(payload)
    path.chmod(0o600)
    with VERIFIER.W.authorized_roots(tmp_path, ROOT):
        with VERIFIER._bound_archive_descriptor(path) as (fd, length, digest):
            assert length == len(payload) and digest == _digest(payload)
            assert os.lseek(fd, 0, os.SEEK_CUR) == 0
            assert os.read(fd, length) == payload


def test_host_verifier_source_helpers_are_focused_and_documented() -> None:
    source = ast.parse((PACKAGE / "verify_image.py").read_text())
    names = {
        "_bind_source_contract",
        "_bind_provenance_files",
        "_bind_executable_sources",
        "_bind_bootstrap_files",
        "_source_provenance_findings",
        "_inspect_provenance_layer",
        "_record_provenance_member",
        "_confirm_archive_descriptor",
        "_bound_archive_descriptor",
        "_archive_report",
        "_verify_archive",
    }
    functions = {
        node.name: node for node in source.body if isinstance(node, ast.FunctionDef)
    }
    for name in names:
        assert functions[name].end_lineno - functions[name].lineno < 40, name
        assert ast.get_docstring(functions[name]), name
    assert set(name for name in functions if not name.startswith("_")) == {"main"}
    for section in ("Args:", "Returns:", "Raises:"):
        assert section in VERIFIER.main.__doc__


def test_verifier_exported_contracts_have_structured_documentation() -> None:
    for function in (H.inspect, H.bind, H.verify):
        assert function.__doc__
        for heading in ("Args:", "Returns:", "Raises:"):
            assert heading in function.__doc__
    # Public validation stays a narrow orchestration boundary, not a second parser.
    source = ast.parse(Path(H.__file__).read_text())
    exported = {
        node.name
        for node in source.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert exported == {"inspect", "bind", "verify"}


def test_refactored_elf_metadata_preserves_complete_benign_identity() -> None:
    payload = _elf64(
        needed=("libfixture.so",),
        soname="libexample.so",
        rpath="/lib",
        runpath="$ORIGIN:/usr/lib",
    )
    assert H._elf_metadata(payload) == {
        "class": 2,
        "machine": 62,
        "needed": ["libfixture.so"],
        "soname": "libexample.so",
        "rpath": ["/lib"],
        "runpath": ["$ORIGIN", "/usr/lib"],
    }


def test_refactored_record_parser_preserves_exact_counts_and_diagnostics() -> None:
    root = "opt/venv/lib/python3.10/site-packages"
    record = root + "/fixture-1.0.dist-info/RECORD"
    target = root + "/fixture/data.txt"
    payload = b"benign parser fixture\n"
    files = {target: {"sha256": _digest(payload), "size": len(payload)}, record: {}}
    records = [
        (
            record,
            f"fixture/data.txt,{_record_hash(payload)},{len(payload)}\n"
            f"fixture-1.0.dist-info/RECORD,,\n".encode(),
        )
    ]
    contract = {"allowed_missing_python_record_patterns": []}
    findings = []
    assert H._python_records(records, files, contract, findings) == ({target}, 1, 0)
    assert findings == []
    files[target]["size"] += 1
    assert H._python_records(records, files, contract, findings) == (set(), 0, 0)
    assert findings == [{"code": "python_record_hash_mismatch", "path": target}]


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
    contract["corresponding_source_closure"] = {
        "status": "complete-accompanying-source",
        "inventory_sha256": None,
        "records": {},
    }
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
        "usr/share/doc/npa-habitat-sim/fixture-source.txt": b"inert source fixture\n",
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


def _fixture_source_delivery(report: dict[str, object]) -> dict[str, object]:
    return {
        "status": "complete-accompanying-source",
        "inventory_sha256": report["corresponding_source_inventory_sha256"],
        "records": {
            identity: {
                "component": row,
                "artifacts": [
                    {
                        "path": "usr/share/doc/npa-habitat-sim/fixture-source.txt",
                        "bytes": len(b"inert source fixture\n"),
                        "sha256": _digest(b"inert source fixture\n"),
                    }
                ],
            }
            for identity, row in report["corresponding_source_inventory"].items()
        },
    }


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
        if (
            selected_contract["corresponding_source_closure"]["status"]
            == "complete-accompanying-source"
        ):
            selected_contract["corresponding_source_closure"] = (
                _fixture_source_delivery(probe)
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
    contract["corresponding_source_closure"] = _fixture_source_delivery(baseline)
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
    report = Path(analysis_root or fixture["analysis"]) / "cli-report.json"
    command = _cli_arguments(
        wrapper,
        report,
        fixture,
        analysis_root or fixture["analysis"],
        trusted_root,
        archive or fixture["archive"],
    )
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result, json.loads(report.read_text()) if report.exists() else {}


def _codes(report: dict[str, object]) -> set[str]:
    return {row["code"] for row in report["findings"]}


def test_valid_attested_oci_has_complete_graph_and_payload_receipt(tmp_path) -> None:
    report = _verify(tmp_path, [_required_entries()])
    assert report["valid"] is True
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 37
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
    if root_name == "analysis":
        assert report == {}
        assert expected_code in result.stderr
    else:
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
    assert report == {}
    assert "report_directory_permissions" in result.stderr

    wrong_trusted = tmp_path / "wrong-trusted"
    wrong_trusted.mkdir(mode=0o755)
    result, report = _run_cli(tmp_path, fixture, trusted_root=wrong_trusted)
    assert result.returncode == 1
    assert _codes(report) == {"trusted_source_root_mismatch"}


def _report_arguments(tmp_path: Path) -> SimpleNamespace:
    analysis = tmp_path / "analysis"
    analysis.mkdir(mode=0o700)
    return SimpleNamespace(analysis_root=analysis, json=analysis / "report.json")


def test_report_publication_confirms_owner_identity_and_bytes(
    tmp_path, monkeypatch
) -> None:
    """Publish benign JSON only; archive verification is an inert mock."""
    args = _report_arguments(tmp_path)
    report = {"valid": True, "findings": [], "fixture": "inert"}
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    verifier = Mock(return_value=report)
    monkeypatch.setattr(VERIFIER, "_verify_archive", verifier)
    assert VERIFIER.main([]) == 0
    verifier.assert_called_once_with(args)
    assert (
        args.json.read_bytes()
        == (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    info = args.json.stat()
    assert info.st_uid == os.geteuid()
    assert info.st_mode & 0o777 == 0o600
    assert info.st_nlink == 1
    assert list(args.analysis_root.iterdir()) == [args.json]


def test_report_existing_owner_output_refused_before_effects(
    tmp_path, monkeypatch, capsys
) -> None:
    args = _report_arguments(tmp_path)
    args.json.write_bytes(b"existing owner evidence\n")
    before = args.json.stat()
    verifier, publisher = Mock(), Mock()
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    monkeypatch.setattr(VERIFIER, "_verify_archive", verifier)
    monkeypatch.setattr(VERIFIER, "_publish_report", publisher)
    assert VERIFIER.main([]) == 1
    verifier.assert_not_called()
    publisher.assert_not_called()
    assert "report_output_exists" in capsys.readouterr().err
    assert args.json.read_bytes() == b"existing owner evidence\n"
    assert args.json.stat() == before


@pytest.mark.parametrize(
    "code",
    ["report_output_scope", "report_parent_component", "report_directory_permissions"],
)
def test_report_validation_refusal_precedes_scanner_and_publication(
    tmp_path, monkeypatch, capsys, code
) -> None:
    """Mock rejected destination classification, without executing unsafe paths."""
    args = _report_arguments(tmp_path)
    verifier, publisher = Mock(), Mock()
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    monkeypatch.setattr(
        VERIFIER, "_open_report_parent", Mock(side_effect=H.W.ScanError(code))
    )
    monkeypatch.setattr(VERIFIER, "_verify_archive", verifier)
    monkeypatch.setattr(VERIFIER, "_publish_report", publisher)
    assert VERIFIER.main([]) == 1
    verifier.assert_not_called()
    publisher.assert_not_called()
    assert code in capsys.readouterr().err
    assert list(args.analysis_root.iterdir()) == []


@pytest.mark.parametrize("boundary", ["link", "readback", "content"])
def test_report_publication_failure_cleans_only_created_output(
    tmp_path, monkeypatch, boundary
) -> None:
    """Inject a failure, not a race; only benign task-created files are touched."""
    args = _report_arguments(tmp_path)
    marker = args.analysis_root / "owner-marker.txt"
    marker.write_bytes(b"retained owner marker\n")
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    monkeypatch.setattr(VERIFIER, "_verify_archive", lambda _: {"valid": True})
    if boundary == "link":
        monkeypatch.setattr(
            VERIFIER.os, "link", Mock(side_effect=OSError("inert refusal"))
        )
    elif boundary == "readback":
        monkeypatch.setattr(
            VERIFIER,
            "_confirm_report",
            Mock(side_effect=H.W.ScanError("report_bytes_changed")),
        )
    else:
        monkeypatch.setattr(
            VERIFIER.W, "descriptor_bytes", lambda _: b"different inert bytes"
        )
    assert VERIFIER.main([]) == 1
    assert list(args.analysis_root.iterdir()) == [marker]
    assert marker.read_bytes() == b"retained owner marker\n"


@pytest.mark.parametrize("error_type", [H.W.ScanError, RuntimeError, KeyError])
def test_report_scanner_failure_is_fail_closed_and_diagnostic(
    tmp_path, monkeypatch, capsys, error_type
) -> None:
    args = _report_arguments(tmp_path)
    monkeypatch.setattr(VERIFIER, "_arguments", lambda _: args)
    monkeypatch.setattr(
        VERIFIER, "_verify_archive", Mock(side_effect=error_type("inert_failure"))
    )
    assert VERIFIER.main([]) == 1
    report = json.loads(args.json.read_bytes())
    assert report["valid"] is False
    if error_type is H.W.ScanError:
        assert _codes(report) == {"inert_failure"}
        assert "scanner_error_type" not in report
    else:
        assert _codes(report) == {"unreadable_or_incomplete_image_evidence"}
        assert report["scanner_error_type"] == error_type.__name__
        assert error_type.__name__ in capsys.readouterr().err


def test_report_cleanup_attempts_close_after_unlink_failure(
    monkeypatch, capsys
) -> None:
    """Mock all filesystem effects; a cleanup error cannot replace the result."""
    info = SimpleNamespace(st_dev=1, st_ino=2)
    close = Mock()
    monkeypatch.setattr(VERIFIER.os, "open", Mock(return_value=42))
    monkeypatch.setattr(VERIFIER.os, "fstat", Mock(return_value=info))
    monkeypatch.setattr(VERIFIER.os, "stat", Mock(return_value=info))
    monkeypatch.setattr(
        VERIFIER.os, "unlink", Mock(side_effect=OSError("inert cleanup refusal"))
    )
    monkeypatch.setattr(VERIFIER.os, "close", close)
    with pytest.raises(RuntimeError, match="original failure"):
        with VERIFIER._temporary_report(41):
            raise RuntimeError("original failure")
    close.assert_called_once_with(42)
    assert "owned report cleanup failed" in capsys.readouterr().err


def test_report_cleanup_does_not_remove_unmatched_identity(monkeypatch, capsys) -> None:
    """An inert identity mismatch must never authorize unlink."""
    unlink = Mock()
    monkeypatch.setattr(
        VERIFIER.os, "fstat", Mock(return_value=SimpleNamespace(st_dev=1, st_ino=2))
    )
    monkeypatch.setattr(
        VERIFIER.os, "stat", Mock(return_value=SimpleNamespace(st_dev=1, st_ino=3))
    )
    monkeypatch.setattr(VERIFIER.os, "unlink", unlink)
    VERIFIER._cleanup_owned_report(41, "inert.pending", 42)
    unlink.assert_not_called()
    assert "cleanup identity changed" in capsys.readouterr().err


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


@pytest.mark.parametrize("error_type", [H.W.ScanError, RuntimeError, KeyError])
def test_package_list_only_classifies_expected_validation_errors(
    monkeypatch, error_type
) -> None:
    """Mock an in-memory package-list parser; no image or filesystem access."""
    monkeypatch.setattr(
        H.W, "safe_name", Mock(side_effect=error_type("inert validation boundary"))
    )
    findings = []
    args = (
        {"var/lib/dpkg/info/fixture.list": b"/usr/lib/fixture.so\n"},
        {"fixture": {}},
        {},
        {},
        findings,
    )
    if error_type is H.W.ScanError:
        assert H._dpkg_file_owners(*args) == {}
        assert findings == [
            {"code": "runtime_package_file_list_invalid", "package": "fixture"}
        ]
    else:
        with pytest.raises(error_type, match="inert validation boundary"):
            H._dpkg_file_owners(*args)
        assert findings == []


@pytest.mark.parametrize("error_type", [H.W.ScanError, RuntimeError, KeyError])
def test_projection_path_validation_only_classifies_expected_errors(
    monkeypatch, error_type
) -> None:
    """Mock the parser boundary; never extract or touch a filesystem path."""
    inventory = js(
        {
            "schema_version": "npa.habitat-sim.source-projection.v1",
            "files": [{"path": "inert.txt", "bytes": 0, "sha256": "a" * 64}],
        }
    )
    tracked = {
        "manifest.json": js(
            {
                "expected_projection": {
                    "inventory_sha256": _digest(inventory),
                    "file_count": 1,
                }
            }
        ),
        "inventory.json": inventory,
    }
    contract = {
        "source_projection": {
            "manifest_path": "/manifest.json",
            "inventory_path": "/inventory.json",
            "source_root": "/usr/src/habitat-sim",
        }
    }

    def refuse(_path):
        raise error_type("inert validation boundary")

    monkeypatch.setattr(H.W, "safe_name", refuse)
    if error_type is H.W.ScanError:
        findings, count = H._source_projection_findings({}, {}, tracked, contract)
        assert findings == [{"code": "source_projection_unsafe_path"}]
        assert count == 0
    else:
        with pytest.raises(error_type, match="inert validation boundary"):
            H._source_projection_findings({}, {}, tracked, contract)


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
