"""Architecture gates must reject malformed binary evidence without a GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import zipfile

import pytest


@pytest.fixture
def measure():
    path = Path(__file__).resolve().parents[2] / "scripts/measure_extension_arches.py"
    spec = importlib.util.spec_from_file_location("measure_extension_arches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(*, kind=2, arch=100, header_size=64, payload_size=8) -> bytes:
    header = bytearray(64)
    struct.pack_into("<HHIQ", header, 0, kind, 0x101, header_size, payload_size)
    struct.pack_into("<I", header, 28, arch)
    # Deliberately no ELF magic: compressed entries are still measurable.
    return bytes(header) + b"payload!"


def _container(entries: bytes, *, version=1) -> bytes:
    return struct.pack("<IHHQ", 0xBA55ED50, version, 16, len(entries)) + entries


def test_counts_sass_and_ptx_in_multiple_embedded_containers(measure) -> None:
    blob = b"ELF-prefix" + _container(_entry() + _entry(kind=1, arch=90))
    blob += b"padding" + _container(_entry(arch=120) + _entry())
    sass, ptx = measure.scan(blob)
    assert sass == {100: 2, 120: 1}
    assert ptx == {90: 1}


@pytest.mark.parametrize("length", range(4, 16))
def test_truncated_container_header_does_not_crash(measure, length) -> None:
    assert measure.scan(_container(_entry())[:length]) == ({}, {})


@pytest.mark.parametrize("entry", [
    _entry(kind=99), _entry(header_size=0), _entry(header_size=16),
    _entry(header_size=4097), _entry(payload_size=0),
    _entry(payload_size=9), _entry(arch=0), _entry()[:31],
])
def test_malformed_entries_never_prove_architecture_coverage(measure, entry) -> None:
    assert measure.scan(_container(entry)) == ({}, {})
    # A valid prefix cannot hide an invalid trailing entry in the same container.
    assert measure.scan(_container(_entry() + entry)) == ({}, {})


def test_invalid_container_version_and_truncated_payload_are_ignored(measure) -> None:
    assert measure.scan(_container(_entry(), version=2)) == ({}, {})
    assert measure.scan(_container(_entry())[:-1]) == ({}, {})


def test_later_valid_container_is_measured_after_invalid_magic(measure) -> None:
    blob = measure.FATBIN_MAGIC + b"garbage garbage" + _container(_entry())
    assert measure.scan(blob) == ({100: 1}, {})


def test_require_fails_for_malformed_binary(measure, tmp_path, capsys) -> None:
    target = tmp_path / "broken.so"
    target.write_bytes(_container(_entry(payload_size=9)))
    assert measure.main([str(target), "--require", "sm_100", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)[str(target)]["missing"] == ["sm_100"]


def test_wheel_and_directory_measurement_and_exact_sass_requirement(measure, tmp_path, capsys) -> None:
    binary = tmp_path / "extension.so"
    binary.write_bytes(_container(_entry() + _entry(kind=1, arch=120)))
    wheel = tmp_path / "extension.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(binary, "package/extension.so")
    assert measure.main([str(wheel), "--min-size-mb", "0", "--require", "100", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["extension.whl:package/extension.so"]["sass"] == ["sm_100"]
    # PTX and forward-compatible SASS do not satisfy an exact --require gate.
    assert measure.main([str(tmp_path), "--min-size-mb", "0", "--require", "120"]) == 1
    assert measure.main([str(binary), "--require", "103"]) == 1


def test_empty_and_missing_targets_fail(measure, tmp_path) -> None:
    assert measure.main([str(tmp_path)]) == 2
    assert measure.main([str(tmp_path / "missing.so")]) == 2
