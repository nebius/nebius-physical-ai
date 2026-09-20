"""Exercise neutral bootstrap source delivery and fail-closed layer inspection."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "docker/workbench/habitat-sim"


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))
    spec = importlib.util.spec_from_file_location("habitat_bootstrap_verifier", PACKAGE / "verify_bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def tar_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def saved_bootstrap(tmp_path, *, changed_source=False, absent_source=False, baked=False, whiteout=None):
    payload = b"official source fixture\n"
    sha = hashlib.sha256(payload).hexdigest()
    descriptor = f"Source: sample\nVersion: 1.0\nChecksums-Sha256:\n {sha} {len(payload)} sample_1.0.tar.gz\n".encode()
    artifacts = [
        {"name": "sample_1.0.tar.gz", "bytes": len(payload), "sha256": sha},
        {"name": "sample_1.0.dsc", "bytes": len(descriptor), "sha256": hashlib.sha256(descriptor).hexdigest()},
    ]
    doc = "usr/share/doc/npa-habitat-sim/"
    files = {
        "var/lib/dpkg/status": b"Package: sample\nStatus: install ok installed\nVersion: 1.0\n",
        doc + "bootstrap-sources.json": json.dumps([{"source": "sample", "version": "1.0", "artifacts": artifacts}]).encode(),
        doc + "ubuntu-sources/sample/1.0/sample_1.0.dsc": descriptor,
    }
    if not absent_source:
        files[doc + "ubuntu-sources/sample/1.0/sample_1.0.tar.gz"] = b"changed" if changed_source else payload
    if baked:
        files["opt/venv/lib/python3.10/site-packages/habitat_sim/_ext.so"] = b"forbidden payload"
    image = {
        "layer.tar": tar_bytes(files),
        "config.json": json.dumps({"architecture": "amd64", "rootfs": {"type": "layers"}}).encode(),
        "manifest.json": json.dumps([{"Config": "config.json", "Layers": ["layer.tar"]}]).encode(),
    }
    if whiteout:
        image["deletion.tar"] = tar_bytes({whiteout: b""})
        image["manifest.json"] = json.dumps([
            {"Config": "config.json", "Layers": ["layer.tar", "deletion.tar"]}
        ]).encode()
    path = tmp_path / "image.tar"
    path.write_bytes(tar_bytes(image))
    return path


def test_exact_source_and_absence_evidence(verifier, tmp_path):
    report = verifier.inspect(saved_bootstrap(tmp_path))
    assert report["valid"] and report["source_components"] == 1
    assert report["source_artifacts"] == 2


@pytest.mark.parametrize("mutation", ["changed_source", "absent_source"])
def test_source_must_reach_recipient_unchanged(verifier, tmp_path, mutation):
    with pytest.raises(ValueError, match="source artifact identity mismatch"):
        verifier.inspect(saved_bootstrap(tmp_path, **{mutation: True}))


@pytest.mark.parametrize("whiteout", [
    ".wh.usr",
    "usr/.wh.share",
    "usr/share/.wh.doc",
    "usr/share/doc/.wh.npa-habitat-sim",
    ".wh..wh..opq",
    "usr/share/doc/.wh..wh..opq",
    "usr/share/doc/npa-habitat-sim/.wh..wh..opq",
    "usr/share/doc/npa-habitat-sim/.wh.ubuntu-sources",
    "usr/share/doc/npa-habitat-sim/ubuntu-sources/sample/1.0/.wh.sample_1.0.tar.gz",
])
def test_source_deleted_by_later_layer_is_refused(verifier, tmp_path, whiteout):
    with pytest.raises(ValueError, match="whiteout affects delivered bootstrap source"):
        verifier.inspect(saved_bootstrap(tmp_path, whiteout=whiteout))


def test_unrelated_layer_whiteout_keeps_source_delivery_valid(verifier, tmp_path):
    assert verifier.inspect(saved_bootstrap(tmp_path, whiteout="var/cache/.wh.apt"))["valid"]


def test_legacy_simulator_payload_cannot_enter_neutral_image(verifier, tmp_path):
    report = verifier.inspect(saved_bootstrap(tmp_path, baked=True))
    assert not report["valid"]
    assert report["findings"][0]["code"] == "baked_runtime_payload"


def test_original_and_superseded_source_versions_are_both_required(verifier):
    status = "Package: sample\nStatus: install ok installed\nVersion: 2\nSource: original (1:1.2)\n"
    assert verifier.package_identities(status) == {("original", "1:1.2")}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["-m", "npa.workflows.habitat_sim_smoke", "--help"], "system"),
        (["-c", "print(1)"], "system"),
        (["-m", "npa.workflows.habitat_sim_smoke", "--run-id", "fixture"], "runtime"),
    ],
)
def test_shim_keeps_help_and_other_python_commands_offline(tmp_path, arguments, expected):
    source = (PACKAGE / "python-shim.sh").read_text()
    for command, label in [("/usr/bin/python3", "system"), ("/usr/local/libexec/npa-habitat-runtime", "runtime")]:
        target = tmp_path / label
        target.write_text(f"#!/bin/sh\nprintf '%s\\n' {label}\n")
        target.chmod(0o700)
        source = source.replace(command, str(target))
    shim = tmp_path / "shim.sh"
    shim.write_text(source)
    result = subprocess.run(["bash", str(shim), *arguments], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected
