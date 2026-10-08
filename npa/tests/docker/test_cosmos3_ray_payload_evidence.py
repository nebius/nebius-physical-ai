"""Prove Ray payload reports identify blocked bytes without exposing contents."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCANNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "scan_image_cosmos3_ray_serve_payload.py"
)
SPEC = importlib.util.spec_from_file_location("ray_payload_evidence", SCANNER_PATH)
assert SPEC is not None and SPEC.loader is not None
SCANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCANNER
SPEC.loader.exec_module(SCANNER)


def _tar_entries(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _image(tmp_path, layers, config=None):
    layer_bytes = [_tar_entries(entries) for entries in layers]
    config_bytes = json.dumps(config or {"config": {}, "history": []}).encode()
    names = [f"layer-{index}.tar" for index in range(len(layers))]
    manifest = [{"Config": "config.json", "RepoTags": [], "Layers": names}]
    entries = [
        ("manifest.json", json.dumps(manifest).encode()),
        ("config.json", config_bytes),
    ]
    entries.extend(zip(names, layer_bytes))
    path = tmp_path / "image.tar"
    path.write_bytes(_tar_entries(entries))
    return path, layer_bytes, config_bytes


@pytest.mark.parametrize("name", ["source.py", "payload.bin"])
@pytest.mark.parametrize(
    "payload,kind",
    [
        (b"hf_token='inert-fixture-value'\n", "credential_assignment"),
        (b"AKIA" + b"Z" * 16, "aws_access_key_id"),
        (b"-----BEGIN PRIVATE KEY-----\n" + b"QUJD", "private_key_content"),
    ],
)
def test_reports_exact_member_and_layer_bytes(tmp_path, name, payload, kind):
    path, layers, config = _image(tmp_path, [[(name, payload)]])
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "restricted-payload-detected"
    assert report["report_scope"] == "full"
    assert report["credential_hits"] == [f"{kind}:{name}"]
    assert report["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["config_sha256"] == hashlib.sha256(config).hexdigest()
    assert report["credential_members"] == [
        {
            "kind": kind,
            "path": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "layer_index": 0,
            "layer_sha256": hashlib.sha256(layers[0]).hexdigest(),
            "member_index": 0,
            "detection_reason": {"source": "member-content", "rule": kind},
        }
    ]
    assert payload.decode() not in json.dumps(report)


def test_hashes_tail_after_first_chunk_credential_finding(tmp_path):
    payload = b"AKIA" + b"Z" * 16 + b"x" * (2 * 1024 * 1024) + b"distinct-end"
    path, _, _ = _image(tmp_path, [[("payload.bin", payload)]])
    finding = SCANNER.scan_tarball(path)["credential_members"][0]
    assert finding["sha256"] == hashlib.sha256(payload).hexdigest()
    assert finding["size"] == len(payload)


def test_boundary_credential_keeps_complete_member_hash(tmp_path):
    payload = b"x" * (1024 * 1024 - 10) + b"AKIA" + b"Z" * 16 + b"suffix"
    path, _, _ = _image(tmp_path, [[("boundary.bin", payload)]])
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "restricted-payload-detected"
    assert (
        report["credential_members"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    )


def test_hashes_path_credential_even_without_content_marker(tmp_path):
    payload = b"inert-path-only-credential-fixture"
    path, _, _ = _image(tmp_path, [[("etc/ssh/ssh_host_rsa_key", payload)]])
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "restricted-payload-detected"
    member = report["credential_members"][0]
    assert member["kind"] == "ssh_host_key"
    assert member["sha256"] == hashlib.sha256(payload).hexdigest()
    assert member["detection_reason"] == {"source": "path", "rule": "ssh_host_key"}


@pytest.mark.parametrize("same_layer", [False, True])
def test_same_path_members_retain_each_physical_byte_identity(tmp_path, same_layer):
    first = ("source.py", b"hf_token='inert-first-value'\n")
    second = ("source.py", b"hf_token='inert-second-value'\n")
    layers = [[first, second]] if same_layer else [[first], [second]]
    path, raw_layers, _ = _image(tmp_path, layers)
    report = SCANNER.scan_tarball(path)
    assert len(report["credential_hits"]) == 1
    assert len(report["credential_members"]) == 2
    for index, (_, payload) in enumerate((first, second)):
        member = report["credential_members"][index]
        layer_index = 0 if same_layer else index
        assert member["sha256"] == hashlib.sha256(payload).hexdigest()
        assert member["layer_index"] == layer_index
        assert member["member_index"] == (index if same_layer else 0)
        assert (
            member["layer_sha256"]
            == hashlib.sha256(raw_layers[layer_index]).hexdigest()
        )


def test_deleted_ancestor_credential_remains_blocking_and_identifiable(tmp_path):
    payload = b"hf_token='inert-deleted-value'\n"
    path, layers, _ = _image(
        tmp_path, [[("source.py", payload)], [(".wh.source.py", b"")]]
    )
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "restricted-payload-detected"
    assert report["credential_members"][0]["layer_index"] == 0
    assert (
        report["credential_members"][0]["layer_sha256"]
        == hashlib.sha256(layers[0]).hexdigest()
    )
    assert (
        report["credential_members"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    )


def test_clean_report_has_no_credential_members(tmp_path):
    path, _, _ = _image(tmp_path, [[("source.py", b"value = 42\n")]])
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "clean"
    assert report["credential_hits"] == report["credential_members"] == []


def test_history_finding_remains_blocking_without_member_evidence(tmp_path):
    config = {"config": {"Env": ["HF_TOKEN=inert-fixture-value"]}, "history": []}
    path, _, _ = _image(tmp_path, [[("source.py", b"value = 42\n")]], config)
    report = SCANNER.scan_tarball(path)
    assert report["verdict"] == "restricted-payload-detected"
    assert report["history_hits"] and report["credential_members"] == []
    assert "inert-fixture-value" not in json.dumps(report)


def test_hash_stream_consumes_short_reads_through_eof():
    class ShortReads(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 7))

    payload = b"short-read-fixture" * 23
    assert (
        SCANNER._stream_sha256(ShortReads(payload))
        == hashlib.sha256(payload).hexdigest()
    )


def test_hash_failure_cannot_return_clean_report(tmp_path, monkeypatch):
    path, _, _ = _image(tmp_path, [[("source.py", b"value = 42\n")]])

    def unreadable(_):
        raise OSError("synthetic unreadable stream")

    monkeypatch.setattr(SCANNER, "_stream_sha256", unreadable)
    with pytest.raises(OSError, match="synthetic unreadable"):
        SCANNER.scan_tarball(path)


def test_missing_docker_save_manifest_has_a_consistent_error(tmp_path):
    path = tmp_path / "missing-manifest.tar"
    path.write_bytes(_tar_entries([]))

    with pytest.raises(RuntimeError, match="missing docker-save manifest"):
        SCANNER.scan_tarball(path)


def test_missing_docker_save_config_has_a_consistent_error(tmp_path):
    manifest = [{"Config": "missing-config.json", "RepoTags": [], "Layers": []}]
    path = tmp_path / "missing-config.tar"
    path.write_bytes(_tar_entries([("manifest.json", json.dumps(manifest).encode())]))

    with pytest.raises(RuntimeError, match="missing image config"):
        SCANNER.scan_tarball(path)


def test_missing_docker_save_layer_has_a_consistent_error(tmp_path):
    manifest = [{"Config": "config.json", "RepoTags": [], "Layers": ["missing.tar"]}]
    config = json.dumps({"config": {}, "history": []}).encode()
    path = tmp_path / "missing-layer.tar"
    path.write_bytes(
        _tar_entries(
            [("manifest.json", json.dumps(manifest).encode()), ("config.json", config)]
        )
    )

    with pytest.raises(RuntimeError, match="missing layer missing.tar"):
        SCANNER.scan_tarball(path)


@pytest.mark.parametrize(
    "manifest, config, error",
    [
        (b"not-json", None, "malformed docker-save manifest"),
        (
            json.dumps([{"Config": "config.json"}]).encode(),
            json.dumps({"config": {}, "history": []}).encode(),
            "malformed docker-save manifest",
        ),
        (
            json.dumps([{"Config": "config.json", "Layers": []}]).encode(),
            b"not-json",
            "malformed image config",
        ),
    ],
)
def test_malformed_docker_save_metadata_has_a_consistent_error(
    tmp_path, manifest, config, error
):
    entries = [("manifest.json", manifest)]
    if config is not None:
        entries.append(("config.json", config))
    path = tmp_path / "malformed-metadata.tar"
    path.write_bytes(_tar_entries(entries))

    with pytest.raises(RuntimeError, match=error):
        SCANNER.scan_tarball(path)


def test_native_cli_keeps_blocking_exit_and_outputs_metadata_only(tmp_path):
    payload = b"hf_token='inert-native-cli-value'\n"
    path, _, _ = _image(tmp_path, [[("source.py", payload)]])
    output = tmp_path / "report.json"
    output.write_text("previous report", encoding="utf-8")
    output.chmod(0o644)
    result = subprocess.run(
        [
            sys.executable,
            str(SCANNER_PATH),
            "--tarball",
            str(path),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert (
        report["credential_members"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    )
    stdout_report = json.loads(result.stdout)
    assert all(
        report[key] == stdout_report[key]
        for key in stdout_report
        if key != "report_scope"
    )
    assert report["report_scope"] == "full"
    assert stdout_report["report_scope"] == "redacted-summary"
    assert stdout_report["credential_hits"] == report["credential_hits"]
    assert "credential_members" not in stdout_report
    assert "archive_sha256" not in stdout_report
    assert "config_sha256" not in stdout_report
    assert hashlib.sha256(payload).hexdigest() not in result.stdout
    assert "inert-native-cli-value" not in result.stdout
    assert result.stderr == ""

    full_result = subprocess.run(
        [sys.executable, str(SCANNER_PATH), "--tarball", str(path)],
        capture_output=True,
        text=True,
    )
    assert full_result.returncode == 1
    assert (
        json.loads(full_result.stdout)["credential_members"][0]["sha256"]
        == hashlib.sha256(payload).hexdigest()
    )


def test_private_json_report_does_not_follow_a_symlink(tmp_path):
    path, _, _ = _image(tmp_path, [[("source.py", b"value = 42\n")]])
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "report.json"
    output.symlink_to(target)

    result = subprocess.run(
        [
            sys.executable,
            str(SCANNER_PATH),
            "--tarball",
            str(path),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == "unchanged"
