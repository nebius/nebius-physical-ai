"""Exercise OpenPI image ancestry using synthetic bytes and independent archives."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/openpi"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load("openpi_image_verifier", IMAGE / "verify_image.py")
# These builders construct archives independently of either verifier and accept
# explicit bytes. No cuRobo production policy or vendor fixture is reused.
ARCHIVES = _load("openpi_archive_fixture_builders", Path(__file__).with_name("test_curobo_image_verifier.py"))


@pytest.fixture
def payload():
    contract = copy.deepcopy(json.loads((IMAGE / "runtime-payload.json").read_text()))
    entries, excluded = [], []
    for index, row in enumerate(contract["required"]):
        data = f"Synthetic {row['kind']} fixture {index}".encode()
        row.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        entries.append(ARCHIVES.entry(row["path"], data))
    for index, row in enumerate(contract["excluded_sdk"]):
        data = f"Synthetic prohibited header {index}".encode()
        row.update(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        excluded.append(data)
    return contract, entries, excluded


def _verify(tmp_path, payload, layers=None, **kwargs):
    archive, image_id = ARCHIVES.save_image(tmp_path, layers or [payload[1]], **kwargs)
    return VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def _codes(report):
    return {row["code"] for row in report["findings"]}


@pytest.mark.parametrize("compressed", [False, True])
def test_complete_layer_inventory_and_all_bytes(tmp_path, payload, compressed):
    neutral = b"large neutral byte population\n" * 400_000
    report = _verify(tmp_path, payload, [[ARCHIVES.entry("opt/neutral", neutral), *payload[1]]], compressed=compressed)
    assert report["valid"]
    assert report["retained_runtime_count"] == 8
    assert report["verified_torch_adapter_count"] == 53
    assert report["required_payload_count"] == 124
    assert report["regular_files_read"] == 125
    assert report["content_bytes_read"] == len(neutral) + sum(len(row[1]) for row in payload[1])


@pytest.mark.parametrize("docker_media,compressed", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("use_config_id", [False, True])
def test_oci_and_classic_identity_bind_repeated_blobs(tmp_path, payload, docker_media, compressed, use_config_id):
    archive, manifest_id, config_id = ARCHIVES.save_oci_image(
        tmp_path, [payload[1], payload[1]], docker_media=docker_media, compressed=compressed,
    )
    report = VERIFIER.verify_image(archive, expected_image_id=config_id if use_config_id else manifest_id, contract=payload[0])
    assert report["valid"]
    assert report["layer_count"] == 2
    assert report["regular_files_read"] == 248
    assert report["image_config_digest"] == config_id
    assert report["image_manifest_digest"] == manifest_id


@pytest.mark.parametrize("renamed", [False, True])
def test_sdk_bytes_stay_rejected_after_whiteout(tmp_path, payload, renamed):
    path = "opt/renamed-private-header" if renamed else "usr/include/cudnn.h"
    whiteout = str(Path(path).parent / (".wh." + Path(path).name))
    report = _verify(tmp_path, payload, [[ARCHIVES.entry(path, payload[2][0])], [ARCHIVES.entry(whiteout), *payload[1]]])
    assert not report["valid"]
    assert "excluded_cudnn_sdk_bytes" in _codes(report)


@pytest.mark.parametrize("kind", ["cudnn_runtime", "cudnn_license", "nvshmem_notice", "nccl_license", "torch_adapter", "torch_license"])
def test_old_changed_payload_cannot_be_hidden_by_correct_replacement(tmp_path, payload, kind):
    row = next(row for row in payload[0]["required"] if row["kind"] == kind)
    report = _verify(tmp_path, payload, [[ARCHIVES.entry(row["path"], b"changed ancestor bytes")], payload[1]])
    assert not report["valid"]
    assert "retained_payload_hash_mismatch" in _codes(report)


@pytest.mark.parametrize("kind", ["cudnn_license", "nvshmem_notice", "nccl_license", "torch_license"])
def test_notice_whiteout_invalidates_final_retention(tmp_path, payload, kind):
    path = next(row["path"] for row in payload[0]["required"] if row["kind"] == kind)
    whiteout = str(Path(path).parent / (".wh." + Path(path).name))
    report = _verify(tmp_path, payload, [payload[1], [ARCHIVES.entry(whiteout)]])
    assert not report["valid"]
    assert "required_payload_missing" in _codes(report)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_required_payload_replacements_must_be_regular(tmp_path, payload, kind):
    report = _verify(tmp_path, payload, [payload[1], [ARCHIVES.entry(payload[1][0][0], kind=kind, link="other")]])
    assert not report["valid"]
    assert "retained_payload_not_regular" in _codes(report)
    assert "required_payload_missing" in _codes(report)


def test_parent_symlink_cannot_preserve_required_byte_proof(tmp_path, payload):
    report = _verify(tmp_path, payload, [payload[1], [ARCHIVES.entry("opt/venv", kind=tarfile.SYMTYPE, link="other")]])
    assert "required_payload_ancestor_not_directory" in _codes(report)
    assert "required_payload_missing" in _codes(report)


@pytest.mark.parametrize("marker_first", [False, True])
@pytest.mark.parametrize("opaque", [False, True])
def test_whiteouts_preserve_same_layer_additions_in_either_order(tmp_path, payload, marker_first, opaque):
    path = Path(payload[1][0][0])
    marker = ARCHIVES.entry(str(path.parent / (".wh..wh..opq" if opaque else ".wh." + path.name)))
    current = [marker, *payload[1]] if marker_first else [*payload[1], marker]
    report = _verify(tmp_path, payload, [payload[1], current])
    assert report["valid"]
    assert report["retained_runtime_count"] == 8


def test_opaque_whiteout_removes_lower_layer_required_payloads(tmp_path, payload):
    report = _verify(tmp_path, payload, [payload[1], [ARCHIVES.entry("opt/.wh..wh..opq")]])
    assert not report["valid"]
    assert "required_payload_missing" in _codes(report)
    assert report["retained_runtime_count"] == 0


def test_same_layer_ancestor_replacement_invalidates_current_proof(tmp_path, payload):
    report = _verify(tmp_path, payload, [[*payload[1], ARCHIVES.entry("opt/venv", kind=tarfile.SYMTYPE, link="other")]])
    assert not report["valid"]
    assert "required_payload_ancestor_not_directory" in _codes(report)
    assert "required_payload_missing" in _codes(report)


@pytest.mark.parametrize("extra,code", [
    (ARCHIVES.entry("opt/venv/lib/python3.11/site-packages/nvidia/cudnn/unknown", b"new"), "unreviewed_cudnn_namespace_payload"),
    (ARCHIVES.entry("tmp/nvidia_cudnn_unreviewed.whl", b"cache"), "cached_cudnn_wheel"),
    (ARCHIVES.entry("usr/lib/libcudnn.so.9", b"ELF"), "unexpected_cudnn_runtime_location"),
    (ARCHIVES.entry("opt/.wh..", b""), "malformed_whiteout"),
])
def test_unknown_payload_and_malformed_whiteout_fail_closed(tmp_path, payload, extra, code):
    report = _verify(tmp_path, payload, [[*payload[1], extra]])
    assert not report["valid"]
    assert code in _codes(report)


def test_duplicate_layer_paths_fail_even_with_identical_bytes(tmp_path, payload):
    report = _verify(tmp_path, payload, [[*payload[1], payload[1][0]]])
    assert not report["valid"]
    assert "duplicate_layer_path" in _codes(report)
    assert report["regular_files_read"] == 125


def test_renamed_full_cudnn_wheel_bytes_are_rejected(tmp_path, payload):
    body = b"Synthetic full cuDNN wheel fixture with a neutral filename"
    payload[0]["source_artifacts"]["cudnn"]["sha256"] = hashlib.sha256(body).hexdigest()
    report = _verify(tmp_path, payload, [[*payload[1], ARCHIVES.entry("opt/cache/object", body)]])
    assert not report["valid"]
    assert "cached_cudnn_wheel_bytes" in _codes(report)


@pytest.mark.parametrize("mutation", [
    lambda manifest: manifest["config"].update(size=1),
    lambda manifest: manifest["layers"][0].update(digest="sha256:" + "0" * 64),
    lambda manifest: manifest["layers"][0].update(mediaType="unsupported"),
])
def test_oci_descriptor_corruption_is_never_a_pass(tmp_path, payload, mutation):
    archive, image_id, _ = ARCHIVES.save_oci_image(tmp_path, [payload[1]], update_manifest=mutation)
    with pytest.raises((KeyError, ValueError)):
        VERIFIER.verify_image(archive, expected_image_id=image_id, contract=payload[0])


def test_cli_incomplete_evidence_emits_sanitized_failure(tmp_path):
    archive, report = tmp_path / "broken.tar", tmp_path / "report.json"
    archive.write_bytes(b"synthetic private-looking untrusted payload must not echo")
    result = subprocess.run(
        [sys.executable, str(IMAGE / "verify_image.py"), "--docker-save", str(archive),
         "--json", str(report), "--expected-image-id", "sha256:" + "0" * 64],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert json.loads(report.read_text())["valid"] is False
    assert "private-looking" not in report.read_text() + result.stdout + result.stderr
