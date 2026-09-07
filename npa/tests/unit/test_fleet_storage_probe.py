"""Exercise host and shared visibility probes without live filesystem access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.fleet import storage_probe


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    root = tmp_path / "mount"
    root.mkdir()
    mountinfo = tmp_path / "mountinfo"
    device = root.stat().st_dev
    device_text = f"{os.major(device)}:{os.minor(device)}"
    mountinfo.write_text(f"31 20 {device_text} / /mnt/data rw,relatime - virtiofs data rw\n")
    self_mountinfo = tmp_path / "self-mountinfo"
    self_mountinfo.write_text(f"31 20 {device_text} /csi-mounted-fs-path-data/synthetic-volume {root} rw - virtiofs data rw\n")
    fstab = tmp_path / "fstab"
    fstab.write_text("data /mnt/data virtiofs defaults,nofail 0 2\n")
    monkeypatch.setattr(storage_probe.os, "fstatvfs", lambda _: SimpleNamespace(
        f_frsize=4096, f_blocks=1024**3 // 4096,
    ))
    return {
        "action": "host", "root_path": str(root), "run_id": "a" * 32,
        "node_token": "b" * 32, "node_tokens": ["b" * 32, "c" * 32],
        "mount_path": "/mnt/data", "mount_tag": "data", "requested_gibibytes": 1,
        "mountinfo_path": str(mountinfo), "fstab_path": str(fstab),
        "self_mountinfo_path": str(self_mountinfo),
    }


def _action(configuration, action, **changes):
    return storage_probe.execute_probe({**configuration, "action": action, **changes})


def test_host_verifies_mount_capacity_write_read_and_delete(configuration):
    result = _action(configuration, "host")
    assert result["passed"] and result["probe_deleted"] and result["nofail"]
    assert result["capacity_bytes"] == result["requested_bytes"] == 1024**3
    assert len(result["checksum"]) == 64
    assert result["source_matches"] and result["read_write"]
    assert _action(configuration, "cleanup")["run_directory_absent"]


@pytest.mark.parametrize(("mount_record", "category"), [
    ("31 20 0:27 / /other rw - virtiofs data rw\n", "missing_or_ambiguous_mount"),
    ("31 20 0:27 / /mnt/data rw - ext4 data rw\n", "wrong_filesystem_type"),
    ("31 20 0:27 / /mnt/data rw - virtiofs wrong rw\n", "wrong_mount_source"),
    ("31 20 0:27 / /mnt/data ro - virtiofs data rw\n", "mount_read_only"),
    ("31 20 0:27 / /mnt/data rw - virtiofs data ro\n", "mount_read_only"),
    ("malformed\n", "missing_or_ambiguous_mount"),
])
def test_host_rejects_missing_wrong_or_read_only_mount(configuration, mount_record, category):
    Path(configuration["mountinfo_path"]).write_text(mount_record)
    with pytest.raises(storage_probe.StorageProbeError, match=category):
        _action(configuration, "host")


@pytest.mark.parametrize(("record", "category"), [
    ("", "missing_or_ambiguous_persistence"),
    ("data /mnt/data virtiofs defaults 0 2\n", "unsafe_mount_persistence"),
    ("data /mnt/data virtiofs defaults,nofail,ro 0 2\n", "unsafe_mount_persistence"),
    ("other /mnt/data virtiofs defaults,nofail 0 2\n", "persistence_identity_mismatch"),
    ("data /mnt/data ext4 defaults,nofail 0 2\n", "persistence_identity_mismatch"),
])
def test_host_requires_exact_reboot_safe_fstab(configuration, record, category):
    Path(configuration["fstab_path"]).write_text(record)
    with pytest.raises(storage_probe.StorageProbeError, match=category):
        _action(configuration, "host")


@pytest.mark.parametrize("filename", ["mountinfo_path", "fstab_path"])
def test_host_rejects_ambiguous_mount_evidence(configuration, filename):
    path = Path(configuration[filename])
    path.write_text(path.read_text() * 2)
    with pytest.raises(storage_probe.StorageProbeError, match="ambiguous"):
        _action(configuration, "host")


@pytest.mark.parametrize(("fragment", "blocks", "category"), [
    (0, 1, "unknown_capacity"), (4096, 0, "unknown_capacity"),
    (4096, 10**9 // 4096, "capacity_mismatch"),
    (4096, 2 * 1024**3 // 4096, "capacity_mismatch"),
    (4096, 1024**3 // 4096 - 1, "capacity_mismatch"),
])
def test_capacity_rejects_unknown_decimal_and_mismatched_size(
    configuration, monkeypatch, fragment, blocks, category,
):
    monkeypatch.setattr(storage_probe.os, "fstatvfs", lambda _: SimpleNamespace(
        f_frsize=fragment, f_blocks=blocks,
    ))
    with pytest.raises(storage_probe.StorageProbeError, match=category):
        _action(configuration, "host")


@pytest.mark.parametrize("capacity", [0, -1, True, "1", None])
def test_capacity_requires_positive_exact_integer_gibibytes(configuration, capacity):
    with pytest.raises(storage_probe.StorageProbeError, match="invalid_requested_capacity"):
        _action(configuration, "host", requested_gibibytes=capacity)


def test_every_node_reads_every_unique_payload(configuration):
    expected = {}
    for token in configuration["node_tokens"]:
        expected[token] = _action(configuration, "write", node_token=token)["checksum"]
    assert len(set(expected.values())) == 2
    for token in configuration["node_tokens"]:
        result = _action(configuration, "read", node_token=token, expected_checksums=expected)
        assert result["verified_checksums"] == expected and result["read_count"] == 2
    result = _action(configuration, "cleanup")
    assert result["removed_files"] == 2 and result["absent_files"] == 4
    assert _action(configuration, "audit")["base_directory_absent"]


def test_cross_node_wrong_checksum_fails(configuration):
    token = configuration["node_token"]
    _action(configuration, "write")
    with pytest.raises(storage_probe.StorageProbeError, match="cross_node_checksum_mismatch"):
        _action(configuration, "read", expected_checksums={token: "0" * 64}, node_tokens=[token])
    assert _action(configuration, "cleanup")["removed_files"] == 1


def test_partial_cross_node_evidence_fails_and_cleanup_recovers(configuration):
    checksum = _action(configuration, "write")["checksum"]
    expected = {"b" * 32: checksum, "c" * 32: "0" * 64}
    with pytest.raises(FileNotFoundError):
        _action(configuration, "read", expected_checksums=expected)
    assert _action(configuration, "cleanup")["run_directory_absent"]


def test_concurrent_invocations_keep_independent_ownership(configuration):
    first_checksum = _action(configuration, "write")["checksum"]
    second = {**configuration, "run_id": "d" * 32}
    second_checksum = _action(second, "write")["checksum"]
    assert first_checksum != second_checksum
    assert not _action(configuration, "cleanup")["base_directory_absent"]
    assert _action(second, "read", expected_checksums={"b" * 32: second_checksum}, node_tokens=["b" * 32])["passed"]
    assert _action(second, "cleanup")["base_directory_absent"]


@pytest.mark.parametrize("location", ["parent", "run", "payload"])
def test_symlinks_are_refused_without_reading_target(configuration, tmp_path, location):
    destination = tmp_path / "unrelated"
    destination.mkdir()
    protected = destination / "protected"
    protected.write_bytes(b"keep")
    root = Path(configuration["root_path"])
    parent = root / ".npa-storage-probes"
    run = parent / configuration["run_id"]
    if location == "parent":
        parent.symlink_to(destination, target_is_directory=True)
    elif location == "run":
        parent.mkdir()
        run.symlink_to(destination, target_is_directory=True)
    else:
        _action(configuration, "host")
        (run / ("payload-" + configuration["node_token"])).symlink_to(protected)
    with pytest.raises((OSError, storage_probe.StorageProbeError)):
        _action(configuration, "write")
    assert protected.read_bytes() == b"keep"


def test_cleanup_refuses_unowned_extra_paths_and_retains_marker(configuration):
    _action(configuration, "write")
    directory = Path(configuration["root_path"]) / ".npa-storage-probes" / configuration["run_id"]
    extra = directory / "unrecognized"
    extra.write_bytes(b"keep")
    with pytest.raises(OSError):
        _action(configuration, "cleanup")
    assert extra.read_bytes() == b"keep"
    assert (directory / (".owner-" + configuration["run_id"])).exists()
    extra.unlink()
    assert _action(configuration, "cleanup")["run_directory_absent"]


def test_probe_actions_do_not_list_existing_directories(configuration, monkeypatch):
    def forbid_listing(*args, **kwargs):
        raise AssertionError("filesystem enumeration is prohibited")

    monkeypatch.setattr(storage_probe.os, "listdir", forbid_listing)
    monkeypatch.setattr(storage_probe.os, "scandir", forbid_listing)
    _action(configuration, "host")
    checksum = _action(configuration, "write")["checksum"]
    _action(configuration, "read", expected_checksums={"b" * 32: checksum}, node_tokens=["b" * 32])
    assert _action(configuration, "cleanup")["passed"]


def test_host_cleanup_runs_after_checksum_failure(configuration, monkeypatch):
    monkeypatch.setattr(storage_probe, "_digest_file", lambda *_: "0" * 64)
    with pytest.raises(storage_probe.StorageProbeError, match="probe_payload_mismatch"):
        _action(configuration, "host")
    assert _action(configuration, "cleanup")["removed_files"] == 0


def test_unwritable_probe_and_cleanup_failure_are_reported(configuration, monkeypatch):
    def deny_write(*args):
        raise PermissionError("private filesystem details")

    monkeypatch.setattr(storage_probe, "_write_file", deny_write)
    with pytest.raises(PermissionError):
        _action(configuration, "host")
    monkeypatch.setattr(storage_probe, "_unlink_file", deny_write)
    with pytest.raises(PermissionError):
        _action(configuration, "cleanup")


def test_stdout_failure_is_sanitized(configuration, monkeypatch, capsys):
    configuration["root_path"] = "/does-not-exist/private-customer-name"
    monkeypatch.setattr(storage_probe.sys, "argv", ["probe", json.dumps(configuration)])
    assert storage_probe.main() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"passed": False, "category": "probe_operation_failed"}
    assert not captured.err and "private-customer-name" not in captured.out


def test_stdout_success_contains_no_payload(configuration, monkeypatch, capsys):
    configuration["action"] = "write"
    monkeypatch.setattr(storage_probe.os, "urandom", lambda _: b"x" * 32)
    monkeypatch.setattr(storage_probe.sys, "argv", ["probe", json.dumps(configuration)])
    assert storage_probe.main() == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["checksum"] == hashlib.sha256(b"x" * 32).hexdigest()
    assert "x" * 32 not in captured.out and not captured.err


def test_stale_host_bind_mount_is_rejected(configuration):
    Path(configuration["mountinfo_path"]).write_text("31 20 0:999 / /mnt/data rw - virtiofs data rw\n")
    with pytest.raises(storage_probe.StorageProbeError, match="stale_mount_binding"):
        _action(configuration, "host")


def test_read_requires_every_declared_node_checksum(configuration):
    checksum = _action(configuration, "write")["checksum"]
    with pytest.raises(storage_probe.StorageProbeError, match="missing_cross_node_evidence"):
        _action(configuration, "read", expected_checksums={"b" * 32: checksum})


@pytest.mark.parametrize("backing", ["other/leaf", "csi-mounted-fs-path-data/../leaf",
                                     "csi-mounted-fs-path-data/..", "csi-mounted-fs-path-data/leaf/child"])
def test_backing_audit_rejects_unproven_paths(configuration, backing):
    with pytest.raises(storage_probe.StorageProbeError, match="invalid_backing_path"):
        _action(configuration, "audit_backing", backing_relative_path=backing)


def test_backing_audit_checks_exact_owned_leaf_without_listing(configuration, monkeypatch):
    parent = Path(configuration["root_path"]) / "csi-mounted-fs-path-data"
    parent.mkdir()
    leaf = parent / "synthetic-volume"
    leaf.mkdir()
    sibling = parent / "unrelated-volume"
    sibling.mkdir()
    settings = {"backing_relative_path": "csi-mounted-fs-path-data/synthetic-volume"}
    with pytest.raises(storage_probe.StorageProbeError, match="backing_cleanup_failed"):
        _action(configuration, "audit_backing", **settings)
    leaf.rmdir()
    assert _action(configuration, "audit_backing", **settings)["backing_directory_absent"]
    assert sibling.is_dir()


def test_write_requires_actual_pvc_mount_with_expected_source(configuration):
    path = Path(configuration["self_mountinfo_path"])
    path.write_text(path.read_text().replace("virtiofs data", "virtiofs different"))
    with pytest.raises(storage_probe.StorageProbeError, match="wrong_mount_source"):
        _action(configuration, "write")


def test_installed_probe_source_runs_without_npa_imports(configuration):
    import subprocess
    import sys

    settings = {**configuration, "action": "audit", "root_path": "relative"}
    source = Path(storage_probe.__file__).read_text()
    result = subprocess.run([sys.executable, "-c", source, json.dumps(settings)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1 and not result.stderr
    assert json.loads(result.stdout) == {"passed": False, "category": "invalid_probe_root"}


def test_csi_cleanup_recovers_backing_identity_after_failed_writes(configuration, monkeypatch):
    captured = []
    original = storage_probe._owned_directory

    def redirect_owned_directory(settings, create):
        captured.append(settings["root_path"])
        return original({**settings, "root_path": configuration["root_path"]}, create)

    original_audit = storage_probe._audit
    monkeypatch.setattr(storage_probe, "_owned_directory", redirect_owned_directory)
    monkeypatch.setattr(storage_probe, "_backing_path", lambda _: "csi-mounted-fs-path-data/synthetic-volume")
    monkeypatch.setattr(storage_probe, "_audit", lambda settings: original_audit({
        **settings, "root_path": configuration["root_path"],
    }))
    result = _action(configuration, "cleanup", root_path="/data")
    assert result["backing_relative_path"] == "csi-mounted-fs-path-data/synthetic-volume"
    assert result["run_directory_absent"] and captured == ["/data"]


def test_host_rejects_bind_of_unexpected_filesystem_subdirectory(configuration):
    path = Path(configuration["mountinfo_path"])
    path.write_text(path.read_text().replace(" / /mnt/data ", " /unexpected /mnt/data "))
    with pytest.raises(storage_probe.StorageProbeError, match="wrong_host_mount_root"):
        _action(configuration, "host")


def test_absence_revalidates_stale_positive_cache_against_server(configuration, monkeypatch):
    calls = []

    def query(directory, filename, flags, mask, output):
        calls.append((directory, filename, flags, mask))
        details = storage_probe.ctypes.cast(
            output, storage_probe.ctypes.POINTER(storage_probe._StatxResult),
        ).contents
        details.mask = storage_probe._STATX_LINK_COUNT
        details.link_count = 0
        return 0

    monkeypatch.setattr(storage_probe.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(statx=query))
    monkeypatch.setattr(storage_probe.os, "stat", lambda *args, **kwargs: SimpleNamespace(st_nlink=2))
    assert storage_probe._absent(17, "owned-probe")
    assert calls == [(17, b"owned-probe", 0x2100, 0x0004)]
    assert storage_probe.ctypes.sizeof(storage_probe._StatxResult) == 256


def test_synced_presence_still_fails_exact_run_directory_audit(configuration, monkeypatch):
    _action(configuration, "host")
    real_query = storage_probe._synchronized_link_count
    observed = []

    def synchronized_query(directory, filename):
        observed.append(filename)
        return real_query(directory, filename)

    monkeypatch.setattr(storage_probe, "_synchronized_link_count", synchronized_query)
    with pytest.raises(storage_probe.StorageProbeError, match="probe_cleanup_failed"):
        _action(configuration, "audit")
    assert configuration["run_id"] in observed
    assert _action(configuration, "cleanup")["run_directory_absent"]


@pytest.mark.parametrize(("error_number", "expected_exception"), [
    (22, storage_probe.StorageProbeError), (38, storage_probe.StorageProbeError),
    (95, storage_probe.StorageProbeError), (13, PermissionError), (5, OSError),
])
def test_absence_query_errors_cannot_be_accepted_as_absent(monkeypatch, error_number, expected_exception):
    def query(*args):
        storage_probe.ctypes.set_errno(error_number)
        return -1

    monkeypatch.setattr(storage_probe.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(statx=query))
    with pytest.raises(expected_exception):
        storage_probe._absent(17, "owned-probe")


def test_absence_query_requires_returned_link_count_field(monkeypatch):
    def query(*args):
        return 0

    monkeypatch.setattr(storage_probe.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(statx=query))
    with pytest.raises(storage_probe.StorageProbeError, match="synchronized_absence_unverified"):
        storage_probe._absent(17, "owned-probe")


def test_absence_query_fails_closed_when_statx_is_unavailable(monkeypatch):
    monkeypatch.setattr(storage_probe.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(storage_probe.StorageProbeError, match="synchronized_absence_unsupported"):
        storage_probe._absent(17, "owned-probe")


def test_absence_query_confirms_missing_exact_path(configuration):
    descriptor = os.open(configuration["root_path"], os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert storage_probe._synchronized_link_count(descriptor, "never-created-owned-probe") is None
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("action", ["cleanup", "audit", "audit_backing"])
@pytest.mark.parametrize(("failure", "category"), [
    ("unmounted", "missing_or_ambiguous_mount"), ("device", "stale_mount_binding"),
    ("type", "wrong_filesystem_type"), ("persistence", "unsafe_mount_persistence"),
    ("capacity", "unknown_capacity"),
])
def test_cleanup_rechecks_host_state_before_removal_or_absence(
    configuration, monkeypatch, action, failure, category,
):
    _action(configuration, "host")
    mountinfo = Path(configuration["mountinfo_path"])
    if failure == "unmounted":
        mountinfo.write_text("")
    elif failure == "device":
        fields = mountinfo.read_text().split()
        fields[2] = "9999:9999"
        mountinfo.write_text(" ".join(fields))
    elif failure == "type":
        mountinfo.write_text(mountinfo.read_text().replace("virtiofs", "ext4"))
    elif failure == "persistence":
        Path(configuration["fstab_path"]).write_text("data /mnt/data virtiofs defaults 0 2\n")
    else:
        monkeypatch.setattr(storage_probe.os, "fstatvfs", lambda _: SimpleNamespace(f_frsize=4096, f_blocks=0))
    touched = []
    monkeypatch.setattr(storage_probe, "_unlink_file", lambda *args: touched.append("remove"))
    monkeypatch.setattr(storage_probe, "_absent", lambda *args: touched.append("absence"))
    with pytest.raises(storage_probe.StorageProbeError, match=category):
        _action(configuration, action, backing_relative_path="csi-mounted-fs-path-data/synthetic-volume")
    assert touched == []
    marker = Path(configuration["root_path"]) / ".npa-storage-probes" / configuration["run_id"]
    assert (marker / (".owner-" + configuration["run_id"])).exists()


@pytest.mark.parametrize("action", ["cleanup", "audit"])
def test_csi_cleanup_rechecks_exact_backing_mount_without_host_fallback(configuration, monkeypatch, action):
    original_open_root = storage_probe._open_root

    def mapped_open_root(path, stack):
        if path == "/data":
            path = configuration["root_path"]
        return original_open_root(path, stack)

    monkeypatch.setattr(storage_probe, "_open_root", mapped_open_root)
    path = Path(configuration["self_mountinfo_path"])
    path.write_text(path.read_text().replace(configuration["root_path"], "/data"))
    assert _action(configuration, "write", root_path="/data")["written"]
    path.write_text("")
    touched = []
    monkeypatch.setattr(storage_probe, "_unlink_file", lambda *args: touched.append("remove"))
    monkeypatch.setattr(storage_probe, "_absent", lambda *args: touched.append("absence"))
    monkeypatch.setattr(storage_probe, "_host_state", lambda *args: touched.append("host_fallback"))
    with pytest.raises(storage_probe.StorageProbeError, match="missing_or_ambiguous_mount"):
        _action(configuration, action, root_path="/data")
    assert touched == []
