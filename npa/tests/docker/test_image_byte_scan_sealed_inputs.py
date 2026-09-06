"""Verified execution bytes stay immutable even before a source audit refuses drift."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import sys

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / "npa/scripts"))
from image_byte_scan import core as W  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(os, "memfd_create"), reason="Linux execution boundary")


def private_file(path, body, *, executable=False):
    path.write_bytes(body)
    path.chmod(0o700 if executable else 0o600)
    return {"path": str(path), "sha256": W.sha(body)}


def assert_closed(fd):
    with pytest.raises(OSError) as caught:
        os.fstat(fd)
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("executable", [False, True])
def test_sealed_copy_preserves_all_bytes_position_and_immutable_content(tmp_path, executable):
    body = bytes(range(256)) * 17
    spec = private_file(tmp_path / "source", body)
    with W.authorized_roots(tmp_path, CHECKOUT), W.bound_open(spec) as (_, source, _):
        with W.sealed_execution_input(source, spec["sha256"], executable=executable) as sealed:
            assert os.lseek(sealed, 0, os.SEEK_CUR) == 0
            assert os.read(sealed, len(body) + 1) == body
            assert W.fcntl.fcntl(sealed, 1034) == 15
            assert os.fstat(sealed).st_mode & 0o777 == (0o500 if executable else 0o400)
            for mutation in (lambda: os.pwrite(sealed, b"changed", 0),
                             lambda: os.ftruncate(sealed, 0),
                             lambda: os.ftruncate(sealed, len(body) + 1)):
                with pytest.raises(OSError) as caught:
                    mutation()
                assert caught.value.errno == errno.EPERM
            assert W.descriptor_digest(sealed) == spec["sha256"]
        assert_closed(sealed)


def test_sealed_copy_handles_short_successful_writes_without_truncation(tmp_path, monkeypatch):
    spec = private_file(tmp_path / "source", bytes(range(256)) * 3)
    real_write = W.os.write
    calls = []

    def short_write(fd, value):
        calls.append(len(value))
        return real_write(fd, value[:7])

    monkeypatch.setattr(W.os, "write", short_write)
    with W.authorized_roots(tmp_path, CHECKOUT), W.bound_open(spec) as (_, source, _):
        with W.sealed_execution_input(source, spec["sha256"]) as sealed:
            assert W.descriptor_digest(sealed) == spec["sha256"]
    assert len(calls) > 1
    assert_closed(sealed)


@pytest.mark.parametrize("failure", ["write", "chmod", "add_seals", "get_seals", "digest"])
def test_failed_copy_or_sealing_closes_its_descriptor(tmp_path, monkeypatch, failure):
    spec = private_file(tmp_path / "source", b"synthetic configuration")
    real_memfd, real_fcntl = W.os.memfd_create, W.fcntl.fcntl
    created = []

    def remember(*args):
        fd = real_memfd(*args)
        created.append(fd)
        return fd

    def fail(*args):
        raise OSError("synthetic operation failure")

    def seal_fault(fd, operation, *args):
        if failure == "add_seals" and operation == 1033:
            return fail()
        if failure == "get_seals" and operation == 1034:
            return 0
        return real_fcntl(fd, operation, *args)

    monkeypatch.setattr(W.os, "memfd_create", remember)
    monkeypatch.setattr(W.fcntl, "fcntl", seal_fault)
    if failure == "write":
        monkeypatch.setattr(W.os, "write", lambda *args: 0)
    if failure == "chmod":
        monkeypatch.setattr(W.os, "fchmod", fail)
    expected = W.sha(b"other bytes") if failure == "digest" else spec["sha256"]
    with W.authorized_roots(tmp_path, CHECKOUT), W.bound_open(spec) as (_, source, _):
        with pytest.raises((OSError, W.ScanError)):
            with W.sealed_execution_input(source, expected):
                pytest.fail("failed input yielded for execution")
    assert len(created) == 1
    assert_closed(created[0])


def helper_fixture(tmp_path):
    script = (
        f"#!{sys.executable}\n"
        "import errno,json,os,sys\n"
        "fd=int(sys.argv[2])\n"
        "with os.fdopen(os.dup(fd),'rb') as stream: config=stream.read().decode()\n"
        "blocked=False\n"
        "try: os.pwrite(fd,b'changed',0)\n"
        "except OSError as error: blocked=error.errno==errno.EPERM\n"
        "print(json.dumps({'helper':'original','config':config,'write_blocked':blocked}),flush=True)\n"
    ).encode()
    return {
        "helper": private_file(tmp_path / "helper", script, executable=True),
        "config": private_file(tmp_path / "config", b"original configuration"),
    }, script


@pytest.mark.parametrize("changed", ["helper", "config", "both"])
def test_actual_exec_never_uses_mutated_source_before_audit_refusal(tmp_path, monkeypatch, changed):
    authorization, original_helper = helper_fixture(tmp_path)
    real_popen = W.subprocess.Popen
    observations, processes, passed = [], [], []

    def mutate_then_spawn(argv, **kwargs):
        passed.extend(kwargs["pass_fds"])
        if changed in ("helper", "both"):
            Path(authorization["helper"]["path"]).write_bytes(original_helper.replace(b"'original'", b"'tampered'"))
        if changed in ("config", "both"):
            Path(authorization["config"]["path"]).write_bytes(b"tampered configuration")
        try:
            process = real_popen(argv, **kwargs)
            processes.append(process)
            observations.append(json.loads(process.stdout.readline()))
            process.wait()
            return process
        finally:
            if changed in ("helper", "both"):
                Path(authorization["helper"]["path"]).write_bytes(original_helper)
            if changed in ("config", "both"):
                Path(authorization["config"]["path"]).write_bytes(b"original configuration")

    monkeypatch.setattr(W.subprocess, "Popen", mutate_then_spawn)
    with W.authorized_roots(tmp_path, CHECKOUT):
        with pytest.raises(W.ScanError, match="input_changed_during_read"):
            W.Detector(authorization, tmp_path / "stderr")
    assert observations == [{"helper": "original", "config": "original configuration", "write_blocked": True}]
    assert len(processes) == 1 and processes[0].returncode == 0
    assert processes[0].stdin.closed and processes[0].stdout.closed
    for fd in passed:
        assert_closed(fd)
    assert W.sha(Path(authorization["helper"]["path"]).read_bytes()) == authorization["helper"]["sha256"]


def test_successful_actual_child_retains_sealed_inputs_after_parent_copy_closes(tmp_path, monkeypatch):
    authorization, _ = helper_fixture(tmp_path)
    passed, observed = [], []
    real_popen = W.subprocess.Popen

    def spawn(argv, **kwargs):
        passed.extend(kwargs["pass_fds"])
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(W.subprocess, "Popen", spawn)
    monkeypatch.setattr(W.Detector, "_validate_ready", lambda self, _: observed.append(json.loads(self.process.stdout.readline())))
    with W.authorized_roots(tmp_path, CHECKOUT):
        detector = W.Detector(authorization, tmp_path / "stderr")
        try:
            assert observed == [{"helper": "original", "config": "original configuration", "write_blocked": True}]
            for fd in passed:
                assert_closed(fd)
            assert detector.process.wait() == 0
        finally:
            detector.abort()
    assert detector.joined


def test_spawn_failure_closes_both_sealed_inputs_and_stderr(tmp_path, monkeypatch):
    authorization, _ = helper_fixture(tmp_path)
    passed = []

    def fail_spawn(*args, **kwargs):
        passed.extend(kwargs["pass_fds"])
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(W.subprocess, "Popen", fail_spawn)
    with W.authorized_roots(tmp_path, CHECKOUT), pytest.raises(OSError):
        W.Detector(authorization, tmp_path / "stderr")
    assert len(passed) == 2
    for fd in passed:
        assert_closed(fd)
    (tmp_path / "stderr").unlink()
