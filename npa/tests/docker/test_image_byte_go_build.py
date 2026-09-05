"""Hermetic bootstrap boundaries. No Go installation, network or image required."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import stat
import sys
import tarfile

import pytest

HELPER = Path(__file__).resolve().parents[2] / "scripts/image_byte_scan/go_helper"
SPEC = importlib.util.spec_from_file_location("image_byte_go_build", HELPER / "build.py")
B = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = B
SPEC.loader.exec_module(B)


@pytest.fixture(autouse=True)
def restore_process_umask():
    """The directly invoked CLI must not change later repository tests' mode."""
    previous = os.umask(0o077)
    os.umask(previous)
    try:
        yield
    finally:
        os.umask(previous)


def archive_bytes(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, kind, content in members:
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            member.type = kind
            member.size = len(content) if kind == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(content) if member.size else None)
    return stream.getvalue()


def prepare_archive(tmp_path, monkeypatch, members):
    raw = archive_bytes(members)
    monkeypatch.setattr(B, "GO_ARCHIVE_SHA256", B.digest(raw))
    archive = tmp_path / "toolchain.tar.gz"
    archive.write_bytes(raw)
    output = B.private_dir(tmp_path / "extracted")
    return archive, output


def test_toolchain_pin_before_extraction(tmp_path):
    archive = tmp_path / "wrong"
    archive.write_bytes(b"untrusted contents")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(B.BuildError, match="^toolchain_digest$"):
        B.extract_toolchain(archive, output)
    assert list(output.iterdir()) == []


def test_exact_archive_extracts_only_regular_bytes(tmp_path, monkeypatch):
    archive, output = prepare_archive(tmp_path, monkeypatch, [
        ("go/", tarfile.DIRTYPE, b""),
        ("go/bin/go", tarfile.REGTYPE, b"synthetic executable bytes")])
    executable = B.extract_toolchain(archive, output)
    assert executable.read_bytes() == b"synthetic executable bytes"
    assert stat.S_IMODE(executable.stat().st_mode) == 0o700


@pytest.mark.parametrize("name,kind", [
    ("/go/bin/go", tarfile.REGTYPE), ("go/../outside", tarfile.REGTYPE),
    ("go/./bin/go", tarfile.REGTYPE), ("go//bin/go", tarfile.REGTYPE),
    ("other/bin/go", tarfile.REGTYPE), ("go\\outside", tarfile.REGTYPE),
    ("go/link", tarfile.SYMTYPE), ("go/link", tarfile.LNKTYPE),
    ("go/fifo", tarfile.FIFOTYPE), ("go/device", tarfile.CHRTYPE)])
def test_archive_rejects_alias_special_and_escape(tmp_path, monkeypatch, name, kind):
    archive, output = prepare_archive(tmp_path, monkeypatch, [(name, kind, b"x")])
    with pytest.raises(B.BuildError, match="^toolchain_member$"):
        B.extract_toolchain(archive, output)


def test_archive_duplicate_rejected(tmp_path, monkeypatch):
    archive, output = prepare_archive(tmp_path, monkeypatch, [
        ("go/bin/go", tarfile.REGTYPE, b"first"), ("go/bin/go", tarfile.REGTYPE, b"second")])
    with pytest.raises(B.BuildError, match="^toolchain_member$"):
        B.extract_toolchain(archive, output)
    assert (output / "go/bin/go").read_bytes() == b"first"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_input_rejects_nonregular_alias(tmp_path, kind):
    target = tmp_path / "target"
    target.write_bytes(b"fixed bytes")
    candidate = tmp_path / "candidate"
    if kind == "symlink":
        candidate.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, candidate)
    elif kind == "fifo":
        os.mkfifo(candidate)
    else:
        candidate.mkdir()
    with pytest.raises(B.BuildError, match="path_symlink|input_not_regular"):
        B.read_regular(candidate)


def test_read_detects_exact_path_replacement(tmp_path, monkeypatch):
    target = tmp_path / "input"
    target.write_bytes(b"old")
    original = B.os.fstat
    calls = 0

    def fstat(fd):
        nonlocal calls
        info = original(fd)
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"new")
            replacement.replace(target)
        return info

    monkeypatch.setattr(B.os, "fstat", fstat)
    with pytest.raises(B.BuildError, match="^input_replaced$"):
        B.read_regular(target)


def test_exclusive_output_never_overwrites(tmp_path):
    target = tmp_path / "receipt"
    B.write_new(target, b"first")
    with pytest.raises(FileExistsError):
        B.write_new(target, b"second")
    assert target.read_bytes() == b"first"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_private_directory_refuses_broad_existing_mode(tmp_path):
    target = tmp_path / "open"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    with pytest.raises(B.BuildError, match="^directory_not_private$"):
        B.private_dir(target)


def test_environment_ignores_ambient_configuration(tmp_path, monkeypatch):
    for name in ("GOFLAGS", "GOTOOLCHAIN", "GOWORK", "GOPROXY", "GOSUMDB",
                 "GOPRIVATE", "NPA_IMAGE_BYTE_TEST_CONFIG", "CUSTOMER_SECRET", "HOME"):
        monkeypatch.setenv(name, "synthetic-sensitive-input")
    env = B.isolated_environment(tmp_path, tmp_path / "config")
    assert env["GOTOOLCHAIN"] == "local"
    assert env["GOFLAGS"] == "-mod=readonly"
    assert env["GOSUMDB"] == "sum.golang.org"
    assert env["GOWORK"] == env["GOENV"] == "off"
    assert "HOME" not in env and "CUSTOMER_SECRET" not in env
    assert "synthetic-sensitive-input" not in json.dumps(env)
    for name in ("GOMODCACHE", "GOCACHE", "GOPATH", "GOTMPDIR", "TMPDIR"):
        assert Path(env[name]).is_relative_to(tmp_path)
        assert stat.S_IMODE(Path(env[name]).stat().st_mode) == 0o700


def ready_raw(**overrides):
    value = {"type": "ready", "protocol": "whole-file-gitleaks.v1", "version": "8.28.0",
             "config_sha256": "a" * 64, "rule_count": 223, "max_target_megabytes": 0,
             "ignore_inline_allow": True, "redact": 100,
             "policy_before_sha256": "b" * 64, "policy_after_sha256": "c" * 64}
    value.update(overrides)
    return B.json_bytes(value) + B.json_bytes({"type": "summary", "files": 0, "bytes": 0, "findings": 0})


def test_ready_preserves_exact_first_line_bytes():
    raw = ready_raw()
    assert B._ready(raw, "a" * 64)["raw"] == raw.splitlines(keepends=True)[0]


@pytest.mark.parametrize("overrides", [
    {"config_sha256": "d" * 64}, {"protocol": "other"}, {"version": "8.27.0"},
    {"max_target_megabytes": 1}, {"ignore_inline_allow": False}, {"redact": 0},
    {"rule_count": 0}, {"policy_after_sha256": "invalid"}])
def test_ready_rejects_changed_policy(overrides):
    with pytest.raises(B.BuildError, match="^helper_handshake_"):
        B._ready(ready_raw(**overrides), "a" * 64)


def test_ready_rejects_missing_terminal_summary():
    with pytest.raises(B.BuildError, match="^helper_handshake_shape$"):
        B._ready(ready_raw().splitlines(keepends=True)[0], "a" * 64)


def test_module_notices_bind_exact_locked_payload(tmp_path):
    cache = tmp_path / "cache"
    folder = cache / "module"
    folder.mkdir(parents=True)
    (folder / "LICENSE").write_bytes(b"synthetic license text")
    output = tmp_path / "notices"
    output.mkdir()
    module = {"Path": "github.com/zricethezav/gitleaks/v8", "Version": B.GITLEAKS_VERSION,
              "Sum": "h1:synthetic-sum", "GoModSum": "h1:synthetic-mod-sum", "Dir": str(folder)}
    sums = f'{module["Path"]} {module["Version"]} {module["Sum"]}\n'.encode()
    records = B.module_notices([module], cache, output, sums)
    assert records[0]["notices"][0]["sha256"] == hashlib.sha256(b"synthetic license text").hexdigest()
    assert Path(records[0]["notices"][0]["path"]).read_bytes() == b"synthetic license text"


@pytest.mark.parametrize("issue", ["unlocked", "replace", "error", "outside", "missing_license"])
def test_module_receipt_rejects_unverified_closure(tmp_path, issue):
    cache = tmp_path / "cache"
    folder = cache / "module"
    folder.mkdir(parents=True)
    (folder / "LICENSE").write_bytes(b"synthetic license")
    module = {"Path": "github.com/zricethezav/gitleaks/v8", "Version": B.GITLEAKS_VERSION,
              "Sum": "h1:synthetic-sum", "Dir": str(folder)}
    sums = f'{module["Path"]} {module["Version"]} {module["Sum"]}\n'.encode()
    if issue == "unlocked":
        sums = b""
    elif issue == "replace":
        module["Replace"] = {"Dir": "."}
    elif issue == "error":
        module["Error"] = "synthetic-sensitive-input"
    elif issue == "outside":
        module["Dir"] = str(tmp_path)
    else:
        (folder / "LICENSE").unlink()
    output = tmp_path / "notices"
    output.mkdir()
    with pytest.raises(B.BuildError, match="module_") as caught:
        B.module_notices([module], cache, output, sums)
    assert "synthetic-sensitive-input" not in str(caught.value)


def test_failed_subprocess_has_private_logs_fixed_boundary(tmp_path, monkeypatch):
    class Result:
        returncode = 17
        pid = 999999999
        stdout = b"synthetic-sensitive-output"
        stderr = b"synthetic-sensitive-error"

        def communicate(self, *args, **kwargs):
            return self.stdout, self.stderr

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        return Result()

    monkeypatch.setattr(B.subprocess, "Popen", popen)
    with pytest.raises(B.BuildError, match="^step_tests_failed$"):
        B.run_step(["unused"], "tests", tmp_path, {}, tmp_path)
    assert (tmp_path / "tests.stdout.log").read_bytes() == Result.stdout
    assert stat.S_IMODE((tmp_path / "tests.stdout.log").stat().st_mode) == 0o600
    assert json.loads((tmp_path / "tests.status.json").read_bytes()) == {"exit_code": 17, "leader_joined": True}


def test_cli_does_not_expose_io_exception(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise OSError("synthetic-sensitive-input")

    monkeypatch.setattr(B, "build", fail)
    assert B.main(["--analysis-root", str(tmp_path), "--trusted-root", str(tmp_path),
                   "--output-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err) == {"status": "failed", "error": "bootstrap_io_or_schema"}


@pytest.mark.parametrize("output_role", ["inside", "outside", "same"])
def test_scope_refusal_before_download(tmp_path, monkeypatch, output_role):
    trusted = tmp_path / "trusted"
    analysis = tmp_path / "analysis"
    output = analysis / "tools"
    if output_role == "inside":
        analysis = trusted / "analysis"
        output = analysis / "tools"
    elif output_role == "outside":
        output = tmp_path / "outside"
    else:
        output = analysis
    monkeypatch.setattr(B.urllib.request, "urlopen", lambda *a: pytest.fail("network reached"))
    with pytest.raises(B.BuildError, match="^output_scope$"):
        B.build(analysis, trusted, output)
    assert not output.exists()


def test_json_stream_preserves_all_module_objects():
    assert B.json_stream(b' {"a":1}\n  {"b":2}\n') == [{"a": 1}, {"b": 2}]
    with pytest.raises(B.BuildError, match="^module_receipt_shape$"):
        B.json_stream(b'[]')


def test_locked_downloads_exclude_mod_only_upstream_test_graph():
    sums = (b'example.org/used v1.0.0 h1:payload\n'
            b'example.org/used v1.0.0/go.mod h1:metadata\n'
            b'example.org/upstream-tests v2.0.0/go.mod h1:metadata\n')
    assert B.locked_modules(sums) == ["example.org/used@v1.0.0"]


@pytest.mark.parametrize("raw", [b'', b'invalid', b'a v1.0.0 unverified',
                                 b'a v1.0.0 h1:a\na v1.0.0 h1:b\n', b'-flag v1.0.0 h1:a'])
def test_locked_downloads_reject_invalid_input(raw):
    with pytest.raises(B.BuildError, match="^module_sum_"):
        B.locked_modules(raw)


def test_repository_closure_is_exactly_payload_pinned():
    raw = (HELPER / "go.sum").read_bytes()
    values = B.locked_modules(raw)
    assert len(values) == 77
    assert f"github.com/zricethezav/gitleaks/v8@{B.GITLEAKS_VERSION}" in values
    assert not any("/go.mod" in value for value in values)


def test_parent_components_rejected_before_normalization(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(B.BuildError, match="^path_parent_component$"):
        B.no_symlinks(tmp_path / "link/../elsewhere")


@pytest.mark.parametrize("operation", ["read", "write"])
def test_parent_swap_cannot_redirect_leaf_io(tmp_path, monkeypatch, operation):
    parent = tmp_path / "parent"
    parent.mkdir()
    evil = tmp_path / "evil"
    evil.mkdir()
    (parent / "input").write_bytes(b"trusted bytes")
    (evil / "input").write_bytes(b"untrusted bytes")
    opened = B.os.open
    switched = False

    def swap(name, flags, *args, **kwargs):
        nonlocal switched
        if name in {"input", "output"} and kwargs.get("dir_fd") is not None and not switched:
            switched = True
            parent.rename(tmp_path / "original")
            parent.symlink_to(evil, target_is_directory=True)
        return opened(name, flags, *args, **kwargs)

    monkeypatch.setattr(B.os, "open", swap)
    if operation == "read":
        with pytest.raises((B.BuildError, OSError)):
            B.read_regular(parent / "input")
        assert (evil / "input").read_bytes() == b"untrusted bytes"
    else:
        B.write_new(parent / "output", b"new trusted bytes")
        assert not (evil / "output").exists()
        assert (tmp_path / "original/output").read_bytes() == b"new trusted bytes"
    assert switched


def test_intermediate_swap_cannot_redirect_component_open(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "nested").mkdir(parents=True)
    evil = tmp_path / "evil"
    (evil / "nested").mkdir(parents=True)
    (parent / "nested/input").write_bytes(b"trusted")
    (evil / "nested/input").write_bytes(b"untrusted")
    opened = B.os.open
    switched = False

    def swap(name, flags, *args, **kwargs):
        nonlocal switched
        if name == "nested" and kwargs.get("dir_fd") is not None and not switched:
            switched = True
            parent.rename(tmp_path / "original")
            parent.symlink_to(evil, target_is_directory=True)
        return opened(name, flags, *args, **kwargs)

    monkeypatch.setattr(B.os, "open", swap)
    with pytest.raises((B.BuildError, OSError)):
        B.read_regular(parent / "nested/input")
    assert switched


@pytest.mark.parametrize("extra", [["--unknown", "synthetic-sensitive-argument"],
                                   ["--analysis-root"], []])
def test_cli_argument_errors_never_echo_values(extra, capsys):
    assert B.main(extra) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err) == {"status": "failed", "error": "arguments_invalid"}
    assert "synthetic-sensitive-argument" not in captured.err


def test_cancellation_restores_handlers():
    previous = {key: signal.getsignal(key) for key in (signal.SIGTERM, signal.SIGINT)}
    with B.cancellation_scope():
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(B.BuildCancelled):
            handler(signal.SIGTERM, None)
        # A second cancellation cannot interrupt cleanup and abandon children.
        handler(signal.SIGTERM, None)
    assert previous == {key: signal.getsignal(key) for key in previous}


def _running(pid):
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return data.rsplit(")", 1)[1].split()[0] != "Z"


def test_sigterm_joins_owned_command_and_stops_ignoring_child(tmp_path):
    marker = tmp_path / "ready.json"
    child = tmp_path / "child.py"
    child.write_text('''import json, os, signal, subprocess, sys, time
from pathlib import Path
armed = str(Path(sys.argv[1]).with_suffix(".armed"))
nested = subprocess.Popen([sys.executable, "-c", "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text('armed'); time.sleep(60)", armed], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not Path(armed).exists(): time.sleep(0.01)
Path(sys.argv[1]).write_text(json.dumps({"leader":os.getpid(),"child":nested.pid}))
time.sleep(60)
''')
    driver = tmp_path / "driver.py"
    driver.write_text('''import importlib.util, pathlib, sys
spec=importlib.util.spec_from_file_location("bridge",sys.argv[1]); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
root=pathlib.Path(sys.argv[2])
try:
    with b.cancellation_scope():
        b.run_step([sys.executable,str(root/"child.py"),str(root/"ready.json")],"cancel",root,{},root)
except b.BuildCancelled:
    raise SystemExit(130)
''')
    process = subprocess.Popen([sys.executable, str(driver), str(HELPER / "build.py"), str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ids = None
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < until:
            assert process.poll() is None
            time.sleep(0.01)
        assert marker.exists(), "owned command did not reach cancellation boundary"
        ids = json.loads(marker.read_text())
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130
        assert not stdout and not stderr
        status = json.loads((tmp_path / "cancel.status.json").read_text())
        assert status["interrupted"] and status["leader_joined"]
        until = time.monotonic() + 5
        while any(_running(pid) for pid in ids.values()) and time.monotonic() < until:
            time.sleep(0.01)
        assert all(not _running(pid) for pid in ids.values())
        assert sibling.poll() is None, "cancellation must not signal a sibling process"
    finally:
        sibling.kill()
        sibling.communicate()
        if process.poll() is None:
            process.kill()
            process.communicate()
        if ids:
            for pid in ids.values():
                if _running(pid):
                    os.kill(pid, signal.SIGKILL)


def test_native_receipt_proves_declared_collection_and_additive_subtests():
    source = b'func TestOne(t *testing.T) {}\nfunc TestTwo(t *testing.T) {}\n'
    rows = [{"Action": "pass", "Test": name} for name in ("TestOne", "TestTwo", "TestOne/sub")]
    rows.append({"Action": "pass", "Package": "synthetic/package"})
    assert B.verify_native_tests(b''.join(B.json_bytes(row) for row in rows), source) == {
        "passed": 3, "declared": 2, "skipped": 0, "failed": 0}


@pytest.mark.parametrize("rows,source,error", [
    ([], b'func TestOne(t *testing.T) {}', "native_tests_empty"),
    ([{"Action": "pass"}], b'', "native_tests_empty"),
    ([{"Action": "pass"}], b'func TestOne(t *testing.T) {}', "native_tests_missing"),
    ([{"Action": "skip"}], b'func TestOne(t *testing.T) {}', "native_tests_skipped"),
    ([{"Action": "fail"}], b'func TestOne(t *testing.T) {}', "native_tests_failed"),
    ([{"Action": "pass", "Test": "TestOne"}], b'func TestOne(t *testing.T) {}',
     "native_tests_package_incomplete")])
def test_native_receipt_refuses_missing_skipped_or_failed_collection(rows, source, error):
    with pytest.raises(B.BuildError, match=f"^{error}$"):
        B.verify_native_tests(b''.join(B.json_bytes(row) for row in rows), source)


@pytest.mark.parametrize("identities", [[], ["a"], ["a", "a"], ["a", "b", "extra"]])
def test_module_receipt_requires_complete_unique_requested_response(identities):
    rows = [{"Path": name, "Version": "v1.0.0"} for name in identities]
    with pytest.raises(B.BuildError, match="^module_response_incomplete$"):
        B.verify_module_set(rows, ["a@v1.0.0", "b@v1.0.0"])


def test_module_response_order_does_not_change_closure():
    B.verify_module_set([{"Path": name, "Version": "v1.0.0"} for name in ("b", "a")],
                        ["a@v1.0.0", "b@v1.0.0"])


def test_successful_leader_cannot_abandon_background_command(tmp_path):
    ready = tmp_path / "child.ready"
    script = tmp_path / "leader.py"
    script.write_text('''import json, pathlib, subprocess, sys, time
ready=pathlib.Path(sys.argv[1])
child=subprocess.Popen([sys.executable,"-c","import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
while not ready.exists(): time.sleep(0.01)
''')
    pid = None
    try:
        with pytest.raises(B.BuildError, match="^step_orphan_unjoined_descendant$"):
            B.run_step([sys.executable, str(script), str(ready)], "orphan", tmp_path, {}, tmp_path)
        pid = int(ready.read_text())
        until = time.monotonic() + 5
        while _running(pid) and time.monotonic() < until:
            time.sleep(0.01)
        assert not _running(pid)
    finally:
        if pid and _running(pid):
            os.kill(pid, signal.SIGKILL)


def test_analysis_ancestor_never_allows_output_in_trusted_checkout(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    output = trusted / "tools"
    monkeypatch.setattr(B.urllib.request, "urlopen", lambda *a: pytest.fail("network reached"))
    with pytest.raises(B.BuildError, match="^output_scope$"):
        B.build(tmp_path, trusted, output)
    assert not output.exists()


def test_signal_during_process_creation_is_deferred_until_handle_is_owned(tmp_path, monkeypatch):
    original = B.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    monkeypatch.setattr(B.subprocess, "Popen", spawn)
    with B.cancellation_scope(), pytest.raises(B.BuildCancelled):
        B.run_step([sys.executable, "-c", "import time; time.sleep(60)"],
                   "creation", tmp_path, {}, tmp_path)
    assert len(children) == 1 and children[0].poll() is not None
    assert json.loads((tmp_path / "creation.status.json").read_text())["leader_joined"]
