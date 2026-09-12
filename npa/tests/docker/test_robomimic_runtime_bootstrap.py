from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
SPEC = importlib.util.spec_from_file_location(
    "verify_robomimic_image", IMAGE_ROOT / "verify_image.py"
)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(tmp_path: Path) -> tuple[Path, Path, str]:
    runtime_root = tmp_path / "runtime"
    interpreter = runtime_root / "payload" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    lock_path = IMAGE_ROOT / "runtime-requirements.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = {
        "schema": "npa.robomimic.runtime-inventory.v1",
        "runtime_id": lock["runtime_id"],
        "lock_sha256": _sha(lock_path),
        "source_revision": lock["source_revision"],
        "abi": lock["abi"],
        "packages": lock["packages"],
        "artifacts": [
            {
                "name": name,
                "version": version,
                "filename": f"{name}-{version}.whl",
                "source": "https://download.pytorch.org/whl/cu128/",
                "sha256": hashlib.sha256(f"{name}=={version}".encode()).hexdigest(),
            }
            for name, version in lock["packages"].items()
        ],
        "files": [
            {
                "path": "payload/bin/python",
                "size": interpreter.stat().st_size,
                "sha256": _sha(interpreter),
            }
        ],
        "symlinks": [],
    }
    inventory_path = runtime_root / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "schema": "npa.robomimic.runtime-ready.v1",
        "runtime_id": lock["runtime_id"],
        "lock_sha256": _sha(lock_path),
        "source_revision": lock["source_revision"],
        "inventory_sha256": _sha(inventory_path),
    }
    (runtime_root / ".ready.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runtime_root, lock_path, _sha(inventory_path)


def _bootstrap_with_fake_snapshot(
    tmp_path: Path, *, expose_launch_window: bool = False
) -> Path:
    fake_verifier = tmp_path / "fake_verifier.py"
    fake_verifier.write_text(
        """\
import os
import sys
import time
from pathlib import Path

destination = Path(sys.argv[sys.argv.index("--destination") + 1])
Path(os.environ["NPA_TEST_SNAPSHOT_RECORD"]).write_text(
    str(destination), encoding="utf-8"
)
Path(os.environ["NPA_TEST_SNAPSHOT_PID"]).write_text(
    str(os.getpid()), encoding="utf-8"
)
if os.environ.get("NPA_TEST_SNAPSHOT_MODE") == "block":
    destination.mkdir(parents=True)
    Path(os.environ["NPA_TEST_SNAPSHOT_STARTED"]).touch()
    while True:
        time.sleep(1)
if os.environ.get("NPA_TEST_SNAPSHOT_MODE") == "fail":
    destination.mkdir(parents=True)
    raise SystemExit(41)
interpreter = destination / "payload" / "bin" / "python"
interpreter.parent.mkdir(parents=True)
interpreter.write_text(
    '''#!/bin/sh
if [ "${1:-}" = "-c" ]; then
    printf '%s' "$$" >"${NPA_TEST_IMPORT_PID}"
    if [ -n "${NPA_TEST_IMPORT_STARTED:-}" ]; then
        : >"${NPA_TEST_IMPORT_STARTED}"
    fi
    case "${NPA_TEST_IMPORT_MODE:-success}" in
        fail) exit 42 ;;
        block) while :; do sleep 1; done ;;
        remove) chmod 000 "$0"; exit 0 ;;
    esac
    exit 0
fi
if [ "${NPA_TEST_EXEC_MODE:-block}" = "ignore" ]; then
    trap '' HUP INT TERM
fi
if [ "${NPA_TEST_EXEC_MODE:-block}" = "observe" ]; then
    trap 'test -d "${NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT}" || exit 90; : >"${NPA_TEST_SNAPSHOT_OBSERVED}"; exit 0' TERM
fi
printf '%s' "$$" >"${NPA_TEST_EXEC_PID}"
: >"${NPA_TEST_EXEC_STARTED}"
case "${NPA_TEST_EXEC_MODE:-block}" in
    exit) exit 0 ;;
    block|ignore|observe) while :; do sleep 1; done ;;
esac
''',
    encoding="utf-8",
)
interpreter.chmod(0o755)
for path in destination.rglob("*"):
    if path.is_file() and not path.is_symlink():
        path.chmod(path.stat().st_mode & 0o555)
for path in sorted(
    (item for item in destination.rglob("*") if item.is_dir()),
    key=lambda item: len(item.parts),
    reverse=True,
):
    path.chmod(0o555)
destination.chmod(0o555)
""",
        encoding="utf-8",
    )
    source = (IMAGE_ROOT / "runtime_bootstrap.sh").read_text(encoding="utf-8")
    source = source.replace(
        'readonly verifier="/opt/npa/robomimic/verify_image.py"',
        f'readonly verifier="{fake_verifier}"',
    ).replace("/usr/local/bin/python3", sys.executable)
    if expose_launch_window:
        source = source.replace(
            '      ) &\n      child_pid="$!"',
            """\
      ) &
      if [[ -n "${NPA_TEST_LAUNCH_WINDOW_STARTED:-}" ]]; then
        : >"${NPA_TEST_LAUNCH_WINDOW_STARTED}"
        while [[ ! -e "${NPA_TEST_LAUNCH_WINDOW_RELEASE}" ]]; do
          :
        done
      fi
      child_pid="$!"
""",
        )
        source = source.replace(
            '        pending_status="$2"\n        return',
            '        pending_status="$2"\n'
            '        : >"${NPA_TEST_SIGNAL_PENDING}"\n'
            "        return",
        )
    script = tmp_path / "runtime_bootstrap.sh"
    script.write_text(source, encoding="utf-8")
    return script


def _kill_process_groups(
    process: subprocess.Popen[str], *, child_process_group: int | None
) -> None:
    process_groups = [process.pid]
    if child_process_group is not None:
        process_groups.insert(0, child_process_group)
    for process_group in process_groups:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for_file(
    path: Path,
    process: subprocess.Popen[str],
    *,
    child_pid_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                child_process_group = (
                    int(child_pid_path.read_text(encoding="utf-8"))
                    if child_pid_path is not None and child_pid_path.is_file()
                    else None
                )
                _kill_process_groups(
                    process, child_process_group=child_process_group
                )
                stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"bootstrap exited before {path.name}: "
                f"status={process.returncode} stdout={stdout!r} stderr={stderr!r}"
            )
        time.sleep(0.01)
    child_process_group = (
        int(child_pid_path.read_text(encoding="utf-8"))
        if child_pid_path is not None and child_pid_path.is_file()
        else None
    )
    _kill_process_groups(process, child_process_group=child_process_group)
    process.communicate(timeout=5)
    pytest.fail(f"bootstrap did not create {path.name}")


def _communicate_or_kill(
    process: subprocess.Popen[str], *, child_process_group: int
) -> tuple[str, str]:
    try:
        return process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_process_groups(process, child_process_group=child_process_group)
        process.communicate(timeout=5)
        raise


def _recorded_pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8"))


def _assert_process_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _assert_process_group_gone(process_group: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.killpg(process_group, 0)


def _remove_snapshot_parent(snapshot_root: Path) -> None:
    if not snapshot_root.parent.exists():
        return
    for path in (snapshot_root, *snapshot_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    shutil.rmtree(snapshot_root.parent, ignore_errors=True)


def _exec_environment(
    tmp_path: Path,
    *,
    import_mode: str,
    snapshot_mode: str = "success",
    exec_mode: str = "block",
) -> dict[str, str]:
    return {
        **os.environ,
        "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_TEST_SNAPSHOT_RECORD": str(tmp_path / "snapshot-record"),
        "NPA_TEST_SNAPSHOT_STARTED": str(tmp_path / "snapshot-started"),
        "NPA_TEST_SNAPSHOT_PID": str(tmp_path / "snapshot-pid"),
        "NPA_TEST_SNAPSHOT_MODE": snapshot_mode,
        "NPA_TEST_IMPORT_STARTED": str(tmp_path / "import-started"),
        "NPA_TEST_IMPORT_PID": str(tmp_path / "import-pid"),
        "NPA_TEST_IMPORT_MODE": import_mode,
        "NPA_TEST_EXEC_STARTED": str(tmp_path / "exec-started"),
        "NPA_TEST_EXEC_PID": str(tmp_path / "exec-pid"),
        "NPA_TEST_EXEC_MODE": exec_mode,
        "NPA_TEST_SNAPSHOT_OBSERVED": str(tmp_path / "snapshot-observed"),
        "NPA_TEST_LAUNCH_WINDOW_STARTED": str(tmp_path / "launch-window-started"),
        "NPA_TEST_LAUNCH_WINDOW_RELEASE": str(tmp_path / "launch-window-release"),
        "NPA_TEST_SIGNAL_PENDING": str(tmp_path / "signal-pending"),
    }


def test_exact_external_runtime_inventory_verifies_without_mutation(
    tmp_path: Path,
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    before = {
        path.relative_to(runtime_root).as_posix(): _sha(path)
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    result = verifier.verify_external_runtime(
        runtime_root=runtime_root,
        runtime_lock_path=lock_path,
        expected_inventory_sha256=inventory_sha256,
        require_read_only_mount=False,
    )
    after = {
        path.relative_to(runtime_root).as_posix(): _sha(path)
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    assert result["package_count"] == 22
    assert result["artifact_count"] == 22
    assert result["payload_file_count"] == 1
    assert before == after


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "wrong-package", "extra"])
def test_runtime_inventory_fails_closed(tmp_path: Path, mutation: str) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    if mutation == "missing":
        (runtime_root / ".ready.json").unlink()
    elif mutation == "corrupt":
        (runtime_root / "payload" / "bin" / "python").write_text("changed")
    elif mutation == "wrong-package":
        inventory_path = runtime_root / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["packages"]["torch"] = "0"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        marker_path = runtime_root / ".ready.json"
        marker = json.loads(marker_path.read_text())
        marker["inventory_sha256"] = _sha(inventory_path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    else:
        (runtime_root / "payload" / "extra").write_text("undeclared")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=False,
        )


@pytest.mark.parametrize("sibling", ["wheelhouse", "cache", "run-output.json"])
def test_runtime_inventory_rejects_undeclared_top_level_objects(
    tmp_path: Path, sibling: str
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    extra = runtime_root / sibling
    if "." in sibling:
        extra.write_text("undeclared", encoding="utf-8")
    else:
        extra.mkdir()

    with pytest.raises(verifier.VerificationError, match="undeclared object"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=False,
        )


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    inventory_path = runtime_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    link = runtime_root / "payload" / "escape"
    os.symlink("../../../outside", link)
    inventory["symlinks"] = [{"path": "payload/escape", "target": "../../../outside"}]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    marker_path = runtime_root / ".ready.json"
    marker = json.loads(marker_path.read_text())
    marker["inventory_sha256"] = _sha(inventory_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="escapes"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=_sha(inventory_path),
            require_read_only_mount=False,
        )


def test_manager_inventory_digest_is_an_external_trust_anchor(tmp_path: Path) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    with pytest.raises(verifier.VerificationError, match="manager-approved digest"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256="0" * 64,
            require_read_only_mount=False,
        )


def test_production_verification_rejects_a_writable_mount(tmp_path: Path) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    with pytest.raises(verifier.VerificationError, match="not mounted read-only"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=True,
        )


def test_runtime_execution_uses_an_atomic_verified_snapshot(tmp_path: Path) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    destination = tmp_path / "private" / "active-runtime"
    result = verifier.materialize_external_runtime(
        runtime_root=runtime_root,
        runtime_lock_path=lock_path,
        expected_inventory_sha256=inventory_sha256,
        destination=destination,
        require_source_read_only=False,
    )
    source_interpreter = runtime_root / "payload" / "bin" / "python"
    snapshot_interpreter = destination / "payload" / "bin" / "python"
    source_interpreter.write_text("changed after snapshot\n", encoding="utf-8")

    assert result["atomic_snapshot_published"] is True
    assert result["snapshot_write_bits_absent"] is True
    assert result["manager_inventory_digest_matched"] is True
    assert result["snapshot_root"] == str(destination)
    assert snapshot_interpreter.read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"
    assert destination.stat().st_mode & 0o222 == 0
    # This is a point-in-time race-resistance copy, not a read-only security
    # boundary: its owner can restore write bits. The separately observed
    # source mount is the authoritative read-only boundary.
    destination.chmod(0o755)
    snapshot_interpreter.chmod(0o755)
    snapshot_interpreter.write_text("owner-restored write access\n", encoding="utf-8")
    assert snapshot_interpreter.read_text(encoding="utf-8") == (
        "owner-restored write access\n"
    )
    for path in destination.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)


def test_runtime_snapshot_never_overwrites_an_existing_destination(
    tmp_path: Path,
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="already exists"):
        verifier.materialize_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            destination=destination,
            require_source_read_only=False,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_symlink_cannot_escape_payload_into_uninventoried_runtime_file(
    tmp_path: Path,
) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    outside_payload = runtime_root / "unreviewed-python"
    outside_payload.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    outside_payload.chmod(0o755)
    interpreter = runtime_root / "payload" / "bin" / "python"
    interpreter.unlink()
    os.symlink("../../unreviewed-python", interpreter)
    inventory_path = runtime_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["files"] = []
    inventory["symlinks"] = [
        {"path": "payload/bin/python", "target": "../../unreviewed-python"}
    ]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    marker_path = runtime_root / ".ready.json"
    marker = json.loads(marker_path.read_text())
    marker["inventory_sha256"] = _sha(inventory_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(verifier.VerificationError, match="undeclared object"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=_sha(inventory_path),
            require_read_only_mount=False,
        )


def test_bootstrap_has_no_fetch_install_or_cache_population_path() -> None:
    text = (IMAGE_ROOT / "runtime_bootstrap.sh").read_text(encoding="utf-8")
    for forbidden in (
        "curl ",
        "wget ",
        "pip install",
        "uv sync",
        "git clone",
        "ensure",
    ):
        assert forbidden not in text
    assert "assert-refusal" in text
    assert "verify" in text
    assert "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256" in text
    assert "--expected-inventory-sha256" in text
    assert '"${verifier}" snapshot' in text
    assert 'NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT="${snapshot_root}"' in text
    for import_gate in (
        "from robomimic.config import config_factory",
        "from robomimic.algo import algo_factory",
        "from robomimic.utils.file_utils import policy_from_checkpoint",
        "from diffusers.training_utils import EMAModel",
    ):
        assert import_gate in text
    assert 'run_child "${snapshot_root}/payload/bin/python" "$@"' in text
    assert 'exec "$@"' in text
    assert 'find "${snapshot_parent}" -type d -exec chmod u+w' in text
    assert '${empty_root}.proof' not in text
    assert "trap - EXIT" not in text


def test_bootstrap_cleans_snapshot_when_import_gate_fails(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(tmp_path, import_mode="fail"),
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == 42, result.stderr
    assert not snapshot_root.parent.exists()


def test_bootstrap_cleans_snapshot_when_snapshot_creation_fails(
    tmp_path: Path,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", snapshot_mode="fail"
        ),
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == 41, result.stderr
    assert not snapshot_root.parent.exists()


@pytest.mark.parametrize(
    ("signal_number", "expected_returncode"),
    [
        (signal.SIGHUP, 129),
        (signal.SIGINT, 130),
        (signal.SIGTERM, 143),
    ],
)
def test_bootstrap_stops_snapshot_creation_before_cleanup(
    tmp_path: Path,
    signal_number: signal.Signals,
    expected_returncode: int,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", snapshot_mode="block"
        ),
        start_new_session=True,
    )
    snapshot_pid: int | None = None
    snapshot_root: Path | None = None
    try:
        _wait_for_file(
            tmp_path / "snapshot-started",
            process,
            child_pid_path=tmp_path / "snapshot-pid",
        )
        snapshot_pid = _recorded_pid(tmp_path / "snapshot-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        signal_started = time.monotonic()
        os.kill(process.pid, signal_number)
        _communicate_or_kill(process, child_process_group=snapshot_pid)

        assert process.returncode == expected_returncode
        assert time.monotonic() - signal_started < 4
        _assert_process_gone(snapshot_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=snapshot_pid)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


def test_signal_pending_across_child_launch_is_forwarded(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path, expose_launch_window=True)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", snapshot_mode="block"
        ),
        start_new_session=True,
    )
    snapshot_pid: int | None = None
    snapshot_root: Path | None = None
    try:
        _wait_for_file(
            tmp_path / "launch-window-started",
            process,
            child_pid_path=tmp_path / "snapshot-pid",
        )
        _wait_for_file(
            tmp_path / "snapshot-pid",
            process,
            child_pid_path=tmp_path / "snapshot-pid",
        )
        snapshot_pid = _recorded_pid(tmp_path / "snapshot-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        os.kill(process.pid, signal.SIGTERM)
        _wait_for_file(
            tmp_path / "signal-pending",
            process,
            child_pid_path=tmp_path / "snapshot-pid",
        )
        (tmp_path / "launch-window-release").touch()
        _communicate_or_kill(process, child_process_group=snapshot_pid)

        assert process.returncode == 143
        _assert_process_group_gone(snapshot_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=snapshot_pid)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


@pytest.mark.parametrize(
    ("signal_number", "expected_returncode"),
    [
        (signal.SIGHUP, 129),
        (signal.SIGINT, 130),
        (signal.SIGTERM, 143),
    ],
)
def test_bootstrap_cleans_snapshot_when_signalled_during_import(
    tmp_path: Path,
    signal_number: signal.Signals,
    expected_returncode: int,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(tmp_path, import_mode="block"),
        start_new_session=True,
    )
    import_pid: int | None = None
    snapshot_root: Path | None = None
    try:
        _wait_for_file(
            tmp_path / "import-started",
            process,
            child_pid_path=tmp_path / "import-pid",
        )
        import_pid = _recorded_pid(tmp_path / "import-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        signal_started = time.monotonic()
        os.kill(process.pid, signal_number)
        _communicate_or_kill(process, child_process_group=import_pid)

        assert process.returncode == expected_returncode
        assert time.monotonic() - signal_started < 4
        _assert_process_gone(import_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=import_pid)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


def test_bootstrap_cleans_snapshot_when_final_exec_fails(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(tmp_path, import_mode="remove"),
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == 126, result.stderr
    assert not snapshot_root.parent.exists()


def test_finite_payload_exit_cleans_snapshot(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", exec_mode="exit"
        ),
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    payload_pid = _recorded_pid(tmp_path / "exec-pid")
    assert result.returncode == 0, result.stderr
    _assert_process_gone(payload_pid)
    assert not snapshot_root.parent.exists()


@pytest.mark.parametrize(
    ("signal_number", "expected_returncode"),
    [
        (signal.SIGHUP, 129),
        (signal.SIGINT, 130),
        (signal.SIGTERM, 143),
    ],
)
def test_supervisor_stops_payload_before_cleaning_snapshot(
    tmp_path: Path,
    signal_number: signal.Signals,
    expected_returncode: int,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(tmp_path, import_mode="success"),
        start_new_session=True,
    )
    snapshot_root: Path | None = None
    payload_pid: int | None = None
    try:
        _wait_for_file(
            tmp_path / "exec-started",
            process,
            child_pid_path=tmp_path / "exec-pid",
        )
        payload_pid = _recorded_pid(tmp_path / "exec-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        assert process.poll() is None
        assert snapshot_root.is_dir()
        assert snapshot_root.stat().st_mode & 0o222 == 0
        signal_started = time.monotonic()
        os.kill(process.pid, signal_number)
        _communicate_or_kill(process, child_process_group=payload_pid)
        assert process.returncode == expected_returncode
        assert time.monotonic() - signal_started < 4
        _assert_process_gone(payload_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        if payload_pid is not None:
            try:
                os.killpg(payload_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


def test_payload_observes_snapshot_until_supervisor_reaps_it(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", exec_mode="observe"
        ),
        start_new_session=True,
    )
    snapshot_root: Path | None = None
    payload_pid: int | None = None
    try:
        _wait_for_file(
            tmp_path / "exec-started",
            process,
            child_pid_path=tmp_path / "exec-pid",
        )
        payload_pid = _recorded_pid(tmp_path / "exec-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        os.kill(process.pid, signal.SIGTERM)
        _communicate_or_kill(process, child_process_group=payload_pid)

        assert process.returncode == 143
        assert (tmp_path / "snapshot-observed").is_file()
        _assert_process_group_gone(payload_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=payload_pid)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


def test_repeated_signal_escalates_and_reaps_ignoring_group(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", exec_mode="ignore"
        ),
        start_new_session=True,
    )
    snapshot_root: Path | None = None
    payload_pid: int | None = None
    try:
        _wait_for_file(
            tmp_path / "exec-started",
            process,
            child_pid_path=tmp_path / "exec-pid",
        )
        payload_pid = _recorded_pid(tmp_path / "exec-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        signal_started = time.monotonic()
        os.kill(process.pid, signal.SIGTERM)
        time.sleep(0.2)
        os.kill(process.pid, signal.SIGHUP)
        _communicate_or_kill(process, child_process_group=payload_pid)
        signal_elapsed = time.monotonic() - signal_started

        assert process.returncode == 143
        assert 5 <= signal_elapsed < 10
        _assert_process_group_gone(payload_pid)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=payload_pid)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            _remove_snapshot_parent(snapshot_root)


def test_shipped_assert_refusal_reaches_missing_ready_marker(tmp_path: Path) -> None:
    source = (IMAGE_ROOT / "runtime_bootstrap.sh").read_text(encoding="utf-8")
    source = source.replace(
        'readonly verifier="/opt/npa/robomimic/verify_image.py"',
        f'readonly verifier="{IMAGE_ROOT / "verify_image.py"}"',
    ).replace(
        'readonly runtime_lock="/opt/npa/robomimic/runtime-requirements.lock"',
        f'readonly runtime_lock="{IMAGE_ROOT / "runtime-requirements.lock"}"',
    ).replace("/usr/local/bin/python3", sys.executable)
    script = tmp_path / "runtime_bootstrap.sh"
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(script), "assert-refusal"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": ""},
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NPA_ROBOMIMIC_RUNTIME_REFUSAL_OK"
