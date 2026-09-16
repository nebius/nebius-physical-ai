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
import traceback
from datetime import datetime, timezone
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
                "size": 1,
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


def _entitlement(
    tmp_path: Path,
    *,
    run_id: str = "entitled-run",
    customer_binding_sha256: str = "b" * 64,
    runtime_inventory_sha256: str = "a" * 64,
) -> tuple[Path, str]:
    lock_path = IMAGE_ROOT / "runtime-requirements.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    contract = lock["customer_entitlement"]
    record = {
        "schema": contract["schema"],
        "decision": "accepted",
        "customer_binding_sha256": customer_binding_sha256,
        "run_id": run_id,
        "source_revision": lock["source_revision"],
        "runtime_id": lock["runtime_id"],
        "runtime_manifest_sha256": runtime_inventory_sha256,
        "runtime_lock_sha256": _sha(lock_path),
        "field_of_use": contract["field_of_use"],
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
        "notice_sha256": verifier._runtime_entitlement_notice_sha256(
            lock=lock, lock_sha256=_sha(lock_path)
        ),
        "accepted_at": "2026-09-16T00:00:00Z",
        "expires_at": "2026-09-16T12:00:00Z",
    }
    path = tmp_path / "entitlement.json"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path, _sha(path)


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

if sys.argv[1] == "entitlement":
    count_path = Path(os.environ["NPA_TEST_ENTITLEMENT_COUNT"])
    count = int(count_path.read_text(encoding="utf-8")) + 1 if count_path.exists() else 1
    count_path.write_text(str(count), encoding="utf-8")
    if os.environ.get("NPA_TEST_ENTITLEMENT_FAIL_AT") == str(count):
        raise SystemExit(78)
    if os.environ.get("NPA_TEST_ENTITLEMENT_MODE") == "fail":
        raise SystemExit(78)
    raise SystemExit(0)

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
    descendant)
        (
            : >"${NPA_TEST_DESCENDANT_STARTED}"
            while [ ! -e "${NPA_TEST_DESCENDANT_RELEASE}" ]; do
                test -d "${NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT}" || exit 90
                sleep 0.05
            done
            test -d "${NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT}" || exit 90
            : >"${NPA_TEST_DESCENDANT_OBSERVED}"
        ) &
        printf '%s' "$!" >"${NPA_TEST_DESCENDANT_PID}"
        exit 0
        ;;
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


def _wait_for_process_gone(pid: int, supervisor: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        assert supervisor.poll() is None
        time.sleep(0.01)
    pytest.fail(f"process {pid} remained alive")


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
        "NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE": str(tmp_path / "entitlement.json"),
        "NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256": "c" * 64,
        "NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256": "b" * 64,
        "NPA_BYOF_RUN_ID": "entitled-run",
        "NPA_TEST_ENTITLEMENT_COUNT": str(tmp_path / "entitlement-count"),
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
        "NPA_TEST_DESCENDANT_STARTED": str(tmp_path / "descendant-started"),
        "NPA_TEST_DESCENDANT_RELEASE": str(tmp_path / "descendant-release"),
        "NPA_TEST_DESCENDANT_OBSERVED": str(tmp_path / "descendant-observed"),
        "NPA_TEST_DESCENDANT_PID": str(tmp_path / "descendant-pid"),
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
    assert result["artifact_payload_bytes"] == 22
    assert result["payload_file_count"] == 1
    assert result["payload_bytes"] == 17
    assert before == after


def test_exact_customer_runtime_entitlement_verifies_without_mutation(
    tmp_path: Path,
) -> None:
    path, record_sha256 = _entitlement(tmp_path)
    before = path.read_bytes()

    result = verifier.verify_customer_runtime_entitlement(
        entitlement_path=path,
        runtime_lock_path=IMAGE_ROOT / "runtime-requirements.lock",
        expected_entitlement_sha256=record_sha256,
        expected_customer_binding_sha256="b" * 64,
        expected_run_id="entitled-run",
        expected_inventory_sha256="a" * 64,
        now=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
    )

    assert result["run_binding_matched"] is True
    assert result["customer_binding_matched"] is True
    assert result["field_of_use"] == "noncommercial"
    assert result["redistribution_granted"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("decision", "binding mismatch: decision"),
        ("customer", "binding mismatch: customer_binding_sha256"),
        ("run", "binding mismatch: run_id"),
        ("manifest", "binding mismatch: runtime_manifest_sha256"),
        ("lock", "binding mismatch: runtime_lock_sha256"),
        ("field", "binding mismatch: field_of_use"),
        ("terms", "binding mismatch: terms"),
        ("responsibilities", "binding mismatch: customer_responsibilities"),
        ("notice", "binding mismatch: notice_sha256"),
        ("stale", "has expired"),
        ("overlong", "validity is out of bounds"),
    ),
)
def test_customer_runtime_entitlement_hostile_bindings_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    path, _ = _entitlement(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "decision":
        record["decision"] = "declined"
    elif mutation == "customer":
        record["customer_binding_sha256"] = "c" * 64
    elif mutation == "run":
        record["run_id"] = "other-run"
    elif mutation == "manifest":
        record["runtime_manifest_sha256"] = "d" * 64
    elif mutation == "lock":
        record["runtime_lock_sha256"] = "e" * 64
    elif mutation == "field":
        record["field_of_use"] = "commercial"
    elif mutation == "terms":
        record["terms"] = []
    elif mutation == "responsibilities":
        record["customer_responsibilities"] = []
    elif mutation == "notice":
        record["notice_sha256"] = "f" * 64
    elif mutation == "stale":
        record["expires_at"] = "2026-09-16T05:59:59Z"
    else:
        record["expires_at"] = "2026-09-17T00:00:01Z"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_customer_runtime_entitlement(
            entitlement_path=path,
            runtime_lock_path=IMAGE_ROOT / "runtime-requirements.lock",
            expected_entitlement_sha256=_sha(path),
            expected_customer_binding_sha256="b" * 64,
            expected_run_id="entitled-run",
            expected_inventory_sha256="a" * 64,
            now=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
        )


def test_bootstrap_refuses_entitlement_before_snapshot_or_payload(
    tmp_path: Path,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    env = _exec_environment(tmp_path, import_mode="success", exec_mode="exit")
    env["NPA_TEST_ENTITLEMENT_MODE"] = "fail"

    result = subprocess.run(
        ["bash", str(script), "exec", "ignored.py"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 78
    assert not (tmp_path / "snapshot-record").exists()
    assert not (tmp_path / "import-started").exists()
    assert not (tmp_path / "exec-started").exists()


@pytest.mark.parametrize("unsafe_kind", ("symlink", "hardlink"))
def test_customer_runtime_entitlement_refuses_linked_metadata(
    tmp_path: Path, unsafe_kind: str
) -> None:
    source, _ = _entitlement(tmp_path)
    unsafe = tmp_path / "unsafe-entitlement.json"
    if unsafe_kind == "symlink":
        unsafe.symlink_to(source)
        message = "required regular file is absent"
    else:
        os.link(source, unsafe)
        message = "must have one link"

    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_customer_runtime_entitlement(
            entitlement_path=unsafe,
            runtime_lock_path=IMAGE_ROOT / "runtime-requirements.lock",
            expected_entitlement_sha256=_sha(source),
            expected_customer_binding_sha256="b" * 64,
            expected_run_id="entitled-run",
            expected_inventory_sha256="a" * 64,
            now=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("unsafe_kind", ("permissive-mode", "wrong-owner"))
def test_customer_runtime_entitlement_requires_owner_private_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unsafe_kind: str
) -> None:
    path, record_sha256 = _entitlement(tmp_path)
    if unsafe_kind == "permissive-mode":
        path.chmod(0o644)
    else:
        observed_uid = path.stat().st_uid
        monkeypatch.setattr(verifier.os, "geteuid", lambda: observed_uid + 1)

    with pytest.raises(verifier.VerificationError, match="must be owner-only"):
        verifier.verify_customer_runtime_entitlement(
            entitlement_path=path,
            runtime_lock_path=IMAGE_ROOT / "runtime-requirements.lock",
            expected_entitlement_sha256=record_sha256,
            expected_customer_binding_sha256="b" * 64,
            expected_run_id="entitled-run",
            expected_inventory_sha256="a" * 64,
            now=datetime(2026, 9, 16, 6, tzinfo=timezone.utc),
        )


def test_customer_runtime_entitlement_fifo_refuses_without_blocking(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "entitlement.fifo"
    os.mkfifo(fifo, mode=0o600)

    started = time.monotonic()
    with pytest.raises(verifier.VerificationError, match="required regular file"):
        verifier._bounded_json_object(
            fifo,
            maximum_size=verifier.RUNTIME_ENTITLEMENT_MAX_BYTES,
            require_single_link=True,
            require_owner_only=True,
        )

    assert time.monotonic() - started < 1.0


def test_entitlement_cli_refusal_redacts_customer_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "private-customer-path-marker"
    missing = tmp_path / marker / "entitlement.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_image.py",
            "entitlement",
            "--entitlement",
            str(missing),
            "--runtime-lock",
            str(IMAGE_ROOT / "runtime-requirements.lock"),
            "--expected-entitlement-sha256",
            "a" * 64,
            "--expected-customer-binding-sha256",
            "b" * 64,
            "--expected-run-id",
            "entitled-run",
            "--expected-inventory-sha256",
            "c" * 64,
        ],
    )

    status = verifier.main()

    output = capsys.readouterr().err
    assert status == verifier.RUNTIME_REFUSAL_STATUS
    assert output.strip() == (
        "NPA_ROBOMIMIC_RUNTIME_REFUSED: customer runtime entitlement refused"
    )
    assert marker not in output


@pytest.mark.parametrize("mode", ("runtime", "snapshot"))
def test_runtime_cli_refusals_never_serialize_rejected_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    marker = f"private-{mode}-path-marker"
    arguments = [
        "verify_image.py",
        mode,
        "--runtime-root",
        str(tmp_path / marker),
        "--runtime-lock",
        str(IMAGE_ROOT / "runtime-requirements.lock"),
        "--expected-inventory-sha256",
        "a" * 64,
    ]
    if mode == "snapshot":
        arguments.extend(["--destination", str(tmp_path / "destination")])
    monkeypatch.setattr(sys, "argv", arguments)

    status = verifier.main()

    output = capsys.readouterr().err
    assert status == verifier.RUNTIME_REFUSAL_STATUS
    assert marker not in output
    assert "Traceback" not in output


def test_entitlement_timestamp_refusal_drops_rejected_value_and_exception_chain() -> None:
    marker = "private-malformed-timestamp-marker"

    with pytest.raises(verifier.VerificationError) as raised:
        verifier._utc_timestamp(marker, field="accepted_at")

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


def test_immutable_url_port_refusal_drops_rejected_value_and_exception_chain() -> None:
    marker = "private-malformed-port-marker"

    with pytest.raises(verifier.VerificationError) as raised:
        verifier._checked_https_url(
            f"https://download.pytorch.org:{marker}/wheel",
            host="download.pytorch.org",
        )

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


def _artifact(index: int, *, size: int) -> dict[str, object]:
    return {
        "name": f"package-{index}",
        "version": "1.0",
        "filename": f"package_{index}-1.0.whl",
        "source": "https://download.pytorch.org/whl/cu128/",
        "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        "size": size,
    }


def test_runtime_artifact_limits_admit_every_positive_boundary() -> None:
    artifact_size = verifier.RUNTIME_PAYLOAD_MAX_BYTES // 32
    artifacts = [_artifact(index, size=artifact_size) for index in range(32)]

    checked, total_bytes = verifier._checked_artifacts(artifacts)

    assert len(checked) == verifier.RUNTIME_ARTIFACT_MAX_COUNT == 32
    assert total_bytes == verifier.RUNTIME_PAYLOAD_MAX_BYTES
    assert artifact_size < verifier.RUNTIME_OBJECT_MAX_BYTES
    assert 22 <= verifier.RUNTIME_ARTIFACT_MAX_COUNT
    assert 1_039_389_795 < verifier.RUNTIME_OBJECT_MAX_BYTES
    assert 3_855_918_950 < verifier.RUNTIME_PAYLOAD_MAX_BYTES


@pytest.mark.parametrize(
    ("artifacts", "message"),
    [
        ([{}] * 33, "artifact object count exceeds limit"),
        (
            [_artifact(0, size=verifier.RUNTIME_OBJECT_MAX_BYTES + 1)],
            "artifact exceeds size limit",
        ),
        (
            [
                _artifact(0, size=verifier.RUNTIME_OBJECT_MAX_BYTES),
                _artifact(1, size=verifier.RUNTIME_OBJECT_MAX_BYTES),
                _artifact(2, size=1),
            ],
            "artifact aggregate exceeds size limit",
        ),
    ],
)
def test_runtime_artifact_limits_fail_closed_before_unbounded_processing(
    artifacts: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(verifier.VerificationError, match=message):
        verifier._checked_artifacts(artifacts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("artifact-count", "artifact object count exceeds limit"),
        ("artifact-size", "artifact exceeds size limit"),
        ("artifact-aggregate", "artifact aggregate exceeds size limit"),
        ("payload-count", "payload entry count exceeds limit"),
        ("payload-size", "payload object exceeds size limit"),
        ("payload-aggregate", "payload aggregate exceeds size limit"),
    ],
)
def test_snapshot_refuses_resource_overflow_before_payload_hash_or_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    inventory_path = runtime_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if mutation == "artifact-count":
        monkeypatch.setattr(verifier, "RUNTIME_ARTIFACT_MAX_COUNT", 21)
    elif mutation == "artifact-size":
        inventory["artifacts"][0]["size"] = verifier.RUNTIME_OBJECT_MAX_BYTES + 1
    elif mutation == "artifact-aggregate":
        monkeypatch.setattr(verifier, "RUNTIME_PAYLOAD_MAX_BYTES", 21)
    elif mutation == "payload-count":
        monkeypatch.setattr(verifier, "RUNTIME_PAYLOAD_MAX_ENTRY_COUNT", 1)
        inventory["symlinks"].append(
            {"path": "payload/bin/python-link", "target": "python"}
        )
    elif mutation == "payload-size":
        inventory["files"][0]["size"] = verifier.RUNTIME_OBJECT_MAX_BYTES + 1
    else:
        monkeypatch.setattr(verifier, "RUNTIME_PAYLOAD_MAX_BYTES", 22)
        inventory["files"][0]["size"] = 23
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    marker_path = runtime_root / ".ready.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["inventory_sha256"] = _sha(inventory_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    original_sha256 = verifier._sha256

    def reject_payload_hash(path: Path) -> str:
        if (runtime_root / "payload") in path.parents:
            pytest.fail("payload hashing started before inventory limits passed")
        return original_sha256(path)

    def reject_copy(*_args: object, **_kwargs: object) -> None:
        pytest.fail("snapshot copying started before inventory limits passed")

    monkeypatch.setattr(verifier, "_sha256", reject_payload_hash)
    monkeypatch.setattr(verifier, "_copy_bounded_regular_file", reject_copy)
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.materialize_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=_sha(inventory_path),
            destination=tmp_path / "snapshot",
            require_source_read_only=False,
        )


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
        inventory_sha256 = _sha(inventory_path)
    else:
        (runtime_root / "payload" / "extra").write_text("undeclared")
    message = (
        "runtime package inventory mismatch"
        if mutation == "wrong-package"
        else None
    )
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=False,
        )


def test_initial_runtime_verification_refuses_fifo_replacement_without_blocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    interpreter = runtime_root / "payload" / "bin" / "python"
    original_open = verifier.os.open
    replaced = False

    def replace_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if Path(path) == interpreter and not replaced:
            replaced = True
            interpreter.unlink()
            os.mkfifo(interpreter)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(verifier.os, "open", replace_before_open)
    started = time.monotonic()
    with pytest.raises(verifier.VerificationError, match="runtime file identity mismatch"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=False,
        )

    assert replaced is True
    assert time.monotonic() - started < 1.0


def test_snapshot_copy_refuses_fifo_replacement_without_blocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"verified")
    destination = tmp_path / "destination"
    original_open = verifier.os.open
    replaced = False

    def replace_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if Path(path) == source and not replaced:
            replaced = True
            source.unlink()
            os.mkfifo(source)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(verifier.os, "open", replace_before_open)
    started = time.monotonic()
    with pytest.raises(verifier.VerificationError, match="source is not regular"):
        verifier._copy_bounded_regular_file(
            source,
            destination,
            expected_size=8,
            expected_sha256=hashlib.sha256(b"verified").hexdigest(),
        )

    assert replaced is True
    assert not destination.exists()
    assert time.monotonic() - started < 1.0


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


def test_runtime_symlink_loop_refusal_is_value_free(tmp_path: Path) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    marker = "private-runtime-symlink-loop-marker"
    link = runtime_root / "payload" / marker
    link.symlink_to(marker)
    inventory_path = runtime_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["symlinks"] = [{"path": f"payload/{marker}", "target": marker}]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    marker_path = runtime_root / ".ready.json"
    ready = json.loads(marker_path.read_text(encoding="utf-8"))
    ready["inventory_sha256"] = _sha(inventory_path)
    marker_path.write_text(json.dumps(ready), encoding="utf-8")

    with pytest.raises(
        verifier.VerificationError, match="runtime symlink resolution failed"
    ) as raised:
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=_sha(inventory_path),
            require_read_only_mount=False,
        )

    serialized = json.dumps(
        {
            "args": raised.value.args,
            "cause": repr(raised.value.__cause__),
            "context": repr(raised.value.__context__),
            "notes": list(getattr(raised.value, "__notes__", ())),
            "traceback": "".join(traceback.format_exception(raised.value)),
        }
    )
    assert marker not in serialized
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not getattr(raised.value, "__notes__", ())


def test_runtime_filesystem_failure_is_value_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    marker = "private-runtime-filesystem-marker"
    original_rglob = Path.rglob

    def refused_rglob(path: Path, pattern: str):
        if path == runtime_root:
            raise OSError(marker)
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", refused_rglob)
    with pytest.raises(
        verifier.VerificationError, match="runtime filesystem verification failed"
    ) as raised:
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            require_read_only_mount=False,
        )

    diagnostics = "".join(traceback.format_exception(raised.value))
    assert marker not in diagnostics
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_runtime_inventory_digest_is_an_external_trust_anchor(tmp_path: Path) -> None:
    runtime_root, lock_path, _ = _runtime(tmp_path)
    with pytest.raises(verifier.VerificationError, match="operator-selected digest"):
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


@pytest.mark.parametrize("metadata_name", (".ready.json", "inventory.json"))
def test_runtime_verification_rejects_oversized_metadata_before_parsing(
    tmp_path: Path, metadata_name: str
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    metadata = runtime_root / metadata_name
    with metadata.open("ab") as handle:
        handle.write(
            b" " * (verifier.RUNTIME_METADATA_MAX_BYTES - metadata.stat().st_size + 1)
        )
    expected_inventory_sha256 = (
        _sha(metadata) if metadata_name == "inventory.json" else inventory_sha256
    )

    with pytest.raises(verifier.VerificationError, match="metadata exceeds size limit"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=expected_inventory_sha256,
            require_read_only_mount=False,
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
    assert result["runtime_manifest_digest_matched"] is True
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


def test_runtime_snapshot_cleans_read_only_staging_after_publication_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    destination = tmp_path / "private" / "active-runtime"
    original_replace = Path.replace

    def collide_on_publish(path: Path, target: Path) -> Path:
        if target == destination:
            destination.mkdir()
            raise FileExistsError("simulated publication race")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", collide_on_publish)
    with pytest.raises(
        verifier.VerificationError, match="runtime snapshot materialization failed"
    ):
        verifier.materialize_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            destination=destination,
            require_source_read_only=False,
        )

    assert destination.is_dir()
    assert list(destination.parent.iterdir()) == [destination]


def test_runtime_snapshot_cleanup_refuses_replaced_staging_identity(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    expected = verifier._owned_snapshot_identity(staging)
    displaced = tmp_path / "displaced"
    staging.rename(displaced)
    staging.mkdir()

    with pytest.raises(verifier.VerificationError, match="snapshot cleanup failed"):
        verifier._cleanup_snapshot_staging(staging, expected)

    assert staging.is_dir()
    assert displaced.is_dir()


def test_runtime_snapshot_rejects_source_growth_after_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root, lock_path, inventory_sha256 = _runtime(tmp_path)
    destination = tmp_path / "private" / "active-runtime"
    source_interpreter = runtime_root / "payload" / "bin" / "python"
    original_copy = verifier._copy_bounded_regular_file
    source_mutated = False

    def grow_declared_file(
        source: Path,
        target: Path,
        *,
        expected_size: int | None,
        expected_sha256: str | None,
        maximum_size: int | None = None,
    ) -> None:
        nonlocal source_mutated
        if source == source_interpreter and not source_mutated:
            source_mutated = True
            with source.open("ab") as handle:
                handle.write(b"unexpected-growth")
        original_copy(
            source,
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            maximum_size=maximum_size,
        )

    monkeypatch.setattr(verifier, "_copy_bounded_regular_file", grow_declared_file)
    with pytest.raises(verifier.VerificationError, match="source size changed"):
        verifier.materialize_external_runtime(
            runtime_root=runtime_root,
            runtime_lock_path=lock_path,
            expected_inventory_sha256=inventory_sha256,
            destination=destination,
            require_source_read_only=False,
        )

    assert source_mutated is True
    assert not destination.exists()
    assert destination.parent.is_dir()
    assert list(destination.parent.iterdir()) == []


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
    assert "EPOCHREALTIME" not in text
    assert "</proc/uptime" in text
    assert "ps -e" not in text


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


def test_bootstrap_rechecks_expiry_after_snapshot_before_import(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    environment = _exec_environment(tmp_path, import_mode="success", exec_mode="exit")
    environment["NPA_TEST_ENTITLEMENT_FAIL_AT"] = "2"

    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == verifier.RUNTIME_REFUSAL_STATUS, result.stderr
    assert (tmp_path / "entitlement-count").read_text(encoding="utf-8") == "2"
    assert not (tmp_path / "import-started").exists()
    assert not (tmp_path / "exec-started").exists()
    assert not snapshot_root.parent.exists()


def test_bootstrap_rechecks_expiry_after_import_before_payload(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    environment = _exec_environment(tmp_path, import_mode="success", exec_mode="exit")
    environment["NPA_TEST_ENTITLEMENT_FAIL_AT"] = "3"

    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == verifier.RUNTIME_REFUSAL_STATUS, result.stderr
    assert (tmp_path / "entitlement-count").read_text(encoding="utf-8") == "3"
    assert (tmp_path / "import-started").is_file()
    assert not (tmp_path / "exec-started").exists()
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


def test_successful_leader_keeps_snapshot_until_descendant_exits(
    tmp_path: Path,
) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_exec_environment(
            tmp_path, import_mode="success", exec_mode="descendant"
        ),
        start_new_session=True,
    )
    snapshot_root: Path | None = None
    payload_group: int | None = None
    try:
        _wait_for_file(
            tmp_path / "descendant-started",
            process,
            child_pid_path=tmp_path / "exec-pid",
        )
        payload_group = _recorded_pid(tmp_path / "exec-pid")
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        _wait_for_process_gone(payload_group, process)
        assert process.poll() is None
        os.killpg(payload_group, 0)
        assert snapshot_root.is_dir()

        (tmp_path / "descendant-release").touch()
        stdout, stderr = _communicate_or_kill(
            process, child_process_group=payload_group
        )

        assert process.returncode == 0, (stdout, stderr)
        assert (tmp_path / "descendant-observed").is_file()
        _assert_process_group_gone(payload_group)
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            _kill_process_groups(process, child_process_group=payload_group)
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


def test_monotonic_deadline_does_not_depend_on_a_hanging_ps(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    hostile_ps = hostile_bin / "ps"
    hostile_ps.write_text(
        '#!/bin/sh\n: >"${NPA_TEST_PS_CALLED}"\nwhile :; do sleep 1; done\n',
        encoding="utf-8",
    )
    hostile_ps.chmod(0o755)
    environment = _exec_environment(
        tmp_path, import_mode="success", exec_mode="ignore"
    )
    environment["PATH"] = f"{hostile_bin}:{environment['PATH']}"
    environment["NPA_TEST_PS_CALLED"] = str(tmp_path / "ps-called")
    process = subprocess.Popen(
        ["bash", str(script), "exec", "smoke.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
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
        assert not (tmp_path / "ps-called").exists()
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
