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


def _bootstrap_with_fake_snapshot(tmp_path: Path) -> Path:
    fake_verifier = tmp_path / "fake_verifier.py"
    fake_verifier.write_text(
        """\
import os
import sys
from pathlib import Path

destination = Path(sys.argv[sys.argv.index("--destination") + 1])
interpreter = destination / "payload" / "bin" / "python"
interpreter.parent.mkdir(parents=True)
interpreter.write_text(
    '''#!/bin/sh
if [ "${1:-}" = "-c" ]; then
    if [ -n "${NPA_TEST_IMPORT_STARTED:-}" ]; then
        : >"${NPA_TEST_IMPORT_STARTED}"
    fi
    case "${NPA_TEST_IMPORT_MODE:-success}" in
        fail) exit 42 ;;
        block) while :; do sleep 1; done ;;
        remove) rm -- "$0"; exit 0 ;;
    esac
    exit 0
fi
: >"${NPA_TEST_EXEC_STARTED}"
while :; do sleep 1; done
''',
    encoding="utf-8",
)
interpreter.chmod(0o755)
Path(os.environ["NPA_TEST_SNAPSHOT_RECORD"]).write_text(
    str(destination), encoding="utf-8"
)
""",
        encoding="utf-8",
    )
    source = (IMAGE_ROOT / "runtime_bootstrap.sh").read_text(encoding="utf-8")
    source = source.replace(
        'readonly verifier="/opt/npa/robomimic/verify_image.py"',
        f'readonly verifier="{fake_verifier}"',
    ).replace("/usr/local/bin/python3", sys.executable)
    script = tmp_path / "runtime_bootstrap.sh"
    script.write_text(source, encoding="utf-8")
    return script


def _wait_for_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                f"bootstrap exited before {path.name}: "
                f"status={process.returncode} stdout={stdout!r} stderr={stderr!r}"
            )
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate(timeout=5)
    pytest.fail(f"bootstrap did not create {path.name}")


def _exec_environment(tmp_path: Path, *, import_mode: str) -> dict[str, str]:
    return {
        **os.environ,
        "NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_TEST_SNAPSHOT_RECORD": str(tmp_path / "snapshot-record"),
        "NPA_TEST_IMPORT_STARTED": str(tmp_path / "import-started"),
        "NPA_TEST_IMPORT_MODE": import_mode,
        "NPA_TEST_EXEC_STARTED": str(tmp_path / "exec-started"),
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
    assert 'exec "${snapshot_root}/payload/bin/python"' in text
    assert "trap - EXIT" not in text


def test_bootstrap_cleans_snapshot_when_import_gate_fails(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(tmp_path, import_mode="fail"),
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == 42, result.stderr
    assert not snapshot_root.parent.exists()


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
    _wait_for_file(tmp_path / "import-started", process)
    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    os.killpg(process.pid, signal_number)
    process.communicate(timeout=5)

    assert process.returncode == expected_returncode
    assert not snapshot_root.parent.exists()


def test_bootstrap_cleans_snapshot_when_final_exec_fails(tmp_path: Path) -> None:
    script = _bootstrap_with_fake_snapshot(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "exec", "smoke.py"],
        check=False,
        capture_output=True,
        text=True,
        env=_exec_environment(tmp_path, import_mode="remove"),
    )

    snapshot_root = Path((tmp_path / "snapshot-record").read_text(encoding="utf-8"))
    assert result.returncode == 127, result.stderr
    assert not snapshot_root.parent.exists()


def test_successful_exec_keeps_snapshot_available_to_payload(tmp_path: Path) -> None:
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
    try:
        _wait_for_file(tmp_path / "exec-started", process)
        snapshot_root = Path(
            (tmp_path / "snapshot-record").read_text(encoding="utf-8")
        )
        assert process.poll() is None
        assert snapshot_root.is_dir()
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 143
        assert not snapshot_root.parent.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        if snapshot_root is not None:
            shutil.rmtree(snapshot_root.parent, ignore_errors=True)


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
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NPA_ROBOMIMIC_RUNTIME_REFUSAL_OK"
