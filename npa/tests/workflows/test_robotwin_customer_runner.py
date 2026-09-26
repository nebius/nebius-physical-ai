"""Customer receipts gate the existing RoboTwin launcher before side effects."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from npa.orchestration.npa_workflow import robotwin_customer as customer
from npa.orchestration.npa_workflow import robotwin_preflight as preflight

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa"))
from tests.orchestration.npa_workflow.test_robotwin_preflight import (  # noqa: E402
    _context_file,
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "robotwin_customer_runner_test", ROOT / "npa/scripts/run_robotwin_customer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, *, wrong_run=False):
    context, raw = _context_file(
        tmp_path,
        runtime_lock_sha256=preflight.RUNTIME_LOCK_SHA256,
        workflow_sha256=preflight.WORKFLOW_SHA256,
    )
    parsed = json.loads(raw)
    private = tmp_path / "customer"
    private.mkdir(mode=0o700)
    receipt = private / "synthetic-test-receipt.json"
    now = datetime.now(timezone.utc)
    value = {
        "schema_version": customer.SCHEMA,
        "issuer": customer.ISSUER,
        "customer_scope_id": parsed["customer_scope_id"],
        "run_id": "robotwin-other" if wrong_run else parsed["run_id"],
        "runtime_manifest_sha256": preflight.RUNTIME_LOCK_SHA256,
        "issued_at": (now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decision": "accepted",
        "intended_activity": customer.ACTIVITY,
        "terms": list(preflight.CUSTOMER_TERMS),
        "assertion_id": "customer-terminal-synthetic-fixture",
        "nonce": "f" * 64,
    }
    receipt.write_text(json.dumps(value))
    receipt.chmod(0o600)
    return context, receipt, parsed


@pytest.mark.parametrize("mode", ("absent", "wrong-run", "declined", "stale"))
def test_customer_refusal_precedes_storage_or_launch(tmp_path, monkeypatch, mode):
    module = _load()
    context, receipt, _ = _inputs(tmp_path, wrong_run=mode == "wrong-run")
    if mode == "absent":
        receipt.unlink()
    elif mode in {"declined", "stale"}:
        value = json.loads(receipt.read_text())
        value["decision" if mode == "declined" else "expires_at"] = (
            "declined" if mode == "declined" else "2000-01-01T00:00:00Z"
        )
        receipt.write_text(json.dumps(value))

    def forbidden(*args, **kwargs):
        pytest.fail("customer refusal must precede storage or launch")

    monkeypatch.setattr(module, "storage_env_for_project", forbidden)
    monkeypatch.setattr(module, "probe_runtime_inputs", forbidden)
    monkeypatch.setattr(module.runner, "_run_authorized_robotwin", forbidden)
    assert (
        module.main(["--context", str(context), "--customer-decision", str(receipt)])
        == 78
    )


@pytest.mark.parametrize("returncode", (0, 1))
def test_customer_runner_binds_config_and_project_storage(
    tmp_path, monkeypatch, returncode
):
    module = _load()
    context, receipt, parsed = _inputs(tmp_path)
    # Synthetic isolation test; no customer CLI, cloud, image, or runtime fetch.
    monkeypatch.setattr(preflight, "RUNTIME_LOCK_STATUS", "complete")
    probes = []

    def probe(path, *, expected_sha256):
        assert path == module.RUNTIME_LOCK
        assert expected_sha256 == preflight.RUNTIME_LOCK_SHA256
        probes.append(expected_sha256)

    monkeypatch.setattr(module, "probe_runtime_inputs", probe)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-canary")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "ambient-session-canary")

    def storage(project, *, allow_host_creds):
        assert probes == [preflight.RUNTIME_LOCK_SHA256]
        assert project == parsed["project"]
        assert allow_host_creds is False
        return {"AWS_ACCESS_KEY_ID": "selected-project-canary"}

    monkeypatch.setattr(module, "storage_env_for_project", storage)
    materialized = []

    def launch(argv, *, authorization):
        preflight._validate_materialized_config_binding(authorization)
        assert authorization.bootstrap_image == parsed["bootstrap_image"]
        assert authorization.kubeconfig_source != parsed["kubeconfig"]
        assert os.environ["AWS_ACCESS_KEY_ID"] == "selected-project-canary"
        assert "AWS_SESSION_TOKEN" not in os.environ
        assert argv[argv.index("--wait-timeout") + 1] == "-1"
        assert "--cleanup" in argv
        assert "--skip-build" not in argv
        materialized.append(Path(authorization.kubeconfig_source).parent)
        return returncode

    monkeypatch.setattr(module.runner, "_run_authorized_robotwin", launch)
    assert module.run(context, receipt) == returncode
    assert os.environ["AWS_ACCESS_KEY_ID"] == "ambient-canary"
    assert os.environ["AWS_SESSION_TOKEN"] == "ambient-session-canary"
    assert materialized[0].exists() is bool(returncode)
    if returncode:
        import shutil

        shutil.rmtree(materialized[0])


def test_unavailable_runtime_stops_before_storage_or_launch(tmp_path, monkeypatch):
    module = _load()
    context, receipt, _ = _inputs(tmp_path)
    monkeypatch.setattr(preflight, "RUNTIME_LOCK_STATUS", "complete")

    def unavailable(*args, **kwargs):
        raise customer.CustomerDecisionError("upstream-payload-unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable runtime must not reach storage or GPU launch")

    monkeypatch.setattr(module, "probe_runtime_inputs", unavailable)
    monkeypatch.setattr(module, "storage_env_for_project", forbidden)
    monkeypatch.setattr(module.runner, "_run_authorized_robotwin", forbidden)
    assert (
        module.main(["--context", str(context), "--customer-decision", str(receipt)])
        == 78
    )


def _tracked_staging(module, tmp_path, monkeypatch):
    directories = []
    original = module.tempfile.mkdtemp

    def create(*, prefix):
        directory = Path(original(prefix=prefix, dir=tmp_path))
        assert directory.stat().st_mode & 0o777 == 0o700
        directories.append(directory)
        return str(directory)

    monkeypatch.setattr(module.tempfile, "mkdtemp", create)
    return directories


@pytest.mark.parametrize("failure_stage", ("materialization", "storage"))
def test_prelaunch_failure_removes_private_staging(
    tmp_path, monkeypatch, capsys, failure_stage
):
    module = _load()
    context, receipt, _ = _inputs(tmp_path)
    monkeypatch.setattr(preflight, "RUNTIME_LOCK_STATUS", "complete")
    monkeypatch.setattr(module, "probe_runtime_inputs", lambda *args, **kwargs: None)
    directories = _tracked_staging(module, tmp_path, monkeypatch)
    materialize = module._materialize_authorization

    def fail_materialization(authorization, directory):
        materialize(authorization, directory)
        raise RuntimeError("materialization failed after writing credentials")

    def storage(*args, **kwargs):
        raise RuntimeError("project storage unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("pre-launch failure must not reach the launcher")

    if failure_stage == "materialization":
        monkeypatch.setattr(module, "_materialize_authorization", fail_materialization)
    monkeypatch.setattr(module, "storage_env_for_project", storage)
    monkeypatch.setattr(module.runner, "_run_authorized_robotwin", forbidden)
    with pytest.raises(RuntimeError):
        module.run(context, receipt)
    assert len(directories) == 1
    assert not directories[0].exists()
    assert context.exists() and receipt.exists()
    assert "recovery configuration retained" not in capsys.readouterr().err


def test_uncertain_launch_failure_retains_private_recovery_config(
    tmp_path, monkeypatch, capsys
):
    module = _load()
    context, receipt, _ = _inputs(tmp_path)
    monkeypatch.setattr(preflight, "RUNTIME_LOCK_STATUS", "complete")
    monkeypatch.setattr(module, "probe_runtime_inputs", lambda *args, **kwargs: None)
    directories = _tracked_staging(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "storage_env_for_project", lambda *args, **kwargs: {})
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-storage-canary")
    captured = []

    def uncertain_launch(argv, *, authorization):
        preflight._validate_materialized_config_binding(authorization)
        captured.append(authorization)
        raise RuntimeError("submission outcome unavailable")

    monkeypatch.setattr(module.runner, "_run_authorized_robotwin", uncertain_launch)
    with pytest.raises(RuntimeError, match="submission outcome unavailable"):
        module.run(context, receipt)
    assert len(directories) == len(captured) == 1
    preflight._validate_materialized_config_binding(captured[0])
    assert directories[0].is_dir()
    assert context.exists() and receipt.exists()
    assert os.environ["AWS_ACCESS_KEY_ID"] == "ambient-storage-canary"
    assert "recovery configuration retained" in capsys.readouterr().err


def test_provider_exception_does_not_disclose_private_values(
    tmp_path, monkeypatch, capsys
):
    module = _load()

    def fail(*args):
        raise RuntimeError("private-endpoint-and-credential-canary")

    monkeypatch.setattr(module, "run", fail)
    assert module.main(["--context", "context", "--customer-decision", "receipt"]) == 1
    assert "private-endpoint-and-credential-canary" not in capsys.readouterr().out
