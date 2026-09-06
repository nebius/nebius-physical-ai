"""Real local processes/sockets verify isolated API ownership and recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from npa.orchestration.skypilot import local_api as api


@pytest.fixture
def local_runtime(tmp_path):
    package = tmp_path / "modules" / "sky" / "server"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "server.py").write_text('''import argparse,json,os,subprocess,sys,re,signal,socket,time\nfrom http.server import BaseHTTPRequestHandler,HTTPServer\np=argparse.ArgumentParser();p.add_argument('--host');p.add_argument('--port',type=int);p.add_argument('--metrics-port');a=p.parse_args()\nqueue=int(re.search(r'port: ([0-9]+)',open(os.environ['SKYPILOT_SERVER_PLUGINS_CONFIG']).read()).group(1))\nsubprocess.Popen([sys.executable,'-c',"import socket,signal,sys;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',int(sys.argv[1])));s.listen();signal.pause()",str(queue)])\ndef stop(signum,frame):\n time.sleep(.1)\n try:\n  with socket.create_connection(('127.0.0.1',queue)):\n   open(os.path.join(os.environ['HOME'],'queue-available-at-parent-exit'),'w').write('yes')\n except OSError:\n  pass\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  if self.path == '/spawn-unmarked':\n   subprocess.Popen([sys.executable,'-c','import signal;signal.pause()'],env={})\n  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'version':'0.12.2','status':'healthy'}).encode())\nHTTPServer.allow_reuse_address=True\nHTTPServer((a.host,a.port),H).serve_forever()\n''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "sky").touch()
    (bin_dir / "sky").chmod(0o700)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    env = {**os.environ, "HOME": str(isolated), "SKYPILOT_USER_ID": "fixture-isolated",
           "PYTHONPATH": str(package.parents[1]), "SKYPILOT_GLOBAL_CONFIG": str(config),
           "AWS_SECRET_ACCESS_KEY": "fixture-secret-value"}
    for key in ("SKYPILOT_API_SERVER_ENDPOINT", "SKYPILOT_DB_CONNECTION_URI", "SKYPILOT_SERVER_PLUGINS_CONFIG"):
        env.pop(key, None)
    env = api.isolated_api_environment(isolated, env)
    values = dict(isolated_dir=isolated, sky_executable=str(bin_dir / "sky"), environment=env, cwd=str(isolated))
    yield values
    api.stop_isolated_api(isolated)


def _record(runtime):
    return json.loads((runtime["isolated_dir"] / "local-api" / "daemon.json").read_text())


def test_real_listener_owned_and_same_process_adopted_on_retry(local_runtime):
    first = api.ensure_isolated_api(**local_runtime)
    record = _record(local_runtime)
    assert first["outcome"] == "owned_isolated_api"
    assert api._listener_owned(record, api._process(record))
    assert api.ensure_isolated_api(**local_runtime) == first
    assert _record(local_runtime)["pid"] == record["pid"]
    assert record["environment_binding"]["HOME"] == hashlib.sha256(local_runtime["environment"]["HOME"].encode()).hexdigest()
    assert "fixture-secret-value" not in json.dumps(record)
    assert len({record["port"], record["queue_port"], record["metrics_port"]}) == 3
    assert record["port"] not in {46580, 50011}
    assert (local_runtime["isolated_dir"] / "local-api" / "daemon.json").stat().st_mode & 0o777 == 0o600


def test_create_response_crash_recovers_exact_existing_process(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    record = _record(local_runtime)
    original_pid = record["pid"]
    record.update(pid=None, start_ticks=None, state="starting")
    api._write(local_runtime["isolated_dir"] / "local-api" / "daemon.json", record)
    api.ensure_isolated_api(**local_runtime)
    assert _record(local_runtime)["pid"] == original_pid


def test_stopped_owned_daemon_restarts_same_endpoint_and_scope(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    first = _record(local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    assert api._process(_record(local_runtime)) is None
    api.ensure_isolated_api(**local_runtime)
    second = _record(local_runtime)
    assert (second["port"], second["marker"]) == (first["port"], first["marker"])
    assert second["pid"] != first["pid"]


def test_foreign_listener_at_reserved_port_is_never_adopted_or_stopped(local_runtime):
    record = _record(local_runtime)
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", record["port"]))
        foreign.listen()
        with pytest.raises(api.IsolatedApiError, match="unowned listener"):
            api.ensure_isolated_api(**local_runtime)
        assert foreign.getsockname()[1] == record["port"]
        assert not _record(local_runtime).get("pid")


def test_live_scope_config_change_refuses_restart(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    first = _record(local_runtime)
    Path(local_runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]).write_text("nebius: {}\n")
    with pytest.raises(api.IsolatedApiError, match="different verified configuration"):
        api.ensure_isolated_api(**local_runtime)
    assert api._process(first)["pid"] == first["pid"]


def test_foreign_configured_endpoint_does_not_get_overwritten(tmp_path):
    with pytest.raises(api.IsolatedApiError, match="different configured API endpoint"):
        api.isolated_api_environment(tmp_path, {"SKYPILOT_API_SERVER_ENDPOINT": "http://127.0.0.1:46580"})
    assert not (tmp_path / "local-api" / "daemon.json").exists()


def test_external_database_rejected_before_process_creation(local_runtime):
    local_runtime["environment"]["SKYPILOT_DB_CONNECTION_URI"] = "secret-database-uri"
    with pytest.raises(api.IsolatedApiError, match="shared external database"):
        api.ensure_isolated_api(**local_runtime)
    assert not _record(local_runtime).get("pid")


def test_tampered_pid_is_not_signaled(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    invalid = {**original, "pid": os.getpid()}
    api._write(local_runtime["isolated_dir"] / "local-api" / "daemon.json", invalid)
    try:
        with pytest.raises(api.IsolatedApiError, match="process lifetime"):
            api.stop_isolated_api(local_runtime["isolated_dir"])
    finally:
        api._write(local_runtime["isolated_dir"] / "local-api" / "daemon.json", original)


def test_corrupt_ownership_record_cannot_fall_back_to_shared_api(local_runtime):
    path = local_runtime["isolated_dir"] / "local-api" / "daemon.json"
    original = path.read_text()
    path.write_text("broken")
    try:
        with pytest.raises(api.IsolatedApiError, match="ownership record is invalid"):
            api.isolated_api_environment(local_runtime["isolated_dir"], local_runtime["environment"])
    finally:
        path.write_text(original)


def test_status_environment_recovers_same_persistent_endpoint_without_submit(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    recovered = api.isolated_api_environment(local_runtime["isolated_dir"], local_runtime["environment"])
    current = _record(local_runtime)
    assert recovered["SKYPILOT_API_SERVER_ENDPOINT"] == api._endpoint(original)
    assert current["pid"] != original["pid"]
    assert current["marker"] == original["marker"]
    assert current["state"] == "ready"


def test_same_path_mutated_kubeconfig_is_not_same_identity(local_runtime):
    config = local_runtime["isolated_dir"] / "kubeconfig"
    config.write_text("fixture-cluster-identity")
    local_runtime["environment"]["KUBECONFIG"] = str(config)
    api.ensure_isolated_api(**local_runtime)
    config.write_text("different-cluster-identity")
    with pytest.raises(api.IsolatedApiError, match="credential configuration changed"):
        api.ensure_isolated_api(**local_runtime)
    # File changes do not remove our ownership of the process for safe cleanup.
    api.stop_isolated_api(local_runtime["isolated_dir"])


@pytest.mark.parametrize("setting", ["AWS_ENDPOINT_URL_S3", "AWS_REGION", "NEBIUS_PROFILE", "NPA_SKYPILOT_PROJECT"])
def test_all_effective_provider_settings_checked_on_adoption(local_runtime, setting):
    api.ensure_isolated_api(**local_runtime)
    local_runtime["environment"][setting] = "different-fixture-setting"
    with pytest.raises(api.IsolatedApiError, match="different executing identity"):
        api.ensure_isolated_api(**local_runtime)


def test_surviving_queue_child_blocks_duplicate_server_then_owned_cleanup(local_runtime):
    import signal

    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    assert len(api._session_members(original)) >= 2
    os.kill(original["pid"], signal.SIGKILL)
    # Wait for the child we killed, rather than asking ownership inspection to
    # classify /proc while the kernel is clearing a dying process's argv/env.
    try:
        os.waitpid(original["pid"], 0)
    except ChildProcessError:
        pass  # Popen's child reaper may already have collected the same exit.
    assert api._process(original) is None
    assert api._session_members(original)
    with pytest.raises(api.IsolatedApiError, match="children survived"):
        api.ensure_isolated_api(**local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    assert api._session_members(original) == []
    api.ensure_isolated_api(**local_runtime)
    assert _record(local_runtime)["pid"] != original["pid"]


def test_foreign_queue_listener_is_rejected_before_api_start(local_runtime):
    record = _record(local_runtime)
    with socket.socket() as foreign:
        foreign.bind(("127.0.0.1", record["queue_port"]))
        foreign.listen()
        with pytest.raises(api.IsolatedApiError, match="unowned listener"):
            api.ensure_isolated_api(**local_runtime)
        assert not _record(local_runtime).get("pid")


def test_fresh_workflow_status_restores_submit_only_resolved_storage_settings(local_runtime, monkeypatch):
    from npa.orchestration.skypilot import workflow
    from types import SimpleNamespace
    import subprocess

    local_runtime["environment"].update(NPA_S3_BUCKET="fixture-task-bucket", NPA_S3_PREFIX="fixture/task-prefix",
                                        AWS_REGION="fixture-region")
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    fresh = dict(local_runtime["environment"])
    for name in ("NPA_S3_BUCKET", "NPA_S3_PREFIX", "AWS_REGION"):
        fresh.pop(name)
    monkeypatch.setattr(workflow, "resolve_config", lambda **kwargs: SimpleNamespace(
        sky_bin=Path(local_runtime["sky_executable"]), isolated_config_dir=local_runtime["isolated_dir"], global_config_path=None))
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda value: value)
    monkeypatch.setattr(workflow, "sky_environment", lambda root: api.isolated_api_environment(root, fresh))
    def queue(argv, **kwargs):
        assert argv[1:3] == ["jobs", "queue"]
        assert kwargs["env"]["NPA_S3_BUCKET"] == "fixture-task-bucket"
        assert kwargs["env"]["NPA_S3_PREFIX"] == "fixture/task-prefix"
        assert _record(local_runtime)["pid"] != original["pid"]
        return subprocess.CompletedProcess(argv, 0, stdout='[{"job_id": 1, "status": "SUCCEEDED"}]', stderr="")
    monkeypatch.setattr(workflow.subprocess, "run", queue)
    outcome = workflow.workflow_status("1", isolated_config_dir=local_runtime["isolated_dir"])
    assert outcome.status == "SUCCEEDED"


def test_default_nebius_aws_profile_mutation_is_not_same_principal(local_runtime):
    profile = Path(local_runtime["environment"]["HOME"]) / ".aws" / "credentials"
    profile.parent.mkdir()
    profile.write_text("[nebius]\naws_access_key_id=fixture-one\n")
    api.ensure_isolated_api(**local_runtime)
    profile.write_text("[nebius]\naws_access_key_id=fixture-two\n")
    with pytest.raises(api.IsolatedApiError, match="credential configuration changed"):
        api.ensure_isolated_api(**local_runtime)


def test_invalid_credential_yaml_diagnostic_does_not_include_source_secret():
    with pytest.raises(api.IsolatedApiError) as raised:
        api._yaml_document("credentials: [fixture-secret-token")
    assert "fixture-secret-token" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("fail_transaction", [False, True])
def test_cleanup_clone_owns_api_preserves_identity_and_snapshots_live_wal(
    local_runtime, monkeypatch, fail_transaction
):
    import shutil
    import sqlite3
    from npa.orchestration.skypilot import cleanup

    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    source_home = Path(local_runtime["environment"]["HOME"])
    source_state = source_home / ".sky"
    source_state.mkdir()
    (source_state / "user_hash").write_text("fixture-isolated")
    connection = sqlite3.connect(source_state / "state.db")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE controllers (name TEXT)")
    connection.execute("INSERT INTO controllers VALUES ('controller-fixture')")
    connection.commit()
    assert (source_state / "state.db-wal").stat().st_size
    (source_state / "api_server").mkdir()
    with sqlite3.connect(source_state / "api_server/requests.db") as requests:
        requests.execute("CREATE TABLE requests (operation TEXT)")
        requests.execute("INSERT INTO requests VALUES ('pending_launch')")
    sky_bin = Path(local_runtime["sky_executable"])
    sky_bin.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "print(json.dumps({key:os.environ[key] for key in "
        "['HOME','SKYPILOT_USER_ID','SKYPILOT_API_SERVER_ENDPOINT','SKYPILOT_GLOBAL_CONFIG']}))\n"
    )
    sky_bin.chmod(0o700)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    clone = None
    clone_record = None
    try:
        try:
            with cleanup._cloned_skypilot_state(local_runtime["isolated_dir"], sky_bin=sky_bin) as clone:
                clone_record = json.loads((clone / "local-api/daemon.json").read_text())
                assert clone_record["pid"] != original["pid"]
                assert api._listener_owned(clone_record, api._process(clone_record))
                assert api._listener_owned(clone_record, api._process(clone_record), port=clone_record["queue_port"])
                client = cleanup._run([str(sky_bin), "status"], isolated_config_dir=clone,
                                      config_path=Path(local_runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]), timeout=10)
                assert client.returncode == 0
                selected = json.loads(client.stdout)
                assert selected["SKYPILOT_USER_ID"] == "fixture-isolated"
                assert selected["HOME"] == str(clone / "home")
                assert selected["SKYPILOT_API_SERVER_ENDPOINT"] == api._endpoint(clone_record)
                assert selected["SKYPILOT_GLOBAL_CONFIG"] == str(clone / "transaction-config.yaml")
                assert "fixture-secret-value" not in json.dumps(clone_record)
                assert not (clone / "home/.sky/api_server/requests.db").exists()
                with sqlite3.connect(clone / "home/.sky/state.db") as db:
                    assert db.execute("SELECT name FROM controllers").fetchall() == [("controller-fixture",)]
                    db.execute("DELETE FROM controllers")
                assert connection.execute("SELECT name FROM controllers").fetchall() == [("controller-fixture",)]
                if fail_transaction:
                    raise RuntimeError("fixture controller refusal")
        except RuntimeError as exc:
            assert fail_transaction and str(exc) == "fixture controller refusal"
        assert clone is not None and not clone.exists()
        assert clone_record is not None and not api._session_members(clone_record)
        assert api._process(original)["pid"] == original["pid"]
    finally:
        connection.close()
        if clone and clone.exists():
            api.stop_isolated_api(clone)
            shutil.rmtree(clone)


def test_cleanup_clone_preserves_ownership_files_when_owned_stop_fails(local_runtime, monkeypatch):
    import shutil
    from npa.orchestration.skypilot import cleanup

    api.ensure_isolated_api(**local_runtime)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    original_stop = api.stop_isolated_api
    clone = None

    def fail_clone_stop(path):
        if path != local_runtime["isolated_dir"]:
            raise api.IsolatedApiError("fixture process ownership uncertain")
        original_stop(path)

    monkeypatch.setattr(api, "stop_isolated_api", fail_clone_stop)
    try:
        with pytest.raises(api.IsolatedApiError, match="ownership uncertain"):
            with cleanup._cloned_skypilot_state(local_runtime["isolated_dir"], sky_bin=local_runtime["sky_executable"]) as clone:
                pass
        assert clone is not None and (clone / "local-api/daemon.json").is_file()
        record = json.loads((clone / "local-api/daemon.json").read_text())
        assert api._process(record)
        assert str(clone.resolve()) not in cleanup._TRANSACTION_ENVIRONMENTS.get()
    finally:
        if clone:
            original_stop(clone)
            shutil.rmtree(clone)


def test_cleanup_clone_rejects_different_kube_identity_before_daemon_creation(local_runtime, monkeypatch, tmp_path):
    from npa.orchestration.skypilot import cleanup

    kube = tmp_path / "original-kube.yaml"
    kube.write_text("{}\n")
    local_runtime["environment"]["KUBECONFIG"] = str(kube)
    api.ensure_isolated_api(**local_runtime)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    with pytest.raises(api.IsolatedApiError, match="original executing identity"):
        with cleanup._cloned_skypilot_state(local_runtime["isolated_dir"], sky_bin=local_runtime["sky_executable"], env_extra={"KUBECONFIG": str(tmp_path / "other-kube.yaml")}):
            pytest.fail("mismatched controller target must not execute")
    assert not (local_runtime["isolated_dir"] / "controller-transactions").exists()


def test_unmarked_executor_child_is_bound_by_lineage_and_persisted_lifetime(local_runtime):
    import signal
    from urllib.request import urlopen

    api.ensure_isolated_api(**local_runtime)
    record = _record(local_runtime)
    with urlopen(api._endpoint(record) + "/spawn-unmarked") as response:
        assert response.status == 200
    members = api._session_members(record)
    assert len(members) >= 3
    assert all(str(pid) in record["session_processes"] for pid in members)
    # This is the exact proof stop persists before signaling its leader.
    api._write(local_runtime["isolated_dir"] / "local-api/daemon.json", record)
    os.kill(record["pid"], signal.SIGKILL)
    try:
        os.waitpid(record["pid"], 0)
    except ChildProcessError:
        pass
    with pytest.raises(api.IsolatedApiError, match="children survived"):
        api.ensure_isolated_api(**local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    assert api._session_members(record) == []


def test_parent_shutdown_can_use_queue_before_owned_children_stop(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    record = _record(local_runtime)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    receipt = Path(local_runtime["environment"]["HOME"]) / "queue-available-at-parent-exit"
    assert receipt.read_text() == "yes"
    assert not api._session_members(record)


def test_stop_recovers_process_created_before_pid_was_saved(local_runtime):
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    interrupted = dict(original, pid=None, start_ticks=None, state="starting")
    api._write(local_runtime["isolated_dir"] / "local-api" / "daemon.json", interrupted)
    api.stop_isolated_api(local_runtime["isolated_dir"])
    assert _record(local_runtime)["state"] == "stopped"
    assert not api._session_members(original)
