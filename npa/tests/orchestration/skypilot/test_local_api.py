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


@pytest.mark.parametrize("resolver", ["resolve_project_storage", "resolve_terraform_state"])
def test_project_resolution_preserves_verified_api_but_rotation_is_rejected(
    local_runtime, tmp_path, monkeypatch, resolver,
):
    from npa.clients import config, credentials, project_credential_store as store

    directory = tmp_path / "credentials"
    path = directory / "credentials.yaml"
    monkeypatch.setattr(credentials, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(store, "_now", lambda: "2025-01-01T00:00:00+00:00")
    store.write_project_credentials(
        "project-fixture",
        {"storage": {"bucket": "fixture-bucket", "aws_access_key_id": "fixture-access",
                     "aws_secret_access_key": "fixture-secret"}},
        alias="fixture",
    )
    # A valid operator-formatted store must retain its exact verified bytes.
    path.write_text("# Operator credential store\n" + path.read_text())
    path.chmod(0o644)
    config_path = tmp_path / "npa-config.yaml"
    config_path.write_text("projects:\n  fixture:\n    project_id: project-fixture\n")
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    local_runtime["environment"]["NPA_CONFIG_DIR"] = str(directory)
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    verified_bytes = path.read_bytes()
    monkeypatch.setattr(store, "_now", lambda: "2025-01-02T00:00:00+00:00")

    for _ in range(2):
        getattr(config, resolver)("fixture")
        assert path.read_bytes() == verified_bytes
        assert path.stat().st_mode & 0o777 == 0o600
        api.ensure_isolated_api(**local_runtime)
        assert _record(local_runtime)["pid"] == original["pid"]

    store.write_project_credentials(
        "project-fixture", {"storage": {"aws_secret_access_key": "rotated-fixture-secret"}},
        alias="fixture",
    )
    with pytest.raises(api.IsolatedApiError, match="credential configuration changed"):
        api.ensure_isolated_api(**local_runtime)
    assert api._process(original, verify_files=False)["pid"] == original["pid"]


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
@pytest.mark.parametrize("runtime_location", ["unset", "absolute", "home_relative"])
def test_cleanup_clone_owns_api_preserves_identity_and_snapshots_live_wal(
    local_runtime, monkeypatch, fail_transaction, runtime_location
):
    import shutil
    import sqlite3
    from npa.orchestration.skypilot import cleanup

    source_home = Path(local_runtime["environment"]["HOME"])
    source_runtime = source_home
    local_runtime["environment"].pop("SKY_RUNTIME_DIR", None)
    if runtime_location == "absolute":
        source_runtime = source_home.parent / "server-runtime"
        local_runtime["environment"]["SKY_RUNTIME_DIR"] = str(source_runtime)
    elif runtime_location == "home_relative":
        source_runtime = source_home / "server-runtime"
        local_runtime["environment"]["SKY_RUNTIME_DIR"] = "~/server-runtime"
    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    home_state = source_home / ".sky"
    home_state.mkdir(exist_ok=True)
    (home_state / "user_hash").write_text("fixture-isolated")
    source_state = source_runtime / ".sky"
    source_state.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(source_state / "state.db")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE controllers (name TEXT)")
    connection.execute("INSERT INTO controllers VALUES ('controller-fixture')")
    connection.commit()
    assert (source_state / "state.db-wal").stat().st_size
    for state in {home_state, source_state}:
        (state / "api_server").mkdir()
        with sqlite3.connect(state / "api_server/requests.db") as requests:
            requests.execute("CREATE TABLE requests (operation TEXT)")
            requests.execute("INSERT INTO requests VALUES ('pending_launch')")
    sky_bin = Path(local_runtime["sky_executable"])
    sky_bin.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "print(json.dumps({key:os.environ[key] for key in "
        "['HOME','SKY_RUNTIME_DIR','SKYPILOT_USER_ID','SKYPILOT_API_SERVER_ENDPOINT','SKYPILOT_GLOBAL_CONFIG']}))\n"
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
                assert (clone / "home/.sky/user_hash").read_text() == "fixture-isolated"
                # The pinned Sky database manager selects SKY_RUNTIME_DIR,
                # independently of the client identity stored under HOME.
                cloned_state = Path(selected["SKY_RUNTIME_DIR"]) / ".sky"
                assert cloned_state.is_relative_to(clone)
                assert not (cloned_state / "api_server/requests.db").exists()
                assert (cloned_state / "state.db").is_file()
                with sqlite3.connect(cloned_state / "state.db") as db:
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


@pytest.mark.parametrize("runtime_value", ["", "relative-runtime", "~other-user/runtime"])
def test_cleanup_clone_refuses_ambiguous_runtime_before_transaction(
    local_runtime, monkeypatch, runtime_value,
):
    from npa.orchestration.skypilot import cleanup

    local_runtime["environment"]["SKY_RUNTIME_DIR"] = runtime_value
    api.ensure_isolated_api(**local_runtime)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    with pytest.raises(api.IsolatedApiError, match="absolute source runtime"):
        with cleanup._cloned_skypilot_state(
            local_runtime["isolated_dir"], sky_bin=local_runtime["sky_executable"],
        ):
            pytest.fail("ambiguous runtime must not start a transaction")
    assert not (local_runtime["isolated_dir"] / "controller-transactions").exists()


@pytest.mark.parametrize("helper_failure", [False, True])
def test_controller_metadata_verified_before_clone_api_starts(
    local_runtime, monkeypatch, helper_failure,
):
    import subprocess
    from npa.orchestration.skypilot import cleanup

    api.ensure_isolated_api(**local_runtime)
    original = _record(local_runtime)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    actual_run = subprocess.run
    prepared = []

    def run(argv, **kwargs):
        if len(argv) < 2 or not str(argv[1]).endswith("controller_clone.py"):
            return actual_run(argv, **kwargs)
        path = Path(argv[2])
        data = json.loads(path.read_text())
        root = Path(data["clone_root"])
        assert path.stat().st_mode & 0o777 == 0o600
        assert data["source_home"] == local_runtime["environment"]["HOME"]
        assert data["controller_names"] == ["sky-jobs-controller-fixture"]
        assert data["context"] == "fixture-context"
        assert kwargs["env"]["HOME"] == str(root / "home")
        assert kwargs["env"]["SKY_RUNTIME_DIR"] == str(root / "sky-runtime")
        assert not json.loads((root / "local-api/daemon.json").read_text()).get("pid")
        prepared.append(root)
        return subprocess.CompletedProcess(argv, int(helper_failure), b"", b"private diagnostic")

    monkeypatch.setattr(subprocess, "run", run)
    try:
        with cleanup._cloned_skypilot_state(
            local_runtime["isolated_dir"], sky_bin=local_runtime["sky_executable"],
            controller_names=["sky-jobs-controller-fixture"], context="fixture-context",
        ) as root:
            assert not helper_failure
            assert prepared == [root]
            assert json.loads((root / "local-api/daemon.json").read_text())["pid"]
    except api.IsolatedApiError as exc:
        assert helper_failure
        assert str(exc) == "controller transaction could not verify its copied controller metadata"
    assert len(prepared) == 1 and not prepared[0].exists()
    assert api._process(original)["pid"] == original["pid"]


@pytest.mark.parametrize("linked_component", ["runtime_root", "parent", "metadata"])
def test_cleanup_clone_refuses_linked_runtime_metadata(
    local_runtime, monkeypatch, tmp_path, linked_component,
):
    from npa.orchestration.skypilot import cleanup

    original_runtime = tmp_path / "original-runtime"
    original_runtime.mkdir()
    original_state = original_runtime / ".sky"
    original_state.mkdir()
    sentinel = original_state / "sentinel"
    sentinel.write_text("original-controller-state")
    if linked_component == "runtime_root":
        selected = tmp_path / "runtime-link"
        selected.symlink_to(original_runtime, target_is_directory=True)
    elif linked_component == "parent":
        parent = tmp_path / "parent-link"
        parent.symlink_to(tmp_path, target_is_directory=True)
        selected = parent / original_runtime.name
    else:
        selected = tmp_path / "selected-runtime"
        selected.mkdir()
        (selected / ".sky").symlink_to(original_state, target_is_directory=True)
    local_runtime["environment"]["SKY_RUNTIME_DIR"] = str(selected)
    api.ensure_isolated_api(**local_runtime)
    monkeypatch.setattr(cleanup, "ensure_skypilot_version", lambda value: Path(value))
    with pytest.raises(api.IsolatedApiError, match="linked source runtime"):
        with cleanup._cloned_skypilot_state(
            local_runtime["isolated_dir"], sky_bin=local_runtime["sky_executable"],
        ):
            pytest.fail("linked runtime metadata must not start a transaction")
    assert sentinel.read_text() == "original-controller-state"
    transactions = local_runtime["isolated_dir"] / "controller-transactions"
    assert not transactions.exists() or not list(transactions.iterdir())


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


@pytest.fixture
def service_account_runtime(local_runtime):
    import yaml
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    home = Path(local_runtime["environment"]["HOME"])
    provider = home / ".nebius"
    provider.mkdir()
    key = home / "selected-private.pem"
    key.write_bytes(rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    profile = {"auth-type": "service account", "service-account-id": "fixture-account",
               "public-key-id": "fixture-key", "private-key-file-path": str(key),
               "endpoint": "fixture.invalid:443", "parent-id": "fixture-project", "tenant-id": "fixture-tenant"}
    (provider / "config.yaml").write_text(yaml.safe_dump({"default": "selected", "profiles": {"selected": profile}}))
    cache = provider / "credentials.yaml"
    cache.write_text(yaml.safe_dump({"tokens": {
        "service-account/fixture-account/fixture-key": {"token": "fixture-old-bearer", "expires_at": 100},
        "service-account/retired-account/retired-key": {"token": "fixture-unrelated-bearer", "expires_at": 50}}}))
    local_runtime["environment"].update(NEBIUS_CONFIG_DIR=str(provider), NPA_CONFIG_DIR=str(home / ".npa"))
    return local_runtime, provider, key, cache


@pytest.mark.parametrize("initial_cache", ["populated", "absent", "empty"])
def test_service_account_cache_refresh_creation_pruning_preserves_owned_pid(service_account_runtime, initial_cache):
    runtime, provider, key, cache = service_account_runtime
    if initial_cache == "absent":
        cache.unlink()
    elif initial_cache == "empty":
        cache.write_text("tokens: {}\n")
    api.ensure_isolated_api(**runtime)
    original = _record(runtime)
    cache.write_text("tokens:\n  service-account/fixture-account/fixture-key:\n    token: fixture-new-bearer\n    expires_at: 200\n")
    assert api.ensure_isolated_api(**runtime)["healthy"]
    assert api._process(original)["pid"] == _record(runtime)["pid"] == original["pid"]
    assert original["identity_files"][str(key)] == hashlib.sha256(key.read_bytes()).hexdigest()
    assert original["identity_files"][str(provider / "config.yaml")] == hashlib.sha256((provider / "config.yaml").read_bytes()).hexdigest()
    assert "fixture-old-bearer" not in json.dumps(original)
    assert "fixture-new-bearer" not in json.dumps(_record(runtime))
    assert key.read_text() not in json.dumps(original)


def test_service_account_private_key_replacement_rejects_owned_api(service_account_runtime):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    runtime, _, key, _ = service_account_runtime
    api.ensure_isolated_api(**runtime)
    original = _record(runtime)
    key.write_bytes(rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    with pytest.raises(api.IsolatedApiError, match="credential configuration changed"):
        api.ensure_isolated_api(**runtime)
    assert api._process(original, verify_files=False)["pid"] == original["pid"]


@pytest.mark.parametrize("change", ["account", "public-key", "endpoint", "key-source", "missing-key", "invalid-key"])
def test_service_account_durable_auth_change_cannot_adopt(service_account_runtime, change):
    import yaml

    runtime, provider, key, _ = service_account_runtime
    api.ensure_isolated_api(**runtime)
    record_path = runtime["isolated_dir"] / "local-api/daemon.json"
    original_record = record_path.read_bytes()
    if change == "missing-key":
        key.unlink()
    elif change == "invalid-key":
        key.write_text("fixture-private-material-must-not-appear-in-error")
    else:
        config = provider / "config.yaml"
        value = yaml.safe_load(config.read_text())
        field = {"account": "service-account-id", "public-key": "public-key-id",
                 "endpoint": "endpoint", "key-source": "private-key-file-path"}[change]
        if change == "key-source":
            replacement = key.with_name("same-bytes-other-path.pem")
            replacement.write_bytes(key.read_bytes())
            value["profiles"]["selected"][field] = str(replacement)
        else:
            value["profiles"]["selected"][field] = "different-fixture-value"
        config.write_text(yaml.safe_dump(value))
    with pytest.raises(api.IsolatedApiError) as raised:
        api.ensure_isolated_api(**runtime)
    assert "fixture-private-material" not in str(raised.value)
    assert record_path.read_bytes() == original_record


@pytest.mark.parametrize("document", [
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, expires_at: true}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, expires_at: 1.5}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, expires_at: null}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, expires_at: -1}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: '', expires_at: 2}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: {}, expires_at: 2}}",
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, expires_at: 2, extra: fixture-x}}",
    "tokens: {federation/fixture: {token: fixture-x, expires_at: 2}}",
    "tokens: {service-account//fixture-key: {token: fixture-x, expires_at: 2}}",
    "tokens: {}\nextra: fixture-x",
    "tokens: {}\ntokens: {}",
    "tokens: {service-account/fixture-account/fixture-key: {token: fixture-x, token: fixture-y, expires_at: 2}}",
    "tokens: [fixture-secret-invalid",
])
def test_unknown_or_malformed_cache_never_uses_derived_identity(service_account_runtime, document):
    runtime, _, _, cache = service_account_runtime
    before = api._identity_files(runtime["environment"])
    cache.write_text(document)
    after = api._identity_files(runtime["environment"])
    assert after[str(cache)] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert before != after
    assert "fixture-secret-invalid" not in json.dumps(after)


def _select_nebius_exec(runtime, args, env=None):
    import yaml

    kube = runtime["isolated_dir"] / "selected-kube.yaml"
    kube.write_text(yaml.safe_dump({"current-context": "fixture-context",
        "contexts": [{"name": "fixture-context", "context": {"user": "fixture-user", "cluster": "fixture-cluster"}}],
        "users": [{"name": "fixture-user", "user": {"exec": {"command": "nebius", "args": args, "env": env}}}]}))
    runtime["environment"]["KUBECONFIG"] = str(kube)
    return kube


@pytest.mark.parametrize("selection", ["explicit", "equals", "short", "exec-env", "env", "config"])
def test_effective_nebius_profile_and_config_precedence(service_account_runtime, selection):
    import yaml

    runtime, provider, key, cache = service_account_runtime
    path = provider / "config.yaml"
    data = yaml.safe_load(path.read_text())
    data["profiles"]["other"] = {"auth-type": "federation", "federation-id": "fixture-federation"}
    data["default"] = "other"
    path.write_text(yaml.safe_dump(data))
    args = ["mk8s", "get-token", "--format", "json"]
    exec_env = None
    if selection == "explicit":
        args += ["--profile", "selected"]
        runtime["environment"]["NEBIUS_PROFILE"] = "other"
    elif selection == "equals":
        args += ["--profile=selected"]
    elif selection == "short":
        args += ["-p", "selected"]
    elif selection == "exec-env":
        exec_env = [{"name": "NEBIUS_PROFILE", "value": "selected"}]
        runtime["environment"]["NEBIUS_PROFILE"] = "other"
    elif selection == "env":
        runtime["environment"]["NEBIUS_PROFILE"] = "selected"
    else:
        custom = runtime["isolated_dir"] / "custom-cli.yaml"
        data["default"] = "selected"
        custom.write_text(yaml.safe_dump(data))
        args += ["--config", str(custom)]
    _select_nebius_exec(runtime, args, exec_env)
    before = api._identity_files(runtime["environment"])
    cache.write_text("tokens: {}\n")
    assert api._identity_files(runtime["environment"]) == before
    assert before[str(key)] == hashlib.sha256(key.read_bytes()).hexdigest()
    if selection == "config":
        assert str(custom) in before
        custom.write_text(custom.read_text() + "# durable config changed\n")
        assert api._identity_files(runtime["environment"]) != before


@pytest.mark.parametrize("unsupported", ["federation", "extra-auth", "duplicate-config", "relative-key", "relative-config", "tilde-key", "tilde-config", "duplicate-selector", "impersonate", "compact-impersonate", "compact-selector", "endpoint", "exec-home", "multiple-profiles", "missing-user", "mixed-missing-user", "missing-context", "mixed-missing-context"])
def test_unsupported_auth_selection_keeps_full_cache_fingerprint(service_account_runtime, unsupported):
    import yaml

    runtime, provider, _, cache = service_account_runtime
    path = provider / "config.yaml"
    data = yaml.safe_load(path.read_text())
    profile = data["profiles"]["selected"]
    if unsupported == "federation":
        profile["auth-type"] = "federation"
    elif unsupported == "extra-auth":
        profile["token-file"] = "fixture-token-file"
    elif unsupported == "relative-key":
        profile["private-key-file-path"] = "selected-private.pem"
    elif unsupported == "tilde-key":
        profile["private-key-file-path"] = "~/selected-private.pem"
    elif unsupported == "endpoint":
        runtime["environment"]["NEBIUS_ENDPOINT"] = "other-fixture.invalid:443"
    path.write_text(yaml.safe_dump(data))
    if unsupported == "duplicate-config":
        path.write_text(path.read_text() + "default: selected\n")
    args = {"relative-config": ["--config", "relative.yaml"], "tilde-config": ["--config", "~/.nebius/config.yaml"],
            "duplicate-selector": ["--profile", "selected", "-p", "selected"], "impersonate": ["-I", "fixture-other-account"],
            "compact-impersonate": ["-Ifixture-other-account"], "compact-selector": ["-pother"]}.get(unsupported, [])
    if args or unsupported == "exec-home":
        _select_nebius_exec(runtime, args, [{"name": "HOME", "value": "/fixture-other-home"}] if unsupported == "exec-home" else None)
    if unsupported == "multiple-profiles":
        data["profiles"]["other"] = dict(profile, **{"service-account-id": "fixture-other-account"})
        path.write_text(yaml.safe_dump(data))
        kube = _select_nebius_exec(runtime, ["--profile", "selected"])
        body = yaml.safe_load(kube.read_text())
        body["contexts"].append({"name": "other-context", "context": {"user": "other-user", "cluster": "fixture-cluster"}})
        body["users"].append({"name": "other-user", "user": {"exec": {"command": "nebius", "args": ["--profile", "other"]}}})
        kube.write_text(yaml.safe_dump(body))
        Path(runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]).write_text(yaml.safe_dump({"kubernetes": {"allowed_contexts": ["fixture-context", "other-context"]}}))
    if unsupported in {"missing-user", "mixed-missing-user", "missing-context", "mixed-missing-context"}:
        kube = _select_nebius_exec(runtime, ["--profile", "selected"])
        body = yaml.safe_load(kube.read_text())
        if unsupported == "missing-user":
            body["users"] = []
        elif unsupported == "missing-context":
            body["current-context"] = "absent-context"
        elif unsupported == "mixed-missing-context":
            Path(runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]).write_text(yaml.safe_dump({"kubernetes": {"allowed_contexts": ["fixture-context", "absent-context"]}}))
        else:
            body["contexts"].append({"name": "missing-user-context", "context": {"user": "missing-user", "cluster": "fixture-cluster"}})
            Path(runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]).write_text(yaml.safe_dump({"kubernetes": {"allowed_contexts": ["fixture-context", "missing-user-context"]}}))
        kube.write_text(yaml.safe_dump(body))
    before = api._identity_files(runtime["environment"])
    assert before[str(cache)] == hashlib.sha256(cache.read_bytes()).hexdigest()
    cache.write_text("tokens: {}\n")
    assert api._identity_files(runtime["environment"]) != before


@pytest.mark.parametrize("role", ["NPA_CONFIG_DIR", "AWS_SHARED_CREDENTIALS_FILE", "NEBIUS_IAM_TOKEN_FILE", "native"])
def test_explicit_credential_file_cannot_be_reclassified_as_provider_cache(service_account_runtime, role):
    runtime, provider, _, cache = service_account_runtime
    if role == "native":
        Path(runtime["environment"]["SKYPILOT_GLOBAL_CONFIG"]).write_text(json.dumps({"workspaces": {"default": {"nebius": {"credentials_file_path": str(cache)}}}}))
    else:
        runtime["environment"][role] = str(provider if role == "NPA_CONFIG_DIR" else cache)
    assert api._identity_files(runtime["environment"])[str(cache)] == hashlib.sha256(cache.read_bytes()).hexdigest()


def test_service_account_legacy_full_hash_record_is_not_silently_rebound(service_account_runtime):
    runtime, _, _, cache = service_account_runtime
    api.ensure_isolated_api(**runtime)
    record = _record(runtime)
    record["identity_files"][str(cache)] = hashlib.sha256(cache.read_bytes()).hexdigest()
    record_path = runtime["isolated_dir"] / "local-api/daemon.json"
    api._write(record_path, record)
    legacy = record_path.read_bytes()
    with pytest.raises(api.IsolatedApiError, match="credential configuration changed"):
        api.ensure_isolated_api(**runtime)
    assert record_path.read_bytes() == legacy


@pytest.mark.parametrize("role", ["npa-config", "npa-json", "npa-token", "provider-json", "provider-token"])
def test_nonderived_identity_alias_remains_byte_strict(service_account_runtime, role):
    runtime, provider, _, cache = service_account_runtime
    base = Path(runtime["environment"]["NPA_CONFIG_DIR"]) if role.startswith("npa-") else provider
    base.mkdir(exist_ok=True)
    name = {"npa-config": "config.yaml", "npa-json": "credentials.json", "npa-token": "NEBIUS_IAM_TOKEN.txt",
            "provider-json": "credentials.json", "provider-token": "NEBIUS_IAM_TOKEN.txt"}[role]
    alias = base / name
    alias.symlink_to(cache)
    before = api._identity_files(runtime["environment"])
    assert before[str(alias)] == before[str(cache)] == hashlib.sha256(cache.read_bytes()).hexdigest()
    cache.write_text("tokens: {}\n")
    assert api._identity_files(runtime["environment"]) != before


def test_designated_provider_cache_alias_uses_same_durable_identity(service_account_runtime):
    runtime, provider, _, cache = service_account_runtime
    alias = runtime["isolated_dir"] / "provider-alias"
    alias.symlink_to(provider, target_is_directory=True)
    runtime["environment"]["NEBIUS_CONFIG_DIR"] = str(alias)
    before = api._identity_files(runtime["environment"])
    assert before[str(alias / "credentials.yaml")] == before[str(cache)]
    cache.write_text("tokens: {}\n")
    assert api._identity_files(runtime["environment"]) == before


def test_unreadable_selected_key_fails_without_credential_diagnostics(service_account_runtime, monkeypatch):
    runtime, _, key, _ = service_account_runtime
    original = Path.read_bytes

    def unreadable(path):
        if path == key:
            raise PermissionError("fixture-secret-error-must-not-leak")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(api.IsolatedApiError) as raised:
        api._identity_files(runtime["environment"])
    assert "fixture-secret-error" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_non_rsa_selected_key_cannot_enable_derived_cache(service_account_runtime):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    runtime, _, key, _ = service_account_runtime
    key.write_bytes(ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    with pytest.raises(api.IsolatedApiError, match="private key cannot be verified"):
        api._identity_files(runtime["environment"])
