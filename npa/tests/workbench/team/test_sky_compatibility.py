"""Check actual pinned SkyPilot behavior without cluster credentials or provisioning."""

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest
import yaml

from npa.workbench.team.deployment import server_config
from npa.workbench.team.ledger import TeamLedger
from npa.workbench.team.sky_backend import SkyBackend
from npa.workbench.team import sky_bridge

pytestmark = pytest.mark.skipif(
    not os.environ.get("NPA_TEAM_SKY_PYTHON"),
    reason="Select an isolated SkyPilot 0.12.2 interpreter for the compatibility test",
)


def test_pinned_sdk_forwards_workspace_for_first_cancel_call(config, binding, tmp_path):
    backend = SkyBackend(config, binding, TeamLedger(config.state_dir), "run-context")
    environment = backend._bridge_environment()
    program = r"""
import runpy, sys
import sky
from sky.server.requests import payloads
from unittest.mock import patch
import requests
response = requests.Response()
response.status_code = 200
response._content = b'{"status":"healthy","api_version":"1","version":"0.12.2"}'
def cancel(**kwargs):
    body = payloads.JobsCancelBody(**kwargs)
    assert body.override_skypilot_config['active_workspace'].startswith('team-')
    assert body.job_ids == [7]
    return 'request'
bridge = runpy.run_path(sys.argv[1])
with patch('sky.server.common.make_authenticated_request', return_value=response), patch('sky.jobs.cancel', side_effect=cancel), patch('sky.get', return_value=None):
    assert bridge['_execute']({'operation':'cancel','job_ids':[7]}) == {'cancel_requested':True}
"""
    result = subprocess.run(
        [os.environ["NPA_TEAM_SKY_PYTHON"], "-c", program, sky_bridge.__file__],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pinned_log_paths_expand_inside_the_run_home(config, binding, tmp_path):
    backend = SkyBackend(config, binding, TeamLedger(config.state_dir), "run-logs")
    program = r"""
import os, runpy, sys
from unittest.mock import patch
from pathlib import Path
bridge = runpy.run_path(sys.argv[1])
destination = Path.home() / 'sky_logs'
with patch('sky.api_info'), patch('sky.jobs.download_logs', return_value={7:'~/sky_logs/job/run'}):
    result = bridge['_execute']({'operation':'logs','job_id':7,'directory':str(destination)})
assert result['paths'][7] == str(destination / 'job/run')
"""
    result = subprocess.run(
        [os.environ["NPA_TEAM_SKY_PYTHON"], "-c", program, sky_bridge.__file__],
        env=backend._bridge_environment(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pinned_sdk_refreshes_rotated_kubernetes_token_files(tmp_path):
    program = _token_rotation_program()
    environment = {**os.environ, "SKYPILOT_KUBECONFIG_REFRESH_INTERVAL_SECONDS": "60"}
    result = subprocess.run(
        [os.environ["NPA_TEAM_SKY_PYTHON"], "-c", program],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _token_rotation_program():
    return r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from kubernetes import client, config
from sky.adaptors import kubernetes as adapter
with TemporaryDirectory() as directory:
    token = Path(directory) / 'token'
    token.write_text('initial-synthetic-token')
    configuration = {'apiVersion':'v1','kind':'Config','current-context':'test',
        'clusters':[{'name':'test','cluster':{'server':'https://unused.example.test'}}],
        'users':[{'name':'test','user':{'tokenFile':str(token)}}],
        'contexts':[{'name':'test','context':{'cluster':'test','user':'test'}}]}
    def getter():
        settings = client.Configuration()
        config.load_kube_config_from_dict(configuration, client_configuration=settings)
        return SimpleNamespace(read=lambda: settings.get_api_key_with_prefix('authorization'))
    stale = getter()
    wrapped = adapter.RetryableClientWrapper(getter(), getter, (), {})
    assert adapter._get_kubeconfig_refresh_interval_seconds() == 60
    assert wrapped.read() == 'Bearer initial-synthetic-token'
    token.write_text('rotated-synthetic-token')
    assert stale.read() == 'Bearer initial-synthetic-token'
    assert wrapped.read() == 'Bearer initial-synthetic-token'
    wrapped._last_refresh_time -= 61
    assert wrapped.read() == 'Bearer rotated-synthetic-token'
"""


def test_real_private_server_accepts_new_synthetic_identity(config, binding, tmp_path):
    executable = Path(os.path.abspath(os.environ["NPA_TEAM_SKY_PYTHON"]))
    settings = tmp_path / "sky-server.yaml"
    settings.write_text(yaml.safe_dump(server_config(config)))
    home = tmp_path / "sky-home"
    (home / ".sky").mkdir(parents=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "IS_SKYPILOT_SERVER": "true",
        "SKYPILOT_GLOBAL_CONFIG": str(settings),
        "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
        "KUBECONFIG": str(tmp_path / "no-cluster-credentials"),
    }
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "-c",
                _isolated_server_entrypoint(),
                str(port),
            ],
            env=environment,
            cwd=home,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            _ready(process, endpoint)
            selected = config.model_copy(
                update={"sky_python": executable, "sky_endpoint": endpoint}
            )
            backend = SkyBackend(
                selected, binding, TeamLedger(config.state_dir), "run-compatibility"
            )
            assert backend.call("queue", job_ids=[42]) == {"jobs": []}
        finally:
            _terminate(process)


def _isolated_server_entrypoint():
    # The public queue extension keeps the test away from an operator's local
    # SkyPilot queue-manager port. Production pods have their own network stack.
    return (
        "import runpy,sys; "
        "from sky.server.requests.queues.base import LocalQueueFactory,set_queue_backend_factory; "
        "set_queue_backend_factory(LocalQueueFactory()); "
        "sys.argv=['sky.server.server','--host','127.0.0.1','--port',sys.argv[1]]; "
        "runpy.run_module('sky.server.server',run_name='__main__')"
    )


def _ready(process, endpoint):
    opener = build_opener(ProxyHandler({}))
    while process.poll() is None:
        try:
            with opener.open(endpoint + "/api/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.1)
    raise AssertionError("private SkyPilot server exited before readiness")


def _terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    process.wait()
