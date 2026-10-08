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

pytestmark = pytest.mark.skipif(
    not os.environ.get("NPA_TEAM_SKY_PYTHON"),
    reason="Select an isolated SkyPilot 0.12.2 interpreter for the compatibility test",
)


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
