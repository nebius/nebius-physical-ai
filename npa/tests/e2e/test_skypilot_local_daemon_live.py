"""Opt-in real local SkyPilot repair, without cloud credentials or workloads.

Run in a fresh PID/network namespace with loopback enabled. Set
NPA_E2E_SKYPILOT_LOCAL_DAEMON=1 and NPA_SKYPILOT_BIN to the pinned isolated
SkyPilot executable. The test refuses to start beside an existing API daemon.
"""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import pytest

from npa.orchestration.skypilot._bin import ensure_skypilot_version
from npa.orchestration.skypilot.workflow import (
    _ensure_local_api_daemon_cwd,
    _probe_local_api_daemon_cwd,
)


pytestmark = pytest.mark.e2e_skypilot


@pytest.mark.parametrize("stale_key", ["HOME", "SKYPILOT_USER_ID", "KUBECONFIG"])
def test_real_local_daemon_repairs_environment_before_status(tmp_path, stale_key) -> None:
    if os.environ.get("NPA_E2E_SKYPILOT_LOCAL_DAEMON") != "1":
        pytest.skip("NPA_E2E_SKYPILOT_LOCAL_DAEMON not set")
    sky_bin = os.environ.get("NPA_SKYPILOT_BIN", "")
    if not sky_bin:
        pytest.fail("Set NPA_SKYPILOT_BIN to the isolated pinned runtime")
    assert Path("/proc").is_dir(), "Live daemon repair requires Linux procfs"
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            cmdline = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        assert not (b"-m" in cmdline and b"sky.server.server" in cmdline), (
            "Refusing live validation beside an existing API daemon"
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 46580))

    tmp_path.chmod(0o700)
    home = tmp_path / "home"
    stale_home = tmp_path / "stale-home"
    runtime = tmp_path / "runtime"
    for directory in (home, stale_home, runtime):
        directory.mkdir(mode=0o700)
    kubeconfig = tmp_path / "kubeconfig"
    stale_kubeconfig = tmp_path / "stale-kubeconfig"
    for config in (kubeconfig, stale_kubeconfig):
        config.write_text("apiVersion: v1\nkind: Config\nclusters: []\ncontexts: []\nusers: []\n")
    # Allowlist local process essentials; no operator credentials or cloud
    # configuration enter the real SkyPilot process tree.
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "SKY_RUNTIME_DIR": str(runtime),
        "SKYPILOT_USER_ID": "local-daemon-validation",
        "KUBECONFIG": str(kubeconfig),
        "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
    }
    stale_env = {
        **env,
        stale_key: {
            "HOME": str(stale_home),
            "SKYPILOT_USER_ID": "stale-local-daemon-validation",
            "KUBECONFIG": str(stale_kubeconfig),
        }[stale_key],
    }
    ensure_skypilot_version(sky_bin)
    started = False
    try:
        start = subprocess.run(
            [sky_bin, "api", "start"], env=stale_env, cwd=tmp_path,
            text=True, capture_output=True, check=False,
        )
        started = True
        assert start.returncode == 0, start.stderr or start.stdout
        before = _probe_local_api_daemon_cwd(
            sky_bin, expected_home=env["HOME"],
            expected_user_id=env["SKYPILOT_USER_ID"],
            expected_kubeconfig=env["KUBECONFIG"],
            expected_runtime_dir=env["SKY_RUNTIME_DIR"],
        )
        assert not before.healthy and before.outcome == "stale_runtime_environment"
        repaired = _ensure_local_api_daemon_cwd(sky_bin, env=env, cwd=str(tmp_path))
        assert repaired.healthy and repaired.outcome == "restarted_from_durable_cwd"
        status = subprocess.run(
            [sky_bin, "status", "--output", "json"], env=env, cwd=tmp_path,
            text=True, capture_output=True, check=False,
        )
        assert status.returncode == 0, status.stderr or status.stdout
        assert status.stdout.strip() == "[]"
    finally:
        if started:
            stop = subprocess.run(
                [sky_bin, "api", "stop"], env=env, cwd=tmp_path,
                text=True, capture_output=True, check=False,
            )
            assert stop.returncode == 0, stop.stderr or stop.stdout
            assert _probe_local_api_daemon_cwd(sky_bin).outcome == "absent"
