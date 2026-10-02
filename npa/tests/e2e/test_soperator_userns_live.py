"""Prove the Soperator user-namespace fix reaches every selected live worker."""

import json
import os
from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.e2e


def _command(context, *args):
    env = dict(os.environ)
    for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE"):
        env.pop(key, None)
    result = subprocess.run(
        ["kubectl", "--context", context, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "live Soperator command failed; inspect private logs"
    return result.stdout


def _workers(context, namespace, expected):
    result = json.loads(
        _command(
            context,
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "slurm.nebius.ai/worker=true",
            "-o",
            "json",
        )
    )
    workers = result["items"]
    assert len(workers) == expected
    for worker in workers:
        assert any(
            condition["type"] == "Ready" and condition["status"] == "True"
            for condition in worker["status"]["conditions"]
        )
    return workers


def _probe_worker(context, namespace, worker):
    command = ["exec", "-n", namespace, worker["metadata"]["name"], "--"]
    gate = _command(
        context,
        *command,
        "cat",
        "/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
    )
    assert gate.strip() == "0"
    uid = _command(
        context,
        *command,
        "chroot",
        "/mnt/jail",
        "id",
        "-u",
        "soperatorchecks",
    ).strip()
    assert uid.isdecimal() and int(uid) > 0
    # Linux separately forbids creating a user namespace from an active chroot.
    # Exercise the host gate as the non-root health-check identity outside it.
    _command(
        context,
        *command,
        "setpriv",
        f"--reuid={uid}",
        "--regid=0",
        "--clear-groups",
        "unshare",
        "--user",
        "--map-root-user",
        "/bin/true",
    )


def test_unprivileged_user_namespace_creation_on_every_worker():
    context = os.environ.get("NPA_E2E_SOPERATOR_CONTEXT")
    expected = os.environ.get("NPA_E2E_SOPERATOR_WORKER_COUNT")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not context or not expected:
        pytest.skip("select a dedicated Soperator context and exact worker count")
    namespace = os.environ.get("NPA_E2E_SOPERATOR_NAMESPACE", "soperator")
    workers = _workers(context, namespace, int(expected))
    assert workers
    for worker in workers:
        _probe_worker(context, namespace, worker)
    evidence = os.environ.get("NPA_E2E_SOPERATOR_EVIDENCE_DIR")
    if evidence:
        output = Path(evidence)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        receipt = output / "userns-live.json"
        with receipt.open("x") as stream:
            json.dump({"workers": len(workers), "unprivileged_userns": "PASS"}, stream)
            stream.write("\n")
        receipt.chmod(0o600)
