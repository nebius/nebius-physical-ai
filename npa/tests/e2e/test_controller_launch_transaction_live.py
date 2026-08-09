"""Operator-authorized fresh-cluster controller launch transaction regression.

This test creates one tiny managed job and may create SkyPilot's Kubernetes jobs
controller. It is excluded from hermetic suites and requires explicit selectors;
it never guesses a project, context, kubeconfig, or run ID and performs no fuzzy
cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.e2e_skypilot


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to an exact operator-reviewed live selector")
    return value


def test_fresh_cluster_controller_launch_records_stability_and_reconciliation(
    tmp_path: Path,
) -> None:
    if os.environ.get("NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION") != "1":
        pytest.skip(
            "set NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION=1 to authorize one live "
            "managed-job/controller launch"
        )
    project = _required("NPA_E2E_PROJECT")
    context = _required("NPA_E2E_CLUSTER_CONTEXT")
    run_id = _required("NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID")
    kubeconfig = _required("KUBECONFIG")
    real_sky_bin = Path(_required("NPA_SKYPILOT_BIN")).resolve()
    if os.pathsep in kubeconfig:
        pytest.skip("live regression requires one exact KUBECONFIG path")

    task = tmp_path / "controller-transaction-live.yaml"
    task.write_text(
        "name: npa-controller-transaction-live\n"
        "resources:\n"
        "  cloud: kubernetes\n"
        "  cpus: 1\n"
        "run: |\n"
        "  printf 'controller launch transaction live regression\\n'\n",
        encoding="utf-8",
    )
    # Test-only adapter: fail exactly the first client-side launch before it can
    # reach SkyPilot.  Queue reconciliation must prove authoritative absence,
    # re-establish API stability, and retry the same logical identity.  This
    # neither damages the live control plane nor adds a production fault hook.
    wrapper_dir = tmp_path / "fault-wrapper"
    wrapper_dir.mkdir(mode=0o700)
    marker = wrapper_dir / "first-launch-failed"
    wrapper = wrapper_dir / "sky"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"real = {str(real_sky_bin)!r}\n"
        f"marker = {str(marker)!r}\n"
        "if sys.argv[1:3] == ['jobs', 'launch']:\n"
        "    try:\n"
        "        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
        "    except FileExistsError:\n"
        "        pass\n"
        "    else:\n"
        "        os.close(fd)\n"
        "        print('Kubernetes API connection refused during controller launch', file=sys.stderr)\n"
        "        raise SystemExit(1)\n"
        "os.execv(real, [real, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    # ``ensure_skypilot_version`` validates the companion interpreter next to the
    # selected CLI. A symlink placed outside the venv loses Python's ``pyvenv.cfg``
    # discovery and therefore cannot import the pinned dependencies, so delegate
    # argv to the real companion interpreter explicitly.
    python_wrapper = wrapper_dir / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"real = {str(real_sky_bin.parent / 'python')!r}\n"
        "os.execv(real, [real, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    python_wrapper.chmod(0o700)
    env = dict(os.environ)
    env["KUBECONFIG"] = kubeconfig
    env["NPA_SKYPILOT_BIN"] = str(wrapper)
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("npa")),
            "workbench",
            "workflow",
            "submit",
            str(task),
            "--project",
            project,
            "--run-id",
            run_id,
            "--infra",
            f"k8s/{context}",
            "--output-format",
            "json",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    transaction = payload["launch_transaction"]
    assert transaction["state"] in {"submitted", "adopted"}
    assert transaction["job_id"] == payload["job_id"]
    assert marker.is_file()
    assert transaction["launch_sequence"] == 2
    assert transaction["category"] == "kubernetes_transport"
    assert "connection refused" in transaction["primary_error"].lower()
    assert transaction["readiness"]
    last_readiness = transaction["readiness"][-1]
    assert last_readiness["state"] == "ready"
    assert last_readiness["consecutive_successes"] >= 3
    states = [item["state"] for item in transaction["reconciliations"]]
    assert states.count("found") == 1
    assert states.count("absent") >= 2
    assert states[-1] == "found"
    # Deliberately no automatic teardown: the operator can inspect the durable
    # evidence, then cancel only payload["job_id"] through the documented exact-ID
    # workflow cancellation path.
