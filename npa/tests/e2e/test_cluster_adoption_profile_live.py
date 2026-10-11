"""Exercise profile-bound adoption of an operator-selected existing cluster."""

import os
import subprocess
import sys
import uuid

import pytest
import yaml

from npa.clients.config import resolve_environment

pytestmark = pytest.mark.e2e


def _adopt_with_conflicting_cli_profile(cluster_name, project_id, context, destination):
    env = {
        **os.environ,
        "NEBIUS_PROFILE": "npa-unused-profile-" + uuid.uuid4().hex,
        "NPA_CONFIG_DIR": str(destination.parent / "npa"),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "npa",
            "cluster",
            "kubeconfig",
            "--cluster-name",
            cluster_name,
            "--project-id",
            project_id,
            "--context",
            context,
            "--kubeconfig",
            str(destination),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


def test_adoption_uses_selected_profile_with_a_conflicting_cli_selector(tmp_path):
    cluster_name = os.environ.get("NPA_E2E_ADOPT_CLUSTER_NAME", "")
    project = os.environ.get("NPA_E2E_PROJECT", "")
    profile = os.environ.get("NPA_NEBIUS_PROFILE", "")
    if not all((cluster_name, project, profile)):
        pytest.skip("Set the private cluster name, project alias and NPA profile")
    context = "npa-adoption-check-" + uuid.uuid4().hex
    destination = tmp_path / "kubeconfig"
    project_id = resolve_environment(project).project_id
    result = _adopt_with_conflicting_cli_profile(
        cluster_name, project_id, context, destination
    )
    (tmp_path / "adoption.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (
        "Live cluster adoption failed; inspect adoption.log in pytest's private directory"
    )
    config = yaml.safe_load(destination.read_text())
    assert config["current-context"] == context
    user_name = next(
        x["context"]["user"] for x in config["contexts"] if x["name"] == context
    )
    user = next(x["user"] for x in config["users"] if x["name"] == user_name)
    args = user["exec"]["args"]
    assert args[args.index("--profile") + 1] == profile
    assert destination.is_file()
