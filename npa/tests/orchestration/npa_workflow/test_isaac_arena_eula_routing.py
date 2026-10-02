"""Preserve Arena acceptance and explicit refusal across real workflow routes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOWS = REPO_ROOT / "workflows/testing"
OPAQUE_IMAGE = "registry.example/operator-worker:immutable"
ARENA_IMAGE = "registry.example/npa-isaac-arena@sha256:" + "a" * 64


@pytest.fixture(params=["state", "video", "raw"])
def arena_route(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, str]:
    if request.param == "video":
        return WORKFLOWS / "isaac-arena-evaluation-rtxpro.yaml", OPAQUE_IMAGE
    state_path = WORKFLOWS / "isaac-arena-evaluation-b200.yaml"
    if request.param == "state":
        return state_path, OPAQUE_IMAGE
    raw = yaml.safe_load(state_path.read_text())
    raw["initial"] = "evaluate"
    raw["states"] = {
        "evaluate": {
            "resources": "gpu",
            "run": {
                "argv": [
                    "bash",
                    "/opt/npa/docker/workbench/isaac-arena/smoke_functional.sh",
                ]
            },
            "terminal": True,
        }
    }
    path = tmp_path / "arena-raw.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, ARENA_IMAGE


@pytest.mark.parametrize("accepted", [True, False])
def test_arena_render_preserves_acceptance_and_bootstrap_refusal(
    arena_route: tuple[Path, str], accepted: bool, tmp_path: Path
) -> None:
    path, image = arena_route
    spec = load_spec(path)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="arena-consent"),
        run_id="arena-consent",
        options=SkypilotRenderOptions(
            image_overrides={"*": image},
            materialize_registry_secrets=False,
            accept_eula=accepted,
        ),
    )
    tasks = [
        doc for doc in yaml.safe_load_all(rendered) if doc and doc.get("resources")
    ]
    assert tasks
    assert all(
        task["envs"].get("ACCEPT_EULA") == ("Y" if accepted else "") for task in tasks
    )
    if not accepted:
        cache = tmp_path / "isaac-cache"
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "npa/docker/workbench/common/isaac_bootstrap.sh"),
                "ensure",
            ],
            env={
                **os.environ,
                "ACCEPT_EULA": tasks[0]["envs"]["ACCEPT_EULA"],
                "NPA_ISAAC_CACHE_DIR": str(cache),
                "NPA_ISAAC_BOOTSTRAP_OFFLINE": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 78
        assert "explicitly disabled" in result.stderr
        assert not cache.exists()


def test_arena_cli_opt_out_stops_before_submission(
    arena_route: tuple[Path, str],
) -> None:
    path, image = arena_route
    with patch("npa.orchestration.skypilot.workflow.submit_workflow") as submit:
        result = CliRunner().invoke(
            app,
            [
                "workbench",
                "workflow",
                "submit",
                str(path),
                "--run-id",
                "arena-explicit-optout",
                "--image",
                image,
                "--no-accept-eula",
            ],
        )
    assert result.exit_code == 1
    assert "Isaac" in result.output
    assert "No expensive action has begun" in result.output
    submit.assert_not_called()
