"""Exercise Wan generation overrides through the planned worker shell boundary."""

from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec

ROOT = Path(__file__).resolve().parents[3]
SPECS = ("byof-wan2.2.yaml", "byof-wan2.2-multigpu.yaml")


def _planned_smoke(filename: str, overrides: dict) -> tuple[str, dict]:
    spec = load_spec(ROOT / "workflows" / "testing" / filename)
    spec.config.update(overrides)
    argv = build_plan(spec, run_id="generation-controls").steps[0].argv
    return argv[argv.index("--smoke-command") + 1], spec.config


def _run_preflight(tmp_path: Path, filename: str, overrides: dict):
    smoke, config = _planned_smoke(filename, overrides)
    contract = ROOT / "npa/docker/workbench/wan2-2/input_contract.py"
    shutil.copyfile(contract, tmp_path / "npa_wan_input_contract.py")
    runtime = tmp_path / "wan-runtime"
    runtime.write_text('#!/bin/sh\nprintf called > "$PROBE_FILE"\n')
    runtime.chmod(0o700)
    environment = dict(os.environ)
    environment.update(
        PATH=f"{tmp_path}:{environment.get('PATH', '')}",
        PYTHONPATH=str(tmp_path),
        NPA_SMOKE_OUTPUT_DIR=str(tmp_path / "output"),
        WAN22_CACHE_DIR=str(tmp_path / "cache"),
        PROBE_FILE=str(tmp_path / "runtime-called"),
        BYOF_CAPABILITY_NAME=config["capability_name"],
        BYOF_SMOKE_ARTIFACT_NAME=config["smoke_artifact_name"],
    )
    script = smoke.split("wan-runtime ensure", 1)[0] + "wan-runtime ensure\n"
    script = script.replace("/opt/wan-base/bin/python", shlex.quote(sys.executable))
    return subprocess.run(
        ["bash", "-c", script], env=environment, text=True, capture_output=True
    )


@pytest.mark.parametrize("filename", SPECS)
@pytest.mark.parametrize("overrides", [{}, {"frames": 121, "steps": 50, "seed": 0}])
def test_generation_request_reaches_runtime_with_exact_integers(
    tmp_path: Path, filename: str, overrides: dict
) -> None:
    result = _run_preflight(tmp_path, filename, overrides)
    assert result.returncode == 0, result.stderr
    requested = json.loads(
        (tmp_path / "output/wan2_2_generation_request.json").read_text()
    )
    assert requested == (overrides or {"frames": 17, "steps": 8, "seed": 42})
    assert all(type(value) is int for value in requested.values())
    assert (tmp_path / "runtime-called").is_file()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("filename", SPECS)
@pytest.mark.parametrize(
    "overrides",
    [
        {"frames": 1},
        {"frames": 120},
        {"frames": 17.5},
        {"frames": True},
        {"steps": 0},
        {"steps": -1},
        {"steps": False},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": None},
    ],
)
def test_invalid_generation_stops_before_runtime_fetch(
    tmp_path: Path, filename: str, overrides: dict
) -> None:
    result = _run_preflight(tmp_path, filename, overrides)
    assert result.returncode != 0
    assert "Wan " in result.stderr
    assert not (tmp_path / "runtime-called").exists()
    assert not (tmp_path / "output/wan2_2_generation_request.json").exists()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("filename", SPECS)
@pytest.mark.parametrize("key", ["frames", "steps", "seed"])
def test_generation_controls_are_data_at_the_worker_shell_boundary(
    tmp_path: Path, filename: str, key: str
) -> None:
    marker = tmp_path / "unexpected-shell-command"
    hostile = f'"; touch {shlex.quote(str(marker))}; echo "$(touch {marker})"'
    result = _run_preflight(tmp_path, filename, {key: hostile})
    assert result.returncode != 0
    assert "must be a non-negative integer" in result.stderr
    assert not marker.exists()
    assert not (tmp_path / "runtime-called").exists()


def test_distributed_wrapper_passes_requested_controls_as_upstream_argv() -> None:
    smoke, _ = _planned_smoke(SPECS[1], {"frames": 121, "steps": 50, "seed": 7})
    wrapper = smoke.split("cat > \"${WRAPPER}\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    tree = ast.parse(wrapper)
    append = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "sys.argv.extend"
    )
    requested = {"frames": 121, "steps": 50, "seed": 7}
    script = (
        "import json, sys\n"
        f"generation = {requested!r}\n"
        f"{ast.unparse(append)}\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    arguments = json.loads(result.stdout)
    assert arguments == [
        "--frame_num",
        "121",
        "--sample_steps",
        "50",
        "--base_seed",
        "7",
    ]


@pytest.mark.parametrize("filename", SPECS)
def test_worker_image_and_byof_verifier_share_the_accepted_digest(filename: str) -> None:
    from npa.deploy.images import DEFAULT_PUBLIC_CONTAINER_REGISTRY, wan_accepted_image_manifest
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    expected = (
        f"{DEFAULT_PUBLIC_CONTAINER_REGISTRY}/npa-wan2-2@"
        f"{wan_accepted_image_manifest()['oci_digest']}"
    )
    spec = load_spec(ROOT / "workflows" / "testing" / filename)
    plan = build_plan(spec, run_id="immutable-worker-image")
    step = plan.steps[0]
    assert step.argv[step.argv.index("--base-image") + 1] == expected
    assert step.resources_profile["image"] == expected
    rendered = render_skypilot_yaml(
        spec, plan, run_id="immutable-worker-image",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    task = [document for document in yaml.safe_load_all(rendered) if document][-1]
    assert task["resources"]["image_id"] == f"docker:{expected}"
