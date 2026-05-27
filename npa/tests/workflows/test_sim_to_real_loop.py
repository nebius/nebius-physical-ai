from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from npa.workflows.sereact_sim_to_real import (
    ControllerSettings,
    build_controller_plan,
    run_controller,
)


ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sim-to-real-loop.yaml"
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_sim_to_real_loop.py"


def _docs() -> list[dict]:
    return [doc for doc in yaml.safe_load_all(YAML_PATH.read_text(encoding="utf-8")) if doc]


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_sim_to_real_loop", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sim_to_real_loop_yaml_uses_controller_job_pattern() -> None:
    docs = _docs()

    assert docs[0] == {"name": "sereact-sim-to-real-loop", "execution": "serial"}
    assert len(docs[1:]) == 1
    task = docs[1]
    assert task["name"] == "sereact-controller-loop"
    assert task["resources"] == {"cloud": "kubernetes", "cpus": 8, "memory": 32}
    assert "npa.workflows.sereact_sim_to_real" in task["run"]
    assert "NPA_DRY_RUN" in task["envs"]


def test_sim_to_real_wrapper_renders_run_scoped_paths(capsys) -> None:
    wrapper = _load_wrapper_module()

    rc = wrapper.main(
        [
            "--yaml",
            str(YAML_PATH),
            "--render-only",
            "--controller-dry-run",
            "--run-id",
            "sereact-test",
            "--bucket",
            "bucket",
            "--source-uri",
            "s3://bucket/raw/",
            "--max-iterations",
            "2",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    rendered = Path(payload["rendered_yaml"])
    docs = [doc for doc in yaml.safe_load_all(rendered.read_text(encoding="utf-8")) if doc]
    envs = docs[1]["envs"]
    assert envs["NPA_PIPELINE_RUN_ID"] == "sereact-test"
    assert envs["SEREACT_SOURCE_URI"] == "s3://bucket/raw/"
    assert envs["SEREACT_OUTPUT_URI"] == "s3://bucket/sereact-sim-to-real/sereact-test/"
    assert envs["SEREACT_MAX_ITERATIONS"] == "2"
    assert envs["NPA_DRY_RUN"] == "1"


def test_sereact_controller_plan_contains_data_cosmos_and_vlm_steps() -> None:
    result = build_controller_plan(
        ControllerSettings(
            run_id="run-1",
            input_path="s3://bucket/raw/",
            output_path="s3://bucket/out/",
            max_iterations=2,
            dry_run=True,
        )
    )

    stages = [step["stage"] for step in result.steps]
    assert stages == [
        "data_import",
        "cosmos_generate",
        "vlm_eval",
        "data_import",
        "cosmos_generate",
        "vlm_eval",
    ]
    commands = json.dumps([step["command"] for step in result.steps]).lower()
    assert "lightwheel" not in commands
    assert "onnx" not in commands


def test_sereact_controller_stops_when_vlm_stub_passes() -> None:
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        stdout = '{"passed": true}' if "vlm-eval" in command else "{}"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    result = run_controller(
        ControllerSettings(
            run_id="run-1",
            input_path="s3://bucket/raw/",
            output_path="s3://bucket/out/",
            max_iterations=3,
            dry_run=False,
        ),
        runner=fake_runner,
    )

    assert result.status == "succeeded"
    assert result.iterations_planned == 1
    assert len(calls) == 3
