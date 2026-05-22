from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SINGLE_YAML = ROOT / "npa" / "workflows" / "skypilot" / "isaac-lab-rl-train.yaml"
SWEEP_YAML = ROOT / "npa" / "workflows" / "skypilot" / "isaac-lab-rl-sweep.yaml"
WRAPPER_PATH = ROOT / "npa" / "scripts" / "run_isaac_lab_rl.py"


def _docs(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc is not None]


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("run_isaac_lab_rl", WRAPPER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_isaac_lab_single_job_yaml_uses_rt_core_gpu_and_rsl_rl_entrypoint() -> None:
    docs = _docs(SINGLE_YAML)

    assert docs[0] == {"name": "isaac-lab-rl-train", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "L40S:1"
    assert task["resources"]["cpus"] == 16
    assert task["resources"]["memory"] == 64
    assert "npa-isaac-lab:2.3.2.post1" in task["resources"]["image_id"]
    assert "scripts/reinforcement_learning/rsl_rl/train.py" in task["run"]
    assert "--num_envs" in task["run"]
    assert "--max_iterations" in task["run"]
    assert "agent.save_interval=1" in task["envs"]["ISAAC_LAB_HYDRA_OVERRIDES"]


def test_isaac_lab_sweep_yaml_uses_parallel_group_and_distinct_variants() -> None:
    docs = _docs(SWEEP_YAML)
    tasks = docs[1:]

    assert docs[0] == {"name": "isaac-lab-rl-sweep", "execution": "parallel"}
    assert len(tasks) == 4
    variants = [task["envs"]["RUN_VARIANT"] for task in tasks]
    assert variants == ["lr-1e-3", "lr-3e-4", "entropy-0", "entropy-0-01"]
    for task in tasks:
        assert task["resources"]["accelerators"] == "L40S:1"
        assert "scripts/reinforcement_learning/rsl_rl/train.py" in task["run"]
        assert task["envs"]["S3_OUTPUT_PREFIX"].endswith(f"/{task['envs']['RUN_VARIANT']}/")


def test_isaac_lab_runner_renders_and_submits(monkeypatch, tmp_path, capsys) -> None:
    wrapper = _load_wrapper_module()
    sky_bin = tmp_path / "sky"
    sky_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    sky_bin.chmod(sky_bin.stat().st_mode | stat.S_IXUSR)
    captured = {}

    def fake_submit_workflow(yaml_path, run_id, **kwargs):
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        captured["docs"] = [doc for doc in yaml.safe_load_all(Path(yaml_path).read_text(encoding="utf-8")) if doc is not None]
        return wrapper.WorkflowResult(status="SUBMITTED", job_id="42", returncode=0, log_paths={"config": str(tmp_path / "config.yaml")})

    def fake_workflow_status(job_id, **kwargs):
        return wrapper.WorkflowResult(status="SUCCEEDED", job_id=job_id, returncode=0)

    monkeypatch.setattr(wrapper, "submit_workflow", fake_submit_workflow)
    monkeypatch.setattr(wrapper, "workflow_status", fake_workflow_status)

    rc = wrapper.main(
        [
            "--yaml",
            str(SINGLE_YAML),
            "--run-id",
            "isaac-test-run",
            "--task",
            "Isaac-Cartpole-v0",
            "--iterations",
            "3",
            "--output-root",
            "s3://bucket/isaac-lab-rl",
            "--image",
            "registry.example/npa-isaac-lab:test",
            "--sky-bin",
            str(sky_bin),
            "--poll-interval",
            "0",
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["outputs"]["checkpoint"] == "s3://bucket/isaac-lab-rl/isaac-test-run/npa_isaac_lab_checkpoint.pt"
    assert captured["run_id"] == "isaac-test-run"
    rendered_task = captured["docs"][1]
    assert rendered_task["envs"]["ISAAC_LAB_ITERATIONS"] == "3"
    assert rendered_task["envs"]["S3_OUTPUT_PREFIX"] == "s3://bucket/isaac-lab-rl/isaac-test-run/"
    assert rendered_task["resources"]["image_id"] == "docker:registry.example/npa-isaac-lab:test"


def test_isaac_lab_runner_render_only_keeps_rendered_yaml(capsys) -> None:
    wrapper = _load_wrapper_module()

    rc = wrapper.main(
        [
            "--yaml",
            str(SINGLE_YAML),
            "--run-id",
            "isaac-render-only",
            "--task",
            "Isaac-Cartpole-v0",
            "--iterations",
            "2",
            "--render-only",
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    rendered = Path(output["rendered_yaml"])
    assert rendered.is_file()
    docs = [doc for doc in yaml.safe_load_all(rendered.read_text(encoding="utf-8")) if doc is not None]
    assert docs[1]["envs"]["NPA_ISAAC_LAB_RUN_ID"] == "isaac-render-only"
