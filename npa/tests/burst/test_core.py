from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.burst import core
from npa.burst.core import BurstConfigError, BurstJobHandle, BurstSpec
from npa.orchestration.skypilot import _bin as bin_module


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _isolated_skypilot_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)


def test_task_yaml_uses_requested_accelerator_and_docker_image() -> None:
    spec = BurstSpec(
        image="registry.example/npa-train:latest",
        num_nodes=2,
        gpu_per_node="CUSTOMGPU:4",
        entrypoint="python train.py --epochs 1",
        name="burst-test",
    )

    rendered = core.task_yaml(spec)
    data = yaml.safe_load(rendered)

    assert data["name"] == "burst-test"
    assert data["num_nodes"] == 2
    assert data["resources"] == {
        "cloud": "nebius",
        "accelerators": "CUSTOMGPU:4",
        "image_id": "docker:registry.example/npa-train:latest",
    }
    assert data["envs"]["BURST_ENTRYPOINT"] == "python train.py --epochs 1"
    assert "torchrun" in data["run"]
    assert "SKYPILOT_NODE_RANK" in data["run"]
    assert "SKYPILOT_NUM_NODES" in data["run"]
    assert "SKYPILOT_NODE_IPS" in data["run"]
    assert "SKYPILOT_NUM_GPUS_PER_NODE" in data["run"]
    assert "L40S" not in rendered
    assert "H100" not in rendered


@pytest.mark.parametrize("gpu_per_node", ["1", "GPU", "GPU:0", "GPU:-1", ""])
def test_gpu_per_node_requires_sky_accelerator_spec(gpu_per_node: str) -> None:
    with pytest.raises(BurstConfigError):
        core.build_task_spec(
            BurstSpec(
                image="example.invalid/train:latest",
                num_nodes=2,
                gpu_per_node=gpu_per_node,
                entrypoint="python train.py",
            )
        )


def test_submit_invokes_skypilot_python_api_not_sky_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    sky_bin = _executable(tmp_path / "bin" / "sky")
    sky_python = _executable(sky_bin.parent / "python")
    calls: list[list[str]] = []
    config = tmp_path / "base-config.yaml"
    config.write_text("jobs:\n  controller:\n    resources:\n      cloud: kubernetes\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        if cmd == [str(sky_python), "-c", "import sky; print(getattr(sky, '__version__', 'unknown'))"]:
            return core.subprocess.CompletedProcess(cmd, 0, stdout="0.12.2\n", stderr="")
        assert cmd[:3] == [str(sky_python), "-m", "npa.burst._sky_api"]
        assert cmd[3] == "launch"
        payload = json.loads(kwargs["input"])
        task = yaml.safe_load(Path(payload["yaml_path"]).read_text(encoding="utf-8"))
        assert task["num_nodes"] == 2
        assert task["resources"]["accelerators"] == "CUSTOMGPU:1"
        return core.subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"job_ids": [123], "output": "submitted"}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    handle = core.submit(
        image="example.invalid/train:latest",
        num_nodes=2,
        gpu_per_node="CUSTOMGPU:1",
        entrypoint="python train.py",
        name="burst-test",
        sky_bin=sky_bin,
        config_path=config,
        isolated_config_dir=tmp_path / "sky-state",
    )

    assert handle.job_id == "123"
    assert handle.name == "burst-test"
    assert handle.sky_bin == str(sky_bin.resolve())
    assert all("sky" not in call[0] for call in calls)


def test_status_and_logs_use_handle_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "bin").mkdir(exist_ok=True)
    sky_bin = _executable(tmp_path / "bin" / "sky")
    sky_python = _executable(sky_bin.parent / "python")
    handle = BurstJobHandle(
        job_id="42",
        name="burst-test",
        sky_bin=str(sky_bin),
        isolated_config_dir=str(tmp_path / "sky-state"),
    )

    def fake_run(cmd, **kwargs):
        if cmd == [str(sky_python), "-c", "import sky; print(getattr(sky, '__version__', 'unknown'))"]:
            return core.subprocess.CompletedProcess(cmd, 0, stdout="0.12.2\n", stderr="")
        action = cmd[3]
        if action == "queue":
            return core.subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {"records": [{"job_id": 42, "status": "RUNNING", "resources": "CUSTOMGPU:1"}]}
                ),
                stderr="",
            )
        if action == "logs":
            return core.subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps({"logs": "rank=0 world_size=2\n", "exit_code": None}),
                stderr="",
            )
        raise AssertionError(action)

    monkeypatch.setattr(core.subprocess, "run", fake_run)

    assert core.status(handle).status == "RUNNING"
    assert "world_size=2" in core.logs(handle, tail=10).text
