from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path
import stat
import textwrap

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from npa.workbench.sonic import export_onnx
from npa.workbench.sonic.eval import (
    EVAL_RESULT_FORMAT,
    EVAL_RESULT_SCHEMA,
    evaluate_onnx_policy,
)


class TinyEvalPolicy(torch.nn.Module):
    observation_dim = 4
    action_dim = 2
    control_dt = 0.02
    obs_spec = {
        "name": "actor_obs",
        "shape": [4],
        "fields": [
            {"name": "base_ang_vel", "dim": 2},
            {"name": "joint_pos", "dim": 2},
        ],
    }
    action_spec = {
        "name": "joint_targets",
        "shape": [2],
        "fields": [
            {"name": "left_hip_pitch", "dim": 1},
            {"name": "right_hip_pitch", "dim": 1},
        ],
    }

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.5, -0.25, 0.75, 0.1], [-0.4, 0.2, 0.3, 0.6]])
            )
            self.linear.bias.copy_(torch.tensor([0.05, -0.15]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


def test_sonic_eval_reference_smoke_writes_schema_result(tmp_path: Path) -> None:
    output = tmp_path / "policy.onnx"
    result_path = tmp_path / "eval.json"
    export = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinyEvalPolicy().eval(),
        verify=False,
    )

    result = evaluate_onnx_policy(
        onnx=export.onnx_path,
        metadata=export.metadata_path,
        backend="reference",
        episodes=3,
        env="smoke",
        output=str(result_path),
    )

    assert result["format"] == EVAL_RESULT_FORMAT
    assert result["backend"] == "reference"
    assert result["mode"] == "smoke"
    assert result["smoke_level"] is True
    assert result["policy"]["obs_dim"] == 4
    assert result["policy"]["action_dim"] == 2
    assert result["metrics"]["episodes"] == 3
    assert result["metrics"]["valid_action_rate"] == 1.0
    assert "No reference simulator env configured" in result["warnings"][0]
    assert (
        json.loads(result_path.read_text(encoding="utf-8"))["format"]
        == EVAL_RESULT_FORMAT
    )
    assert EVAL_RESULT_SCHEMA["properties"]["metrics"]["required"] == [
        "episode_return_mean",
        "distance_mean",
        "fall_rate",
        "termination_rate",
        "episode_length_mean",
        "valid_action_rate",
    ]


def test_sonic_eval_reference_locomotion_rollout_writes_metrics(tmp_path: Path) -> None:
    output = tmp_path / "policy.onnx"
    result_path = tmp_path / "locomotion-eval.json"
    export = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinyEvalPolicy().eval(),
        verify=False,
    )

    result = evaluate_onnx_policy(
        onnx=export.onnx_path,
        metadata=export.metadata_path,
        backend="reference",
        episodes=3,
        env="sonic-locomotion-smoke",
        output=str(result_path),
    )

    assert result["format"] == EVAL_RESULT_FORMAT
    assert result["backend"] == "reference"
    assert result["mode"] == "sim"
    assert result["smoke_level"] is False
    assert result["metrics"]["episodes"] == 3
    assert result["metrics"]["distance_mean"] > 0.0
    assert result["metrics"]["fall_rate"] == 0.0
    assert result["metrics"]["valid_action_rate"] == 1.0
    assert [episode["episode_length"] for episode in result["episodes"]] == [
        32,
        32,
        32,
    ]
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["metrics"]["distance_mean"] == result["metrics"]["distance_mean"]


def test_sonic_eval_reference_applies_sidecar_normalization(tmp_path: Path) -> None:
    output = tmp_path / "sidecar.onnx"
    export = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinyEvalPolicy().eval(),
        config={
            "normalization": {
                "mean": [1.0, 2.0, 3.0, 4.0],
                "std": [2.0, 2.0, 2.0, 2.0],
            }
        },
        normalize="sidecar",
    )

    result = evaluate_onnx_policy(
        onnx=export.onnx_path,
        metadata=export.metadata_path,
        backend="reference",
        episodes=2,
        env="smoke",
    )

    assert result["policy"]["normalize"] == "sidecar"
    assert result["metrics"]["valid_action_rate"] == 1.0
    assert [episode["episode_length"] for episode in result["episodes"]] == [1, 1]


def test_sonic_eval_container_backend_uses_configured_io_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "policy.onnx"
    export = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinyEvalPolicy().eval(),
        verify=False,
    )
    runtime = _fake_container_runtime(tmp_path)
    result_path = tmp_path / "container-result.json"

    result = evaluate_onnx_policy(
        onnx=export.onnx_path,
        metadata=export.metadata_path,
        backend="container",
        episodes=2,
        env="locomotion-smoke",
        output=str(result_path),
        container_image="mock-sonic-eval:latest",
        container_runtime=str(runtime),
        container_policy_path="/contract/input/exported_policy.onnx",
        container_metadata_path="/contract/input/exported_policy.metadata.json",
        container_output_path="/contract/output/results.json",
    )

    assert result["backend"] == "container"
    assert result["container"] == {
        "image": "mock-sonic-eval:latest",
        "runtime": str(runtime),
        "gpus": "all",
        "driver_capabilities": "all",
        "vulkan_icd": "/etc/vulkan/icd.d/nvidia_icd.json",
        "glx_vendor": "nvidia",
        "devices": [],
        "render_frames": 8,
        "policy_path": "/contract/input/exported_policy.onnx",
        "metadata_path": "/contract/input/exported_policy.metadata.json",
        "output_path": "/contract/output/results.json",
    }
    assert result["render"] == {"backend": "mock", "graphics_api": "vulkan", "frames": 8}
    assert result["metrics"]["episode_return_mean"] == 1.25
    assert result["metrics"]["distance_mean"] == 2.5
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["backend"] == "container"
    assert written["metrics"]["valid_action_rate"] == 1.0


def test_sonic_eval_container_backend_accepts_nvidia_cdi_gpu_request(
    tmp_path: Path,
) -> None:
    output = tmp_path / "policy.onnx"
    export = export_onnx(
        checkpoint="in-memory-policy",
        output=str(output),
        policy=TinyEvalPolicy().eval(),
        verify=False,
    )
    runtime = _fake_container_runtime(tmp_path, name="docker")

    result = evaluate_onnx_policy(
        onnx=export.onnx_path,
        metadata=export.metadata_path,
        backend="container",
        episodes=1,
        env="isaac-lab-headless-render",
        container_image="mock-sonic-eval:latest",
        container_runtime=str(runtime),
        container_gpus="nvidia.com/gpu=all",
        container_args=["eval"],
    )

    argv = result["diagnostics"]["argv"]
    env = result["diagnostics"]["env"]
    assert "--gpus" not in argv
    assert argv[0:4] == ["run", "--rm", "--runtime", "nvidia"]
    assert env["NVIDIA_VISIBLE_DEVICES"] == "nvidia.com/gpu=all"
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert result["container"]["gpus"] == "nvidia.com/gpu=all"
    assert result["render"] == {
        "backend": "mock",
        "graphics_api": "vulkan",
        "frames": 8,
    }


def _fake_container_runtime(
    tmp_path: Path, *, name: str = "fake-container-runtime.py"
) -> Path:
    runtime = tmp_path / name
    runtime.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            volumes = {}
            env = {}
            idx = 0
            while idx < len(args):
                if args[idx] == "-v":
                    host, container, *_mode = args[idx + 1].split(":")
                    volumes[container] = host
                    idx += 2
                    continue
                if args[idx] == "-e":
                    key, value = args[idx + 1].split("=", 1)
                    env[key] = value
                    idx += 2
                    continue
                idx += 1

            def host_path(container_path: str) -> Path:
                matches = sorted(volumes.items(), key=lambda item: len(item[0]), reverse=True)
                for container_root, host_root in matches:
                    root = container_root.rstrip("/")
                    if container_path == root or container_path.startswith(root + "/"):
                        rel = container_path[len(root):].lstrip("/")
                        return Path(host_root) / rel
                raise SystemExit(f"unmounted path: {container_path}")

            policy = host_path(env["NPA_SONIC_ONNX"])
            metadata = host_path(env["NPA_SONIC_METADATA"])
            output = host_path(env["NPA_SONIC_OUTPUT"])
            if not policy.exists() or not metadata.exists():
                raise SystemExit("missing staged eval input")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "status": "completed",
                "metrics": {
                    "episode_return_mean": 1.25,
                    "distance_mean": 2.5,
                    "fall_rate": 0.0,
                    "termination_rate": 0.0,
                    "episode_length_mean": 4.0,
                    "valid_action_rate": 1.0
                },
                "episodes": [
                    {"episode_index": 0, "episode_return": 1.0, "distance": 2.0},
                    {"episode_index": 1, "episode_return": 1.5, "distance": 3.0}
                ],
                "render": {"backend": "mock", "graphics_api": "vulkan", "frames": 8},
                "diagnostics": {"argv": args, "env": env},
                "warnings": []
            }), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    runtime.chmod(runtime.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return runtime
