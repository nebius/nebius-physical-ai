from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from npa.orchestration.skypilot.capacity import is_capacity_error


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[3]
BLUEPRINT = (
    ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "sonic-export-eval.yaml"
)
DEFAULT_IMAGE = "docker:python:3.11-slim"
DEFAULT_GPU = "L40S:1"
DEFAULT_GPU_CANDIDATES = (DEFAULT_GPU, "RTX_PRO_6000_BLACKWELL:1", "H100:1")
DEFAULT_CLOUD = "nebius"
DEFAULT_TIMEOUT_SECONDS = 5400
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass
class LiveSkyRun:
    sky_bin: str
    cluster_name: str
    evidence_dir: Path


@pytest.fixture
def live_sky_run(tmp_path: Path) -> LiveSkyRun:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")

    sky_bin = _resolve_sky_bin()
    evidence_dir = Path(os.environ.get("NPA_SONIC_E2E_EVIDENCE_DIR", tmp_path))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    check = _run([sky_bin, "check", "nebius"], timeout=300)
    _write_capture(evidence_dir / "sky-check-nebius.json", check)
    if check.returncode != 0:
        pytest.skip(f"SkyPilot Nebius check is not enabled: {check.stderr or check.stdout}")

    cluster_name = f"npa-sonic-e2e-{uuid.uuid4().hex[:8]}"
    try:
        yield LiveSkyRun(
            sky_bin=sky_bin,
            cluster_name=cluster_name,
            evidence_dir=evidence_dir,
        )
    finally:
        down = _run([sky_bin, "down", "--yes", cluster_name], timeout=900)
        _write_capture(evidence_dir / "sky-down.json", down)
        status = _run([sky_bin, "status", "--refresh"], timeout=300)
        _write_capture(evidence_dir / "sky-status-after-down.json", status)


def test_sonic_export_eval_e2e_runs_live_reference_rollouts_on_nebius(
    tmp_path: Path,
    live_sky_run: LiveSkyRun,
) -> None:
    timeout = int(os.environ.get("NPA_SONIC_E2E_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    candidates = _gpu_candidates()
    attempts: list[dict[str, Any]] = []
    selected_gpu = candidates[0]
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    for gpu in candidates:
        selected_gpu = gpu
        workflow = _write_live_workflow(tmp_path, live_sky_run.cluster_name, gpu=gpu)
        current = _run(
            [
                live_sky_run.sky_bin,
                "launch",
                "-c",
                live_sky_run.cluster_name,
                "--yes",
                str(workflow),
            ],
            timeout=timeout,
        )
        attempts.append({"gpu": gpu, "returncode": current.returncode})
        _write_capture(live_sky_run.evidence_dir / f"sky-launch-{_gpu_slug(gpu)}.json", current)
        if current.returncode == 0:
            result = current
            break
        if not _is_capacity_error(current):
            result = current
            break
    latency_seconds = round(time.monotonic() - started, 3)
    if result is None:
        pytest.skip(
            "SkyPilot could not provision any configured GPU candidate due to capacity constraints"
        )
    if result.returncode != 0 and _is_capacity_error(result):
        pytest.skip(
            f"SkyPilot capacity unavailable for all GPU candidates: {', '.join(item['gpu'] for item in attempts)}"
        )
    assert result.returncode == 0, _format_result(result)
    output = _strip_ansi(f"{result.stdout}\n{result.stderr}")
    export_result = _extract_json(
        output,
        "NPA_SONIC_E2E_EXPORT_JSON_BEGIN",
        "NPA_SONIC_E2E_EXPORT_JSON_END",
    )
    metrics = _extract_json(
        output,
        "NPA_SONIC_E2E_METRICS_JSON_BEGIN",
        "NPA_SONIC_E2E_METRICS_JSON_END",
    )

    assert export_result["status"] == "exported"
    assert export_result["onnx_path"].endswith("sonic_policy.onnx")
    assert export_result["metadata_path"].endswith("sonic_policy.metadata.json")
    assert export_result["parity"]["passed"] is True

    assert metrics["format"] == "npa_sonic_eval_result_v1"
    assert metrics["status"] == "completed"
    assert metrics["backend"] == "reference"
    assert metrics["mode"] == "sim"
    assert metrics["smoke_level"] is False
    assert metrics["eval"]["env"] == "sonic-locomotion-smoke"
    assert metrics["metrics"]["episodes"] == 3
    assert metrics["metrics"]["distance_mean"] > 0.0
    assert metrics["metrics"]["fall_rate"] == 0.0
    assert metrics["metrics"]["valid_action_rate"] == 1.0
    assert len(metrics["episodes"]) == 3

    summary = {
        "cluster_name": live_sky_run.cluster_name,
        "gpu": selected_gpu,
        "cloud": os.environ.get("NPA_SONIC_E2E_CLOUD", DEFAULT_CLOUD),
        "latency_seconds": latency_seconds,
        "attempts": attempts,
        "export": export_result,
        "metrics": metrics,
    }
    (live_sky_run.evidence_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_live_workflow(tmp_path: Path, cluster_name: str, *, gpu: str) -> Path:
    docs = [
        doc
        for doc in yaml.safe_load_all(BLUEPRINT.read_text(encoding="utf-8"))
        if doc is not None
    ]
    task = copy.deepcopy(docs[1])
    task["name"] = cluster_name
    task["resources"]["cloud"] = os.environ.get("NPA_SONIC_E2E_CLOUD", DEFAULT_CLOUD)
    task["resources"]["accelerators"] = gpu
    task["resources"]["image_id"] = os.environ.get("NPA_SONIC_E2E_IMAGE", DEFAULT_IMAGE)
    if task["resources"]["cloud"] == "nebius":
        task["resources"]["cpus"] = "8+"
        task["resources"]["memory"] = "32+"
    task["file_mounts"] = {"/tmp/npa-repo": str(_stage_minimal_repo(tmp_path))}
    task["envs"].update(
        {
            "POLICY_CKPT": "/tmp/npa-sonic-e2e-input/tiny_policy.pt",
            "OUTPUT_DIR": "/tmp/npa-sonic-e2e-output",
            "EVAL_BACKEND": "reference",
            "EVAL_ENV": "sonic-locomotion-smoke",
            "EPISODES": "3",
            "CONTAINER_IMAGE": "",
            "GPU": task["resources"]["accelerators"],
            "NPA_REPO_DIR": "/tmp/npa-repo",
            "NPA_PYTHON_BIN": "python3",
            "PYTHONPATH": "/tmp/npa-sonic-e2e-input",
            "SONIC_CONFIG": "/tmp/npa-sonic-e2e-input/tiny_policy_config.yaml",
            "SONIC_VERIFY": "1",
        }
    )
    task["setup"] = task["setup"] + _tiny_policy_setup()

    path = tmp_path / "sonic-export-eval-live.yaml"
    path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return path


def _gpu_candidates() -> list[str]:
    raw = os.environ.get("NPA_SONIC_E2E_GPU_CANDIDATES", "").strip()
    if raw:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if parsed:
            return parsed
    single = os.environ.get("NPA_SONIC_E2E_GPU", "").strip()
    if single:
        return [single]
    return list(DEFAULT_GPU_CANDIDATES)


# Broader SkyPilot provisioning markers specific to this export/eval GPU chain,
# layered on top of the shared high-confidence capacity classifier.
_EXPORT_EVAL_EXTRA_CAPACITY_MARKERS = (
    "insufficient",
    "resource unavailable",
    "catalog does not contain",
    "cannot be scheduled",
    "failed to provision",
    "no feasible",
    "could not provision",
)


def _is_capacity_error(result: subprocess.CompletedProcess[str]) -> bool:
    text = _strip_ansi(f"{result.stdout}\n{result.stderr}")
    if is_capacity_error(text):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _EXPORT_EVAL_EXTRA_CAPACITY_MARKERS)


def _gpu_slug(gpu: str) -> str:
    return gpu.lower().replace(":", "-").replace("_", "-")


def _stage_minimal_repo(tmp_path: Path) -> Path:
    staged = tmp_path / "repo" / "npa"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "npa" / "pyproject.toml", staged / "pyproject.toml")
    shutil.copytree(ROOT / "npa" / "src", staged / "src", dirs_exist_ok=True)
    return staged.parent


def _tiny_policy_setup() -> str:
    return r'''

# Stage 0b: create the representative policy checkpoint used by the live e2e.
mkdir -p /tmp/npa-sonic-e2e-input
cat > /tmp/npa-sonic-e2e-input/tiny_policy.py <<'PY'
import torch


class TinySonicPolicy(torch.nn.Module):
    observation_dim = 8
    action_dim = 2
    control_dt = 0.02
    obs_spec = {
        "name": "actor_obs",
        "shape": [8],
        "fields": [
            {"name": "velocity", "dim": 1, "units": "m/s"},
            {"name": "pitch", "dim": 1, "units": "rad"},
            {"name": "x_position", "dim": 1, "units": "m"},
            {"name": "time", "dim": 1, "units": "s"},
            {"name": "episode_bias", "dim": 1},
            {"name": "phase_sin", "dim": 1},
            {"name": "phase_cos", "dim": 1},
            {"name": "command", "dim": 1},
        ],
    }
    action_spec = {
        "name": "joint_targets",
        "shape": [2],
        "fields": [
            {"name": "left_hip_pitch", "dim": 1, "units": "rad"},
            {"name": "right_hip_pitch", "dim": 1, "units": "rad"},
        ],
    }

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.12, -0.03, 0.02, 0.01, 0.04, 0.06, 0.03, 0.35],
                        [0.10, 0.02, -0.01, 0.01, 0.03, -0.04, 0.05, 0.30],
                    ]
                )
            )
            self.linear.bias.copy_(torch.tensor([0.05, 0.04]))

    def forward(self, obs):
        return torch.tanh(self.linear(obs))
PY

python3 <<'PY'
from pathlib import Path

import torch
from tiny_policy import TinySonicPolicy

root = Path("/tmp/npa-sonic-e2e-input")
policy = TinySonicPolicy().eval()
torch.save({"actor_model_state_dict": policy.state_dict()}, root / "tiny_policy.pt")
(root / "tiny_policy_config.yaml").write_text(
    """
policy:
  class: tiny_policy.TinySonicPolicy
  kwargs: {}
control_dt: 0.02
""".strip()
    + "\n",
    encoding="utf-8",
)
PY
'''


def _resolve_sky_bin() -> str:
    configured = os.environ.get("NPA_SKYPILOT_BIN", "").strip()
    if configured:
        return configured
    discovered = shutil.which("sky")
    if discovered:
        return discovered
    pytest.skip("SkyPilot CLI not found via NPA_SKYPILOT_BIN or PATH")


def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def _write_capture(path: Path, result: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "cmd": result.args,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _extract_json(output: str, start_marker: str, end_marker: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"{re.escape(start_marker)}\s*(.*?)\s*{re.escape(end_marker)}",
        re.DOTALL,
    )
    match = pattern.search(output)
    assert match, f"missing marker pair {start_marker}/{end_marker}"
    payload = _remove_sky_log_prefixes(match.group(1))
    start = payload.find("{")
    end = payload.rfind("}")
    assert start >= 0 and end > start, f"no JSON object found between {start_marker}"
    return json.loads(payload[start : end + 1])


def _strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def _remove_sky_log_prefixes(value: str) -> str:
    lines = []
    for line in _strip_ansi(value).splitlines():
        lines.append(re.sub(r"^\([^)]*\)\s*", "", line))
    return "\n".join(lines)


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"cmd={result.args!r}\n"
        f"returncode={result.returncode}\n"
        f"stdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
