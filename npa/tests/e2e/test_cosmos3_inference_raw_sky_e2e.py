from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[3]
YAML_PATH = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "skypilot"
    / "cosmos3-text-to-image-inference.yaml"
)
GPU_CHAIN = ("H100:1", "H200:1", "A100:1", "L40S:1", "RTX6000:1")
PUBLIC_SOURCE_REPO = "https://github.com/NVIDIA/cosmos-framework.git"
PUBLIC_MODEL_ID = "nvidia/Cosmos3-Nano"


def test_cosmos3_text_to_image_raw_sky_public_defaults(tmp_path: Path) -> None:
    """Run Cosmos3 text-to-image inference via raw `sky launch`."""

    env_names = _require_public_runtime()
    sky_bin = _sky_bin()
    image_id = os.environ.get("NPA_COSMOS3_E2E_IMAGE_ID", "")
    run_id = f"cosmos3-infer-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    evidence_dir = Path(
        os.environ.get("NPA_COSMOS3_E2E_EVIDENCE_DIR", str(tmp_path / "evidence"))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    workdir = _copy_clean_workdir(tmp_path / "workdir")
    yaml_path = (
        workdir
        / "npa"
        / "workflows"
        / "workbench"
        / "skypilot"
        / "cosmos3-text-to-image-inference.yaml"
    )
    attempts: list[dict[str, Any]] = []

    for gpu in _gpu_chain():
        cluster = _cluster_name(run_id, gpu)
        stdout_path = evidence_dir / f"{cluster}.stdout.txt"
        stderr_path = evidence_dir / f"{cluster}.stderr.txt"
        cmd = _sky_launch_command(
            sky_bin=sky_bin,
            yaml_path=yaml_path,
            workdir=workdir,
            cluster=cluster,
            run_id=run_id,
            gpu=gpu,
            image_id=image_id,
            env_names=env_names,
        )
        attempt = {
            "cluster": cluster,
            "gpu": gpu,
            "command": _redact_command(cmd, image_id=image_id),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.environ.get("NPA_COSMOS3_E2E_TIMEOUT_SECONDS", "21600")),
                check=False,
            )
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            attempt["returncode"] = result.returncode
            if result.returncode == 0 and '"status": "ok"' in result.stdout:
                attempt["status"] = "passed"
                attempts.append(attempt)
                _write_evidence(evidence_dir, run_id=run_id, attempts=attempts)
                return
            attempt["status"] = "failed"
        finally:
            _sky_down_and_poll(sky_bin, cluster, evidence_dir=evidence_dir)
        attempts.append(attempt)

    _write_evidence(evidence_dir, run_id=run_id, attempts=attempts)
    pytest.fail(
        f"Cosmos3 inference raw SkyPilot run failed on all GPU tiers; evidence={evidence_dir}"
    )


def _require_public_runtime() -> dict[str, str]:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_COSMOS3_E2E") != "1":
        pytest.skip("NPA_COSMOS3_E2E not set")
    missing: list[str] = []
    github_env = os.environ.get("NPA_COSMOS3_GITHUB_TOKEN_ENV", "GITHUB_TOKEN")
    hf_env = os.environ.get("NPA_COSMOS3_HF_TOKEN_ENV", "HF_TOKEN")
    ngc_env = os.environ.get("NPA_COSMOS3_NGC_API_KEY_ENV", "NGC_API_KEY")
    if not os.environ.get(hf_env):
        missing.append(hf_env)
    if os.environ.get("NPA_COSMOS3_REQUIRE_NGC") == "1" and not os.environ.get(ngc_env):
        missing.append(ngc_env)
    if missing:
        pytest.skip(
            "Cosmos3 runtime env is incomplete: " + ", ".join(sorted(set(missing)))
        )
    return {"github": github_env, "hf": hf_env, "ngc": ngc_env}


def _sky_bin() -> str:
    sky_bin = os.environ.get(
        "NPA_SKYPILOT_BIN", "/home/ubuntu/.npa/skypilot-venv/bin/sky"
    )
    if not Path(sky_bin).exists():
        pytest.skip(f"SkyPilot binary not found: {sky_bin}")
    return sky_bin


def _gpu_chain() -> tuple[str, ...]:
    configured = os.environ.get("NPA_COSMOS3_E2E_GPU_CHAIN", "")
    if not configured.strip():
        return GPU_CHAIN
    return tuple(gpu.strip() for gpu in configured.split(",") if gpu.strip())


def _copy_clean_workdir(target: Path) -> Path:
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "*.pyc"
        ),
    )
    return target


def _sky_launch_command(
    *,
    sky_bin: str,
    yaml_path: Path,
    workdir: Path,
    cluster: str,
    run_id: str,
    gpu: str,
    image_id: str,
    env_names: dict[str, str],
) -> list[str]:
    cache_dir = f"/tmp/npa-cosmos3-cache-{run_id}"
    output_dir = f"/tmp/npa-cosmos3-inference-{run_id}"
    cmd = [
        sky_bin,
        "launch",
        "--yes",
        "--cluster",
        cluster,
        "--name",
        cluster,
        "--workdir",
        str(workdir),
        "--infra",
        os.environ.get("NPA_COSMOS3_E2E_INFRA", "nebius/eu-north1"),
        "--gpus",
        gpu,
        "--env",
        f"NPA_COSMOS3_CACHE={cache_dir}",
        "--env",
        f"NPA_COSMOS3_OUTPUT_DIR={output_dir}",
        "--env",
        f"NPA_COSMOS3_OUTPUT_IMAGE={output_dir}/text-to-image.png",
        "--env",
        f"NPA_COSMOS3_SUCCESS_JSON={output_dir}/success.json",
        "--env",
        f"NPA_COSMOS3_GITHUB_TOKEN_ENV={env_names['github']}",
        "--env",
        f"NPA_COSMOS3_HF_TOKEN_ENV={env_names['hf']}",
        "--env",
        f"NPA_COSMOS3_NGC_API_KEY_ENV={env_names['ngc']}",
        "--env",
        f"NPA_COSMOS3_REQUIRE_NGC={os.environ.get('NPA_COSMOS3_REQUIRE_NGC', '0')}",
        "--env",
        f"NPA_COSMOS3_UV_GROUP={os.environ.get('NPA_COSMOS3_UV_GROUP', 'cu130-train')}",
        "--env",
        f"NPA_COSMOS3_INFER_PROMPT={os.environ.get('NPA_COSMOS3_INFER_PROMPT', 'a small robot arm sorting colored blocks on a workbench')}",
        "--secret",
        env_names["hf"],
        str(yaml_path),
    ]
    if image_id:
        cmd[-1:-1] = ["--image-id", image_id]
    for name, default in (
        ("NPA_COSMOS3_SOURCE_REPO", PUBLIC_SOURCE_REPO),
        ("NPA_COSMOS3_MODEL_ID", PUBLIC_MODEL_ID),
        ("NPA_COSMOS3_INFER_COMMAND", ""),
        ("NPA_COSMOS3_OUTPUT_S3_URI", ""),
    ):
        value = os.environ.get(name, "")
        if value and value != default:
            cmd[-1:-1] = ["--env", f"{name}={value}"]
    if os.environ.get(env_names["github"]):
        cmd[-1:-1] = ["--secret", env_names["github"]]
    if os.environ.get("NPA_COSMOS3_REQUIRE_NGC") == "1":
        cmd[-1:-1] = ["--secret", env_names["ngc"]]
    return cmd


def _sky_down_and_poll(sky_bin: str, cluster: str, *, evidence_dir: Path) -> None:
    down = subprocess.run(
        [sky_bin, "down", "--yes", cluster],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(os.environ.get("NPA_COSMOS3_E2E_TEARDOWN_TIMEOUT_SECONDS", "900")),
        check=False,
    )
    (evidence_dir / f"{cluster}.down.stdout.txt").write_text(
        down.stdout, encoding="utf-8"
    )
    (evidence_dir / f"{cluster}.down.stderr.txt").write_text(
        down.stderr, encoding="utf-8"
    )
    deadline = time.monotonic() + int(
        os.environ.get("NPA_COSMOS3_E2E_TEARDOWN_POLL_TIMEOUT_SECONDS", "1200")
    )
    while time.monotonic() < deadline:
        status = subprocess.run(
            [sky_bin, "status", "--refresh"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        (evidence_dir / f"{cluster}.status-after-down.txt").write_text(
            status.stdout + status.stderr, encoding="utf-8"
        )
        if cluster not in status.stdout:
            return
        time.sleep(float(os.environ.get("NPA_COSMOS3_E2E_TEARDOWN_POLL_SECONDS", "30")))
    pytest.fail(f"SkyPilot cluster still present after teardown timeout: {cluster}")


def _cluster_name(run_id: str, gpu: str) -> str:
    return (run_id + "-" + gpu.lower().replace(":", ""))[:63]


def _redact_command(cmd: list[str], *, image_id: str) -> list[str]:
    redacted: list[str] = []
    for part in cmd:
        if part.endswith("/sky"):
            redacted.append("<sky>")
        elif image_id and part == image_id:
            redacted.append("<image-id>")
        else:
            redacted.append(part)
    return redacted


def _write_evidence(
    evidence_dir: Path, *, run_id: str, attempts: list[dict[str, Any]]
) -> None:
    evidence = {
        "run_id": run_id,
        "storage": "node-local with optional configured S3 upload",
        "s3": os.environ.get("NPA_COSMOS3_OUTPUT_S3_URI", "not configured"),
        "attempts": attempts,
    }
    (evidence_dir / "cosmos3-inference-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
