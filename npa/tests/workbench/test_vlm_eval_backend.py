from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
import time
import uuid

import numpy as np
import pytest
from PIL import Image
from PIL import ImageDraw

from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_BENCHMARK_PATH,
    VlmStructuredResponse,
    benchmark_vlm_eval,
    evaluate_stub,
    evaluate_vlm,
    load_benchmark_dataset,
    parse_structured_response,
    select_rollout_frames,
)

LIVE_GPU_ACCELERATORS = ("H100:1", "H200:1", "A100:1", "L40S:1", "RTX6000:1")
LIVE_GPU_PORT = 8000
LIVE_GPU_LAUNCH_TIMEOUT_S = int(
    os.environ.get("NPA_VLM_EVAL_LIVE_LAUNCH_TIMEOUT_SECONDS", "3600")
)
LIVE_GPU_HEALTH_TIMEOUT_S = int(
    os.environ.get("NPA_VLM_EVAL_LIVE_HEALTH_TIMEOUT_SECONDS", "1800")
)
LIVE_GPU_REQUEST_TIMEOUT_S = float(
    os.environ.get("NPA_VLM_EVAL_LIVE_REQUEST_TIMEOUT_SECONDS", "240")
)
_CREDENTIAL_ERROR_PATTERNS = (
    "authentication",
    "authorize",
    "credential",
    "credentials",
    "login",
    "no cloud",
    "not configured",
    "permission",
)


@dataclass(frozen=True)
class LiveGpuEndpoint:
    cluster_name: str
    endpoint_url: str
    gpu_type: str
    startup_log: str


@pytest.fixture
def live_gpu_endpoint(tmp_path_factory: pytest.TempPathFactory) -> LiveGpuEndpoint:
    """Serve Qwen2-VL-7B through vLLM on a live SkyPilot GPU cluster."""

    sky_bin = _resolve_sky_bin()
    if sky_bin is None:
        pytest.skip("SkyPilot CLI not found; set NPA_SKYPILOT_BIN to run the live GPU test")

    _require_skypilot_credentials(sky_bin)
    reuse_cluster_name = os.environ.get("NPA_VLM_EVAL_LIVE_REUSE_CLUSTER", "").strip()
    if reuse_cluster_name:
        try:
            yield _existing_live_gpu_endpoint(sky_bin, reuse_cluster_name)
        finally:
            _sky_down(sky_bin, reuse_cluster_name)
        return

    work_dir = tmp_path_factory.mktemp("vlm-eval-live-gpu")
    cluster_name = f"npa-vlm-live-{uuid.uuid4().hex[:8]}"
    try:
        yield _launch_live_gpu_endpoint(sky_bin, cluster_name, work_dir)
    finally:
        _sky_down(sky_bin, cluster_name)


def test_golden_set_scores_known_good_and_bad_rollouts(monkeypatch, tmp_path: Path) -> None:
    good_rollout = _write_image_rollout(tmp_path / "good", [(20, 20, 20), (40, 120, 40), (20, 220, 20)])
    bad_rollout = _write_image_rollout(tmp_path / "bad", [(20, 20, 20), (120, 40, 40), (220, 20, 20)])

    def fake_vlm_call(**kwargs):
        assert kwargs["model"] == DEFAULT_MODEL
        assert kwargs["backend"] == "self-hosted"
        assert kwargs["frames"]
        assert all(frame.media_type == "image/png" for frame in kwargs["frames"])
        score = 0.92 if "completed placement" in kwargs["prompt"] else 0.18
        return VlmStructuredResponse(
            success=score >= 0.8,
            score=score,
            rationale="golden-set fixture",
        )

    monkeypatch.setattr(vlm_eval, "_call_openai_compatible", fake_vlm_call)

    good = evaluate_vlm(
        input_path=str(good_rollout),
        output_path=str(tmp_path / "good-out"),
        task="completed placement",
        success_threshold=0.8,
    )
    bad = evaluate_vlm(
        input_path=str(bad_rollout),
        output_path=str(tmp_path / "bad-out"),
        task="failed placement",
        success_threshold=0.8,
    )

    assert good.passed is True
    assert bad.passed is False
    assert good.score >= good.success_threshold
    assert bad.score < bad.success_threshold
    assert all(0.0 <= result.score <= 1.0 for result in (good, bad))
    assert good.frame_selection == "keyframes"
    assert good.frame_count == 3


def test_contract_matches_stub_scalar_score_range(tmp_path: Path) -> None:
    stub = evaluate_stub(
        input_path="rollouts",
        output_path=str(tmp_path / "stub"),
        score=0.61,
    )
    real = evaluate_vlm(
        input_path="rollouts",
        output_path=str(tmp_path / "real"),
        score=0.61,
    )

    assert set(asdict(real)) == set(asdict(stub))
    for result in (stub, real):
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)


def test_parse_structured_response_clamps_score() -> None:
    parsed = parse_structured_response(
        '{"success": true, "score": 1.4, "rationale": "clear completion"}'
    )

    assert parsed.success is True
    assert parsed.score == 1.0
    assert parsed.rationale == "clear completion"


def test_mocked_self_hosted_endpoint_returns_structured_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rollout = _write_image_rollout(
        tmp_path / "mock-rollout",
        [(240, 240, 240), (120, 220, 120), (40, 180, 40)],
    )

    def fake_vlm_call(**kwargs):
        assert kwargs["backend"] == "self-hosted"
        assert kwargs["endpoint_url"] == "http://mock-vllm.local/v1"
        assert kwargs["frames"]
        return VlmStructuredResponse(
            success=True,
            score=0.74,
            rationale="mocked endpoint saw the expected frame sequence",
        )

    monkeypatch.setattr(vlm_eval, "_call_openai_compatible", fake_vlm_call)

    result = evaluate_vlm(
        input_path=str(rollout),
        output_path=str(tmp_path / "mock-out"),
        task="confirm the green object reaches the target",
        backend="self-hosted",
        endpoint_url="http://mock-vllm.local/v1",
        success_threshold=0.5,
    )
    structured = _structured_eval_payload(result)

    assert structured == {
        "success": True,
        "score": 0.74,
        "rationale": "mocked endpoint saw the expected frame sequence",
    }
    assert 0.0 <= structured["score"] <= 1.0


@pytest.mark.gpu
@pytest.mark.e2e
def test_live_gpu_self_hosted_vlm_eval_returns_structured_score(
    live_gpu_endpoint: LiveGpuEndpoint, tmp_path: Path
) -> None:
    rollout = _write_pick_place_rollout(tmp_path / "live-rollout")
    started_at = time.monotonic()

    result = evaluate_vlm(
        input_path=str(rollout),
        output_path=str(tmp_path / "live-out"),
        task=(
            "A robot should move the red block into the green target zone. "
            "Score completion from the image sequence."
        ),
        backend="self-hosted",
        model=DEFAULT_MODEL,
        endpoint_url=live_gpu_endpoint.endpoint_url,
        frame_selection="keyframes",
        max_frames=4,
        success_threshold=0.8,
        timeout_s=LIVE_GPU_REQUEST_TIMEOUT_S,
    )
    latency_s = time.monotonic() - started_at
    structured = _structured_eval_payload(result)

    print(f"NPA_VLM_LIVE_GPU_TYPE={live_gpu_endpoint.gpu_type}")
    print(f"NPA_VLM_LIVE_ENDPOINT={live_gpu_endpoint.endpoint_url}")
    print(f"NPA_VLM_LIVE_LATENCY_SECONDS={latency_s:.2f}")
    print(f"NPA_VLM_LIVE_EVAL_OUTPUT={structured}")
    print(f"NPA_VLM_LIVE_VLLM_STARTUP={live_gpu_endpoint.startup_log[-1000:]}")

    assert set(structured) == {"success", "score", "rationale"}
    assert isinstance(structured["success"], bool)
    assert isinstance(structured["score"], float)
    assert 0.0 <= structured["score"] <= 1.0
    assert isinstance(structured["rationale"], str)
    assert structured["rationale"].strip()
    assert result.backend == "self-hosted"
    assert result.model == DEFAULT_MODEL
    assert result.frame_count in {2, 3, 4}


def test_select_rollout_frames_from_numpy_final_frame(tmp_path: Path) -> None:
    rollout = tmp_path / "episode_0000"
    rollout.mkdir()
    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(5, dtype=np.uint8).reshape(5, 1, 1)
    np.save(rollout / "obs_workspace.npy", frames)

    selected = select_rollout_frames(rollout, frame_selection="final", max_frames=4)

    assert len(selected) == 1
    assert selected[0].label == "obs_workspace.npy:4"
    assert selected[0].media_type == "image/png"
    assert selected[0].data.startswith(b"\x89PNG")

def _structured_eval_payload(result) -> dict[str, object]:
    return {
        "success": bool(result.passed),
        "score": float(result.score),
        "rationale": result.rationale,
    }


def _resolve_sky_bin() -> str | None:
    configured = os.environ.get("NPA_SKYPILOT_BIN", "").strip()
    if configured:
        return configured
    return shutil.which("sky")


def _require_skypilot_credentials(sky_bin: str) -> None:
    result = _run(
        [sky_bin, "check"],
        timeout_s=180,
    )
    if result.returncode == 0:
        print("NPA_VLM_LIVE_SKY_CHECK=ok")
        return

    output = f"{result.stdout}\n{result.stderr}".lower()
    if any(pattern in output for pattern in _CREDENTIAL_ERROR_PATTERNS):
        pytest.skip("SkyPilot/Nebius credentials are not configured for the live GPU test")
    pytest.fail(
        "sky check failed before the live GPU test:\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )


def _launch_live_gpu_endpoint(
    sky_bin: str,
    cluster_name: str,
    work_dir: Path,
) -> LiveGpuEndpoint:
    candidates = _available_gpu_candidates(sky_bin)
    print(f"NPA_VLM_LIVE_GPU_CANDIDATES={','.join(candidates)}")
    errors: list[str] = []

    for accelerator in candidates:
        task_file = work_dir / f"vlm-eval-{accelerator.replace(':', '-')}.yaml"
        task_file.write_text(_live_gpu_task_yaml(accelerator), encoding="utf-8")
        result = _run(
            [
                sky_bin,
                "launch",
                "-c",
                cluster_name,
                str(task_file),
                "-y",
                "--detach-run",
            ],
            timeout_s=LIVE_GPU_LAUNCH_TIMEOUT_S,
        )
        if result.returncode != 0:
            errors.append(_format_attempt_error(accelerator, result.stdout, result.stderr))
            _sky_down(sky_bin, cluster_name)
            continue

        try:
            endpoint = _wait_for_endpoint(sky_bin, cluster_name)
        except AssertionError as exc:
            logs = _sky_logs(sky_bin, cluster_name)
            errors.append(f"{accelerator} endpoint did not become healthy: {exc}\n{logs}")
            _sky_down(sky_bin, cluster_name)
            continue

        startup_log = _sky_logs(sky_bin, cluster_name)
        return LiveGpuEndpoint(
            cluster_name=cluster_name,
            endpoint_url=f"{endpoint.rstrip('/')}/v1",
            gpu_type=accelerator,
            startup_log=startup_log,
        )

    pytest.fail("No live GPU type could serve Qwen2-VL-7B via vLLM:\n" + "\n\n".join(errors))


def _existing_live_gpu_endpoint(sky_bin: str, cluster_name: str) -> LiveGpuEndpoint:
    endpoint = _wait_for_endpoint(sky_bin, cluster_name)
    return LiveGpuEndpoint(
        cluster_name=cluster_name,
        endpoint_url=f"{endpoint.rstrip('/')}/v1",
        gpu_type=_cluster_gpu_type(sky_bin, cluster_name),
        startup_log=_sky_logs(sky_bin, cluster_name),
    )


def _available_gpu_candidates(sky_bin: str) -> list[str]:
    result = _run(
        [sky_bin, "show-gpus", "--cloud", "nebius", "--all"],
        timeout_s=180,
    )
    if result.returncode != 0:
        pytest.fail(
            "sky show-gpus failed before the live GPU test:\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    output = f"{result.stdout}\n{result.stderr}".upper()
    available = [
        accelerator
        for accelerator in LIVE_GPU_ACCELERATORS
        if accelerator.split(":", 1)[0] in output
    ]
    return available or list(LIVE_GPU_ACCELERATORS)


def _wait_for_endpoint(sky_bin: str, cluster_name: str) -> str:
    deadline = time.monotonic() + LIVE_GPU_HEALTH_TIMEOUT_S
    last_status = ""
    while time.monotonic() < deadline:
        status = _run(
            [sky_bin, "status", "--endpoint", str(LIVE_GPU_PORT), cluster_name],
            timeout_s=60,
        )
        last_status = f"{status.stdout}\n{status.stderr}"
        endpoint = _extract_first_endpoint(last_status)
        if endpoint and _endpoint_is_healthy(endpoint):
            return endpoint
        job_status = _run([sky_bin, "logs", cluster_name, "--status"], timeout_s=60)
        if job_status.returncode in {100, 103}:
            logs = _sky_logs(sky_bin, cluster_name)
            raise AssertionError(
                "detached SkyPilot vLLM job failed before the endpoint became healthy\n"
                f"{logs}"
            )
        time.sleep(10)
    raise AssertionError(
        f"endpoint did not report /health within {LIVE_GPU_HEALTH_TIMEOUT_S}s; "
        f"last sky status output:\n{last_status[-2000:]}"
    )


def _cluster_gpu_type(sky_bin: str, cluster_name: str) -> str:
    result = _run([sky_bin, "status", cluster_name], timeout_s=60)
    match = re.search(r"gpus=([A-Z0-9]+:\d+)", f"{result.stdout}\n{result.stderr}")
    if match is None:
        return os.environ.get("NPA_VLM_EVAL_LIVE_REUSE_GPU_TYPE", "unknown")
    return match.group(1)


def _endpoint_is_healthy(endpoint: str) -> bool:
    try:
        import httpx

        response = httpx.get(f"{endpoint.rstrip('/')}/health", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _extract_first_endpoint(text: str) -> str:
    match = re.search(r"https?://[^\s]+", text)
    if match is not None:
        return match.group(0).rstrip(".,")
    host_port = re.search(
        r"((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|\d{1,3}(?:\.\d{1,3}){3}):\d+",
        text,
    )
    if host_port is None:
        return ""
    return f"http://{host_port.group(0).rstrip('.,')}"


def _sky_logs(sky_bin: str, cluster_name: str) -> str:
    result = _run(
        [sky_bin, "logs", cluster_name, "--no-follow", "--tail", "200"],
        timeout_s=120,
    )
    return f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"


def _sky_down(sky_bin: str, cluster_name: str) -> None:
    if not cluster_name:
        return
    result = _run([sky_bin, "down", "--yes", cluster_name], timeout_s=600)
    print(
        f"NPA_VLM_LIVE_SKY_DOWN cluster={cluster_name} "
        f"returncode={result.returncode}"
    )


def _run(cmd: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            124,
            stdout=(exc.stdout or ""),
            stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout_s}s",
        )


def _format_attempt_error(accelerator: str, stdout: str, stderr: str) -> str:
    return (
        f"{accelerator} launch failed\n"
        f"stdout:\n{stdout[-2000:]}\n"
        f"stderr:\n{stderr[-4000:]}"
    )


def _live_gpu_task_yaml(accelerator: str) -> str:
    return textwrap.dedent(
        f"""\
        name: npa-vlm-eval-live-gpu
        resources:
          cloud: nebius
          accelerators: {accelerator}
          ports: {LIVE_GPU_PORT}
          disk_size: 256
        envs:
          VLM_MODEL: "{DEFAULT_MODEL}"
        setup: |
          set -euo pipefail
          python3 -m pip install --upgrade pip
          python3 -m pip install "vllm>=0.8.5" "transformers>=4.49.0" qwen-vl-utils pillow
        run: |
          set -euo pipefail
          pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
          rm -f vlm-server.log
          nohup python3 -m vllm.entrypoints.openai.api_server \\
            --host 0.0.0.0 \\
            --port {LIVE_GPU_PORT} \\
            --model "${{VLM_MODEL}}" \\
            --served-model-name "${{VLM_MODEL}}" \\
            --trust-remote-code \\
            --max-model-len 4096 \\
            --gpu-memory-utilization 0.9 \\
            --limit-mm-per-prompt '{{"image": 4}}' \\
            > vlm-server.log 2>&1 &
          echo "$!" > vlm-server.pid
          for _attempt in $(seq 1 240); do
            if curl -fsS http://127.0.0.1:{LIVE_GPU_PORT}/health >/dev/null 2>&1; then
              echo "NPA_VLM_EVAL_SERVER_READY model=${{VLM_MODEL}}"
              while true; do
                if ! kill -0 "$(cat vlm-server.pid)" >/dev/null 2>&1; then
                  tail -200 vlm-server.log >&2 || true
                  exit 1
                fi
                sleep 30
              done
            fi
            if ! kill -0 "$(cat vlm-server.pid)" >/dev/null 2>&1; then
              tail -200 vlm-server.log >&2 || true
              exit 1
            fi
            sleep 5
          done
          tail -200 vlm-server.log >&2 || true
          exit 1
        """
    )


def test_sample_benchmark_fixture_reports_best_threshold() -> None:
    report = benchmark_vlm_eval(
        dataset=str(DEFAULT_SAMPLE_BENCHMARK_PATH),
        backend="stub",
        thresholds=[0.5, 0.8, 0.9],
        rubrics=["default", "strict"],
        models=[DEFAULT_MODEL],
    )

    assert report.item_count == 4
    assert report.best_config.config.success_threshold == 0.8
    assert report.best_config.metrics.accuracy == 1.0
    assert report.best_config.metrics.precision == 1.0
    assert report.best_config.metrics.recall == 1.0
    assert report.best_config.metrics.true_positives == 2
    assert report.best_config.metrics.true_negatives == 2
    assert all(0.0 <= case.score <= 1.0 for case in report.best_config.results)
    assert {case.score_source for case in report.best_config.results} == {"fixture"}


def test_load_benchmark_dataset_resolves_relative_rollouts() -> None:
    dataset = load_benchmark_dataset(str(DEFAULT_SAMPLE_BENCHMARK_PATH))

    assert dataset.format == "npa_vlm_eval_benchmark_v1"
    assert len(dataset.items) == 4
    assert all(Path(item.rollout).exists() for item in dataset.items)
    assert {"default", "strict"} <= set(dataset.rubrics)


def test_select_rollout_frames_accepts_sample_ppm_fixture() -> None:
    dataset = load_benchmark_dataset(str(DEFAULT_SAMPLE_BENCHMARK_PATH))

    selected = select_rollout_frames(dataset.items[0].rollout, frame_selection="keyframes", max_frames=4)

    assert len(selected) == 1
    assert selected[0].media_type == "image/png"
    assert selected[0].data.startswith(b"\x89PNG")


def _write_image_rollout(root: Path, colors: list[tuple[int, int, int]]) -> Path:
    root.mkdir(parents=True)
    for index, color in enumerate(colors):
        image = Image.new("RGB", (16, 16), color)
        image.save(root / f"frame-{index:03d}.png")
    return root


def _write_pick_place_rollout(root: Path) -> Path:
    root.mkdir(parents=True)
    positions = ((12, 54), (38, 43), (62, 32), (72, 28))
    for index, (x_pos, y_pos) in enumerate(positions):
        image = Image.new("RGB", (96, 96), (245, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.rectangle((64, 20, 88, 44), outline=(30, 140, 70), width=4)
        draw.rectangle((8, 72, 88, 84), fill=(95, 95, 95))
        draw.rectangle((x_pos, y_pos, x_pos + 16, y_pos + 16), fill=(210, 40, 40))
        draw.line((20, 72, x_pos + 8, y_pos + 16), fill=(55, 80, 140), width=3)
        image.save(root / f"frame-{index:03d}.png")
    return root
