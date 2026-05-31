from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import pytest
from PIL import Image
from PIL import ImageDraw

from npa.workbench.vlm_eval import DEFAULT_MODEL, DEFAULT_RUBRIC, evaluate_vlm, write_result
from test_vlm_eval_backend import LIVE_GPU_REQUEST_TIMEOUT_S
from test_vlm_eval_backend import live_gpu_endpoint as live_gpu_endpoint


pytestmark = [pytest.mark.gpu, pytest.mark.e2e]


def test_vlm_eval_loop_e2e_scores_rollouts_and_reports(
    live_gpu_endpoint, tmp_path: Path
) -> None:
    rollouts = _write_representative_rollouts(tmp_path / "rollouts")
    output_dir = tmp_path / "vlm-eval-loop"
    success_threshold = 0.5
    started_at = time.monotonic()

    per_rollout: list[dict[str, object]] = []
    for rollout_id, rollout_path, task in rollouts:
        result = evaluate_vlm(
            input_path=str(rollout_path),
            output_path=str(output_dir / rollout_id),
            task=task,
            backend="self-hosted",
            model=DEFAULT_MODEL,
            endpoint_url=live_gpu_endpoint.endpoint_url,
            frame_selection="keyframes",
            max_frames=4,
            rubric=DEFAULT_RUBRIC,
            success_threshold=success_threshold,
            timeout_s=LIVE_GPU_REQUEST_TIMEOUT_S,
        )
        write_result(asdict(result), result_uri=result.result_uri)
        per_rollout.append(
            {
                "rollout_id": rollout_id,
                "success": bool(result.passed),
                "score": float(result.score),
                "rationale": result.rationale,
                "status": result.status,
                "frame_count": result.frame_count,
                "result_uri": result.result_uri,
            }
        )

    report = _aggregate_report(
        per_rollout,
        model=DEFAULT_MODEL,
        success_threshold=success_threshold,
        latency_s=time.monotonic() - started_at,
    )
    report_path = output_dir / "task_success_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"NPA_VLM_LOOP_E2E_GPU_TYPE={live_gpu_endpoint.gpu_type}")
    print(f"NPA_VLM_LOOP_E2E_MODEL={DEFAULT_MODEL}")
    print(f"NPA_VLM_LOOP_E2E_LATENCY_SECONDS={report['latency_s']:.2f}")
    print(
        "NPA_VLM_LOOP_E2E_VLLM_STARTUP="
        + _compact_startup_log(live_gpu_endpoint.startup_log)
    )
    print("NPA_VLM_LOOP_E2E_ROLLOUT_OUTPUTS=" + json.dumps(per_rollout, sort_keys=True))
    print("NPA_VLM_LOOP_E2E_AGGREGATE_REPORT=" + json.dumps(report, sort_keys=True))

    assert report_path.exists()
    assert report["total_rollouts"] == 2
    assert 0.0 <= report["mean_score"] <= 1.0
    assert 0.0 <= report["success_rate"] <= 1.0
    assert isinstance(report["task_success"], bool)

    for item in per_rollout:
        structured = {
            "success": item["success"],
            "score": item["score"],
            "rationale": item["rationale"],
        }
        assert set(structured) == {"success", "score", "rationale"}
        assert isinstance(structured["success"], bool)
        assert isinstance(structured["score"], float)
        assert 0.0 <= structured["score"] <= 1.0
        assert isinstance(structured["rationale"], str)
        assert structured["rationale"].strip()
        assert isinstance(item["frame_count"], int)
        assert item["frame_count"] > 0
        assert Path(str(item["result_uri"])).exists()


def _aggregate_report(
    per_rollout: list[dict[str, object]],
    *,
    model: str,
    success_threshold: float,
    latency_s: float,
) -> dict[str, object]:
    total = len(per_rollout)
    passed = sum(1 for item in per_rollout if item["success"] is True)
    mean_score = sum(float(item["score"]) for item in per_rollout) / total
    success_rate = passed / total
    return {
        "status": "completed",
        "model": model,
        "success_threshold": success_threshold,
        "total_rollouts": total,
        "passed_rollouts": passed,
        "success_rate": round(success_rate, 4),
        "mean_score": round(mean_score, 4),
        "task_success": mean_score >= success_threshold,
        "latency_s": round(latency_s, 2),
        "rollouts": per_rollout,
    }


def _write_representative_rollouts(root: Path) -> list[tuple[str, Path, str]]:
    task = "Move the red block into the green target zone."
    return [
        (
            "block-in-target",
            _write_rollout(
                root / "block-in-target",
                ((12, 58), (34, 48), (56, 38), (68, 30)),
                final_in_target=True,
                task=task,
            ),
            task,
        ),
        (
            "block-misses-target",
            _write_rollout(
                root / "block-misses-target",
                ((12, 58), (22, 58), (32, 58), (42, 58)),
                final_in_target=False,
                task=task,
            ),
            task,
        ),
    ]


def _write_rollout(
    root: Path,
    positions: tuple[tuple[int, int], ...],
    *,
    final_in_target: bool,
    task: str,
) -> Path:
    root.mkdir(parents=True)
    for index, (x_pos, y_pos) in enumerate(positions):
        image = Image.new("RGB", (96, 96), (246, 247, 249))
        draw = ImageDraw.Draw(image)
        draw.rectangle((64, 20, 88, 44), outline=(28, 138, 70), width=4)
        draw.rectangle((8, 72, 88, 84), fill=(96, 96, 96))
        draw.line((20, 72, x_pos + 8, y_pos + 16), fill=(55, 80, 140), width=3)
        draw.rectangle((x_pos, y_pos, x_pos + 16, y_pos + 16), fill=(210, 40, 40))
        if final_in_target and index == len(positions) - 1:
            draw.text((56, 50), "target reached", fill=(28, 100, 55))
        image.save(root / f"frame-{index:03d}.png")
    (root / "manifest.json").write_text(
        json.dumps({"task": task, "format": "npa_vlm_eval_rollout_fixture_v1"}) + "\n",
        encoding="utf-8",
    )
    return root


def _compact_startup_log(startup_log: str) -> str:
    interesting = [
        line.strip()
        for line in startup_log.splitlines()
        if "NPA_VLM_EVAL_SERVER_READY" in line
        or "Uvicorn running" in line
        or "Qwen/Qwen2-VL-7B-Instruct" in line
    ]
    text = " ".join(interesting or startup_log.splitlines()[-8:])
    return " ".join(text.split())[-1000:]
