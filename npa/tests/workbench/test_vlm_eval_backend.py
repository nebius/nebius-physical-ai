from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

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
