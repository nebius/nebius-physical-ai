"""Verify that visual coverage, disagreement, and ambiguity remain meaningful qualification gates."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import numpy as np

from npa.workbench.vlm_eval.temporal import RUBRIC
from npa.workflows.franka_rl import _recipe
from npa.workflows.franka_rl_vlm import summarize_visual


@pytest.fixture
def visual_case():
    recipe = _recipe(
        SimpleNamespace(
            run_id="test-visual",
            seed=42,
            iterations=1500,
            num_envs=4096,
            eval_episodes=128,
            minimum_success=0.7,
        )
    )
    recipe["visual_eval"] = {
        "arms": ["initial", "trained"],
        "minimum_lift_agreement": 0.8,
        "model": "MiniMaxAI/MiniMax-M3",
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
    }
    rows = []
    for arm in recipe["visual_eval"]["arms"]:
        for condition in recipe["conditions"]:
            for index in range(recipe["capture_episodes"]):
                positive = arm == "trained"
                verdict = {
                    name: {"verdict": "yes" if positive else "no"}
                    for name in ("lifted", "held_at_end")
                }
                verdict.update(
                    scene_disturbed={"verdict": "no"}, failure_modes=["none"]
                )
                rows.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "capture_index": index,
                        "reference_lifted": positive,
                        "reference_valid": True,
                        "visual": {
                            "backend": "token_factory",
                            "verdict": verdict,
                            "model": recipe["visual_eval"]["model"],
                            "rubric_sha256": recipe["visual_eval"]["rubric_sha256"],
                        },
                    }
                )
    return recipe, rows


def test_perfect_visual_agreement_is_separate_from_robot_readiness(visual_case):
    recipe, rows = visual_case
    summary = summarize_visual(rows, recipe)
    assert summary["episodes"] == 32
    assert summary["lift_balanced_accuracy"] == 1.0
    assert summary["visual_audit_passed"] and summary["visual_task_passed"]
    assert not summary["calibrated_on_independent_human_labels"]
    assert (
        "simulation_qualified" not in summary
        and "ready_for_robot_deployment" not in summary
    )


def test_false_positive_lifts_cannot_be_hidden_in_mean_score(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["visual"]["verdict"]["lifted"]["verdict"] = "yes"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_sensitivity"] == 1.0 and summary["lift_specificity"] == 0.0
    assert summary["lift_confusion"]["negative_yes"] == 16
    assert not summary["visual_audit_passed"]


def test_uncertain_lifts_count_against_agreement(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["visual"]["verdict"]["lifted"]["verdict"] = "uncertain"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_balanced_accuracy"] == 0.0
    assert not summary["visual_audit_passed"]


@pytest.mark.parametrize("verdict", ["yes", "uncertain"])
def test_scene_disturbance_or_ambiguity_keeps_visual_task_gate_closed(
    visual_case, verdict
):
    recipe, rows = visual_case
    rows[-1]["visual"]["verdict"]["scene_disturbed"]["verdict"] = verdict
    summary = summarize_visual(rows, recipe)
    assert summary["visual_audit_passed"]
    assert not summary["visual_task_passed"]


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "model", "rubric", "stub", "reference"]
)
def test_incomplete_or_mixed_visual_evidence_is_rejected(visual_case, mutation):
    recipe, rows = deepcopy(visual_case)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "reference":
        rows[-1]["reference_lifted"] = "true"
    else:
        key, value = {
            "model": ("model", "substitute"),
            "rubric": ("rubric_sha256", "x" * 64),
            "stub": ("backend", "stub"),
        }[mutation]
        rows[-1]["visual"][key] = value
    with pytest.raises(ValueError):
        summarize_visual(rows, recipe)


def test_audit_without_both_reference_classes_is_unqualified(visual_case):
    recipe, rows = visual_case
    for row in rows:
        row["reference_lifted"] = False
        row["visual"]["verdict"]["lifted"]["verdict"] = "no"
    summary = summarize_visual(rows, recipe)
    assert summary["lift_balanced_accuracy"] is None
    assert not summary["visual_audit_passed"]


@pytest.mark.parametrize("validity", [False, None])
def test_invalid_or_unverified_physics_never_calibrates_visual_judge(
    visual_case, validity
):
    recipe, rows = visual_case
    for row in rows:
        if validity is None:
            row.pop("reference_valid")
        else:
            row["reference_valid"] = validity
    summary = summarize_visual(rows, recipe)
    assert summary["invalid_or_unverified_physics_episodes"] == 32
    assert summary["reference_valid_episodes"] == 0
    assert summary["lift_confusion"] == {}
    assert summary["lift_balanced_accuracy"] is None
    assert not summary["visual_audit_passed"] and not summary["visual_task_passed"]


@pytest.mark.parametrize(
    "elevated,expected", [([3, 4], False), ([35], False), ([0, 35], True)]
)
def test_elevation_reference_uses_only_frames_supplied_to_judge(
    tmp_path, monkeypatch, elevated, expected
):
    from npa.workflows import franka_rl_vlm

    trajectory = tmp_path / "trajectories/episode_000000"
    trajectory.mkdir(parents=True)
    np.save(trajectory / "rgb.npy", np.zeros((250, 2, 2, 3), dtype=np.uint8))
    geometry = np.zeros((250, 3))
    geometry[elevated, 2] = 0.2
    np.save(trajectory / "object_metrics.npy", geometry)
    supplied = [0, 35, 249]
    monkeypatch.setattr(
        franka_rl_vlm,
        "judge_manipulation",
        lambda **kwargs: {"frames": [{"index": index} for index in supplied]},
    )
    recipe = {
        "assets": {"description": "orange spool"},
        "minimum_object_height_m": 0.1,
        "visual_eval": {"frame_count": 16, "model": "test"},
    }
    metadata = {"episode_results": [{"length": 250}], "fps": 50}
    row = franka_rl_vlm._judge_episode(
        tmp_path, tmp_path / "output", 0, metadata, recipe, None
    )
    assert row["reference_lifted"] is expected
    assert row["reference_sample_indices"] == supplied


@pytest.mark.parametrize("publication_fails", [False, True])
def test_failed_judge_retains_diagnostics_without_success_manifest(
    tmp_path, monkeypatch, publication_fails
):
    from npa.workflows import franka_rl, franka_rl_vlm
    from npa.workflows.lerobot_transfer_data import materialize

    def fail_judge(prepared, output, **kwargs):
        episode = output / "episode-000000"
        episode.mkdir(parents=True)
        (episode / "request.json").write_text('{"model":"test"}')
        (episode / "response.json").write_text('{"choices":[]}')
        raise ValueError("invalid judge response")

    monkeypatch.setattr(franka_rl_vlm, "evaluate_captures", fail_judge)
    monkeypatch.setattr(franka_rl, "materialize", lambda *args: tmp_path)
    if publication_fails:

        def fail_publish(*args):
            raise OSError("storage unavailable")

        monkeypatch.setattr(franka_rl, "publish", fail_publish)
    with pytest.raises(ValueError, match="invalid judge response"):
        franka_rl.main(
            [
                "visual-evaluate",
                "--input-path",
                str(tmp_path),
                "--output-path",
                str(tmp_path / "visual"),
            ]
        )
    assert not (tmp_path / "visual").exists()
    if not publication_fails:
        evidence = next((tmp_path / "visual-failures").iterdir())
        materialize(str(evidence), tmp_path / "unused")
        assert (
            evidence / "episode-000000/response.json"
        ).read_text() == '{"choices":[]}'
        assert (evidence / "failure.json").is_file()
        assert not (evidence / "visual-evaluation.json").exists()


def test_failure_directory_error_does_not_replace_judge_error(
    tmp_path, monkeypatch, capsys
):
    from pathlib import Path
    from npa.workflows.franka_rl import _publish_stage_failure

    def fail_mkdir(*args, **kwargs):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    _publish_stage_failure(
        tmp_path, str(tmp_path / "visual"), ValueError("invalid judge")
    )
    assert '"failure_evidence_published": false' in capsys.readouterr().out


def test_invalid_judgment_remains_in_denominator_and_closes_both_gates(visual_case):
    recipe, rows = visual_case
    rows[-1]["visual"].update(
        status="invalid_response",
        verdict=None,
        validation_error={"type": "evidence_contract", "message": "Invalid hold"},
    )
    summary = summarize_visual(rows, recipe)
    assert summary["episodes"] == 32 and summary["invalid_responses"] == 1
    assert summary["lift_confusion"]["positive_invalid"] == 1
    assert summary["lift_sensitivity"] == 15 / 16
    condition = summary["outcomes"]["trained"][rows[-1]["condition"]]
    assert condition["episodes"] == 4 and condition["invalid_count"] == 1
    assert condition["held_at_end_rate"] == 3 / 4
    assert not summary["visual_audit_passed"] and not summary["visual_task_passed"]


def test_visual_stage_continues_after_invalid_response_and_reports_failed_gate(
    tmp_path, monkeypatch, visual_case
):
    import json
    from npa.workflows import franka_rl_vlm, franka_rl_report
    from npa.workflows.lerobot_transfer_data import file_sha256

    recipe, rows = visual_case
    rows[0]["visual"].update(
        status="invalid_response",
        verdict=None,
        validation_error={"type": "evidence_contract", "message": "Invalid hold"},
    )
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "evaluation.json").write_text("{}")
    (tmp_path / "trajectories/meta.json").write_text("{}")
    recipe["capture_seed"] = 123
    metadata = {"num_episodes": len(rows)}
    monkeypatch.setattr(
        franka_rl_vlm, "_capture_contract", lambda path: ({}, metadata, recipe)
    )
    monkeypatch.setattr(
        franka_rl_vlm,
        "TokenFactoryClient",
        lambda: SimpleNamespace(list_models=lambda: [recipe["visual_eval"]["model"]]),
    )
    seen = []

    def judge(evaluated, output, index, metadata, recipe, client, previous):
        seen.append(index)
        return dict(rows[index], episode_index=index)

    monkeypatch.setattr(franka_rl_vlm, "_judge_episode", judge)
    visual = tmp_path / "visual"
    franka_rl_vlm.evaluate_captures(tmp_path, visual)
    assert sorted(seen) == list(range(32))
    audit = json.loads((visual / "visual-evaluation.json").read_text())
    assert audit["summary"]["invalid_responses"] == 1
    report = {"recipe": recipe, "simulation_qualified": True}
    output = tmp_path / "report"
    output.mkdir()
    franka_rl_report._attach_visual(report, tmp_path, visual, output)
    assert report["physics_qualified"] and not report["simulation_qualified"]
    assert report["visual_evaluation_sha256"] == file_sha256(
        visual / "visual-evaluation.json"
    )


def test_replay_requires_exact_capture_contract(tmp_path):
    import json
    from npa.workflows.franka_rl_vlm import _prior_responses

    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "capture-contract.json").write_text(
        json.dumps({"evaluation_sha256": "wrong", "capture_sha256": "wrong"})
    )
    (tmp_path / "evaluation.json").write_text("{}")
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "trajectories/meta.json").write_text("{}")
    with pytest.raises(ValueError, match="different evaluation or capture"):
        _prior_responses(previous, tmp_path, {"num_episodes": 32})


def _recovery_inputs(tmp_path, recipe, rows):
    import json
    from npa.workflows.lerobot_transfer_data import file_sha256

    recipe["assets"] = {"description": "orange spool"}
    recipe["visual_eval"]["frame_count"] = 16
    metadata = {"num_episodes": len(rows), "episode_results": [], "fps": 2}
    for index, row in enumerate(rows):
        trajectory = tmp_path / "trajectories" / f"episode_{index:06d}"
        trajectory.mkdir(parents=True)
        np.save(trajectory / "rgb.npy", np.zeros((10, 16, 16, 3), dtype=np.uint8))
        geometry = np.zeros((10, 3))
        geometry[:, 2] = 0.2 if row["arm"] == "trained" else 0
        np.save(trajectory / "object_metrics.npy", geometry)
        metadata["episode_results"].append(
            {key: row[key] for key in ("arm", "condition", "capture_index")}
            | {"length": 10}
        )
    (tmp_path / "evaluation.json").write_text(json.dumps({"recipe": recipe}))
    (tmp_path / "trajectories/meta.json").write_text(json.dumps(metadata))
    contract = {
        "evaluation_sha256": file_sha256(tmp_path / "evaluation.json"),
        "capture_sha256": file_sha256(tmp_path / "trajectories/meta.json"),
    }
    return metadata, contract


def test_recovery_replays_all_first_responses_and_requests_only_missing_episodes(
    tmp_path, monkeypatch, visual_case
):
    import json
    from npa.workflows import franka_rl_vlm

    recipe, rows = visual_case
    metadata, contract = _recovery_inputs(tmp_path, recipe, rows)
    calls = []
    invalid_response = {
        "model": recipe["visual_eval"]["model"],
        "id": "fixture-response",
        "choices": [],
    }

    def complete(**kwargs):
        calls.append(kwargs)
        return invalid_response

    client = SimpleNamespace(
        list_models=lambda: [recipe["visual_eval"]["model"]],
        chat_completion=complete,
        last_request_metrics={"attempts": 1},
    )
    monkeypatch.setattr(franka_rl_vlm, "TokenFactoryClient", lambda: client)
    monkeypatch.setattr(
        franka_rl_vlm, "_capture_contract", lambda path: ({}, metadata, recipe)
    )
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "capture-contract.json").write_text(json.dumps(contract))
    for index in range(22):
        franka_rl_vlm._judge_episode(
            tmp_path, previous, index, metadata, recipe, client
        )
    (previous / "episode-000021/verdict.json").unlink()
    assert len(calls) == 22
    output = tmp_path / "output"
    franka_rl_vlm.evaluate_captures(tmp_path, output, previous=previous)
    audit = json.loads((output / "visual-evaluation.json").read_text())
    assert len(calls) == 32 and audit["summary"]["invalid_responses"] == 32
    assert [row["visual"]["response_source"] for row in audit["episodes"]] == [
        "replayed"
    ] * 22 + ["live"] * 10
    assert audit["episodes"][21]["visual"]["transport"] is None
    for index in range(22):
        name = f"episode-{index:06d}/response.json"
        assert (previous / name).read_bytes() == (output / name).read_bytes()
