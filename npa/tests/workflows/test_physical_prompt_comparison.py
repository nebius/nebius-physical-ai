"""Verify physical-prompt conditioning, sealed artifacts, and complete blinded comparisons."""

import json
from pathlib import Path

import pytest

from npa.workflows import physical_prompt_comparison_contract as contract
from npa.workflows import physical_prompt_comparison_evaluate as evaluation
from npa.workflows import physical_prompt_comparison_generate as generation
from npa.workflows.physical_prompt_comparison_artifacts import (
    materialize,
    publish,
    write_json,
)


def recipe():
    return {
        "schema": "npa.physical-prompt-comparison.recipe.v1",
        "solution": contract.SOLUTION,
        "arms": list(contract.ARMS),
        "seeds": [0, 1],
        "evaluation_frames": 16,
        "cases": [
            {
                "id": name,
                "prompt": prompt,
                "assertions": list(assertions),
                "physics_reasoning": "A force produces continuous movement.",
                "physics_negative_prompt": "Motion before contact, disappearing objects.",
            }
            for name, prompt, assertions in contract.SCENARIOS
        ],
    }


def completion(content, finish="stop", model="test-model"):
    return {
        "id": "request-1",
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def test_conditioning_arms_preserve_base_scenario_and_isolate_negative_guidance():
    case = recipe()["cases"][0]
    baseline = contract.arm_prompts(case, "baseline")
    physical = contract.arm_prompts(case, "physics")
    negative = contract.arm_prompts(case, "physics-negative")
    assert baseline == (case["prompt"], "")
    assert (
        physical[0] == negative[0] == case["prompt"] + " " + case["physics_reasoning"]
    )
    assert physical[1] == ""
    assert negative[1] == case["physics_negative_prompt"]


@pytest.mark.parametrize(
    "payload",
    [
        completion("{}", finish="length"),
        completion("{}", model="different-model"),
        completion("```json\n{}\n```"),
        completion('{"score": NaN}'),
        completion('{"score": 1, "score": 0}'),
        completion("[]"),
    ],
)
def test_provider_output_is_not_repaired_into_success(payload):
    with pytest.raises(ValueError):
        contract.completion_json(payload, "test-model")


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"physics_reasoning": "reason"},
        {"physics_reasoning": "", "physics_negative_prompt": "negative"},
        {"physics_reasoning": "word " * 101, "physics_negative_prompt": "negative"},
    ],
)
def test_invalid_physical_prompts_are_rejected(fields):
    with pytest.raises(ValueError):
        contract.validate_physics(fields)


def test_rejected_provider_response_is_preserved(monkeypatch, tmp_path):
    response = completion(
        json.dumps(
            {
                "physics_reasoning": "word " * 101,
                "physics_negative_prompt": "disappearing objects",
            }
        )
    )

    class Client:
        def chat_completion(self, **kwargs):
            return response

    monkeypatch.setattr("npa.clients.token_factory.TokenFactoryClient", Client)
    with pytest.raises(ValueError, match="100 words"):
        contract.prepare(tmp_path, "test", [0, 1], "test-model")
    persisted = json.loads((tmp_path / "responses/domino-contact.json").read_text())
    assert persisted == response
    assert not (tmp_path / "recipe.json").exists()


def test_artifact_readback_detects_changed_or_added_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "recipe.json", recipe())
    destination = tmp_path / "destination"
    publish(source, str(destination))
    assert materialize(str(destination), tmp_path / "unused") == destination
    (destination / "injected.txt").write_text("unsealed bytes")
    with pytest.raises(ValueError, match="checksum"):
        materialize(str(destination), tmp_path / "unused")
    (destination / "injected.txt").unlink()
    (destination / "recipe.json").write_text("changed bytes")
    with pytest.raises(ValueError, match="checksum"):
        materialize(str(destination), tmp_path / "unused")


@pytest.mark.parametrize("verdict", ["yes", True, 0.8])
def test_judge_rejects_non_contract_verdicts(verdict):
    with pytest.raises(ValueError):
        evaluation.validate_judgment(
            {"assertions": [{"index": 0, "verdict": verdict, "evidence": "Frame 0"}]}, 1
        )


def test_judge_requires_exact_assertion_coverage():
    row = {
        "index": 0,
        "verdict": "unknown",
        "evidence": "Contact occurs between sampled frames.",
    }
    assert evaluation.validate_judgment({"assertions": [row]}, 1) == [row]
    with pytest.raises(ValueError):
        evaluation.validate_judgment({"assertions": [row, row]}, 2)
    with pytest.raises(ValueError):
        evaluation.validate_judgment({"assertions": []}, 1)


def test_summary_preserves_negative_results_and_unknown_assertions():
    spec = recipe()
    judgments = []
    for name, seed, arm in generation.expected_grid(spec):
        score = 1.0 if arm == "baseline" else 0.0
        judgments.append(
            {
                "case_id": name,
                "seed": seed,
                "arm": arm,
                "score": score,
                "assertions": [{"verdict": "pass" if score else "unknown"}],
            }
        )
    result = evaluation.summarize(spec, judgments)
    assert len(judgments) == 36
    assert len(result["pairs"]) == 12
    assert result["paired_mean_difference"] == -1
    assert result["unknown_assertions"] == 24
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.summarize(spec, judgments[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.summarize(spec, judgments + [judgments[0]])


def test_gpu_worker_reuses_model_and_pairs_every_scenario(monkeypatch, tmp_path):
    prepared, output = tmp_path / "prepared", tmp_path / "generated"
    write_json(prepared / "recipe.json", recipe())
    loads, calls = [], []
    model = object()
    monkeypatch.setattr(
        generation, "load_video_pipeline", lambda name: loads.append(name) or model
    )
    monkeypatch.setattr(generation, "_check_tokens", lambda *args: None)

    def native(solution, prompt, seed, path, negative_prompt, pipeline):
        calls.append((solution, prompt, seed, negative_prompt, pipeline))
        return {"seed": seed, "observed": {"frame_count": 81}}

    monkeypatch.setattr(generation, "generate_video", native)
    generation.generate(prepared, output, 1)
    assert loads == [contract.SOLUTION]
    assert len(calls) == 18
    assert all(call[2] == 1 and call[4] is model for call in calls)
    manifest = json.loads((output / "generation.json").read_text())
    assert {(r["case_id"], r["seed"], r["arm"]) for r in manifest["videos"]} == {
        key for key in generation.expected_grid(recipe()) if key[1] == 1
    }


def test_invalid_seed_fails_before_gpu_initialization(monkeypatch, tmp_path):
    write_json(tmp_path / "recipe.json", recipe())
    monkeypatch.setattr(
        generation, "load_video_pipeline", lambda _: pytest.fail("loaded GPU")
    )
    with pytest.raises(ValueError, match="seed"):
        generation.generate(tmp_path, tmp_path / "output", 55)


def test_shards_cannot_substitute_recipes_or_omit_pairs(monkeypatch, tmp_path):
    spec = recipe()
    write_json(tmp_path / "recipe.json", spec)
    manifest = {
        "schema": "npa.physical-prompt-comparison.generation.v1",
        "status": "completed",
        "seed": 0,
        "recipe_sha256": evaluation.file_hash(tmp_path / "recipe.json"),
        "videos": [],
    }
    write_json(tmp_path / "generation.json", manifest)
    with pytest.raises(ValueError, match="coverage"):
        evaluation.collect_shards([tmp_path])
    manifest["recipe_sha256"] = "0" * 64
    write_json(tmp_path / "generation.json", manifest)
    with pytest.raises(ValueError, match="recipe"):
        evaluation.collect_shards([tmp_path])


def test_judge_request_is_blind_to_prompt_arm(monkeypatch, tmp_path):
    case = recipe()["cases"][0]
    expected = [
        {"index": i, "verdict": "unknown", "evidence": "Frame 0 is inconclusive."}
        for i in range(3)
    ]
    calls = []

    class Client:
        def chat_completion(self, **kwargs):
            calls.append(kwargs)
            return completion(json.dumps({"assertions": expected}))

    monkeypatch.setattr(
        evaluation,
        "_frames",
        lambda *args: ([{"type": "text", "text": "Frame 0"}], [0]),
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"decoded-video-test")
    row = {
        "case_id": case["id"],
        "seed": 0,
        "arm": "physics-negative",
        "relative_video": "hidden/video.mp4",
    }
    result = evaluation._judge(Client(), "test-model", case, row, video, 16, tmp_path)
    serialized = json.dumps(calls[0])
    assert "physics-negative" not in serialized
    assert case["physics_reasoning"] not in serialized
    assert case["physics_negative_prompt"] not in serialized
    assert result["score"] == 0


def test_workflow_plans_full_parallel_gpu_comparison():
    import yaml

    root = Path(__file__).resolve().parents[3]
    spec = yaml.safe_load(
        (root / "workflows/testing/physical-prompt-comparison.yaml").read_text()
    )
    assert spec["metadata"]["executionMode"] == "runtime"
    assert spec["states"]["generate-pairs"]["parallel"] == ["generate-a", "generate-b"]
    assert spec["resources"]["gpu"]["accelerators"] == "RTXPRO6000:1"
    for name in ("generate-a", "generate-b"):
        argv = spec["states"][name]["run"]["argv"]
        assert argv[2:4] == ["npa.workflows.physical_prompt_comparison", "generate"]
        assert "--seed" in argv and "--input-path" in argv and "--output-path" in argv


@pytest.mark.parametrize(
    "response", [completion("{", finish="length"), completion("{}")]
)
def test_rejected_judge_response_is_preserved(monkeypatch, tmp_path, response):
    class Client:
        def chat_completion(self, **kwargs):
            return response

    monkeypatch.setattr(evaluation, "_frames", lambda *args: ([], [0]))
    case = recipe()["cases"][0]
    with pytest.raises(ValueError):
        evaluation._judge(
            Client(),
            "test-model",
            case,
            {"relative_video": "clip/video.mp4"},
            tmp_path / "video.mp4",
            16,
            tmp_path,
        )
    saved = list((tmp_path / "responses").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text()) == response
