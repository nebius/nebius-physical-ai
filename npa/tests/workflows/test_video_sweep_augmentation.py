"""Verify grounded prompt preparation, malformed responses and input binding."""

import copy
import json
from types import SimpleNamespace

import pytest

from npa.workflows.video_sweep import (
    artifacts,
    augmentation,
    execution,
    matrix,
    matrix_view,
    planning,
)


def test_description_frames_have_matching_timestamps():
    images = [{"type": "image_url", "image_url": {"url": str(i)}} for i in range(2)]
    blocks = planning._timed_frames(
        images, {"sample_indices": [0, 24], "sample_times": [0.0, 1.0]}
    )
    assert blocks[1::2] == images
    assert blocks[0]["text"] == "Source frame 0; time 0.0 seconds"
    assert blocks[2]["text"] == "Source frame 24; time 1.0 seconds"


@pytest.fixture
def brief():
    return {
        "scene": "A loaded forklift moves along a warehouse aisle.",
        "appearance": "Warm light with consistent contact shadows.",
        "preserve": ["Vehicle geometry", "Supported pallet", "Camera and motion"],
        "avoid": ["Floating tires", "Deformed forks", "Temporal flicker"],
    }


def _client(answer, observed):
    def call(**request):
        observed.append(request)
        return {
            "model": request["model"],
            "id": "synthetic-test-request",
            "choices": [{"finish_reason": "stop", "message": {"content": answer}}],
        }

    return SimpleNamespace(chat_completion=call)


def test_augmented_appearance_reaches_cosmos_with_protected_source_invariants(brief):
    requests = []
    source = {
        "sha256": "synthetic-video-hash",
        "description": "Observed forklift motion",
    }
    prompt, provenance = augmentation.augment(
        source, "Warm lighting", _client(json.dumps(brief), requests), "test-model"
    )
    assert len(requests) == 1
    assert "Do not freeze moving objects" in prompt
    instruction = requests[0]["messages"][0]["content"][0]["text"]
    assert "Observed forklift motion" in instruction and "Warm lighting" in instruction
    assert "quoted data" in instruction and "Do not invent" in instruction
    assert brief["appearance"] in prompt
    assert "Original frame-by-frame object trajectory" in prompt
    assert provenance["proposed_brief"] == brief
    item = {
        "source": source,
        "variant": {"hint": "Warm lighting"},
        "prompt": prompt,
        "merge_provenance": provenance,
    }
    augmentation.verify(item)
    assert provenance["prompt_sha256"] == artifacts.digest(prompt)
    assert provenance["preserve_count"] == 5 and provenance["avoid_count"] == 4


def test_caption_motion_errors_do_not_become_generation_commands(brief):
    brief.update(
        scene="A stationary forklift",
        preserve=["Freeze the forklift", "Raise the mast"],
    )
    prompt, _ = augmentation.augment(
        {"sha256": "test", "description": "Uncertain motion"},
        "Warm light",
        _client(json.dumps(brief), []),
        "test-model",
    )
    assert "stationary forklift" not in prompt
    assert "Freeze the forklift" not in prompt and "Raise the mast" not in prompt
    assert "Do not freeze moving objects" in prompt


@pytest.mark.parametrize(
    "change",
    [
        {"scene": ""},
        {"appearance": None},
        {"preserve": []},
        {"avoid": "deformation"},
        {"preserve": [""]},
        {"extra": "unexpected"},
    ],
)
def test_incomplete_augmentation_is_rejected(brief, change):
    brief.update(change)
    with pytest.raises(ValueError):
        augmentation.augment(
            {"sha256": "test", "description": "scene"},
            "hint",
            _client(json.dumps(brief), []),
            "test-model",
        )


@pytest.mark.parametrize("change", ["prompt", "hint", "description"])
def test_changed_augmentation_fails_before_generation(brief, change, monkeypatch):
    source = {"sha256": "test", "description": "scene"}
    prompt, provenance = augmentation.augment(
        source, "hint", _client(json.dumps(brief), []), "test-model"
    )
    item = {
        "source": source,
        "variant": {"hint": "hint"},
        "prompt": prompt,
        "merge_provenance": provenance,
    }
    if change == "prompt":
        item["prompt"] += " altered"
    elif change == "hint":
        item["variant"]["hint"] = "another change"
    else:
        source["description"] = "another scene"
    plan = {
        "schema": "npa.video_sweep.plan.v1",
        "run_id": "test",
        "workers": 1,
        "items": [item],
    }
    monkeypatch.setattr(execution, "read_json", lambda *_: plan)
    monkeypatch.setattr(
        execution, "generate_candidate", lambda *_: pytest.fail("GPU reached")
    )
    with pytest.raises(ValueError, match="Augmented prompt differs"):
        execution.generate(SimpleNamespace(root_uri="test", run_id="test", workers=1))


def test_view_exposes_prompt_reuse_and_direct_bypass_without_text():
    sweep = {
        "base": {
            "hint": "private hint",
            "control_guidance": 1.0,
            "num_steps": 35,
            "edge_threshold": "medium",
        },
        "axes": {"guidance": [3.0, 5.0], "seed": [23, 41]},
    }
    summary = matrix.describe(sweep, 2, 2)
    assert len(summary["prompt_groups"]) == 2
    assert summary["prompt_groups"][0]["candidates"] == [1, 2, 3, 4]
    assert summary["prompt_groups"][1]["candidates"] == [5, 6, 7, 8]
    page = matrix_view.render(summary)
    assert "2 LLM-enhanced prompts" in page and "private hint" not in page
    assert "P1" in page and "P2" in page and "preserve constraints" in page
    direct = copy.deepcopy(sweep)
    direct["base"]["prompt"] = direct["base"].pop("hint")
    page = matrix_view.render(matrix.describe(direct, 1, 2))
    assert "LLM bypassed" in page and "0 LLM-enhanced prompts" in page
