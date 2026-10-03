"""Check variable film durations, audience briefs, callouts and audio slicing."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def film(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "npa/src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "brief_film_render", directory / "render.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("duration", [1, 30, 60, 121, 180, 600])
def test_full_timeline_accepts_short_and_long_films(film, duration):
    story = {
        "duration": duration,
        "scenes": [{"id": "one", "duration": duration, "assets": []}],
    }
    film._validate_storyboard(story)
    assert film._validate(story, {}) == set()
    del story["duration"]
    film._validate_storyboard(story)


@pytest.mark.parametrize("duration", [0, -1, 0.5, True, "30"])
def test_timeline_rejects_invalid_scene_durations(film, duration):
    with pytest.raises(ValueError, match="positive whole seconds"):
        film._validate_storyboard({"scenes": [{"id": "one", "duration": duration}]})


def test_empty_timeline_and_conflicting_declared_duration_fail(film):
    with pytest.raises(ValueError, match="at least one scene"):
        film._validate_storyboard({"scenes": []})
    with pytest.raises(ValueError, match="sum of scene durations"):
        film._validate_storyboard(
            {"duration": 120, "scenes": [{"id": "one", "duration": 30}]}
        )


@pytest.mark.parametrize(
    "name, duration", [("storyboard.json", 30), ("storyboard-technical.json", 60)]
)
def test_example_narratives_match_their_brief_and_layout_contract(film, name, duration):
    story = json.loads((film._ROOT / name).read_text())
    film._validate_storyboard(story)
    assert (
        sum(s["duration"] for s in story["scenes"])
        == story["brief"]["duration_seconds"]
        == duration
    )
    for scene in story["scenes"]:
        assert (
            len(scene["assets"]) == len(scene["labels"]) == len(film._rectangles(scene))
        )
        assert scene["narration"].strip()
    story["brief"]["duration_seconds"] += 1
    with pytest.raises(ValueError, match="creative brief duration_seconds"):
        film._validate_storyboard(story)


def test_new_prompt_replaces_executive_guidance_and_exposes_missing_provenance(film):
    from film_brief import _authoring_packet

    story = json.loads((film._ROOT / "storyboard.json").read_text())
    assets = {
        "unreviewed": {
            "path": "/private/source.mp4",
            "kind": "video",
            "sha256": "a" * 64,
        },
        "reviewed": {
            "kind": "video",
            "sha256": "b" * 64,
            "provenance": {
                "tool": "RoboCasa",
                "artifact_role": "output",
                "limitations": ["random actions; zero reward"],
            },
        },
    }
    packet = _authoring_packet(
        story, assets, {"prompt": "Teach artifact verification", "duration_seconds": 45}
    )
    assert packet["brief"]["duration_seconds"] == 45
    assert packet["brief"]["prompt"] == "Teach artifact verification"
    assert "executive" not in packet["brief"]["audience"].lower()
    assert packet["available_assets"]["unreviewed"]["provenance"] == {
        "status": "unattributed"
    }
    assert packet["available_assets"]["reviewed"]["provenance"]["limitations"] == [
        "random actions; zero reward"
    ]
    assert "/private/" not in json.dumps(packet)
    assert story["brief"]["duration_seconds"] == 30
    assert _authoring_packet(story, {}, {})["brief"] == story["brief"]


def test_brief_overrides_change_its_identity_and_validate_duration(film):
    from film_brief import _authoring_packet

    story = json.loads((film._ROOT / "storyboard.json").read_text())
    original = _authoring_packet(story, {}, {})
    changed = _authoring_packet(
        story, {}, {"audience": "Research scientists", "duration_seconds": 300}
    )
    assert changed["brief_sha256"] != original["brief_sha256"]
    assert changed["brief"]["audience"] == "Research scientists"
    with pytest.raises(ValueError, match="positive whole seconds"):
        _authoring_packet(story, {}, {"duration_seconds": 0})
    with pytest.raises(ValueError, match="nonempty prompt"):
        _authoring_packet(story, {}, {"prompt": " "})


@pytest.mark.parametrize(
    "duration, offset, sound_duration, codec",
    [
        (30, 0, 30.01, "copy"),
        (120, 0, 120, "copy"),
        (180, 0, 180, "copy"),
        (120, 0, 180, "aac"),
        (120, 12, 180, "aac"),
        (10, 5, 30, "aac"),
    ],
)
def test_audio_copy_depends_on_full_track_selection_not_two_minutes(
    film,
    tmp_path,
    monkeypatch,
    duration,
    offset,
    sound_duration,
    codec,
):
    source = tmp_path / "part.mp4"
    source.write_bytes(b"video")
    calls = []
    monkeypatch.setattr(film, "_duration", lambda path: sound_duration)
    monkeypatch.setattr(
        film.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    film._join([source], tmp_path / "mix.m4a", tmp_path, duration, offset, "A new film")
    command = calls[0]
    assert command[command.index("-c:a") + 1] == codec
    assert command[command.index("-ss") + 1] == str(offset)
    assert command[command.index("-t") + 1] == str(duration)


def test_reason_layout_uses_authored_callouts_without_implying_nvidia_origin(
    film, monkeypatch
):
    import graphics

    text = []
    monkeypatch.setattr(
        graphics, "_text", lambda draw, position, value, *args: text.append(value)
    )
    monkeypatch.setattr(
        graphics, "_paragraph", lambda draw, position, value, *args: text.append(value)
    )
    graphics._details(
        None,
        {
            "layout": "reason",
            "quote_heading": "EVALUATION NOTE",
            "quote": "No successful trials.",
            "source_notes": ["LeRobot evaluation"],
        },
    )
    assert text == ["EVALUATION NOTE", "No successful trials.", "LeRobot evaluation"]
    text.clear()
    graphics._details(None, {"layout": "reason"})
    assert not any(text)


def test_layout_rejects_unrenderable_asset_counts_and_excess_callouts(film):
    story = json.loads((film._ROOT / "storyboard-technical.json").read_text())
    scene = story["scenes"][0]
    scene["assets"].pop()
    with pytest.raises(ValueError, match="asset roles and labels"):
        film._validate_storyboard(story)
    scene["assets"].append("replacement")
    scene["pipeline_labels"] = ["too many"] * 4
    with pytest.raises(ValueError, match="at most 3 pipeline_labels"):
        film._validate_storyboard(story)
