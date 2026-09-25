"""Verify synthesis settings, cache invalidation and preservation of supplied speech."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def speech(monkeypatch, tmp_path):
    renderer = Path(__file__).resolve().parents[2] / "src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(renderer))
    voice = importlib.import_module("film_voice")
    narrate = importlib.import_module("narrate")
    scene = {"id": "opening", "narration": "Build what comes next."}
    for suffix in ("mp3", "srt"):
        (tmp_path / f"opening.{suffix}").write_bytes(b"original recording")
    args = SimpleNamespace(
        output_dir=tmp_path,
        voice="voice-a",
        force=False,
        recorded=False,
        rate="+8%",
        pitch="+2Hz",
    )
    return voice, narrate, scene, args


@pytest.mark.parametrize("changed", [{"rate": "+9%"}, {"pitch": "+3Hz"}])
def test_prosody_change_regenerates_speech_then_reuses_it(speech, monkeypatch, changed):
    voice, narrate, scene, args = speech
    record = voice._voice_record(
        scene, args.output_dir, args.voice, rate=args.rate, pitch=args.pitch
    )
    for name, value in changed.items():
        setattr(args, name, value)
    calls = []

    async def synthesize(scene, directory, selected_voice, **settings):
        calls.append(settings)
        for suffix in ("mp3", "srt"):
            (directory / f"opening.{suffix}").write_bytes(b"new delivery")
        return voice._voice_record(scene, directory, selected_voice, **settings)

    monkeypatch.setattr(narrate, "_scene", synthesize)
    updated = asyncio.run(narrate._update_scene(scene, args, record))
    assert calls == [{"rate": args.rate, "pitch": args.pitch}]
    assert (args.output_dir / "opening.mp3").read_bytes() == b"new delivery"
    assert asyncio.run(narrate._update_scene(scene, args, updated)) == updated
    assert len(calls) == 1


def test_supplied_recording_ignores_synthesis_settings(speech, monkeypatch):
    voice, narrate, scene, args = speech
    record = voice._voice_record(scene, args.output_dir, "recorded")
    monkeypatch.setattr(
        narrate, "_scene", lambda *a, **k: pytest.fail("Must not synthesize")
    )
    assert asyncio.run(narrate._update_scene(scene, args, record)) == record
    assert (args.output_dir / "opening.mp3").read_bytes() == b"original recording"


def test_old_manifest_reuses_only_neutral_delivery(speech):
    voice, _, scene, args = speech
    record = voice._voice_record(scene, args.output_dir, args.voice)
    del record["rate"], record["pitch"]
    assert voice._voice_matches(
        scene, args.output_dir, record, rate="+0%", pitch="+0Hz"
    )
    assert not voice._voice_matches(
        scene, args.output_dir, record, rate="+8%", pitch="+0Hz"
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"rate": "8%"},
        {"rate": "+8.5%"},
        {"pitch": "+2%"},
        {"pitch": 2},
        {"rate": None},
        {"pitch": "+2Hz\n"},
    ],
)
def test_invalid_settings_fail_before_synthesis(speech, settings):
    voice, _, _, _ = speech
    with pytest.raises(ValueError, match="Voice (rate|pitch) must"):
        voice._voice_settings(**settings)


def test_project_passes_negative_settings_as_single_arguments(speech, tmp_path):
    edit = importlib.import_module("edit")
    path = tmp_path / "film-project.json"
    path.write_text(
        json.dumps(
            {
                "storyboard": "story.json",
                "assets": "assets.json",
                "voice_dir": "narration",
                "output_dir": "renders",
                "voice_rate": "-4%",
                "voice_pitch": "+2Hz",
            }
        )
    )
    project = edit._project(path)
    command = edit._narrate_command(
        SimpleNamespace(recorded=False, force=False), project
    )
    assert "--rate=-4%" in command and "--pitch=+2Hz" in command
    path.write_text(path.read_text().replace('"-4%"', '"brighter"'))
    with pytest.raises(ValueError, match="Voice rate"):
        edit._project(path)


def test_service_receives_rate_and_pitch(speech, monkeypatch):
    import sys

    voice, narrate, scene, args = speech
    calls = []

    class Communication:
        def __init__(self, text, selected_voice, **settings):
            calls.append((text, selected_voice, settings))

        async def stream(self):
            yield {"type": "audio", "data": b"synthesized"}

    service = SimpleNamespace(
        Communicate=Communication,
        SubMaker=lambda: SimpleNamespace(get_srt=lambda: "captions"),
    )
    monkeypatch.setitem(sys.modules, "edge_tts", service)
    record = asyncio.run(
        narrate._scene(
            scene, args.output_dir, args.voice, rate=args.rate, pitch=args.pitch
        )
    )
    assert calls == [(scene["narration"], args.voice, {"rate": "+8%", "pitch": "+2Hz"})]
    assert voice._voice_matches(
        scene, args.output_dir, record, rate="+8%", pitch="+2Hz"
    )
