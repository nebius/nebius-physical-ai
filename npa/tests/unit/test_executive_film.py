"""Check the executive film's timing, media identity, and portable asset loading."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def film(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "docs/demos/executive-film"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("executive_film_render", directory / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_storyboard_has_exact_runtime_and_a_panel_for_every_label(film):
    story = json.loads((film._ROOT / "storyboard.json").read_text())
    assert sum(scene["duration"] for scene in story["scenes"]) == 120
    assert story["fps"] * story["duration"] == 3600
    for scene in story["scenes"]:
        assert len(scene["assets"]) == len(scene["labels"]) == len(film._rectangles(scene))
        assert scene["narration"].strip()
        for x, y, width, height in film._rectangles(scene):
            assert width % 2 == height % 2 == 0
            assert x + width <= story["width"]
            assert y + height <= story["height"]


def test_changed_asset_is_rejected_before_media_probe(film, tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"changed after approval")
    monkeypatch.setattr(film, "_probe", lambda path: pytest.fail("must verify identity first"))
    assets = {"source": {"path": str(source), "kind": "video", "sha256": "0" * 64}}
    story = {"scenes": [{"duration": 120, "assets": ["source"]}]}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        film._validate(story, assets)


def test_relative_assets_are_resolved_against_manifest_location(film, tmp_path, monkeypatch):
    package = tmp_path / "delivery"
    package.mkdir()
    source = package / "source.png"
    Image.new("RGB", (64, 40)).save(source)
    manifest = package / "assets.json"
    manifest.write_text(json.dumps({"source": {
        "path": "source.png", "kind": "image",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}))
    monkeypatch.chdir(tmp_path)
    assert film._load_assets(manifest)["source"]["path"] == str(source)


def test_crop_outside_source_is_rejected(film, tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 40)).save(source)
    assets = {"source": {"path": str(source), "kind": "image", "crop": [40, 0, 30, 20],
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    monkeypatch.setattr(film, "_probe", lambda path: {
        "streams": [{"codec_type": "video", "width": 64, "height": 40}],
    })
    with pytest.raises(ValueError, match="Crop exceeds source bounds"):
        film._validate({"scenes": [{"duration": 120, "assets": ["source"]}]}, assets)


def test_captions_follow_scene_boundaries_and_do_not_run_past_them(film, tmp_path, monkeypatch):
    scenes = [{"id": "first", "duration": 10}, {"id": "second", "duration": 10}]
    for scene in scenes:
        (tmp_path / f"{scene['id']}.srt").write_text(
            "1\n00:00:00,000 --> 00:00:20,000\nAn intentionally long cue.\n"
        )
    monkeypatch.setattr(film, "_duration", lambda path: 8)
    film._captions(scenes, tmp_path, tmp_path)
    captions = (tmp_path / "workbench-executive-film.srt").read_text()
    assert "00:00:00,350 --> 00:00:10,000" in captions
    assert "00:00:10,350 --> 00:00:20,000" in captions


def test_preview_panels_stay_even_and_inside_the_frame(film):
    story = json.loads((film._ROOT / "storyboard.json").read_text())
    profile = film._PROFILES["preview"]
    for scene in story["scenes"]:
        for rectangle in film._rectangles(scene):
            x, y, width, height = film._scaled_rectangle(rectangle, profile)
            assert width % 2 == height % 2 == 0
            assert 0 <= x < x + width <= profile["width"]
            assert 0 <= y < y + height <= profile["height"]


def test_one_scene_keeps_its_full_film_audio_offset(film):
    story = json.loads((film._ROOT / "storyboard.json").read_text())
    selected, offset = film._selection(story, "07-robotics")
    assert offset == 62
    assert selected[0][0] == 6
    assert selected[0][1]["duration"] == 10
    with pytest.raises(ValueError, match="Unknown scene"):
        film._selection(story, "nonexistent")


def test_cache_rejects_modified_output_and_keeps_failed_builds_unpublished(film, tmp_path):
    from film_cache import _build_cached, _cached

    inputs = {"title": "A title"}
    first, reused = _build_cached(tmp_path, "scenes", inputs, lambda p: (p / "clip.mp4").write_bytes(b"complete"))
    assert not reused
    (first / "clip.mp4").write_bytes(b"truncated")
    assert not _cached(first)

    def fail(staging):
        (staging / "clip.mp4").write_bytes(b"partial replacement")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        _build_cached(tmp_path, "scenes", inputs, fail)
    assert (first / "clip.mp4").read_bytes() == b"truncated"
    assert not _cached(first)
    repaired, reused = _build_cached(tmp_path, "scenes", inputs, lambda p: (p / "clip.mp4").write_bytes(b"complete"))
    assert not reused and _cached(repaired)
    _, reused = _build_cached(tmp_path, "scenes", inputs, lambda p: pytest.fail("must reuse repaired result"))
    assert reused


@pytest.fixture
def editing_session(film, tmp_path, monkeypatch):
    story = {"title": "Film", "scenes": [
        {"id": "first", "duration": 60, "title": ["First"], "narration": "Words", "assets": ["robot"]},
        {"id": "second", "duration": 60, "title": ["Second"], "narration": "Words", "assets": ["robot"]},
    ]}
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    for scene in story["scenes"]:
        for suffix in ["mp3", "srt"]:
            (voice_dir / f"{scene['id']}.{suffix}").write_bytes(b"recorded")
    args = SimpleNamespace(output_dir=tmp_path / "output", voice_dir=voice_dir,
                           storyboard=tmp_path / "story.json", profile="preview", scene=None, workers=2)
    args.storyboard.write_text(json.dumps(story))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-a")
    assets = {"robot": {"path": str(source), "kind": "video", "sha256": film._hash(source)}}
    calls = {"visual": [], "audio": 0, "assembly": 0}
    _patch_encoding(film, monkeypatch, calls)
    return film, args, story, assets, calls


def _patch_encoding(film, monkeypatch, calls):
    def visual(scene, index, total, assets, directory, profile):
        calls["visual"].append(scene["id"])
        payload = {"scene": {k: v for k, v in scene.items() if k != "narration"}, "assets": assets, "profile": profile}
        (directory / f"{scene['id']}.mp4").write_text(json.dumps(payload))

    def audio(scenes, voice_dir, directory, cache_root):
        calls["audio"] += 1
        (directory / "mix.m4a").write_bytes(b"".join((voice_dir / f"{s['id']}.mp3").read_bytes() for s in scenes))

    def assemble(args, storyboard, assets, parts, audio, selected, offset, staging):
        calls["assembly"] += 1
        (staging / "workbench-executive-film.mp4").write_bytes(b"".join(p.read_bytes() for p in parts))

    monkeypatch.setattr(film, "_render_scene", visual)
    monkeypatch.setattr(film, "_mix", audio)
    monkeypatch.setattr(film, "_assemble", assemble)
    monkeypatch.setattr(film, "_publish", lambda *args: None)


def _edit_run(session):
    film, args, story, assets, calls = session
    args.storyboard.write_text(json.dumps(story))
    (args.output_dir / args.profile).mkdir(parents=True, exist_ok=True)
    film._render_film(args, story, assets, {"ffmpeg": "test version"})
    return json.loads((args.output_dir / args.profile / "render-timing.json").read_text())


def test_title_edit_rebuilds_one_scene_and_revert_reuses_prior_variant(editing_session):
    film, args, story, assets, calls = editing_session
    assert _edit_run(editing_session)["scenes_rendered"] == 2
    assert _edit_run(editing_session)["assembly_reused"]
    story["scenes"][0]["title"] = ["Changed title"]
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == report["scenes_reused"] == 1
    assert report["audio_reused"] and not report["assembly_reused"]
    story["scenes"][0]["title"] = ["First"]
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 0 and report["assembly_reused"]
    assert sorted(calls["visual"]) == ["first", "first", "second"]
    assert calls["audio"] == 1


def test_narration_and_caption_edits_do_not_reencode_visuals(editing_session):
    film, args, story, assets, calls = editing_session
    _edit_run(editing_session)
    story["scenes"][0]["narration"] = "New spoken words"
    (args.voice_dir / "first.mp3").write_bytes(b"new recording")
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 0 and not report["audio_reused"]
    (args.voice_dir / "first.srt").write_bytes(b"new captions")
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 0 and report["audio_reused"]
    assert not report["assembly_reused"]
    assert calls["audio"] == 2 and calls["assembly"] == 3


def test_changed_asset_or_profile_never_reuses_stale_visuals(editing_session):
    film, args, story, assets, calls = editing_session
    _edit_run(editing_session)
    source = Path(assets["robot"]["path"])
    source.write_bytes(b"replacement-source")
    assets["robot"]["sha256"] = film._hash(source)
    assert _edit_run(editing_session)["scenes_rendered"] == 2
    args.profile = "final"
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 2 and report["audio_reused"]


def test_stale_spoken_words_are_rejected_before_rendering(film, tmp_path):
    from film_voice import _verify_narration, _voice_record

    scene = {"id": "first", "narration": "Original words"}
    for suffix in ["mp3", "srt"]:
        (tmp_path / f"first.{suffix}").write_bytes(b"recording")
    record = _voice_record(scene, tmp_path, "recorded")
    (tmp_path / "voice-manifest.json").write_text(json.dumps([record]))
    _verify_narration([scene], tmp_path)
    scene["narration"] = "Changed words"
    with pytest.raises(ValueError, match="Narration is missing or stale for first"):
        _verify_narration([scene], tmp_path)


@pytest.mark.parametrize("voice", ["voice-a", "recorded"])
def test_narration_generation_reuses_audio_until_words_change(film, tmp_path, monkeypatch, voice):
    import asyncio

    import narrate
    from film_voice import _voice_record

    scene = {"id": "first", "narration": "Original words"}
    for suffix in ["mp3", "srt"]:
        (tmp_path / f"first.{suffix}").write_bytes(b"recording")
    record = _voice_record(scene, tmp_path, voice)
    args = SimpleNamespace(output_dir=tmp_path, voice="voice-a", force=False, recorded=False)
    generated = []

    async def generate(scene, directory, voice):
        generated.append(scene["id"])
        for suffix in ["mp3", "srt"]:
            (directory / f"first.{suffix}").write_bytes(b"updated")
        return _voice_record(scene, directory, voice)

    monkeypatch.setattr(narrate, "_scene", generate)
    asyncio.run(narrate._update_scene(scene, args, record))
    assert generated == []
    scene["narration"] = "New words"
    record = asyncio.run(narrate._update_scene(scene, args, record))
    assert generated == ["first"]
    assert (tmp_path / "first.mp3").read_bytes() == b"updated"
    assert record["text_sha256"] == hashlib.sha256(b"New words").hexdigest()


def test_scene_ids_cannot_collide_or_escape_output_directory(film):
    with pytest.raises(ValueError, match="Scene IDs"):
        film._validate_storyboard({"scenes": [{"id": "../outside", "duration": 120}]})
    with pytest.raises(ValueError, match="Scene IDs"):
        film._validate_storyboard({"scenes": [{"id": "same", "duration": 60}] * 2})
