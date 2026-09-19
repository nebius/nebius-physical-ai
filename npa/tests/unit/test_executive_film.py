"""Check the executive film's timing, media identity, and portable asset loading."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def film(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "npa/src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "executive_film_render", directory / "render.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_storyboard_has_exact_runtime_and_a_panel_for_every_label(film):
    story = json.loads((film._ROOT / "storyboard.json").read_text())
    assert sum(scene["duration"] for scene in story["scenes"]) == 30
    assert story["fps"] * story["duration"] == 900
    for scene in story["scenes"]:
        assert (
            len(scene["assets"]) == len(scene["labels"]) == len(film._rectangles(scene))
        )
        assert scene["narration"].strip()
        for x, y, width, height in film._rectangles(scene):
            assert width % 2 == height % 2 == 0
            assert x + width <= story["width"]
            assert y + height <= story["height"]


@pytest.mark.parametrize(
    "timing",
    [
        {"title_delay_seconds": "1"},
        {"title_duration_seconds": float("nan")},
        {"title_delay_seconds": -1},
        {"title_duration_seconds": 0.1},
        {"title_delay_seconds": 8, "title_duration_seconds": 3},
    ],
)
def test_film_title_timing_is_rejected_before_rendering(film, timing):
    scene = {
        "id": "opening",
        "layout": "film",
        "duration": 10,
        "assets": ["scene"],
        "labels": [""],
        "title": ["Before the first move."],
        **timing,
    }
    with pytest.raises(ValueError, match="Film title"):
        film._validate_layout(scene)


@pytest.mark.parametrize("size", [0, 17, 1024 * 1024 + 17])
def test_asset_hash_supports_python_310_and_chunk_boundaries(
    film, tmp_path, monkeypatch, size
):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    payload = (bytes(range(256)) * (size // 256 + 1))[:size]
    source = tmp_path / "asset.bin"
    source.write_bytes(payload)
    assert film._hash(source) == hashlib.sha256(payload).hexdigest()


def test_changed_asset_is_rejected_before_media_probe(film, tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"changed after approval")
    monkeypatch.setattr(
        film, "_probe", lambda path: pytest.fail("must verify identity first")
    )
    assets = {"source": {"path": str(source), "kind": "video", "sha256": "0" * 64}}
    story = {"scenes": [{"duration": 120, "assets": ["source"]}]}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        film._validate(story, assets)


@pytest.mark.parametrize("second_role", ["source", "alternate"])
def test_unique_sources_rejects_reused_bytes_even_with_another_name_or_trim(
    film, tmp_path, monkeypatch, second_role
):
    original = tmp_path / "source.mp4"
    original.write_bytes(b"the same approved source")
    alternate = tmp_path / "different-name.mp4"
    shutil.copy2(original, alternate)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assets = {
        "source": {
            "path": str(original),
            "kind": "video",
            "sha256": digest,
            "trim": [0, 2],
        },
        "alternate": {
            "path": str(alternate),
            "kind": "video",
            "sha256": digest,
            "trim": [4, 6],
        },
    }
    story = {
        "unique_sources": True,
        "scenes": [
            {"id": "opening", "duration": 2, "assets": ["source"]},
            {"id": "closing", "duration": 2, "assets": [second_role]},
        ],
    }
    monkeypatch.setattr(
        film, "_probe", lambda path: pytest.fail("reject before probing")
    )
    with pytest.raises(
        ValueError, match="Repeated source media: opening/source and closing/"
    ):
        film._validate(story, assets)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_unique_source_policy_preserves_optional_comparison_workflows(film, enabled):
    story = {} if enabled is None else {"unique_sources": enabled}
    assets = {"first": {"sha256": "a" * 64}, "second": {"sha256": "b" * 64}}
    scenes = [
        (0, {"id": "opening", "assets": ["first"]}),
        (1, {"id": "closing", "assets": ["second"]}),
    ]
    film._validate_unique_sources(story, assets, scenes)
    if not enabled:
        assets["second"]["sha256"] = assets["first"]["sha256"]
        film._validate_unique_sources(story, assets, scenes)


@pytest.mark.parametrize("enabled", ["true", 1, None])
def test_unique_source_policy_rejects_non_boolean_values(film, enabled):
    with pytest.raises(ValueError, match="unique_sources must be a boolean"):
        film._validate_unique_sources({"unique_sources": enabled}, {}, [])


def test_relative_assets_are_resolved_against_manifest_location(
    film, tmp_path, monkeypatch
):
    package = tmp_path / "delivery"
    package.mkdir()
    source = package / "source.png"
    Image.new("RGB", (64, 40)).save(source)
    manifest = package / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {
                    "path": "source.png",
                    "kind": "image",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    assert film._load_assets(manifest)["source"]["path"] == str(source)


def test_scene_preview_can_verify_its_media_before_other_shots_arrive(
    film, tmp_path, monkeypatch
):
    source = tmp_path / "review.png"
    Image.new("RGB", (64, 40)).save(source)
    assets = {
        "review": {"path": str(source), "kind": "image", "sha256": film._hash(source)}
    }
    story = {
        "scenes": [
            {"id": "review", "duration": 10, "assets": ["review"]},
            {"id": "generation", "duration": 110, "assets": ["pending_gpu_output"]},
        ]
    }
    monkeypatch.setattr(
        film,
        "_probe",
        lambda path: {
            "streams": [{"codec_type": "video", "width": 64, "height": 40}],
        },
    )
    assert film._validate(story, assets, "review") == {"review"}
    with pytest.raises(ValueError, match="Missing asset roles.*pending_gpu_output"):
        film._validate(story, assets)
    source.write_bytes(b"changed review media")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        film._validate(story, assets, "review")


def test_crop_outside_source_is_rejected(film, tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 40)).save(source)
    assets = {
        "source": {
            "path": str(source),
            "kind": "image",
            "crop": [40, 0, 30, 20],
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    }
    monkeypatch.setattr(
        film,
        "_probe",
        lambda path: {
            "streams": [{"codec_type": "video", "width": 64, "height": 40}],
        },
    )
    with pytest.raises(ValueError, match="Crop exceeds source bounds"):
        film._validate({"scenes": [{"duration": 120, "assets": ["source"]}]}, assets)


@pytest.mark.parametrize(
    "trim", [[-1, 2], [2, 2], [0, 4], [True, 2], [0, float("nan")], None]
)
def test_invalid_excerpt_is_rejected_before_render(film, trim):
    with pytest.raises(ValueError, match="trim|Trim"):
        film._validate_trim(
            {"kind": "video", "trim": trim}, {"format": {"duration": "3"}}, "demo"
        )


def test_excerpt_cannot_silently_loop_the_entire_recording(film):
    with pytest.raises(ValueError, match="requires hold"):
        film._validate_trim(
            {"kind": "video", "trim": [1, 2], "playback": "loop"},
            {"format": {"duration": "3"}},
            "demo",
        )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
def test_excerpt_decodes_only_selected_frames_and_holds_last_frame(film, tmp_path):
    source = tmp_path / "recording.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=64x64:r=30:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=lime:s=64x64:r=30:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=64x64:r=30:d=1",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
            "-map",
            "[out]",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
    )
    asset = {"kind": "video", "path": str(source), "trim": [1, 2]}
    film._validate_trim(asset, film._probe(source), "demo")
    filters = film._media_filter(0, asset, (0, 0, 64, 64), 2, {"width": 64})
    frames = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            *film._input(asset),
            "-filter_complex",
            filters[0],
            "-map",
            "[media0]",
            "-t",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    assert len(frames) == 60 * 64 * 64 * 3
    for frame in range(60):
        offset = (frame * 64 * 64 + 32 * 64 + 32) * 3
        red, green, blue = frames[offset : offset + 3]
        assert green > 240 and red < 10 and blue < 10


def test_captions_follow_scene_boundaries_and_do_not_run_past_them(
    film, tmp_path, monkeypatch
):
    scenes = [{"id": "first", "duration": 10}, {"id": "second", "duration": 10}]
    for scene in scenes:
        (tmp_path / f"{scene['id']}.srt").write_text(
            "1\n00:00:00,000 --> 00:00:20,000\nAn intentionally long cue.\n"
        )
    monkeypatch.setattr(film, "_duration", lambda path: 8)
    film._captions(scenes, tmp_path, tmp_path)
    captions = (tmp_path / "film.srt").read_text()
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
    story = {
        "scenes": [
            {"id": "opening", "duration": 12},
            {"id": "generation", "duration": 20},
            {"id": "robotics", "duration": 10},
        ]
    }
    selected, offset = film._selection(story, "robotics")
    assert offset == 32
    assert selected[0][0] == 2
    assert selected[0][1]["duration"] == 10
    with pytest.raises(ValueError, match="Unknown scene"):
        film._selection(story, "nonexistent")


def test_cache_rejects_modified_output_and_keeps_failed_builds_unpublished(
    film, tmp_path
):
    from film_cache import _build_cached, _cached

    inputs = {"title": "A title"}
    first, reused = _build_cached(
        tmp_path, "scenes", inputs, lambda p: (p / "clip.mp4").write_bytes(b"complete")
    )
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
    repaired, reused = _build_cached(
        tmp_path, "scenes", inputs, lambda p: (p / "clip.mp4").write_bytes(b"complete")
    )
    assert not reused and _cached(repaired)
    _, reused = _build_cached(
        tmp_path, "scenes", inputs, lambda p: pytest.fail("must reuse repaired result")
    )
    assert reused


@pytest.fixture
def editing_session(film, tmp_path, monkeypatch):
    story = {
        "title": "Film",
        "scenes": [
            {
                "id": "first",
                "duration": 60,
                "title": ["First"],
                "narration": "Words",
                "assets": ["robot"],
            },
            {
                "id": "second",
                "duration": 60,
                "title": ["Second"],
                "narration": "Words",
                "assets": ["robot"],
            },
        ],
    }
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    for scene in story["scenes"]:
        for suffix in ["mp3", "srt"]:
            (voice_dir / f"{scene['id']}.{suffix}").write_bytes(b"recorded")
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        voice_dir=voice_dir,
        storyboard=tmp_path / "story.json",
        profile="preview",
        scene=None,
        workers=2,
    )
    args.storyboard.write_text(json.dumps(story))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-a")
    assets = {
        "robot": {"path": str(source), "kind": "video", "sha256": film._hash(source)}
    }
    calls = {"visual": [], "audio": 0, "assembly": 0}
    _patch_encoding(film, monkeypatch, calls)
    return film, args, story, assets, calls


def _patch_encoding(film, monkeypatch, calls):
    def visual(scene, index, total, assets, directory, profile):
        calls["visual"].append(scene["id"])
        payload = {
            "scene": {k: v for k, v in scene.items() if k != "narration"},
            "assets": assets,
            "profile": profile,
        }
        (directory / f"{scene['id']}.mp4").write_text(json.dumps(payload))

    def audio(scenes, voice_dir, directory, cache_root, music_path=None):
        calls["audio"] += 1
        payload = b"".join((voice_dir / f"{s['id']}.mp3").read_bytes() for s in scenes)
        (directory / "mix.m4a").write_bytes(
            payload + (music_path.read_bytes() if music_path else b"")
        )

    def assemble(args, storyboard, assets, parts, audio, selected, offset, staging):
        calls["assembly"] += 1
        (staging / "film.mp4").write_bytes(b"".join(p.read_bytes() for p in parts))

    monkeypatch.setattr(film, "_render_scene", visual)
    monkeypatch.setattr(film, "_mix", audio)
    monkeypatch.setattr(film, "_assemble", assemble)
    monkeypatch.setattr(film, "_publish", lambda *args: None)


def _edit_run(session):
    film, args, story, assets, calls = session
    args.storyboard.write_text(json.dumps(story))
    (args.output_dir / args.profile).mkdir(parents=True, exist_ok=True)
    film._render_film(args, story, assets, {"ffmpeg": "test version"})
    return json.loads(
        (args.output_dir / args.profile / "render-timing.json").read_text()
    )


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


def test_music_edit_rebuilds_audio_and_retains_visuals(editing_session, tmp_path):
    film, args, story, assets, calls = editing_session
    args.music_path = tmp_path / "score.wav"
    args.music_path.write_bytes(b"original score")
    _edit_run(editing_session)
    args.music_path.write_bytes(b"revised score")
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 0 and report["scenes_reused"] == 2
    assert not report["audio_reused"] and not report["assembly_reused"]
    assert calls["audio"] == 2


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


def test_footer_changes_invalidate_only_affected_visuals(editing_session):
    film, args, story, assets, calls = editing_session
    _edit_run(editing_session)
    story["scenes"][0]["footer"] = "CUSTOM SCENE FOOTER"
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == report["scenes_reused"] == 1
    story["footer"] = "NEW FILM FOOTER"
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == report["scenes_reused"] == 1
    assert report["audio_reused"]


def test_provenance_correction_refreshes_manifest_without_reencoding_media(
    editing_session,
):
    film, args, story, assets, calls = editing_session
    _edit_run(editing_session)
    assets["robot"]["provenance"] = {
        "tool": "RoboCasa",
        "limitations": ["random actions"],
    }
    report = _edit_run(editing_session)
    assert report["scenes_rendered"] == 0 and report["audio_reused"]
    assert not report["assembly_reused"] and calls["assembly"] == 2
    assert _edit_run(editing_session)["assembly_reused"]


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
def test_narration_generation_reuses_audio_until_words_change(
    film, tmp_path, monkeypatch, voice
):
    import asyncio

    import narrate
    from film_voice import _voice_record

    scene = {"id": "first", "narration": "Original words"}
    for suffix in ["mp3", "srt"]:
        (tmp_path / f"first.{suffix}").write_bytes(b"recording")
    record = _voice_record(scene, tmp_path, voice)
    args = SimpleNamespace(
        output_dir=tmp_path, voice="voice-a", force=False, recorded=False
    )
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
