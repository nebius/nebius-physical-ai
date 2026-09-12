"""Check the executive film's timing, media identity, and portable asset loading."""

import hashlib
import importlib.util
import json
from pathlib import Path

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
