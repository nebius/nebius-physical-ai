"""Verify offline player timelines, literal caption text, and render publication."""

import html
import importlib.util
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def player(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "docs/demos/executive-film"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("test_film_player", root / "film_player.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _story():
    return {"title": "A film", "scenes": [
        {"id": "first", "duration": 8, "title": ["First", "chapter"], "subtitle": "Opening"},
        {"id": "second", "duration": 12, "title": ["Second"], "subtitle": "Closing"},
    ]}


def _data(document):
    payload = re.search(r'<script id="film-data" type="application/json">(.*?)</script>',
                        document, flags=re.DOTALL)
    assert payload is not None
    assert "<" not in payload[1]
    return json.loads(payload[1])


def _write_inputs(directory, story, captions):
    (directory / "render-storyboard.json").write_text(json.dumps(story))
    (directory / "workbench-executive-film.srt").write_text(captions)


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attributes):
        self.elements.append((tag, dict(attributes)))


def test_player_escapes_titles_subtitles_and_native_caption_markup(player):
    story = _story()
    hostile = '</script><img src="x" onerror="alert(1)"> & literal text'
    story["title"] = hostile
    story["subtitle"] = hostile
    story["scenes"][0]["title"] = [hostile]
    story["scenes"][0]["subtitle"] = hostile
    document = player._player_html(story, f"1\n00:00:00,100 --> 00:00:02,200\n{hostile}\n")
    parser = _Elements()
    parser.feed(document)
    assert [tag for tag, _ in parser.elements].count("script") == 2
    assert not any(tag == "img" for tag, _ in parser.elements)
    assert not any(key.startswith("on") for _, attributes in parser.elements for key in attributes)
    assert _data(document)["chapters"][0]["title"] == hostile
    assert _data(document)["captions"][0]["text"] == html.escape(hostile, quote=False)
    assert html.escape(hostile) in document
    locations = [value for _, attributes in parser.elements
                 for key, value in attributes.items() if key in {"src", "href", "poster"}]
    assert set(locations) == {"poster.png", "workbench-executive-film.mp4", "workbench-executive-film.srt"}


def test_player_refreshes_chapters_and_captions_from_the_rendered_snapshot(player, tmp_path):
    story = _story()
    _write_inputs(tmp_path, story, "1\n00:00:08,100 --> 00:00:09,000\nOriginal sentence.\n")
    player._write_player(tmp_path)
    initial = _data((tmp_path / "watch.html").read_text())
    assert [chapter["start"] for chapter in initial["chapters"]] == [0, 8]
    assert initial["duration"] == 20
    story["scenes"] = [story["scenes"][1]]
    story["scenes"][0]["title"] = ["Revised scene"]
    _write_inputs(tmp_path, story, "1\n00:00:00,100 --> 00:00:01,000\nRevised sentence.\n")
    player._write_player(tmp_path)
    changed = _data((tmp_path / "watch.html").read_text())
    assert changed["duration"] == 12
    assert changed["chapters"] == [{"start": 0, "duration": 12,
                                     "title": "Revised scene", "subtitle": "Closing"}]
    assert changed["captions"] == [{"start": 0.1, "end": 1.0, "text": "Revised sentence."}]


@pytest.mark.parametrize("cue", [
    "1\n00:00:00,000 --> 00:00:21,000\nOutside the film.",
    "1\n00:00:01,000 --> 00:00:00,000\nReversed.",
    "1\n00:00:00,000 --> 00:00:00,000\nEmpty interval.",
    "1\n00:65:00,000 --> 00:66:00,000\nInvalid clock.",
    "Not SRT.",
])
def test_invalid_caption_timing_cannot_publish_a_player(player, tmp_path, cue):
    _write_inputs(tmp_path, _story(), cue)
    output = tmp_path / "watch.html"
    output.write_text("last good player")
    with pytest.raises(ValueError, match="Player captions"):
        player._write_player(tmp_path)
    assert output.read_text() == "last good player"


def test_player_javascript_seeks_and_creates_native_cues_without_fetching(player):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional for the offline player behavior check")
    payload = _data(player._player_html(_story(), "1\n00:00:00,100 --> 00:00:01,000\nHello & goodbye.\n"))
    harness = """
const assert = require('node:assert/strict');
class Element {
  constructor() { this.events = {}; this.attributes = {}; }
  addEventListener(name, action) { this.events[name] = action; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
}
const controls = [new Element(), new Element()];
const toggle = new Element(); const playback = new Element();
const captionTrack = {cues: [], addCue(cue) { this.cues.push(cue); }};
const media = new Element(); media.currentTime = 0;
media.textTracks = new Element(); media.addTextTrack = () => captionTrack;
media.play = () => { media.played = true; return Promise.resolve(); };
global.VTTCue = class { constructor(start, end, text) { Object.assign(this, {start, end, text}); }};
global.document = {querySelector: () => media, querySelectorAll: () => controls,
  getElementById: id => ({'film-data': {textContent: JSON.stringify(DATA)}, captions: toggle, playback})[id]};
"""
    checks = """
assert.deepEqual(captionTrack.cues.map(c => [c.start, c.end, c.text]), [[0.1, 1, 'Hello &amp; goodbye.']]);
controls[1].events.click(); assert.equal(media.currentTime, 8); assert.equal(media.played, true);
media.events.timeupdate(); assert.equal(playback.textContent, '0:08 / 0:20');
assert.equal(controls[1].attributes['aria-current'], 'true');
assert.equal(controls[0].attributes['aria-current'], undefined);
toggle.events.click(); assert.equal(captionTrack.mode, 'showing');
assert.equal(toggle.attributes['aria-pressed'], 'true');
"""
    subprocess.run([node, "-e", f"const DATA = {json.dumps(payload)};\n" + harness + player._SCRIPT + checks], check=True)


def test_assembly_publishes_player_using_only_selected_scene_snapshot(player, tmp_path, monkeypatch):
    import render

    staging, audio, destination = (tmp_path / name for name in ("staging", "audio", "delivery"))
    staging.mkdir()
    audio.mkdir()
    for name in ("narration.wav", "original-score.wav"):
        (audio / name).write_bytes(b"audio")

    def join(parts, sound, directory, duration, offset, title):
        target = directory / "workbench-executive-film.mp4"
        target.write_bytes(b"assembled video")
        return target

    def evidence(target, snapshot, assets, directory, recipe):
        (directory / "poster.png").write_bytes(b"poster")
        (directory / "render-manifest.json").write_text("{}")

    def captions(scenes, voice_dir, directory):
        (directory / "workbench-executive-film.srt").write_text("1\n00:00:00,100 --> 00:00:01,000\nSelected.\n")

    monkeypatch.setattr(render, "_join", join)
    monkeypatch.setattr(render, "_evidence", evidence)
    monkeypatch.setattr(render, "_captions", captions)
    story = _story()
    render._assemble(SimpleNamespace(voice_dir=tmp_path), story, {}, [], audio,
                     [(1, story["scenes"][1])], 8, staging)
    render._publish(staging, destination)
    output = _data((destination / "watch.html").read_text())
    assert output["chapters"] == [{"start": 0, "duration": 12, "title": "Second", "subtitle": "Closing"}]
    assert output["captions"][0]["text"] == "Selected."
    assert "film_player.py" in render._LOADED_SOURCES
