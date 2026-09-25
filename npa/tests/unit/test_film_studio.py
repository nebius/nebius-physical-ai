"""Check independent film selection, narration-free drafts and debounced local editing."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def studio(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "npa/src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("studio")


@pytest.fixture
def project(studio, tmp_path):
    draft = importlib.import_module("film_draft")
    source = tmp_path / "source.png"
    source.write_bytes(b"source image bytes")
    assets = {
        "source": {"path": "source.png", "kind": "image", "sha256": draft._hash(source)}
    }
    (tmp_path / "assets.json").write_text(json.dumps(assets))
    scene = {
        "id": "opening",
        "duration": 4,
        "layout": "screen",
        "title": ["A goal"],
        "assets": ["source"],
        "labels": ["SIMULATION"],
        "narration": "Unrecorded text",
    }
    pending = {**scene, "id": "later", "assets": ["not-produced-yet"]}
    (tmp_path / "story.json").write_text(
        json.dumps({"title": "Film", "duration": 8, "scenes": [scene, pending]})
    )
    path = tmp_path / "film-project.json"
    path.write_text(
        json.dumps(
            {
                "storyboard": "story.json",
                "assets": "assets.json",
                "voice_dir": "missing-narration",
                "output_dir": "renders",
            }
        )
    )
    return path, draft


def test_registry_resolves_independent_projects_relative_to_registry(
    studio, tmp_path, monkeypatch
):
    registry = tmp_path / "studio.json"
    registry.write_text(
        json.dumps(
            {
                "projects": {
                    "exec": "exec/project.json",
                    "inference": "inference/project.json",
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    projects = studio._projects(registry)
    assert projects["exec"] == tmp_path / "exec/project.json"
    assert projects["inference"] == tmp_path / "inference/project.json"
    assert projects["exec"] != projects["inference"]


def test_architecture_draft_needs_no_media_or_credentials(studio, tmp_path):
    draft = importlib.import_module("film_draft")
    scene = {
        "id": "infrastructure",
        "duration": 12,
        "layout": "architecture",
        "title": ["Connect the workflow"],
        "subtitle": "",
        "eyebrow": "",
        "assets": [],
        "labels": [],
        "narration": "Use the right infrastructure.",
        "controller": "Workbench",
        "architecture_note": "Reference deployment.",
        "compute_nodes": [
            {"title": "Compute", "purpose": "Inference", "models": ["Model"]}
        ],
        "storage_node": {"title": "Storage", "detail": "Run outputs"},
    }
    story = tmp_path / "story.json"
    assets = tmp_path / "assets.json"
    story.write_text(json.dumps({"title": "Film", "scenes": [scene]}))
    assets.write_text("{}")
    _, _, selected, required = draft._inputs(
        {"storyboard": story, "assets": assets}, "infrastructure"
    )
    assert selected == {**scene, "footer": "Film"}
    assert required == {}
    assert draft.render._rectangles(scene) == []
    scene["compute_nodes"][0]["models"] = "not a list"
    with pytest.raises(ValueError, match="model labels"):
        draft.render._validate_layout(scene)


@pytest.mark.parametrize(
    "projects",
    [
        [],
        {"list": "p.json"},
        {"init": "p.json"},
        {"search": "p.json"},
        {"exec": {}},
        {"Exec": "p.json"},
    ],
)
def test_invalid_registry_fails_before_running_a_command(studio, tmp_path, projects):
    path = tmp_path / "studio.json"
    path.write_text(json.dumps({"projects": projects}))
    with pytest.raises(ValueError):
        studio._projects(path)


def test_studio_preserves_freeform_authoring_options_and_selected_project(
    studio, tmp_path
):
    project = tmp_path / "project with spaces.json"
    options = ["--prompt", "A technical film about simulation", "--duration", "75"]
    command = studio._command(project, "brief", options)
    assert command[2:] == ["brief", "--project", str(project), *options]
    assert studio._command(project, "draft", ["--scene", "one"])[1].endswith(
        "film_draft.py"
    )
    with pytest.raises(ValueError, match="selects --project"):
        studio._command(project, "final", ["--project=another.json"])


def test_selected_command_help_reaches_the_draft_parser(studio, tmp_path, monkeypatch):
    registry = tmp_path / "studio.json"
    registry.write_text(json.dumps({"projects": {"inference": "project.json"}}))
    monkeypatch.setattr(
        studio.sys,
        "argv",
        ["studio.py", "--registry", str(registry), "inference", "draft", "--help"],
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(studio.subprocess, "run", run)
    with pytest.raises(SystemExit) as stopped:
        studio._main()
    assert stopped.value.code == 0
    assert calls[0][1].endswith("film_draft.py") and calls[0][-1] == "--help"


def _fake_clip(draft, tmp_path, monkeypatch):
    clip = tmp_path / "cached.mp4"
    clip.write_bytes(b"verified scene")
    monkeypatch.setattr(
        draft.render,
        "_probe",
        lambda path: {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 960,
                    "height": 540,
                    "nb_frames": "120",
                    "r_frame_rate": "30/1",
                }
            ],
        },
    )
    monkeypatch.setattr(draft, "_environment", lambda: {})
    monkeypatch.setattr(draft.render, "_scene_job", lambda *args: (clip, True))
    return clip


def test_draft_works_without_narration_or_other_scenes_media(project, monkeypatch):
    path, draft = project
    clip = _fake_clip(draft, path.parent, monkeypatch)
    target = draft._draft(path, "opening")
    assert target.read_bytes() == clip.read_bytes()
    report = json.loads((target.parent / "draft.json").read_text())
    assert report["audio"] is False and report["scene_reused"] is True
    assert set(report["assets"]) == {"source"}
    assert not (path.parent / "missing-narration").exists()
    assert not (path.parent / "renders/final").exists()


@pytest.mark.parametrize("changed", ["source.png", "story.json", "film-project.json"])
def test_draft_retains_previous_output_when_inputs_change_during_render(
    project, monkeypatch, changed
):
    path, draft = project
    clip = _fake_clip(draft, path.parent, monkeypatch)
    target = draft._draft(path, "opening")
    original = target.read_bytes()

    def edit_while_rendering(*args):
        source = path.parent / changed
        source.write_bytes(source.read_bytes() + b" ")
        return clip, False

    monkeypatch.setattr(draft.render, "_scene_job", edit_while_rendering)
    with pytest.raises(ValueError, match="changed during the draft"):
        draft._draft(path, "opening")
    assert target.read_bytes() == original


def test_watcher_tracks_selected_media_and_keeps_paths_during_partial_json_edit(
    project,
):
    path, _ = project
    watch = importlib.import_module("film_watch")
    paths = watch._watch_paths(path, "opening")
    assert path.parent / "source.png" in paths
    assert not any("missing-narration" in str(p) for p in paths)
    original = watch._signature(paths)
    (path.parent / "source.png").write_bytes(b"updated source bytes")
    assert watch._signature(paths) != original
    (path.parent / "story.json").write_text("{")
    assert paths <= watch._watch_paths(path, "opening", paths)


def test_watch_debounce_coalesces_edits_and_waits_after_failed_attempt(studio):
    watch = importlib.import_module("film_watch")
    changes = watch._Changes(0.5)
    assert not changes.ready("initial", 0)
    assert not changes.ready("edited", 0.4)
    assert not changes.ready("edited", 0.8)
    assert changes.ready("edited", 0.91)
    assert not changes.ready("edited", 2)
    assert not changes.ready("corrected", 3)
    assert changes.ready("corrected", 3.51)
