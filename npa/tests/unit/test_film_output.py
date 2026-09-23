"""Check generic rendering output options and preserve offline planning behavior."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def renderer(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "npa/src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("edit"), importlib.import_module("render")


@pytest.mark.parametrize("command", ["preview", "final"])
def test_output_options_reach_the_generic_renderer(renderer, command, tmp_path):
    edit, _ = renderer
    args = edit._parser().parse_args(
        [
            command,
            "--output-path",
            "s3://test-bucket/clip.mp4",
            "--storage-project",
            "test-storage",
            "--scene",
            "opening",
        ]
    )
    project = {
        name: tmp_path / name
        for name in ("storyboard", "assets", "voice_dir", "output_dir")
    }
    argv = edit._render_command(args, project)
    assert argv[argv.index("--output-path") + 1] == "s3://test-bucket/clip.mp4"
    assert argv[argv.index("--storage-project") + 1] == "test-storage"
    assert argv[argv.index("--scene") + 1] == "opening"


@pytest.mark.parametrize(
    "options",
    [
        ["narrate", "--output-path", "s3://test-bucket/clip.mp4"],
        ["final", "--storage-project", "test-storage"],
        ["final", "--output-path", ""],
        [
            "final",
            "--output-path",
            "s3://test-bucket/clip.mp4",
            "--storage-project",
            "",
        ],
        ["final", "--output-path", "https://example.invalid/clip.mp4"],
    ],
)
def test_invalid_options_fail_before_loading_or_rendering(
    renderer, monkeypatch, options
):
    edit, _ = renderer
    monkeypatch.setattr(edit.sys, "argv", ["edit.py", *options])
    monkeypatch.setattr(
        edit, "_project", lambda *a: pytest.fail("must reject before loading")
    )
    with pytest.raises(SystemExit) as result:
        edit._main()
    assert result.value.code == 2


def _mock_render(renderer, tmp_path, monkeypatch, *, plan=False, check=False):
    _, render = renderer
    story = tmp_path / "story.json"
    story.write_text(json.dumps({"scenes": [{"id": "opening"}]}))
    args = SimpleNamespace(
        storyboard=story,
        assets=None,
        scene=None,
        voice_dir=tmp_path / "voice",
        output_dir=tmp_path / "renders",
        plan=plan,
        check=check,
        output_path="s3://test-bucket/clip.mp4",
        storage_project="test-storage",
    )
    monkeypatch.setattr(render, "_arguments", lambda: args)
    monkeypatch.setattr(render, "_validate_storyboard", lambda *a: None)
    monkeypatch.setattr(render, "_load_assets", lambda *a: {})
    monkeypatch.setattr(render, "_validate", lambda *a: [])
    monkeypatch.setattr(render, "_environment", lambda: {})
    monkeypatch.setattr(render, "_visual_plan", lambda *a: [])
    monkeypatch.setattr(render, "_verify_narration", lambda *a: None)
    return render, args


@pytest.mark.parametrize("mode", ["plan", "check"])
def test_read_only_render_modes_never_upload(renderer, tmp_path, monkeypatch, mode):
    render, _ = _mock_render(renderer, tmp_path, monkeypatch, **{mode: True})
    monkeypatch.setattr(
        render, "_deliver_output", lambda *a: pytest.fail("must not upload")
    )
    monkeypatch.setattr(
        render, "_render_film", lambda *a: pytest.fail("must not render")
    )
    render._main()


def test_only_successful_render_is_delivered(renderer, tmp_path, monkeypatch):
    render, args = _mock_render(renderer, tmp_path, monkeypatch)
    events = []
    target = tmp_path / "finished.mp4"

    def finish(*unused):
        target.write_bytes(b"completed video")
        events.append("render")
        return target

    def deliver(options, source):
        assert options is args and source.read_bytes() == b"completed video"
        events.append("delivery")

    monkeypatch.setattr(render, "_render_film", finish)
    monkeypatch.setattr(render, "_deliver_output", deliver)
    render._main()
    assert events == ["render", "delivery"]


def test_render_failure_never_uploads_previous_delivery(
    renderer, tmp_path, monkeypatch
):
    render, _ = _mock_render(renderer, tmp_path, monkeypatch)

    def fail(*args):
        raise ValueError("render failure")

    monkeypatch.setattr(render, "_render_film", fail)
    monkeypatch.setattr(
        render, "_deliver_output", lambda *a: pytest.fail("must not upload")
    )
    with pytest.raises(ValueError, match="render failure"):
        render._main()


def test_local_only_output_does_not_import_cloud_code(renderer, monkeypatch, tmp_path):
    output = importlib.import_module("film_output")
    import builtins

    original = builtins.__import__

    def require_offline(name, *args, **kwargs):
        if name.startswith(("npa.video_output", "boto", "npa.clients")):
            pytest.fail("local output must not load cloud configuration")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", require_offline)
    output._deliver_output(SimpleNamespace(output_path=None), tmp_path / "film.mp4")
