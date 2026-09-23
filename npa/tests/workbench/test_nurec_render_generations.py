"""NuRec render attempts preserve earlier media and publish their own output."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path
import threading

import npa.cli.nurec as cli
import pytest
from npa.workbench.nurec.nurec import NurecConfig, render_novel_views
from typer.testing import CliRunner


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def _media(root: Path) -> dict[str, bytes]:
    return {
        name: data
        for name, data in _tree(root).items()
        if Path(name).suffix in {".png", ".mp4"}
    }


def _seed_output(root: Path) -> dict[str, bytes | None]:
    (root / "old-camera").mkdir(parents=True)
    (root / "old-camera/000000.png").write_bytes(b"previous image")
    (root / "old-camera.mp4").write_bytes(b"previous video")
    (root / "prior-receipt.json").write_text('{"status":"ok"}')
    return _tree(root)


def _preserved(root: Path, previous: dict[str, bytes | None]) -> None:
    current = _tree(root)
    assert all(current.get(name) == data for name, data in previous.items())


def _native_output(command: list[str], marker: bytes = b"new") -> Path:
    output = Path(command[command.index("--output-dir") + 1])
    camera = output / "new-camera"
    camera.mkdir(parents=True, exist_ok=True)
    (camera / "000000.png").write_bytes(marker + b" first")
    (camera / "000001.png").write_bytes(marker + b" last")
    (output / "new-camera.mp4").write_bytes(marker + b" video")
    return output


@pytest.fixture
def render_inputs(tmp_path: Path):
    artifact = tmp_path / "trained.usdz"
    artifact.write_bytes(b"existing trained scene")
    output = tmp_path / "renders"
    previous = _seed_output(output)
    config = NurecConfig.from_env(environ={}, entrypoint="native-render-test")
    return config, artifact, output, previous


def _render(inputs, runner, *, dry_run=False):
    config, artifact, output, _ = inputs
    return render_novel_views(
        config,
        artifact_path=str(artifact),
        output_dir=str(output),
        camera_ids=["new-camera"],
        runner=runner,
        environ={},
        dry_run=dry_run,
    )


def test_render_reports_only_new_media_and_preserves_previous_output(render_inputs):
    calls = []

    def runner(command, **_kwargs):
        calls.append(_native_output(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    result = _render(render_inputs, runner)

    assert result.ok
    assert len(calls) == 1
    assert Path(result.output_dir) == calls[0]
    assert result.frame_count == 2
    assert result.video_count == 1
    assert _media(Path(result.output_dir)) == {
        "new-camera/000000.png": b"new first",
        "new-camera/000001.png": b"new last",
        "new-camera.mp4": b"new video",
    }
    _preserved(render_inputs[2], render_inputs[3])


def test_successful_native_exit_without_new_media_does_not_reuse_old_frames(
    render_inputs,
):
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    result = _render(render_inputs, runner)

    assert len(calls) == 1
    assert not result.ok
    assert result.errors
    _preserved(render_inputs[2], render_inputs[3])


def test_failed_native_render_keeps_previous_output(render_inputs):
    def runner(command, **_kwargs):
        _native_output(command, b"partial")
        return subprocess.CompletedProcess(command, 7, "", "native render failed")

    result = _render(render_inputs, runner)

    assert not result.ok
    assert result.errors
    assert not result.output_uri
    _preserved(render_inputs[2], render_inputs[3])


def test_successive_renders_keep_their_distinct_media(render_inputs):
    calls = []

    def runner(command, **_kwargs):
        calls.append(_native_output(command, str(len(calls)).encode()))
        return subprocess.CompletedProcess(command, 0, "", "")

    first = _render(render_inputs, runner)
    first_files = _tree(Path(first.output_dir))
    second = _render(render_inputs, runner)

    assert first.ok and second.ok
    assert len(calls) == 2
    assert first.output_dir != second.output_dir
    _preserved(Path(first.output_dir), first_files)
    _preserved(render_inputs[2], render_inputs[3])
    assert _media(Path(first.output_dir)) != _media(Path(second.output_dir))


def test_render_dry_run_does_not_create_generation_or_modify_files(render_inputs):
    parent = render_inputs[2].parent
    before = _tree(parent)

    def runner(*_args, **_kwargs):
        pytest.fail("dry run invoked native renderer")

    result = _render(render_inputs, runner, dry_run=True)

    assert result.ok
    assert _tree(parent) == before


def _controlled_cli(monkeypatch, config):
    original_publish = cli._publish
    published_sources = []

    def native(command, **_kwargs):
        _native_output(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def controlled(*args, **kwargs):
        return render_novel_views(*args, **kwargs, runner=native, environ={})

    def publish(source, destination):
        published_sources.append(Path(source))
        return original_publish(source, destination)

    monkeypatch.setattr(cli, "_config", lambda **_kwargs: config)
    monkeypatch.setattr(cli, "render_novel_views", controlled)
    monkeypatch.setattr(cli, "_publish", publish)
    return published_sources


def _invoke_cli(artifact, output, destination):
    return CliRunner().invoke(
        cli.app,
        [
            "render",
            "--artifact-path",
            str(artifact),
            "--output-dir",
            str(output),
            "--output-uri",
            str(destination),
            "--camera-id",
            "new-camera",
            "--output",
            "json",
        ],
    )


def test_cli_publishes_the_returned_generation(render_inputs, tmp_path, monkeypatch):
    config, artifact, output, previous = render_inputs
    published_sources = _controlled_cli(monkeypatch, config)
    result = _invoke_cli(artifact, output, tmp_path / "published")

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert published_sources == [Path(payload["output_dir"])]
    assert _media(Path(payload["output_uri"])) == _media(Path(payload["output_dir"]))
    assert len(_media(Path(payload["output_uri"]))) == 3
    _preserved(output, previous)


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_initial_renders_reserve_distinct_generations(tmp_path, existing):
    artifact = tmp_path / "trained.usdz"
    artifact.write_bytes(b"scene")
    output = tmp_path / "renders"
    if existing:
        output.mkdir()
    inputs = (
        NurecConfig.from_env(environ={}, entrypoint="native-test"),
        artifact,
        output,
        {},
    )
    ready = threading.Barrier(2)

    def invoke(marker):
        def native(command, **_kwargs):
            # Both invocations choose their output before either produces media.
            ready.wait(timeout=10)
            _native_output(command, marker)
            return subprocess.CompletedProcess(command, 0, "", "")

        return _render(inputs, native)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(invoke, b"first")
        second = executor.submit(invoke, b"second")
        results = [first.result(), second.result()]
    assert all(result.ok for result in results)
    assert results[0].output_dir != results[1].output_dir
    for result, marker in zip(results, [b"first", b"second"], strict=True):
        assert result.frame_count == 2 and result.video_count == 1
        assert _media(Path(result.output_dir)) == {
            "new-camera/000000.png": marker + b" first",
            "new-camera/000001.png": marker + b" last",
            "new-camera.mp4": marker + b" video",
        }


def test_prior_nonmedia_output_cannot_be_overwritten(tmp_path):
    artifact = tmp_path / "trained.usdz"
    artifact.write_bytes(b"scene")
    output = tmp_path / "renders"
    output.mkdir()
    previous = output / "metadata.json"
    previous.write_bytes(b"previous native metadata")
    config = NurecConfig.from_env(environ={}, entrypoint="native-test")

    def native(command, **_kwargs):
        generation = _native_output(command)
        (generation / "metadata.json").write_bytes(b"current native metadata")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = _render((config, artifact, output, {}), native)
    assert result.ok
    assert previous.read_bytes() == b"previous native metadata"
    assert (
        Path(result.output_dir) / "metadata.json"
    ).read_bytes() == b"current native metadata"
