"""Exercise RoboCasa's build snapshot through hermetic Git and Docker boundaries."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "npa/docker/workbench/robocasa/build.sh"
SPEC = importlib.util.spec_from_file_location(
    "robocasa_build_boundaries", Path(__file__).with_name("test_curobo_build_script.py")
)
BOUNDARIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARIES)


@pytest.fixture
def build_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(BOUNDARIES, "BUILD", BUILD)
    run = BOUNDARIES.build_boundary.__wrapped__(tmp_path)
    git = tmp_path / "bin/git"
    git.write_text(git.read_text().replace(
        "npa/docker/workbench/curobo/Dockerfile", "npa/docker/workbench/robocasa/Dockerfile"
    ))
    return run


@pytest.mark.parametrize("mode", ["clean", "dirty-unrelated"])
def test_build_uses_exact_archived_source_and_shared_filter(build_boundary, mode):
    result, calls, docker = build_boundary(mode)
    assert result.returncode == 0, result.stderr
    assert docker["dockerfile"] == "FROM committed-base\n"
    assert docker["context"] != str(ROOT / "npa")
    assert not Path(docker["context"]).exists()
    assert "npa-robocasa:dev-" + "a" * 40 in docker["args"]
    assert "NPA_SOURCE_SHA=" + "a" * 40 in docker["args"]
    assert "src/npa/workflows/main/sim2real.yaml" in docker["catalog"]
    for call in calls:
        if call[0] in {"diff", "ls-files", "archive"}:
            assert "npa/docker/workbench/curobo/filter_cudnn_runtime.py" in call
            assert "npa/docker/workbench/robocasa" in call


@pytest.mark.parametrize("mode", [
    "bad-sha", "missing-dockerfile", "dirty-source", "staged-source",
    "dirty-catalog", "staged-catalog", "diff-read-failure", "untracked-source",
    "untracked-catalog", "untracked-read-failure", "archive-failure",
    "staging-failure", "missing-catalog", "missing-python",
])
def test_unreviewed_inputs_never_reach_docker(build_boundary, mode, tmp_path):
    result, _calls, docker = build_boundary(mode)
    assert result.returncode != 0
    assert docker is None
    assert not list(tmp_path.glob("npa-robocasa-build.*"))


@pytest.mark.parametrize("args", [("--push",), ("--tag", "latest"), ("--tag", "0.1.0")])
def test_publication_and_mutable_tags_are_refused(build_boundary, args):
    result, _calls, docker = build_boundary("clean", *args)
    assert result.returncode != 0
    assert docker is None
