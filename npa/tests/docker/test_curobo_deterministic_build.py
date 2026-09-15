"""Exercise the trusted build step with isolated process boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/curobo"


@pytest.mark.parametrize("tool", ["curobo", "lerobot", "cosmos3-serving"])
def test_trusted_workflow_passes_commit_epoch_only_to_curobo(tmp_path, tool):
    steps = yaml.safe_load(
        (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    )["jobs"]["build-development"]["steps"]
    script = next(
        s["run"]
        for s in steps
        if s.get("name") == "Build immutable development image locally"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in {
        "git": "import json,os,sys;from pathlib import Path\nassert sys.argv[1:]==['show','-s','--format=%ct','a'*40]\nPath(os.environ['OBSERVED_GIT']).write_text(json.dumps(sys.argv[1:]))\nprint('1700000000')\n",
        "docker": "import json,os,sys;from pathlib import Path\nPath(os.environ['OBSERVED_DOCKER']).write_text(json.dumps(sys.argv[1:]))\n",
    }.items():
        path = bindir / name
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o700)
    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "TOOL": tool,
        "DEVELOPMENT_SHA": "a" * 40,
        "DOCKERFILE": "Dockerfile",
        "GITHUB_REPOSITORY": "example/project",
        "IMAGE": "test-image:dev-" + "a" * 40,
        "OBSERVED_GIT": str(tmp_path / "git.json"),
        "OBSERVED_DOCKER": str(tmp_path / "docker.json"),
    }
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "docker.json").read_text())
    assert "NPA_SOURCE_SHA=" + "a" * 40 in args
    assert ("SOURCE_DATE_EPOCH=1700000000" in args) == (tool == "curobo")
    assert (tmp_path / "git.json").exists() == (tool == "curobo")
    assert "--push" not in args


@pytest.mark.parametrize(
    "epoch", ["", "0", "-1", "0123", "not-an-epoch", "123\n456", "1700000000"]
)
def test_dockerfile_validates_build_only_epoch_before_any_install(epoch):
    text = (IMAGE / "Dockerfile").read_text()
    assert text.index("ARG SOURCE_DATE_EPOCH") < text.index("RUN pip install")
    assert not any(
        line.lstrip().startswith("ENV SOURCE_DATE_EPOCH") for line in text.splitlines()
    )
    start = text.index("RUN [[")
    end = text.index("\nRUN ", start)
    instruction = text[start + 4 : end]
    env = {**os.environ, "NPA_SOURCE_SHA": "a" * 40, "SOURCE_DATE_EPOCH": epoch}
    result = subprocess.run(
        ["bash", "-c", instruction],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (epoch == "1700000000")
