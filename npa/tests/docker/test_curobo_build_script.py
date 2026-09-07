"""Exercise the real shell entrypoint with hermetic Git and Docker boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "npa/docker/workbench/curobo/build.sh"
SOURCE_SHA = "a" * 40
INPUTS = {
    "npa/src/npa",
    "npa/pyproject.toml",
    "npa/README.md",
    "npa/.dockerignore",
    "npa/docker/workbench/curobo",
    "workflows/main",
    "workflows/testing",
}


@pytest.fixture
def build_boundary(tmp_path):
    """No repository/branch creation or actual image build is performed."""
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    git = executable_dir / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        """
import io
import json
import os
import sys
import tarfile
from pathlib import Path

args = sys.argv[1:]
assert args[:1] == ["-C"]
args = args[2:]
with Path(os.environ["TEST_GIT_CALLS"]).open("a") as log:
    log.write(json.dumps(args) + "\\n")
mode = os.environ.get("TEST_GIT_MODE", "clean")
if args[0] == "rev-parse":
    print("bad-sha" if mode == "bad-sha" else "a" * 40)
elif args[0] == "show":
    assert args == ["show", "-s", "--format=%ct", "a" * 40]
    if mode == "epoch-read-failure":
        sys.exit(1)
    print({"bad-epoch": "invalid", "zero-epoch": "0", "negative-epoch": "-1", "empty-epoch": "", "leading-zero-epoch": "0123"}.get(mode, "1700000000"))
elif args[0] == "cat-file":
    sys.exit(1 if mode == "missing-dockerfile" else 0)
elif args[0] == "diff":
    changed = mode in {"dirty-source", "staged-source", "dirty-catalog", "staged-catalog", "dirty-unrelated"}
    if changed:
        path = "docs/unrelated.md" if mode == "dirty-unrelated" else (
            "workflows/testing/curobo-benchmark.yaml" if mode.endswith("catalog") else "npa/src/npa/malicious.py"
        )
        scopes = args[args.index("--") + 1:]
        applies = any(path == p or path.startswith(p + "/") for p in scopes)
        cached = "--cached" in args
        sys.exit(int(applies and (cached == mode.startswith("staged-"))))
    sys.exit(1 if mode == "diff-read-failure" else 0)
elif args[0] == "ls-files":
    if mode == "untracked-source":
        print("npa/src/npa/untracked_payload.py")
    elif mode == "untracked-catalog":
        print("workflows/testing/untracked.yaml")
    elif mode == "untracked-read-failure":
        sys.exit(1)
elif args[0] == "archive":
    if mode == "archive-failure":
        sys.exit(1)
    # The mocked commit deliberately differs from the mutable working tree.
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, payload in {
            "npa/docker/workbench/curobo/Dockerfile": b"FROM committed-base\\n",
            "npa/src/npa/committed.py": b"COMMITTED = True\\n",
            "npa/pyproject.toml": b"[project]\\nname = 'synthetic'\\n",
            "npa/src/npa/workflow_build.py": (
                b"raise RuntimeError('staging failed')\\n" if mode == "staging-failure"
                else Path(os.environ["TEST_WORKFLOW_BUILD"]).read_bytes()
            ),
            "workflows/main/sim2real.yaml": b"metadata: {name: committed-sim2real}\\n",
            "workflows/main/paidf-cosmos3.yaml": b"metadata: {name: committed-paidf-cosmos3}\\n",
            "workflows/testing/physical-ai-data-factory.yaml": b"metadata: {name: committed-paidf}\\n",
            "workflows/testing/curobo-benchmark.yaml": b"metadata: {name: committed-curobo}\\n",
        }.items():
            if mode == "missing-catalog" and name.startswith("workflows/"):
                continue
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
else:
    raise AssertionError(args)
"""
    )
    git.chmod(0o700)
    docker = executable_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        """
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
context = Path(args[-1])
dockerfile = Path(args[args.index("--file") + 1])
Path(os.environ["TEST_DOCKER_CALL"]).write_text(json.dumps({
    "args": args,
    "context": str(context),
    "dockerfile": dockerfile.read_text(),
    "files": sorted(str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()),
    "catalog": {str(p.relative_to(context)): p.read_text() for p in context.rglob("*.yaml")},
}))
"""
    )
    docker.chmod(0o700)
    workflow_build = tmp_path / "workflow_build.py"
    workflow_build.write_bytes((ROOT / "npa/src/npa/workflow_build.py").read_bytes())
    environment = {
        **os.environ,
        "PATH": str(executable_dir) + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(tmp_path),
        "TEST_GIT_CALLS": str(tmp_path / "git.jsonl"),
        "TEST_DOCKER_CALL": str(tmp_path / "docker.json"),
        "TEST_WORKFLOW_BUILD": str(workflow_build),
        "NPA_PYTHON_BIN": sys.executable,
    }

    def run(mode="clean", *args):
        result = subprocess.run(
            ["bash", str(BUILD), *args],
            env={**environment, "TEST_GIT_MODE": mode,
                 "NPA_PYTHON_BIN": str(tmp_path / "missing-python") if mode == "missing-python" else sys.executable},
            capture_output=True,
            text=True,
            check=False,
        )
        calls = [json.loads(line) for line in (tmp_path / "git.jsonl").read_text().splitlines()]
        docker_call = tmp_path / "docker.json"
        return result, calls, json.loads(docker_call.read_text()) if docker_call.exists() else None

    return run


@pytest.mark.parametrize("mode", ["clean", "dirty-unrelated"])
def test_only_exact_commit_snapshot_reaches_docker(build_boundary, mode):
    result, calls, docker = build_boundary(mode)
    assert result.returncode == 0, result.stderr
    assert docker["dockerfile"] == "FROM committed-base\n"
    catalog = {
        "src/npa/workflows/main/sim2real.yaml": "metadata: {name: committed-sim2real}\n",
        "src/npa/workflows/main/paidf-cosmos3.yaml": "metadata: {name: committed-paidf-cosmos3}\n",
        "src/npa/workflows/testing/physical-ai-data-factory.yaml": "metadata: {name: committed-paidf}\n",
        "src/npa/workflows/testing/curobo-benchmark.yaml": "metadata: {name: committed-curobo}\n",
    }
    assert docker["catalog"] == catalog
    assert docker["files"] == sorted([
        "docker/workbench/curobo/Dockerfile", "pyproject.toml",
        "src/npa/committed.py", "src/npa/workflow_build.py", *catalog,
    ])
    assert docker["context"] != str(ROOT / "npa")
    assert not Path(docker["context"]).exists(), "The run-owned build snapshot must be removed"
    assert f"npa-curobo:dev-{SOURCE_SHA}" in docker["args"]
    assert f"NPA_SOURCE_SHA={SOURCE_SHA}" in docker["args"]
    assert "SOURCE_DATE_EPOCH=1700000000" in docker["args"]
    label_index = docker["args"].index("--label")
    assert docker["args"][label_index + 1] == (
        "org.opencontainers.image.source=https://github.com/nebius/nebius-physical-ai"
    )
    assert ["show", "-s", "--format=%ct", SOURCE_SHA] in calls
    assert "--push" not in docker["args"]
    for call in calls:
        if call[0] in {"diff", "ls-files", "archive"}:
            assert set(call[call.index("--") + 1 :]) == INPUTS
    assert ["cat-file", "-e", f"{SOURCE_SHA}:npa/docker/workbench/curobo/Dockerfile"] in calls
    assert next(c for c in calls if c[0] == "archive")[:3] == ["archive", SOURCE_SHA, "--"]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bad-sha", "Full source SHA is required"),
        ("bad-epoch", "Positive source commit epoch is required"),
        ("zero-epoch", "Positive source commit epoch is required"),
        ("negative-epoch", "Positive source commit epoch is required"),
        ("empty-epoch", "Positive source commit epoch is required"),
        ("leading-zero-epoch", "Positive source commit epoch is required"),
        ("epoch-read-failure", ""),
        ("missing-dockerfile", "Dockerfile must be checked in"),
        ("dirty-source", "Commit the cuRobo build input changes"),
        ("staged-source", "Commit the cuRobo build input changes"),
        ("dirty-catalog", "Commit the cuRobo build input changes"),
        ("staged-catalog", "Commit the cuRobo build input changes"),
        ("diff-read-failure", "Commit the cuRobo build input changes"),
        ("untracked-source", "Untracked cuRobo build inputs"),
        ("untracked-catalog", "Untracked cuRobo build inputs"),
        ("untracked-read-failure", ""),
        ("archive-failure", ""),
        ("staging-failure", "staging failed"),
        ("missing-catalog", "catalog is missing"),
        ("missing-python", "Repository Python is required"),
    ],
)
def test_unreviewed_or_unreadable_build_inputs_never_reach_docker(build_boundary, mode, message, tmp_path):
    result, _, docker = build_boundary(mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert docker is None
    assert not list(tmp_path.glob("npa-curobo-build.*"))


def test_public_push_refuses_before_docker(build_boundary):
    result, _, docker = build_boundary("clean", "--push")
    assert result.returncode != 0
    assert "trusted publication workflow" in result.stderr
    assert docker is None
