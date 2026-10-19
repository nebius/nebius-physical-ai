"""Reject inherited source outside EnvGen's exact pre-flatten COPY closure."""

from pathlib import Path
import subprocess

import pytest


DOCKERFILE = (
    Path(__file__).resolve().parents[2] / "docker/workbench/sim2real-envgen/Dockerfile"
)
RETIRED_FILES = (
    "src/npa/orchestration/skypilot/capacity.py",
    "src/npa/workflows/sim2real/k8s_submit.py",
    "src/npa/workflows/sim2real/materialize.py",
    "src/npa/workflows/sim2real/registry_auth.py",
    "src/npa/workflows/skypilot/README.md",
    "src/npa/workflows/skypilot/cosmos3-generate.yaml",
    "src/npa/workflows/skypilot/nurec-reconstruct.yaml",
)


def _cleanup_command() -> str:
    text = DOCKERFILE.read_text().replace("\\\n", "")
    commands = [
        line.removeprefix("RUN ")
        for line in text.splitlines()
        if line.startswith("RUN test -d /opt/npa")
    ]
    assert len(commands) == 1
    assert "test ! -L /opt/npa" in commands[0]
    assert "rm -rf /opt/npa/src /opt/npa/workflows" in commands[0]
    assert text.index("RUN test -d /opt/npa") < text.index("COPY --chown")
    assert text.index("RUN test -d /opt/npa") < text.index("FROM scratch")
    return commands[0]


def _write(path: Path, content: str = "inherited") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _run_cleanup(root: Path) -> subprocess.CompletedProcess[str]:
    # Substitute only the fixture-owned absolute prefix, not shell syntax.
    command = _cleanup_command().replace("/opt/npa", str(root))
    return subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True)


def test_cleanup_removes_all_seven_inherited_files_before_copy(tmp_path):
    root = tmp_path / "npa"
    for relative in RETIRED_FILES:
        _write(root / relative)
    _write(root / "workflows/stale/catalog.yaml")
    _write(root / "venv/runtime-marker", "preserved runtime")
    _write(tmp_path / "outside-marker", "preserved outside")
    result = _run_cleanup(root)
    assert result.returncode == 0, result.stderr
    assert not (root / "src").exists()
    assert not (root / "workflows").exists()
    assert (root / "venv/runtime-marker").read_text() == "preserved runtime"
    assert (tmp_path / "outside-marker").read_text() == "preserved outside"
    _write(root / "src/npa/current.py", "current source")
    _write(root / "workflows/current.yaml", "current catalog")
    assert sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ) == [
        "src/npa/current.py",
        "venv/runtime-marker",
        "workflows/current.yaml",
    ]


def test_missing_cleanup_leaves_unbound_inherited_files(tmp_path):
    root = tmp_path / "npa"
    for relative in RETIRED_FILES:
        _write(root / relative)
    _write(root / "src/npa/current.py", "current source")
    assert all((root / relative).is_file() for relative in RETIRED_FILES)


@pytest.mark.parametrize("kind", ["missing", "regular_file", "symlink"])
def test_invalid_source_root_refuses_before_deleting_external_bytes(tmp_path, kind):
    root = tmp_path / "npa"
    outside = tmp_path / "external"
    _write(outside / "src/keep", "untouched")
    _write(outside / "workflows/keep", "untouched")
    if kind == "regular_file":
        root.write_text("not a directory")
    elif kind == "symlink":
        root.symlink_to(outside, target_is_directory=True)
    result = _run_cleanup(root)
    assert result.returncode != 0
    assert (outside / "src/keep").read_text() == "untouched"
    assert (outside / "workflows/keep").read_text() == "untouched"


@pytest.mark.parametrize("target", ["src", "workflows"])
def test_owned_leaf_symlink_is_unlinked_not_traversed(tmp_path, target):
    root = tmp_path / "npa"
    root.mkdir()
    outside = tmp_path / "external"
    _write(outside / "keep", "untouched")
    (root / target).symlink_to(outside, target_is_directory=True)
    result = _run_cleanup(root)
    assert result.returncode == 0, result.stderr
    assert not (root / target).exists()
    assert (outside / "keep").read_text() == "untouched"
