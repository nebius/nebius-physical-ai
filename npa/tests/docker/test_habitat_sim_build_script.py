"""Exercise the local-only, full-SHA Habitat image build boundary."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/docker/workbench/habitat-sim/build.sh"
SOURCE_PATHS = (
    "docker/workbench/habitat-sim/Dockerfile",
    "docker/workbench/habitat-sim/REDISTRIBUTION.md",
    "docker/workbench/habitat-sim/THIRD_PARTY_NOTICES.md",
    "docker/workbench/habitat-sim/apt-build.lock",
    "docker/workbench/habitat-sim/apt-runtime.lock",
    "docker/workbench/habitat-sim/build.sh",
    "docker/workbench/habitat-sim/entrypoint.sh",
    "docker/workbench/habitat-sim/licenses.json",
    "docker/workbench/habitat-sim/prepare_source.py",
    "docker/workbench/habitat-sim/requirements-build.lock",
    "docker/workbench/habitat-sim/requirements-runtime.lock",
    "docker/workbench/habitat-sim/runtime-payload.json",
    "docker/workbench/habitat-sim/source-manifest.json",
    "docker/workbench/habitat-sim/verify_apt_artifacts.sh",
    "docker/workbench/habitat-sim/verify_image.py",
    "docker/workbench/packaging-contract.yaml",
    "src/npa/__init__.py",
    "src/npa/workflows/__init__.py",
    "src/npa/workflows/habitat_sim_smoke.py",
)


def _run(
    *args: object,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    script: Path = SCRIPT,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *(str(arg) for arg in args)],
        cwd=cwd or script.parents[4],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _committed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    for path in SOURCE_PATHS:
        source = ROOT / "npa" / path
        destination = repository / "npa" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "npa"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Habitat test",
            "-c",
            "user.email=habitat@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return repository, repository / "npa/docker/workbench/habitat-sim/build.sh"


def _stubbed_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker-argv"
    capture = tmp_path / "projected-source"
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$DOCKER_ARGV_LOG"
previous=
for argument in "$@"; do
  if [[ "$previous" == --build-context ]]; then
    source=${argument#npa-source-provenance=}
    cp -a "$source" "$DOCKER_SOURCE_CAPTURE"
  fi
  previous=$argument
done
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_ARGV_LOG": str(log),
        "DOCKER_SOURCE_CAPTURE": str(capture),
    }
    return env, log, capture


def test_builder_requires_one_new_owner_selected_oci_path(tmp_path: Path) -> None:
    missing = _run()
    assert missing.returncode == 2
    assert "usage:" in missing.stderr

    output = tmp_path / "existing.oci.tar"
    output.write_bytes(b"owner data")
    existing = _run(output)
    assert existing.returncode == 2
    assert "refusing to overwrite" in existing.stderr
    assert output.read_bytes() == b"owner data"

    missing_parent = _run(tmp_path / "missing" / "candidate.oci.tar")
    assert missing_parent.returncode == 2
    assert "parent directory does not exist" in missing_parent.stderr


def test_builder_refuses_existing_relative_output_from_original_cwd(
    tmp_path: Path,
) -> None:
    repository, script = _committed_fixture(tmp_path)
    env, log, _capture = _stubbed_environment(tmp_path)
    caller = tmp_path / "outside"
    caller.mkdir()
    output = repository / "existing.oci.tar"
    output.write_bytes(b"owner data")
    relative_output = os.path.relpath(output, caller)

    result = _run(relative_output, cwd=caller, env=env, script=script)

    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    assert output.read_bytes() == b"owner data"
    assert not log.exists()


def test_builder_keeps_relative_output_bound_to_original_cwd(tmp_path: Path) -> None:
    repository, script = _committed_fixture(tmp_path)
    env, log, _capture = _stubbed_environment(tmp_path)
    caller = tmp_path / "outside"
    caller.mkdir()
    repository_output = repository / "candidate.oci.tar"
    repository_output.write_bytes(b"repository owner data")
    (repository / ".git/info/exclude").write_text(
        "candidate.oci.tar\n", encoding="utf-8"
    )
    caller_output = caller / "candidate.oci.tar"

    result = _run("candidate.oci.tar", cwd=caller, env=env, script=script)

    assert result.returncode == 0, result.stderr
    assert (
        f"type=oci,dest={caller_output}" in log.read_text(encoding="utf-8").splitlines()
    )
    assert repository_output.read_bytes() == b"repository owner data"


def test_builder_and_verifier_share_the_exact_source_input_path_set() -> None:
    tree = ast.parse((SCRIPT.parent / "verify_image.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "NPA_SOURCE_PATHS"
            for target in node.targets
        )
    )
    assert tuple(ast.literal_eval(assignment.value)) == SOURCE_PATHS
    script = SCRIPT.read_text(encoding="utf-8")
    declared = script.split("readonly source_paths=(", 1)[1].split("\n)", 1)[0]
    assert tuple(line.strip() for line in declared.splitlines() if line.strip()) == (
        SOURCE_PATHS
    )


def test_builder_passes_exact_git_sha_to_local_attested_oci_export(
    tmp_path: Path,
) -> None:
    repository, script = _committed_fixture(tmp_path)
    env, log, capture = _stubbed_environment(tmp_path)
    output = tmp_path / "candidate.oci.tar"

    result = _run(output, env=env, script=script)

    assert result.returncode == 0, result.stderr
    argv = log.read_text(encoding="utf-8").splitlines()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=repository, text=True
    ).strip()
    assert argv[:2] == ["buildx", "build"]
    assert argv.count("--platform=linux/amd64") == 1
    assert f"NPA_SOURCE_SHA={revision}" in argv
    manifest = capture / "npa-source-manifest.sha256"
    manifest_digest = subprocess.check_output(
        ["sha256sum", str(manifest)], text=True
    ).split()[0]
    assert f"NPA_SOURCE_MANIFEST_SHA256={manifest_digest}" in argv
    assert f"npa-source-provenance={capture}" not in argv
    build_context = next(
        value for value in argv if value.startswith("npa-source-provenance=")
    )
    assert not Path(build_context.split("=", 1)[1]).exists()
    assert f"type=oci,dest={output}" in argv
    assert "--provenance=mode=max" in argv
    assert "--sbom=true" in argv
    assert "--push" not in argv and "--load" not in argv
    assert argv[-1] == "npa"
    observed = {
        path.relative_to(capture / "inputs").as_posix()
        for path in (capture / "inputs").rglob("*")
        if path.is_file()
    }
    assert observed == set(SOURCE_PATHS)
    assert manifest.read_text(encoding="utf-8").splitlines() == sorted(
        manifest.read_text(encoding="utf-8").splitlines(),
        key=lambda row: row.split("  ", 1)[1],
    )
    for path in SOURCE_PATHS:
        committed = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"HEAD:npa/{path}"]
        )
        assert (capture / "inputs" / path).read_bytes() == committed


def test_builder_refuses_a_committed_noncanonical_platform(tmp_path: Path) -> None:
    repository, script = _committed_fixture(tmp_path)
    env, log, _capture = _stubbed_environment(tmp_path)
    content = script.read_text(encoding="utf-8")
    hostile = content.replace(
        'readonly build_platform="linux/amd64"',
        'readonly build_platform="linux/arm64"',
    )
    assert hostile != content
    script.write_text(hostile, encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "npa"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Habitat test",
            "-c",
            "user.email=habitat@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "hostile platform fixture",
        ],
        check=True,
    )

    result = _run(tmp_path / "refused.oci.tar", env=env, script=script)

    assert result.returncode == 2
    assert "refusing non-canonical Habitat build platform" in result.stderr
    assert not log.exists()


@pytest.mark.parametrize("mutation", ["modified", "staged", "untracked"])
def test_builder_refuses_any_noncommitted_repository_state(
    tmp_path: Path, mutation: str
) -> None:
    repository, script = _committed_fixture(tmp_path)
    env, log, _capture = _stubbed_environment(tmp_path)
    target = repository / "npa/src/npa/__init__.py"
    if mutation == "untracked":
        (repository / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    else:
        target.write_text(target.read_text() + "\n# hostile drift\n", encoding="utf-8")
        if mutation == "staged":
            subprocess.run(
                ["git", "-C", str(repository), "add", "npa/src/npa/__init__.py"],
                check=True,
            )

    result = _run(tmp_path / "refused.oci.tar", env=env, script=script)

    assert result.returncode == 2
    assert not log.exists()
    expected = {
        "modified": "refusing a dirty repository",
        "staged": "refusing staged repository changes",
        "untracked": "refusing untracked repository files",
    }
    assert expected[mutation] in result.stderr
