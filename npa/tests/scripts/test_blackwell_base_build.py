"""Exercise canonical and legacy base-build names without a Docker daemon."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


BASES = Path(__file__).resolve().parents[2] / "docker/workbench/base"
CANONICAL = BASES / "cuda13-blackwell"
LEGACY = BASES / "cuda13-b300"
SUFFIX = "dev-" + "a" * 40


@pytest.fixture
def docker_calls(tmp_path, monkeypatch):
    executable = tmp_path / "docker"
    log = tmp_path / "docker.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_DOCKER_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('TEST_DOCKER_FAIL') in sys.argv[1:]:\n"
        "    sys.exit(17)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_DOCKER_LOG", str(log))
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("TEST_DOCKER_FAIL", raising=False)
    return log


def _run(directory, *args):
    return subprocess.run(
        ["bash", str(directory / "build.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def _tags(args):
    return [args[i + 1] for i, arg in enumerate(args) if arg == "-t"]


@pytest.mark.parametrize("directory", [CANONICAL, LEGACY])
def test_both_paths_build_one_image_with_both_names(directory, docker_calls):
    result = _run(directory, "--tag", SUFFIX)
    assert result.returncode == 0, result.stderr
    calls = _calls(docker_calls)
    assert len(calls) == 1
    assert calls[0][0] == "build"
    assert calls[0][-1] == str(CANONICAL)
    assert _tags(calls[0]) == [
        f"npa-base:cuda13-blackwell-{SUFFIX}",
        f"npa-base:cuda13-b300-{SUFFIX}",
    ]
    assert f"Built: npa-base:cuda13-blackwell-{SUFFIX}" in result.stdout


def test_legacy_directory_is_one_recipe_alias():
    assert LEGACY.is_symlink()
    assert LEGACY.resolve() == CANONICAL
    for relative in (
        "Dockerfile",
        "build.sh",
        "scripts/flash_attn_root.py",
        "scripts/check_torch_gpu_arch.py",
        "scripts/gpu_capability_smoke.py",
    ):
        assert (LEGACY / relative).samefile(CANONICAL / relative)


@pytest.mark.parametrize("context", [None, "test-builder"])
def test_registry_push_uses_same_build_and_context(docker_calls, monkeypatch, context):
    if context:
        monkeypatch.setenv("DOCKER_CONTEXT", context)
    result = _run(
        CANONICAL, "--registry", "registry.example/team/", "--tag", SUFFIX, "--push"
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(docker_calls)
    prefix = ["--context", context] if context else []
    assert len(calls) == 3
    assert calls[0][: len(prefix) + 1] == [*prefix, "build"]
    registry_tags = [
        f"registry.example/team/npa-base:cuda13-blackwell-{SUFFIX}",
        f"registry.example/team/npa-base:cuda13-b300-{SUFFIX}",
    ]
    assert _tags(calls[0])[2:] == registry_tags
    assert calls[1:] == [[*prefix, "push", tag] for tag in registry_tags]


def test_registry_tagging_without_push(docker_calls):
    result = _run(CANONICAL, "--registry", "registry.example/team", "--tag", SUFFIX)
    assert result.returncode == 0, result.stderr
    calls = _calls(docker_calls)
    assert len(calls) == 1
    assert len(_tags(calls[0])) == 4


@pytest.mark.parametrize(("failure", "call_count"), [("build", 1), ("push", 2)])
def test_docker_failure_stops_before_further_pushes(
    docker_calls, monkeypatch, failure, call_count
):
    monkeypatch.setenv("TEST_DOCKER_FAIL", failure)
    result = _run(
        CANONICAL, "--registry", "registry.example/team", "--tag", SUFFIX, "--push"
    )
    assert result.returncode == 17
    assert len(_calls(docker_calls)) == call_count


@pytest.mark.parametrize(
    "args",
    [("--push",), ("--registry",), ("--tag",), ("--arch-list",), ("--unknown",)],
)
def test_invalid_arguments_fail_before_docker(docker_calls, args):
    result = _run(CANONICAL, *args)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr
    assert not docker_calls.exists()


def test_architecture_arguments_survive_legacy_path(docker_calls):
    result = _run(
        LEGACY, "--tag", SUFFIX, "--arch-list", "10.0 12.0", "--require-archs", ""
    )
    assert result.returncode == 0, result.stderr
    args = _calls(docker_calls)[0]
    assert "TORCH_CUDA_ARCH_LIST=10.0 12.0" in args
    assert "REQUIRE_TORCH_ARCHS=" in args


def test_help_explains_both_names_without_docker(docker_calls):
    result = _run(LEGACY, "--help")
    assert result.returncode == 0
    assert "cuda13-blackwell" in result.stdout
    assert "cuda13-b300" in result.stdout
    assert not docker_calls.exists()
