"""Exercise profile rollback through an installed CLI with disposable local config."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from npa.clients import nebius


@pytest.fixture
def native_profile(tmp_path):
    executable = shutil.which("nebius")
    if executable is None:
        pytest.skip("Native Nebius CLI is not installed")
    config = tmp_path / "profile.yaml"
    home = tmp_path / "home"
    home.mkdir()
    environment = {"HOME": str(home), "PATH": os.defpath, "LANG": "C.UTF-8"}

    def invoke(args):
        return subprocess.run(
            [executable, "--config", str(config), "--no-check-update", *args],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    created = invoke(
        [
            "profile",
            "create",
            "review-placeholder",
            "--skip-auth",
            "--endpoint",
            "api.example.invalid",
            "--token-endpoint",
            "http://127.0.0.1:9",
            "--parent-id",
            "project-placeholder",
            "--tenant-id",
            "tenant-placeholder",
        ]
    )
    assert created.returncode == 0, created.stderr
    return invoke


def _runner_with_failed_tenant_write(native_profile, failure_timing, calls):
    failed_write = ["config", "set", "tenant-id", "tenant-new"]

    def isolated_run(args):
        assert args[:2] in (["config", "get"], ["config", "set"], ["config", "unset"])
        calls.append(args)
        failing = args == failed_write
        forwarded = (
            args + ["--unsupported-test-flag"]
            if failing and failure_timing == "before_write"
            else args
        )
        result = native_profile(forwarded)
        if failing:
            if failure_timing == "before_write":
                assert result.returncode != 0
            else:
                assert result.returncode == 0, result.stderr
            raise nebius.NebiusError(
                "Injected second-write failure or lost acknowledgement"
            )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    return isolated_run


@pytest.mark.parametrize("initial_parent", ["project-placeholder", ""])
@pytest.mark.parametrize("failure_timing", ["before_write", "after_write"])
def test_native_profile_second_write_failure_restores_original_values(
    native_profile, monkeypatch, initial_parent, failure_timing
):
    if not initial_parent:
        assert native_profile(["config", "unset", "parent-id"]).returncode == 0
    calls = []
    monkeypatch.setattr(
        nebius,
        "_run",
        _runner_with_failed_tenant_write(native_profile, failure_timing, calls),
    )

    result = nebius.set_profile_project("project-new", "tenant-new")

    assert result is nebius.ProfileMutationResult.RESTORED
    assert ["config", "set", "tenant-id", "tenant-new"] in calls
    for key, expected in (
        ("parent-id", initial_parent),
        ("tenant-id", "tenant-placeholder"),
    ):
        observed = native_profile(["config", "get", key])
        assert observed.returncode == 0, observed.stderr
        assert observed.stdout.strip() == expected
    if not initial_parent:
        assert ["config", "unset", "parent-id"] in calls
