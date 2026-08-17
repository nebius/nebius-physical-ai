"""Every BYOF runner must forward the S3 credentials its profile needs.

Live regression (SkyPilot job 206): a run provisioned, pulled the Isaac image, executed
the relocated profile's script — and then died at the artifact upload with
``botocore.exceptions.NoCredentialsError: Unable to locate credentials``, because the
runner called ``submit_workflow`` without ``secret_envs``. The profiles all upload a
summary to S3, so the credentials are not optional.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "npa" / "scripts"
RUNNERS = ("run_isaac_lab_rl.py", "run_byof_datagen.py", "run_byof_container_verify.py")


def _load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", RUNNERS)
def test_runner_defaults_to_forwarding_s3_credentials(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(name)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None) == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]


@pytest.mark.parametrize("name", RUNNERS)
def test_runner_drops_unset_names(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """SkyPilot rejects a secret it cannot resolve, so an unset name must not be sent."""

    module = _load(name)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None) == ["AWS_SECRET_ACCESS_KEY"]


@pytest.mark.parametrize("name", RUNNERS)
def test_explicit_secret_env_wins_and_dedupes(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(name)
    monkeypatch.setenv("HF_TOKEN", "probe-token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")

    assert module.resolve_secret_envs(["HF_TOKEN", "HF_TOKEN"]) == ["HF_TOKEN"]


@pytest.mark.parametrize("name", RUNNERS)
def test_runner_passes_secret_envs_to_submit_workflow(name: str) -> None:
    """Textual guard: the wiring must not be dropped in a future refactor."""

    text = (SCRIPTS / name).read_text(encoding="utf-8")

    assert "secret_envs=resolve_secret_envs(" in text
    # The runner must name the solution, or every image receives every vendor's
    # operator answers whenever they happen to be set in the shell.
    if name == "run_byof_container_verify.py":
        assert "solution_name=args.solution_name" in text
    assert '"--secret-env"' in text
