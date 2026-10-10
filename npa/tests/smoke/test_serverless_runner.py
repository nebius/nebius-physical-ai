"""Unit tests for npa.smoke.serverless_runner project-id resolution (item 9).

These are import-safe and touch no real infrastructure: they only exercise the
``_project_id`` resolution precedence (explicit > env > config.yaml >
credentials.yaml) that ``npa configure`` relies on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from npa.clients import config as client_config
from npa.smoke import serverless_runner


@pytest.fixture()
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / ".npa" / "config.yaml"
    monkeypatch.setattr(client_config, "CONFIG_PATH", cfg)
    # ``_project_id`` reads ~/.npa/credentials.yaml via expanduser().
    monkeypatch.setenv("HOME", str(tmp_path))
    for env_var in ("NEBIUS_PROJECT_ID", "NPA_PROJECT_ID"):
        monkeypatch.delenv(env_var, raising=False)
    return tmp_path


def _write_config_project(cfg: Path, project_id: str) -> None:
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "default_project": "eu",
                "projects": {"eu": {"project_id": project_id}},
            }
        )
    )


def test_project_id_explicit_wins(isolated_home: Path) -> None:
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    assert serverless_runner._project_id("project-explicit") == "project-explicit"


def test_project_id_env_wins_over_config(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "project-from-env")
    assert serverless_runner._project_id(None) == "project-from-env"


def test_project_id_reads_config_yaml(isolated_home: Path) -> None:
    # The regression this fixes: no env vars, id lives only in config.yaml
    # (where `npa configure` writes it).
    _write_config_project(isolated_home / ".npa" / "config.yaml", "project-from-config")
    assert serverless_runner._project_id(None) == "project-from-config"


def test_project_id_reads_credentials_yaml(isolated_home: Path) -> None:
    creds = isolated_home / ".npa" / "credentials.yaml"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text(yaml.safe_dump({"nebius": {"project_id": "project-from-creds"}}))
    assert serverless_runner._project_id(None) == "project-from-creds"


def test_project_id_error_names_npa_configure(isolated_home: Path) -> None:
    # Nothing configured anywhere -> actionable error that names `npa configure`.
    with pytest.raises(RuntimeError, match="npa configure"):
        serverless_runner._project_id(None)


def test_curobo_golden_image_resolves_exact_digest():
    digest = "sha256:" + "a" * 64
    assert (
        serverless_runner.resolve_golden_image(
            "curobo", registry="ghcr.io/example", tag=digest
        )
        == f"ghcr.io/example/npa-curobo@{digest}"
    )


def test_curobo_serverless_refuses_mutable_tag_before_provider_access(monkeypatch):
    monkeypatch.setattr(
        serverless_runner,
        "_project_id",
        lambda *_: pytest.fail("provider access preceded immutable image gate"),
    )
    with pytest.raises(RuntimeError, match="exact digest"):
        serverless_runner.submit_golden_eval(
            "curobo",
            registry="ghcr.io/example",
            tag="dev-" + "a" * 40,
            wait=False,
        )


def test_curobo_serverless_passes_digest_and_durable_smoke_environment(monkeypatch):
    digest = "sha256:" + "a" * 64
    captured = {}

    class Client:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id="job", status="pending")

    monkeypatch.setattr(serverless_runner, "_project_id", lambda *_: "project")
    monkeypatch.setattr(
        serverless_runner,
        "load_credentials",
        lambda **_kwargs: SimpleNamespace(
            s3_bucket="s3://example-bucket",
            s3_access_key_id="access",
            s3_secret_access_key="secret",
            s3_endpoint="https://s3.example.invalid",
            hf_token=None,
        ),
    )
    monkeypatch.setattr(
        serverless_runner, "resolve_gpu_platform", lambda *_: ("platform", "preset", 1)
    )
    monkeypatch.setattr(serverless_runner, "resolve_subnet", lambda *_: "subnet")
    monkeypatch.setattr(
        serverless_runner, "require_s3_credentials", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        serverless_runner,
        "build_serverless_job_env",
        lambda *, output_path, extra_env, **_kwargs: {
            **extra_env,
            "NPA_OUTPUT_PATH": output_path,
        },
    )
    monkeypatch.setattr(
        serverless_runner, "split_serverless_env", lambda env: (env, {})
    )
    monkeypatch.setattr(serverless_runner, "ServerlessClient", Client)
    result = serverless_runner.submit_golden_eval(
        "curobo",
        project_id="project",
        registry="ghcr.io/example",
        tag=digest,
        expected_image_digest=digest,
        wait=False,
    )
    assert result["image"] == f"ghcr.io/example/npa-curobo@{digest}"
    assert captured["image"] == result["image"]
    assert captured["env"]["NPA_IMAGE_DIGEST"] == digest
    assert captured["env"]["NPA_EXPECTED_IMAGE_DIGEST"] == digest
    assert captured["env"]["NPA_SMOKE_OUTPUT_DIR"] == "/workspace/npa-golden"
    assert captured["env"]["NPA_SMOKE_RUN_ID"] == captured["name"]
    assert captured["env"]["NPA_OUTPUT_PATH"].startswith(
        "s3://example-bucket/golden-evals/"
    )


@pytest.mark.parametrize("expected", [None, "sha256:" + "b" * 64])
def test_curobo_serverless_refuses_unfrozen_or_different_digest_before_access(
    monkeypatch, expected
):
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        serverless_runner,
        "_project_id",
        lambda *_: pytest.fail("provider access preceded candidate digest gate"),
    )
    with pytest.raises(RuntimeError, match="frozen|differs"):
        serverless_runner.submit_golden_eval(
            "curobo",
            registry="ghcr.io/example",
            tag=digest,
            expected_image_digest=expected,
            wait=False,
        )
