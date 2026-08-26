"""Flag alignment and staged-source persistence found by the PAIDF walkthrough."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app


runner = CliRunner()


def _option_names(command_path: list[str]) -> set[str]:
    import typer.main

    command = typer.main.get_command(app)
    for name in command_path:
        command = command.commands[name]  # type: ignore[attr-defined]
    names: set[str] = set()
    for param in command.params:
        # Click keeps the negative half of a `--x/--no-x` flag in secondary_opts,
        # so opts alone silently misses every off-switch.
        names.update(getattr(param, "opts", ()))
        names.update(getattr(param, "secondary_opts", ()))
    return names


def test_cluster_up_accepts_the_same_flag_as_provision_if_absent() -> None:
    # `provision-if-absent --cluster-name` and `cluster up --context` name the
    # same thing, so copying a command between help pages used to fail outright.
    assert "--cluster-name" in _option_names(["provision-if-absent"])
    assert "--cluster-name" in _option_names(["cluster", "up"])
    assert "--context" in _option_names(["cluster", "up"])


def test_gpu_driver_and_health_flags_align_across_direct_entrypoints() -> None:
    expected = {
        "--gpu-driver-mode",
        "--managed-driver-preset",
        "--allow-unsafe-nvswitch-operator",
        "--deny-unsafe-nvswitch-operator",
        "--gpu-health-stabilization-seconds",
        "--gpu-cuda-smoke",
        "--skip-gpu-cuda-smoke",
        "--gpu-cuda-smoke-image",
    }
    assert expected <= _option_names(["cluster", "up"])
    assert expected <= _option_names(["provision-if-absent"])


def test_cluster_up_rejects_an_unknown_flag_but_not_cluster_name(
    tmp_path: Path,
) -> None:
    unknown = runner.invoke(app, ["cluster", "up", "--not-a-flag", "x"])
    assert unknown.exit_code != 0
    assert "No such option" in unknown.output

    # --cluster-name parses; the command then fails on a missing terraform dir,
    # which is a different (and expected) failure.
    accepted = runner.invoke(
        app,
        [
            "cluster",
            "up",
            "--cluster-name",
            "demo",
            "--terraform-dir",
            str(tmp_path / "missing"),
        ],
    )
    assert "No such option" not in accepted.output


def test_configure_persists_the_staged_source_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", config_path)
    config_path.write_text(
        yaml.safe_dump({"default_project": "demo", "projects": {"demo": {}}}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["configure", "--src-s3-uri", "s3://bucket/prefix/npa"]
    )

    assert result.exit_code == 0, result.output
    stored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert stored["projects"]["demo"]["src_s3_uri"] == "s3://bucket/prefix/npa"


def test_configure_rejects_a_non_s3_source_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", tmp_path / "config.yaml")

    result = runner.invoke(app, ["configure", "--src-s3-uri", "/local/path"])

    assert result.exit_code == 1
    assert "must be an s3:// URI" in result.output


def test_render_resolves_the_staged_prefix_from_config_without_an_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import resolve_src_s3_uri

    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "demo",
                "projects": {"demo": {"src_s3_uri": "s3://bucket/prefix/npa"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", config_path)

    assert resolve_src_s3_uri() == "s3://bucket/prefix/npa"


def test_an_exported_prefix_still_wins_over_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import resolve_src_s3_uri

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "demo",
                "projects": {"demo": {"src_s3_uri": "s3://from-config/npa"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", config_path)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://from-env/npa")

    assert resolve_src_s3_uri() == "s3://from-env/npa"


def test_submit_source_resolver_honors_an_explicit_nondefault_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.workflow import _resolve_submit_src_s3_uri

    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "first",
                "projects": {
                    "first": {"src_s3_uri": "s3://first/source"},
                    "selected": {"src_s3_uri": "s3://selected/source"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", config_path)

    assert _resolve_submit_src_s3_uri("selected") == "s3://selected/source"


def test_an_unset_prefix_is_still_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.orchestration.npa_workflow.skypilot_render import resolve_src_s3_uri

    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    monkeypatch.setattr("npa.clients.config.CONFIG_PATH", tmp_path / "missing.yaml")

    assert resolve_src_s3_uri() == ""
