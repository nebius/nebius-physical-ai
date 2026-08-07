"""Unit tests for `npa.cli._typer_defaults`.

Typer commands that are also called as plain Python functions used to leak
``typer.models.OptionInfo`` sentinels into downstream code (Terraform vars,
shell commands, boolean flags). These tests pin the decorator that makes the
direct-call path behave like the CLI path.
"""

from __future__ import annotations

import inspect

import pytest
import typer
from typer.testing import CliRunner

from npa.cli._typer_defaults import (
    DECORATOR_ATTR,
    is_unresolved_option,
    resolve_option_default,
    resolve_typer_defaults,
)


def test_is_unresolved_option_detects_typer_sentinels() -> None:
    assert is_unresolved_option(typer.Option("x", "--x"))
    assert is_unresolved_option(typer.Argument("y"))
    assert not is_unresolved_option("x")
    assert not is_unresolved_option(0)
    assert not is_unresolved_option(None)


def test_resolve_option_default_returns_declared_default() -> None:
    assert resolve_option_default(typer.Option("ubuntu", "--ssh-user")) == "ubuntu"
    assert resolve_option_default(typer.Option(8088, "--port")) == 8088
    assert resolve_option_default(typer.Option(False, "--flag")) is False
    # Plain values pass through untouched.
    assert resolve_option_default("literal") == "literal"


def test_resolve_option_default_raises_for_required_option() -> None:
    with pytest.raises(TypeError, match="missing required argument 'project_id'"):
        resolve_option_default(typer.Option(..., "--project-id"), name="project_id")


def test_direct_call_fills_omitted_option_defaults() -> None:
    """Omitting an option in a direct call yields the declared default."""

    captured: dict = {}

    @resolve_typer_defaults
    def cmd(
        name: str = typer.Option("agent", "--name"),
        port: int = typer.Option(8088, "--port"),
        flag: bool = typer.Option(False, "--flag"),
        items: list[str] = typer.Option([], "--item"),
    ) -> None:
        captured.update(name=name, port=port, flag=flag, items=items)

    cmd()

    assert captured == {"name": "agent", "port": 8088, "flag": False, "items": []}
    assert not any(type(v).__name__ == "OptionInfo" for v in captured.values())


def test_direct_call_keeps_explicit_arguments() -> None:
    captured: dict = {}

    @resolve_typer_defaults
    def cmd(
        name: str = typer.Option("agent", "--name"),
        port: int = typer.Option(8088, "--port"),
    ) -> None:
        captured.update(name=name, port=port)

    cmd(name="custom", port=1234)
    assert captured == {"name": "custom", "port": 1234}

    captured.clear()
    cmd("positional")
    assert captured == {"name": "positional", "port": 8088}


def test_mutable_defaults_are_copied_per_call() -> None:
    """A list default must not be shared between calls."""

    seen: list[list[str]] = []

    @resolve_typer_defaults
    def cmd(items: list[str] = typer.Option([], "--item")) -> None:
        items.append("mutated")
        seen.append(items)

    cmd()
    cmd()

    assert seen[0] == ["mutated"]
    assert seen[1] == ["mutated"]
    assert seen[0] is not seen[1]


def test_direct_call_raises_for_missing_required_option() -> None:
    @resolve_typer_defaults
    def cmd(
        project_id: str = typer.Option(..., "--project-id"),
        name: str = typer.Option("agent", "--name"),
    ) -> None:  # pragma: no cover - never reached
        raise AssertionError("should not run")

    with pytest.raises(TypeError, match="missing required argument"):
        cmd()

    # Supplying it is fine.
    @resolve_typer_defaults
    def ok(
        project_id: str = typer.Option(..., "--project-id"),
        name: str = typer.Option("agent", "--name"),
    ) -> str:
        return f"{project_id}/{name}"

    assert ok(project_id="pid") == "pid/agent"


def test_signature_is_preserved_for_typer_introspection() -> None:
    def cmd(name: str = typer.Option("agent", "--name")) -> None:  # pragma: no cover
        pass

    wrapped = resolve_typer_defaults(cmd)
    assert inspect.signature(wrapped) == inspect.signature(cmd)
    assert wrapped.__name__ == cmd.__name__
    assert getattr(wrapped, DECORATOR_ATTR) is True


def test_cli_path_is_unaffected() -> None:
    """Registering a wrapped function keeps the full Typer CLI behavior."""

    app = typer.Typer()
    captured: dict = {}

    @app.command("run")
    @resolve_typer_defaults
    def run_cmd(
        name: str = typer.Option("agent", "--name", help="Deployment name."),
        port: int = typer.Option(8088, "--port"),
        flag: bool = typer.Option(False, "--flag"),
    ) -> None:
        """Run it."""
        captured.update(name=name, port=port, flag=flag)

    @app.command("other")
    def other_cmd() -> None:
        """Second command so Typer builds a group, not a single root command."""

    runner = CliRunner()

    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert captured == {"name": "agent", "port": 8088, "flag": False}

    captured.clear()
    result = runner.invoke(app, ["run", "--name", "x", "--port", "9", "--flag"])
    assert result.exit_code == 0, result.output
    assert captured == {"name": "x", "port": 9, "flag": True}

    # Help text (docstring + option help) survives the wrapper.
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Deployment name." in result.output
    assert "Run it." in result.output


def test_positional_only_parameters_are_rejected() -> None:
    def cmd(value, /, name: str = typer.Option("a", "--name")) -> None:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="positional-only"):
        resolve_typer_defaults(cmd)
