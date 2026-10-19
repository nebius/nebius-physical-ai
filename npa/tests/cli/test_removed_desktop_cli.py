"""Regression coverage for the extracted coding-desktop CLI surface."""

from typer.testing import CliRunner

from npa.cli.main import app


def test_extracted_desktop_namespace_is_not_registered() -> None:
    """NPA must not silently retain an unavailable coding-desktop entry point."""

    result = CliRunner().invoke(app, ["tools", "desktop", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output
