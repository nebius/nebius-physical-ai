"""Guards for the documented first-time-user onboarding path.

These tests defend the copy-pasteable quickstart so docs that "look right"
cannot silently rot: the setup guidance must stay placeholder-only (public
hygiene), and the advertised first real success must keep working offline.
"""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import DEFAULT_MODEL, DEFAULT_SAMPLE_BENCHMARK_PATH


runner = CliRunner()

# Matches any dotted-quad IPv4 literal, e.g. 203.0.113.10.
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_setup_guidance_points_to_single_configure_entrypoint() -> None:
    """Onboarding guidance should not require separate nebius CLI steps first."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        lowered = result.output.lower()
        assert "npa configure --interactive" in result.output
        assert "nebius profile create" not in lowered
        assert "get-access-token" not in lowered


def test_setup_guidance_contains_no_raw_ip_address() -> None:
    """Setup guidance must use placeholders, never a literal host/IP."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        match = _IPV4.search(result.output)
        assert match is None, (
            f"`npa {command}` guidance leaks a literal IP {match.group(0)!r}; "
            "use a placeholder such as <your-byovm-host> instead."
        )
        assert "<your-byovm-host>" in result.output


def test_quickstart_first_success_fixture_is_packaged() -> None:
    """The fixture the quickstart points at must ship inside the package."""
    assert DEFAULT_SAMPLE_BENCHMARK_PATH.exists(), (
        "Quickstart first-success benchmark fixture is missing: "
        f"{DEFAULT_SAMPLE_BENCHMARK_PATH}"
    )


def test_quickstart_benchmark_command_produces_real_result(tmp_path) -> None:
    """Run the exact documented first-success command end to end, offline."""
    output_path = tmp_path / "vlm-eval-benchmark.json"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "benchmark",
            "--dataset",
            str(DEFAULT_SAMPLE_BENCHMARK_PATH),
            "--output",
            str(output_path),
            "--backend",
            "stub",
            "--thresholds",
            "0.5,0.8,0.9",
            "--rubrics",
            "default,strict",
            "--models",
            DEFAULT_MODEL,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # A real scoring pass over the shipped labeled rollout set, no GPU or creds.
    assert payload["best_config"]["metrics"]["accuracy"] == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["item_count"] == 4
