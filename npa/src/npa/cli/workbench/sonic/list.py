"""SONIC list command."""

from __future__ import annotations

from npa.cli.workbench.sonic.helpers import DEFAULT_MODEL_REPO, EXPECTED_HF_ARTIFACTS, OutputFormat, output, sonic_workbenches
import typer


def list_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "--output", help="Output format."
    ),
) -> None:
    """List configured SONIC workbenches and default model artifacts."""

    payload = {
        "models": [
            {
                "repo": DEFAULT_MODEL_REPO,
                "default_checkpoint": "sonic_release/last.pt",
                "embodiment": "unitree-g1",
                "artifacts": list(EXPECTED_HF_ARTIFACTS),
            }
        ],
        "projects": sonic_workbenches(),
    }
    output(payload, output_format)
