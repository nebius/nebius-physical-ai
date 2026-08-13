"""Typer CLI for ``npa workbench ltx2``.

LTX-2.5 generation itself runs inside the ``npa-ltx2`` container on a GPU node,
launched through ``npa workbench byof run`` like every other BYOF solution. What
lives here is the part that has to be callable from the host and from a
workflow state without a GPU: the licensing surface.

``terms`` prints what the operator is being asked to accept. ``declare``
validates the answers they gave. ``stamp`` records those answers onto a run's
artifacts. ``gate`` reads that record back and refuses to let a trainer consume
LTX Outputs when Attachment A(18) forbids it.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import typer

app = typer.Typer(
    name="ltx2",
    help=(
        "LTX-2.5 licensing surface: declare the LTX-2.x Community License terms, "
        "stamp them onto generated video, and gate downstream training on them."
    ),
    no_args_is_help=True,
)

# Exit codes. 78 is EX_CONFIG, the same code the container's runtime gate uses
# for a missing or invalid declaration, so an operator sees one number for "you
# have not told us your licensing position" wherever it surfaces. A denial under
# Attachment A(18) is a different thing — the declaration was fine, the answer
# was no — and gets its own code.
EXIT_UNDECLARED = 78
EXIT_DENIED = 3


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _emit(payload: dict[str, Any], *, output: OutputFormat, text: str) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@app.command("terms")
def terms_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Print the LTX-2.x licence terms and the declaration this workbench requires."""

    from npa.workbench.ltx2 import licensing

    payload = {
        "license": {
            "name": licensing.LICENSE_NAME,
            "date": licensing.LICENSE_DATE,
            "url": licensing.LICENSE_URL,
            "acceptable_use_policy": licensing.ACCEPTABLE_USE_POLICY_URL,
            "osi_approved": False,
            "commercial_contact": licensing.COMMERCIAL_LICENSE_CONTACT,
            "commercial_revenue_threshold_usd": (
                licensing.COMMERCIAL_REVENUE_THRESHOLD_USD
            ),
        },
        "runtime_fetch": {
            "source": {"repo": licensing.SOURCE_REPO, "ref": licensing.SOURCE_REF},
            "weights": {"repo": licensing.WEIGHTS_REPO, "gated": True},
            "baked_into_image": False,
        },
        "declaration_env": {
            "accept": licensing.ACCEPT_ENV,
            "entity_class": licensing.ENTITY_CLASS_ENV,
            "use_class": licensing.USE_CLASS_ENV,
            "commercial_agreement_ref": licensing.COMMERCIAL_AGREEMENT_ENV,
        },
        "entity_classes": list(licensing.ENTITY_CLASSES),
        "use_classes": list(licensing.USE_CLASSES),
        "output_obligations": list(licensing.OUTPUT_OBLIGATIONS),
    }
    _emit(
        payload,
        output=output,
        text=licensing.refusal_text("Nothing has been requested yet."),
    )


@app.command("declare")
def declare_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Validate the operator's licensing declaration from the environment."""

    from npa.workbench.ltx2.licensing import LtxLicenseError, declaration_from_env

    try:
        declaration = declaration_from_env(os.environ)
    except LtxLicenseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNDECLARED)

    payload = declaration.as_dict()
    payload["derived_model_training"] = declaration.derived_model_training
    _emit(
        payload,
        output=output,
        text=(
            f"entity={declaration.entity_class} use={declaration.use_class} "
            f"derived_model_training={declaration.derived_model_training}"
        ),
    )


@app.command("stamp")
def stamp_cmd(
    run_id: str = typer.Option(
        ..., "--run-id", help="Run id recorded in the manifest."
    ),
    manifest_uri: str = typer.Option(
        ...,
        "--manifest-uri",
        help="S3 prefix or path to write the provenance manifest to.",
    ),
    output_uri: list[str] = typer.Option(
        [],
        "--output-uri",
        help="A generated artifact covered by this manifest; repeatable.",
    ),
    model_file: list[str] = typer.Option(
        [], "--model-file", help="Weights file the run fetched; repeatable."
    ),
    declaration_uri: str = typer.Option(
        "",
        "--declaration-uri",
        help=(
            "The generation's own declaration (`ltx-runtime provenance`). When "
            "given, this state's declaration must match it or the stamp refuses."
        ),
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Stamp the accepted licence terms onto the artifacts a run generated."""

    from npa.workbench.ltx2.gate import stamp_run
    from npa.workbench.ltx2.licensing import LtxLicenseError

    try:
        result = stamp_run(
            run_id=run_id,
            outputs=list(output_uri),
            model_files=list(model_file),
            manifest_uri=manifest_uri,
            env=os.environ,
            declaration_uri=declaration_uri,
        )
    except LtxLicenseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_UNDECLARED)

    payload = result.as_dict()
    disposition = payload["manifest"]["restrictions"]["derived_model_training"]
    _emit(
        payload,
        output=output,
        text=(f"stamped {result.manifest_uri} derived_model_training={disposition}"),
    )


@app.command("gate")
def gate_cmd(
    manifest_uri: str = typer.Option(
        ..., "--manifest-uri", help="Provenance manifest written by `ltx2 stamp`."
    ),
    consumer: str = typer.Option(
        ...,
        "--consumer",
        help="What wants the artifacts, e.g. 'lerobot policy training'.",
    ),
    artifact_uri: list[str] = typer.Option(
        [],
        "--artifact-uri",
        help=(
            "An artifact the consumer intends to use; repeatable. The manifest "
            "must claim it, so another run's manifest cannot clear these bytes."
        ),
    ),
    report_uri: str = typer.Option(
        "", "--report-uri", help="Optional S3 prefix or path for the gate report."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.json, "--output", help="Output format."
    ),
) -> None:
    """Refuse or allow a downstream trainer to consume LTX-2.5 output.

    Exits non-zero when the answer is no, so a workflow state fails rather than
    proceeding past a licence restriction it just printed.
    """

    from npa.workbench.ltx2.gate import gate_run

    result = gate_run(
        manifest_uri=manifest_uri,
        consumer=consumer,
        report_uri=report_uri,
        artifacts=list(artifact_uri),
    )
    payload = result.as_dict()
    text = f"allowed={result.decision.allowed} {result.decision.reason}"
    if result.decision.allowed:
        _emit(payload, output=output, text=text)
        return

    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True), err=True)
    else:
        typer.echo(text, err=True)
    raise typer.Exit(EXIT_DENIED)
