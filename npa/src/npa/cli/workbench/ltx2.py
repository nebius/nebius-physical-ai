"""Typer CLI for ``npa workbench ltx2``.

LTX-2.5 generation itself runs inside the ``npa-ltx2`` container on a GPU node,
launched through ``npa workbench byof run`` like every other BYOF solution. What
lives here is the part that has to be callable from the host without a GPU:
``terms`` prints which licence governs LTX-2.5, where to read it, and what the
workbench needs from the operator before it can fetch anything (a Hugging Face
token with access to the gated weights repository).

There is nothing here to declare or to accept. The LTX-2.x agreement forms by
conduct — "By downloading, using, accessing or distributing any portion or
element of LTX-2.x, you agree ... to be bound by this Agreement" — and access to
the gated repository is granted by Lightricks on Hugging Face, not by us.
Compliance with the agreement is the operator's own responsibility.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import typer

app = typer.Typer(
    name="ltx2",
    help=(
        "LTX-2.5 licence surface: print the LTX-2.x Community License terms, the "
        "pinned upstream source, and the gated weights repository the operator's "
        "own Hugging Face entitlement unlocks."
    ),
    no_args_is_help=True,
)


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
    """Print the LTX-2.x licence terms and what running LTX-2.5 here requires."""

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
            "weights": {
                "repo": licensing.WEIGHTS_REPO,
                "url": licensing.WEIGHTS_REPO_URL,
                "gated": True,
            },
            "baked_into_image": False,
            "requires": "HF_TOKEN",
        },
    }
    _emit(payload, output=output, text=terms_text())


def terms_text() -> str:
    """Return the operator-facing summary of the terms and what they must hold."""

    from npa.workbench.ltx2 import licensing

    return f"""\
npa-ltx2 ships no LTX-2.5 code and no LTX-2.5 weights. Running it asks
Lightricks' own channels to deliver them to you:
  source:  {licensing.SOURCE_REPO} @ {licensing.SOURCE_REF}
  weights: {licensing.WEIGHTS_REPO_URL} (gated)

LTX-2.5 is not OSI open source. It is licensed under the
{licensing.LICENSE_NAME} ({licensing.LICENSE_DATE}):
  {licensing.LICENSE_URL}
  {licensing.ACCEPTABLE_USE_POLICY_URL}

The Agreement binds by use: "By downloading, using, accessing or distributing
any portion or element of LTX-2.x, you agree that you have read and accepted to
be bound by this Agreement." Accept it with Lightricks, on the gated repository
page, with your own Hugging Face account. Both fetches then run under your own
HF_TOKEN, which is the only thing this workbench requires of you.

Two obligations are yours alone, and nothing here checks them for you:
  Section 2.1      an Entity whose annual revenue is at or above
                   ${licensing.COMMERCIAL_REVENUE_THRESHOLD_USD:,}, counting all affiliates under
                   common Control, needs a paid Commercial Use Agreement for any
                   use outside the Section 2.2 non-commercial carve-out.
                   Contact {licensing.COMMERCIAL_LICENSE_CONTACT}.
  Attachment A(18) for commercial use, the Outputs may not be used to train,
                   improve, or fine-tune any other machine learning model. A
                   robot policy is another machine learning model."""
