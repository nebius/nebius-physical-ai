"""The LTX-2.x licence facts are stated once, and every surface repeats them.

Nothing here is a gate. Acceptance of the LTX-2.x Community License Agreement
happens by conduct and with Lightricks, on the gated Hugging Face repository
page; the workbench only has to name the right agreement, the right URLs, and
the two obligations that stay the operator's own.

That makes drift the only real failure mode: the shipped container carries its
own copy of these strings (it can no longer import this module — the image bakes
no Python of ours beyond the video check), and the CLI prints another. A wrong
licence URL in one of them is a factual error about someone's legal position, so
the three are held to the same constants here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.ltx2 import licensing

RUNTIME_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "workbench"
    / "ltx2"
    / "ltx_runtime.sh"
)

#: The facts every surface must agree on. Each is something an operator would
#: act on: which agreement binds them, where to read it, which bytes arrive, and
#: who to contact when Section 2.1 applies to them.
SHARED_FACTS = (
    licensing.LICENSE_NAME,
    licensing.LICENSE_DATE,
    licensing.LICENSE_URL,
    licensing.ACCEPTABLE_USE_POLICY_URL,
    licensing.WEIGHTS_REPO_URL,
    licensing.SOURCE_REF,
    licensing.COMMERCIAL_LICENSE_CONTACT,
)


class TestTheRecordedFacts:
    def test_it_names_the_ltx_2_x_agreement_not_ltx_2(self) -> None:
        """Lightricks reissued the agreement; the older one governs different bytes."""

        assert licensing.LICENSE_NAME == "LTX-2.x Community License Agreement"
        assert licensing.LICENSE_DATE == "2026-08-11"

    def test_the_weights_repo_url_is_derived_from_the_repo_id(self) -> None:
        assert licensing.WEIGHTS_REPO == "Lightricks/LTX-2.5"
        assert licensing.WEIGHTS_REPO_URL.endswith(licensing.WEIGHTS_REPO)

    def test_the_source_ref_is_an_immutable_commit(self) -> None:
        """A branch name would let the recorded provenance change underneath us."""

        assert len(licensing.SOURCE_REF) == 40
        assert set(licensing.SOURCE_REF) <= set("0123456789abcdef")

    def test_nothing_survives_that_would_ask_an_operator_to_self_certify(self) -> None:
        """The declaration machinery is gone, and must not creep back.

        A local `ACCEPT=YES` never formed the agreement — it binds by conduct —
        and a self-typed revenue class was unverifiable. The gated repository
        entitlement is the evidence instead, so this module holds facts only.
        """

        assert not [name for name in dir(licensing) if name.endswith("_ENV")]
        for gone in (
            "declaration_from_env",
            "check_training_consumer",
            "LicenseDeclaration",
            "ProvenanceRecord",
        ):
            assert not hasattr(licensing, gone), gone


class TestEverySurfaceRepeatsTheSameFacts:
    def test_the_cli_terms_text_quotes_them(self) -> None:
        result = CliRunner().invoke(app, ["workbench", "ltx2", "terms"])

        assert result.exit_code == 0, result.output
        for fact in SHARED_FACTS:
            assert fact in result.output, fact
        # The one thing the workbench actually needs from the operator.
        assert "HF_TOKEN" in result.output

    def test_the_cli_terms_json_is_machine_readable(self) -> None:
        result = CliRunner().invoke(
            app, ["workbench", "ltx2", "terms", "--output", "json"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["license"]["name"] == licensing.LICENSE_NAME
        assert payload["license"]["osi_approved"] is False
        assert payload["runtime_fetch"]["baked_into_image"] is False
        assert payload["runtime_fetch"]["weights"]["gated"] is True
        assert payload["runtime_fetch"]["requires"] == "HF_TOKEN"

    @pytest.mark.skipif(
        shutil.which("bash") is None, reason="the shipped script requires bash"
    )
    def test_the_containers_own_terms_text_quotes_them_too(self) -> None:
        """The image cannot import this module, so its copy is checked against it.

        Run rather than read: the script interpolates its own facts, and a
        substring search over the source would pass on a variable that renders
        to the wrong URL.
        """

        result = subprocess.run(
            ["bash", str(RUNTIME_SCRIPT), "terms"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, result.stderr
        for fact in SHARED_FACTS:
            assert fact in result.stdout, fact
        assert "HF_TOKEN" in result.stdout

    def test_the_cli_states_the_two_obligations_that_stay_the_operators(self) -> None:
        """Naming them is all we can do; neither is checkable from here."""

        result = CliRunner().invoke(app, ["workbench", "ltx2", "terms"])

        assert "Section 2.1" in result.output
        assert f"${licensing.COMMERCIAL_REVENUE_THRESHOLD_USD:,}" in result.output
        assert "Attachment A(18)" in result.output
        assert "another machine learning model" in result.output
