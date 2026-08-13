"""Operator licensing declaration and output provenance for LTX-2.5.

LTX-2.5 is *not* OSI-licensed open source, despite being described as such on
Lightricks' marketing pages. Everything Lightricks publishes for it — weights
*and* the ``ltx-core`` / ``ltx-pipelines`` / ``ltx-trainer`` code — is governed
by the LTX-2.x Community License Agreement (Section 1.9 folds
"inference-enabling code, training-enabling code ... accompanying source code"
into the licensed subject matter). That agreement carries two obligations no
container can satisfy on the operator's behalf:

* **Section 2.1** — an Entity with annual revenue of at least $10,000,000 (a
  "Commercial Entity", measured across all affiliates under common Control)
  needs a *paid* Commercial Use Agreement for any use except the Section 2.2
  Non-Commercial Purpose carve-out.
* **Attachment A(18)** — "For commercial use only: To train, improve, or
  fine-tune any other machine learning model, artificial intelligence system, or
  competing model". Section 2.2(c) says the same thing from the other side.

That second one is the constraint that matters most in a physical-AI workbench,
because the obvious reason to want a video world model here is to manufacture
training data for a robot policy — and a robot policy is "any other machine
learning model". For commercial use, that is exactly what the licence forbids.
Prose in a README would not stop it, so the disposition is computed here, stamped
onto every generated artifact, and re-checked by a fail-closed gate before any
downstream trainer is allowed to consume LTX output.

Nothing in this module accepts a licence for the operator or infers a missing
declaration. An absent or unrecognised declaration refuses; that refusal is the
mechanism, so it is tested rather than documented.

This is engineering enforcement of a recorded classification, not legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

# Pinned upstream identity. The licence is versioned by date and Lightricks has
# already reissued it once (the LTX-2 agreement of 2026-01-05 was superseded by
# the LTX-2.x agreement of 2026-08-11, released with LTX-2.5), so record which
# text a run was accepted against rather than just naming "the LTX licence".
LICENSE_NAME = "LTX-2.x Community License Agreement"
LICENSE_DATE = "2026-08-11"
LICENSE_URL = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md"
ACCEPTABLE_USE_POLICY_URL = (
    "https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf"
)
COMMERCIAL_LICENSE_CONTACT = "ltxv-licensing@lightricks.com"
COMMERCIAL_REVENUE_THRESHOLD_USD = 10_000_000

# Upstream source and weights. Neither is baked into the image; both are fetched
# at run time from Lightricks' own distribution channels under the operator's own
# acceptance. See npa/docker/workbench/ltx2/REDISTRIBUTION.md.
SOURCE_REPO = "https://github.com/Lightricks/LTX-2"
SOURCE_REF = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"
WEIGHTS_REPO = "Lightricks/LTX-2.5"
WEIGHTS_REPO_URL = f"https://huggingface.co/{WEIGHTS_REPO}"

# The operator's declaration. Three separate answers, because they are three
# separate questions and collapsing them is how a "yes" to the easy one gets
# silently reused for the hard one.
ACCEPT_ENV = "NPA_LTX_ACCEPT_COMMUNITY_LICENSE"
ENTITY_CLASS_ENV = "NPA_LTX_ENTITY_CLASS"
USE_CLASS_ENV = "NPA_LTX_USE_CLASS"
COMMERCIAL_AGREEMENT_ENV = "NPA_LTX_COMMERCIAL_AGREEMENT_REF"

ENTITY_COMMUNITY = "community"
ENTITY_COMMERCIAL = "commercial"
ENTITY_CLASSES = (ENTITY_COMMUNITY, ENTITY_COMMERCIAL)

USE_NON_COMMERCIAL = "non-commercial"
USE_COMMERCIAL = "commercial"
USE_CLASSES = (USE_NON_COMMERCIAL, USE_COMMERCIAL)

# Disposition of Attachment A(18) for the declared use.
TRAINING_PROHIBITED = "prohibited"
TRAINING_NON_COMMERCIAL_ONLY = "non-commercial-only"

PROVENANCE_SCHEMA = "npa.ltx2.provenance.v1"

# Restrictions that survive into the Output and therefore have to travel with the
# artifacts rather than living in a wiki. Section 6 and Attachment A(19) forbid
# stripping provenance/latent-disclosure markers; Attachment A(5) requires
# machine-generated content to be disclosed as such.
OUTPUT_OBLIGATIONS = (
    "attachment-a-5-disclose-machine-generated",
    "attachment-a-17-no-weapons-or-injury-causing-applications",
    "attachment-a-19-no-stripping-provenance-or-watermarks",
    "section-6-preserve-ai-regulation-disclosures",
)


class LtxLicenseError(RuntimeError):
    """Raised when the operator's LTX licensing declaration is absent or invalid."""


@dataclass(frozen=True)
class LicenseDeclaration:
    """A validated operator declaration under the LTX-2.x Community License."""

    entity_class: str
    use_class: str
    commercial_agreement_ref: str = ""

    @property
    def derived_model_training(self) -> str:
        """Return the Attachment A(18) disposition for training other models.

        Attachment A(18) is scoped "for commercial use only", so the *use*
        decides, not the entity size: a sub-$10M company using Outputs
        commercially is just as barred from training a robot policy on them as a
        Commercial Entity is.
        """

        if self.use_class == USE_COMMERCIAL:
            return TRAINING_PROHIBITED
        return TRAINING_NON_COMMERCIAL_ONLY

    @property
    def requires_paid_license(self) -> bool:
        """Whether Section 2.1 requires a Commercial Use Agreement for this run."""

        return self.entity_class == ENTITY_COMMERCIAL and self.use_class == USE_COMMERCIAL

    @property
    def relies_on_non_commercial_carve_out(self) -> bool:
        """Whether this run depends on the Section 2.2 carve-out to proceed."""

        return (
            self.entity_class == ENTITY_COMMERCIAL
            and self.use_class == USE_NON_COMMERCIAL
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_class": self.entity_class,
            "use_class": self.use_class,
            "commercial_agreement_ref": self.commercial_agreement_ref,
            "requires_paid_license": self.requires_paid_license,
            "relies_on_section_2_2_non_commercial_carve_out": (
                self.relies_on_non_commercial_carve_out
            ),
        }


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _accepted(value: str | None) -> bool:
    return _normalize(value) == "yes"


def refusal_text(reason: str) -> str:
    """Return the operator-facing refusal explaining what to declare and why."""

    return f"""\
npa-ltx2: refusing to fetch or run LTX-2.5. {reason}

This image contains no LTX-2.5 code and no LTX-2.5 weights. The requested
operation would ask Lightricks' own channels to deliver them to you:
  source:  {SOURCE_REPO} @ {SOURCE_REF}
  weights: {WEIGHTS_REPO_URL} (gated; needs your own HF token)

LTX-2.5 is not OSI open source. It is licensed under the
{LICENSE_NAME} ({LICENSE_DATE}):
  {LICENSE_URL}
  {ACCEPTABLE_USE_POLICY_URL}

Nebius cannot accept it for you, and cannot tell whether your organisation
crosses the Section 2.1 revenue threshold. Declare all three:

  {ACCEPT_ENV}=YES
      You have read and accept the Agreement and its Attachment A.

  {ENTITY_CLASS_ENV}={ENTITY_COMMUNITY}|{ENTITY_COMMERCIAL}
      '{ENTITY_COMMUNITY}'  annual revenue below ${COMMERCIAL_REVENUE_THRESHOLD_USD:,},
                  counting all affiliates under common Control (Section 1.6).
      '{ENTITY_COMMERCIAL}' at or above that threshold: a Commercial Entity.

  {USE_CLASS_ENV}={USE_NON_COMMERCIAL}|{USE_COMMERCIAL}
      '{USE_NON_COMMERCIAL}' testing, evaluation, or non-commercial R&D in a
                  non-production or development environment (Section 2.2).
      '{USE_COMMERCIAL}'     anything revenue-generating, user-facing, or
                  production (Section 2.2 says this is not the carve-out).

A Commercial Entity declaring commercial use must also set
{COMMERCIAL_AGREEMENT_ENV} to its Commercial Use Agreement
reference; without a paid licence that combination is prohibited outright
(Section 2.1). Contact {COMMERCIAL_LICENSE_CONTACT}.

Note before you pick '{USE_COMMERCIAL}': Attachment A(18) forbids using LTX-2.5
Outputs to train, improve, or fine-tune any other machine learning model for
commercial use. Robot policies are other machine learning models, so a
commercial declaration makes this run's video unusable as policy training data,
and the workbench will refuse to feed it to a trainer.

Nothing has been downloaded."""


def declaration_from_env(env: Mapping[str, str]) -> LicenseDeclaration:
    """Validate the operator's declaration, or refuse.

    Every failure path refuses before anything is fetched. There is deliberately
    no default for any of the three answers: a default would be Nebius answering
    a licensing question on the operator's behalf, which is the one thing this
    gate exists to prevent.
    """

    if not _accepted(env.get(ACCEPT_ENV)):
        raise LtxLicenseError(
            refusal_text(f"{ACCEPT_ENV} is not set to YES.")
        )

    entity_class = _normalize(env.get(ENTITY_CLASS_ENV))
    if entity_class not in ENTITY_CLASSES:
        raise LtxLicenseError(
            refusal_text(
                f"{ENTITY_CLASS_ENV} must be one of {', '.join(ENTITY_CLASSES)} "
                f"(got {env.get(ENTITY_CLASS_ENV)!r})."
            )
        )

    use_class = _normalize(env.get(USE_CLASS_ENV))
    if use_class not in USE_CLASSES:
        raise LtxLicenseError(
            refusal_text(
                f"{USE_CLASS_ENV} must be one of {', '.join(USE_CLASSES)} "
                f"(got {env.get(USE_CLASS_ENV)!r})."
            )
        )

    agreement_ref = (env.get(COMMERCIAL_AGREEMENT_ENV) or "").strip()
    declaration = LicenseDeclaration(
        entity_class=entity_class,
        use_class=use_class,
        commercial_agreement_ref=agreement_ref,
    )
    if declaration.requires_paid_license and not agreement_ref:
        raise LtxLicenseError(
            refusal_text(
                "A Commercial Entity declaring commercial use needs a paid "
                f"Commercial Use Agreement; {COMMERCIAL_AGREEMENT_ENV} is empty. "
                f"Contact {COMMERCIAL_LICENSE_CONTACT} (Section 2.1)."
            )
        )
    return declaration


@dataclass(frozen=True)
class ProvenanceRecord:
    """The licence terms that travel with a set of generated artifacts."""

    declaration: LicenseDeclaration
    run_id: str
    outputs: tuple[str, ...] = ()
    model_files: tuple[str, ...] = ()
    source_ref: str = SOURCE_REF
    weights_repo: str = WEIGHTS_REPO
    generated_at: str = field(default_factory=lambda: _utc_now())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVENANCE_SCHEMA,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "source": {"repo": SOURCE_REPO, "ref": self.source_ref},
            "weights": {
                "repo": self.weights_repo,
                "url": f"https://huggingface.co/{self.weights_repo}",
                "files": list(self.model_files),
                "delivery": "runtime-fetch-under-operator-hf-token",
            },
            "license": {
                "name": LICENSE_NAME,
                "date": LICENSE_DATE,
                "url": LICENSE_URL,
                "acceptable_use_policy": ACCEPTABLE_USE_POLICY_URL,
                "osi_approved": False,
            },
            "operator_declaration": self.declaration.as_dict(),
            "restrictions": {
                "derived_model_training": self.declaration.derived_model_training,
                "synthetic_content_disclosure": "required",
                "output_obligations": list(OUTPUT_OBLIGATIONS),
            },
            "outputs": [
                {"uri": uri, "machine_generated": True} for uri in self.outputs
            ],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class GateDecision:
    """Result of checking whether a consumer may use LTX-2.5 output."""

    allowed: bool
    reason: str
    disposition: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "derived_model_training": self.disposition,
        }


def check_training_consumer(manifest: Any, *, consumer: str) -> GateDecision:
    """Decide whether *consumer* may train on artifacts described by *manifest*.

    Fail-closed in every direction that is not an explicit permission: a missing
    manifest, a manifest of an unrecognised schema, and an unknown disposition
    all deny. The permissive branch is the narrow one, which is the right way
    round for a restriction whose breach is a licence termination event
    (Section 13).
    """

    if not isinstance(manifest, Mapping):
        return GateDecision(
            allowed=False,
            reason=(
                "No LTX-2.5 provenance manifest accompanies these artifacts. "
                "Attachment A(18) restricts training other models on LTX Outputs, "
                "so unlabelled artifacts are refused rather than assumed free."
            ),
        )

    schema = manifest.get("schema")
    if schema != PROVENANCE_SCHEMA:
        return GateDecision(
            allowed=False,
            reason=(
                f"Unrecognised provenance schema {schema!r}; expected "
                f"{PROVENANCE_SCHEMA}. Refusing rather than guessing the terms."
            ),
        )

    restrictions = manifest.get("restrictions")
    disposition = ""
    if isinstance(restrictions, Mapping):
        disposition = str(restrictions.get("derived_model_training") or "")

    if disposition == TRAINING_NON_COMMERCIAL_ONLY:
        return GateDecision(
            allowed=True,
            reason=(
                f"{consumer} may consume these artifacts: the run was declared "
                "non-commercial, so Attachment A(18) does not bar training on the "
                "Outputs. Any resulting model is a Derivative of LTX-2.x "
                "(Section 1.5) and stays bound by the Agreement, including the "
                "Section 3.5 transfer conditions."
            ),
            disposition=disposition,
        )

    if disposition == TRAINING_PROHIBITED:
        return GateDecision(
            allowed=False,
            reason=(
                f"{consumer} may not consume these artifacts. The run was declared "
                "commercial use, and Attachment A(18) forbids using LTX-2.5 Outputs "
                "to train, improve, or fine-tune any other machine learning model "
                "for commercial use. Re-run under a non-commercial declaration, or "
                f"obtain a Commercial Use Agreement ({COMMERCIAL_LICENSE_CONTACT}) "
                "that covers it."
            ),
            disposition=disposition,
        )

    return GateDecision(
        allowed=False,
        reason=(
            f"Provenance manifest carries no recognised derived_model_training "
            f"disposition (got {disposition!r}). Refusing fail-closed."
        ),
        disposition=disposition,
    )


__all__ = [
    "ACCEPTABLE_USE_POLICY_URL",
    "ACCEPT_ENV",
    "COMMERCIAL_AGREEMENT_ENV",
    "COMMERCIAL_LICENSE_CONTACT",
    "COMMERCIAL_REVENUE_THRESHOLD_USD",
    "ENTITY_CLASSES",
    "ENTITY_CLASS_ENV",
    "ENTITY_COMMERCIAL",
    "ENTITY_COMMUNITY",
    "GateDecision",
    "LICENSE_DATE",
    "LICENSE_NAME",
    "LICENSE_URL",
    "LicenseDeclaration",
    "LtxLicenseError",
    "OUTPUT_OBLIGATIONS",
    "PROVENANCE_SCHEMA",
    "ProvenanceRecord",
    "SOURCE_REF",
    "SOURCE_REPO",
    "TRAINING_NON_COMMERCIAL_ONLY",
    "TRAINING_PROHIBITED",
    "USE_CLASSES",
    "USE_CLASS_ENV",
    "USE_COMMERCIAL",
    "USE_NON_COMMERCIAL",
    "WEIGHTS_REPO",
    "WEIGHTS_REPO_URL",
    "check_training_consumer",
    "declaration_from_env",
    "refusal_text",
]
