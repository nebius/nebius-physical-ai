"""NVIDIA Cosmos Evaluator workbench tool.

Wraps the open-source NVIDIA Cosmos Evaluator checks
(https://github.com/nvidia-cosmos/cosmos-evaluator, Apache-2.0) so the Physical
AI Data Factory blueprint grades its Cosmos Transfer output with the evaluator
NVIDIA ships for exactly that job instead of a generic VLM scorer.

Two checks are wired, both real:

- **Hallucination check** (:mod:`.hallucination`) — compares per-frame dynamic
  masks of the original clip and the augmented clip and scores the augmented
  clip on hallucinated motion. CPU only. Runs upstream's own
  ``HallucinationProcessor`` when an upstream checkout is importable, and
  otherwise runs the in-repo port of the same published algorithm.
- **Attribute verification check** (:mod:`.attribute_verification`) — generates
  one multiple-choice question per augmented attribute with an LLM, then asks a
  VLM to answer it from a frame of the augmented clip. Upstream drives this
  through any OpenAI-compatible endpoint, so NPA points it at Nebius Token
  Factory (zero-GPU, hosted).

:func:`evaluate_run` is the stage entry point: it walks a Physical AI Data
Factory run prefix, pairs every augmented variant with the run's source clip and
its sampled appearance variables, runs both checks per variant, and writes one
``npa.cosmos_evaluator.report.v1`` report whose ``score`` the blueprint's quality
gate reads.
"""

from __future__ import annotations

from npa.workbench.cosmos_evaluator.attribute_verification import (
    AttributeVerificationCheck,
    AttributeVerificationResult,
    verify_attributes,
)
from npa.workbench.cosmos_evaluator.evaluate import (
    RESULT_FILENAME,
    ClipEvaluation,
    EvaluateRunResult,
    evaluate_run,
    report_uri_for,
)
from npa.workbench.cosmos_evaluator.hallucination import (
    HallucinationResult,
    check_hallucination,
)
from npa.workbench.cosmos_evaluator.upstream import (
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosEvaluatorError,
    CosmosEvaluatorStorageError,
    upstream_source_dir,
)

# Engine tags recorded in every report so a reader can tell which code produced
# a score: upstream's own module, or the in-repo port of upstream's algorithm.
ENGINE_UPSTREAM = "cosmos-evaluator-upstream"
ENGINE_PORT = "cosmos-evaluator-npa-port"
ENGINE_UNAVAILABLE = "unavailable"

__all__ = [
    "ENGINE_PORT",
    "ENGINE_UNAVAILABLE",
    "ENGINE_UPSTREAM",
    "RESULT_FILENAME",
    "UPSTREAM_LICENSE",
    "UPSTREAM_REPO",
    "AttributeVerificationCheck",
    "AttributeVerificationResult",
    "ClipEvaluation",
    "CosmosEvaluatorError",
    "CosmosEvaluatorStorageError",
    "EvaluateRunResult",
    "HallucinationResult",
    "check_hallucination",
    "evaluate_run",
    "report_uri_for",
    "upstream_source_dir",
    "verify_attributes",
]
