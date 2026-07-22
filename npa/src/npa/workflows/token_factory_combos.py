"""Infra-free helpers for the Token Factory + Nebius-compute combo workflows.

Four combo workflows compose *real Nebius GPU compute* with *hosted Token
Factory inference* (zero-GPU on the client side):

1. **train-triage (serverless).** A LeRobot **serverless GPU Job** trains/evals a
   policy and writes its run artifacts (configs, logs, metrics) to S3. A Token
   Factory text model then reads those real artifacts and writes a human-readable
   triage + next-steps report. Runner: ``npa/scripts/run_tokenfactory_train_triage.py``.
2. **rollout-judge (kubernetes).** A LeRobot eval rollout renders videos on a
   Nebius **Managed Kubernetes GPU**; ``vlm-eval --backend api`` then scores the
   rollout with a hosted VLM, with no local VLM serving stage. Workflow:
   ``npa/src/npa/workflows/skypilot/tokenfactory-rollout-judge.yaml``.
3. **sim-sweep (serverless fan-out).** A hosted text model designs an experiment
   sweep, a deterministic grid launches one LeRobot **serverless GPU Job** per
   variant, and a hosted text model ranks the completed runs from their real
   artifacts. Runner: ``npa/scripts/run_tokenfactory_sim_sweep.py``.
4. **scene-to-rollout-judge (kubernetes).** A hosted reasoner extracts a plan
   from scene images, a Nebius **Managed Kubernetes GPU** rolls out a policy, and
   a hosted VLM judges the rollout against that plan. Workflow:
   ``npa/src/npa/workflows/skypilot/tokenfactory-scene-to-rollout-judge.yaml``.

This module holds only pure logic (digesting artifacts, building prompts,
deriving run IDs / job names / URIs / variant grids) so it is unit-testable
without SSH, S3, Nebius, or GPUs. All network, storage, and Token Factory calls
live in the runner scripts and the existing client/tool modules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TRIAGE_SYSTEM_PROMPT = (
    "You are a senior robot-learning engineer triaging a training run for a "
    "physical-AI team. You are given the run's configuration and log artifacts. "
    "Write a concise, technical report with these sections: (1) Summary — what "
    "was trained and whether it looks healthy; (2) Signals — concrete numbers or "
    "facts you can cite from the artifacts (losses, steps, durations, config "
    "choices); (3) Risks — anything that looks misconfigured, unstable, or "
    "missing; (4) Next steps — a short, ordered list of what to try next. Only "
    "use facts present in the artifacts; if something is unknown, say so rather "
    "than inventing numbers."
)

# Textual artifact suffixes worth feeding to a text model. Binary weights
# (.safetensors, .pt, .ckpt) and media are intentionally excluded.
_TEXT_ARTIFACT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

# Cap how much artifact text we forward so a noisy run cannot blow the context
# window or the request cost.
DEFAULT_MAX_FILES = 24
DEFAULT_MAX_FILE_BYTES = 4_000
DEFAULT_MAX_TOTAL_BYTES = 24_000


def utc_stamp(now: datetime | None = None) -> str:
    """Return a compact UTC timestamp suitable for run IDs and job names."""

    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_triage_run_id(now: datetime | None = None) -> str:
    return f"tf-train-triage-{utc_stamp(now)}"


def triage_job_name(run_id: str) -> str:
    """Derive a Nebius-safe serverless Job name from a run ID.

    Job names must be lowercase alphanumeric plus hyphens; collapse anything
    else and trim to a conservative length.
    """

    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in run_id.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned:
        cleaned = f"tf-triage-{utc_stamp()}"
    return cleaned[:48].rstrip("-")


def join_uri(base: str, *parts: str) -> str:
    """Join an ``s3://`` (or local) prefix with path parts, normalizing slashes."""

    root = base.rstrip("/")
    tail = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{root}/{tail}" if tail else root + "/"


def triage_prompts_uri(triage_root: str) -> str:
    return join_uri(triage_root, "prompts.jsonl")


def triage_report_uri(triage_root: str) -> str:
    return join_uri(triage_root, "generations.jsonl")


def _is_textual(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_ARTIFACT_SUFFIXES


def _iter_textual_files(local_dir: Path) -> Iterable[Path]:
    if local_dir.is_file():
        if _is_textual(local_dir):
            yield local_dir
        return
    yield from sorted(
        path
        for path in local_dir.rglob("*")
        if path.is_file()
        and _is_textual(path)
        and not any(part.startswith(".") for part in path.relative_to(local_dir).parts)
    )


def summarize_run_artifacts(
    local_dir: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> str:
    """Build a bounded, labelled digest of the textual artifacts under a dir.

    Reads ``.json/.log/.txt/.yaml/...`` files, truncates each to
    ``max_file_bytes``, and stops once ``max_total_bytes`` is reached so the
    prompt stays within a sane size regardless of how chatty the run was.
    """

    base = Path(local_dir)
    if not base.exists():
        raise FileNotFoundError(f"artifact path does not exist: {base}")

    sections: list[str] = []
    total = 0
    file_count = 0
    for path in _iter_textual_files(base):
        if file_count >= max_files or total >= max_total_bytes:
            break
        label = str(path.relative_to(base)) if base.is_dir() else path.name
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippet = raw.strip()
        if len(snippet.encode("utf-8")) > max_file_bytes:
            snippet = snippet.encode("utf-8")[:max_file_bytes].decode("utf-8", errors="ignore")
            snippet = snippet.rstrip() + "\n... [truncated]"
        if not snippet:
            continue
        block = f"### {label}\n{snippet}"
        total += len(block.encode("utf-8"))
        file_count += 1
        sections.append(block)

    if not sections:
        return "(no textual artifacts found in the run output)"
    return "\n\n".join(sections)


def build_triage_prompt(
    *,
    job_name: str,
    output_uri: str,
    artifact_digest: str,
    extra_context: str = "",
) -> str:
    """Compose the user prompt that asks the text model to triage one run."""

    header = (
        f"Triage this robot-policy training run.\n"
        f"- Job name: {job_name}\n"
        f"- Artifacts location: {output_uri}\n"
    )
    if extra_context.strip():
        header += f"- Operator notes: {extra_context.strip()}\n"
    return (
        f"{header}\n"
        "Run artifacts (configs and logs) follow. Base your report only on these.\n\n"
        f"{artifact_digest}\n"
    )


def triage_prompt_record(
    *,
    job_name: str,
    output_uri: str,
    artifact_digest: str,
    extra_context: str = "",
) -> dict[str, str]:
    """Return one ``{"id", "prompt"}`` record for the token-factory generate tool."""

    return {
        "id": f"triage-{triage_job_name(job_name)}",
        "prompt": build_triage_prompt(
            job_name=job_name,
            output_uri=output_uri,
            artifact_digest=artifact_digest,
            extra_context=extra_context,
        ),
    }


def render_triage_prompts_jsonl(records: Iterable[dict[str, str]]) -> str:
    """Serialize prompt records to JSONL text for the generate tool."""

    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


# ---------------------------------------------------------------------------
# sim-sweep combo: Token Factory design -> N serverless GPU trains -> ranking.
#
# Pure logic for the "experiment sweep" composition: a hosted text model
# proposes an experiment rationale, a deterministic grid defines the actual
# hyper-parameters launched on Nebius GPUs (so the GPU stage never depends on
# parsing free-form model output), and a hosted text model finally ranks the
# completed runs from their real artifacts. All network/storage/GPU work lives
# in ``npa/scripts/run_tokenfactory_sim_sweep.py``.
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_DESIGN_SYSTEM_PROMPT = (
    "You are a robot-learning research lead designing a small training sweep for "
    "a physical-AI team. You are given an objective and the concrete variant grid "
    "that will actually run on GPUs. For each variant, write one or two sentences "
    "stating the hypothesis it tests and what signal would confirm or refute it. "
    "Be specific and grounded in the given hyper-parameters; do not invent "
    "variants that are not in the grid."
)

DEFAULT_SWEEP_RANKING_SYSTEM_PROMPT = (
    "You are a senior robot-learning engineer reviewing the results of a training "
    "sweep. You are given the sweep objective and, for each completed variant, its "
    "configuration and log artifacts. Produce a report with: (1) Ranking — order "
    "the variants best-to-worst with a one-line justification each, citing concrete "
    "numbers from the artifacts; (2) Winner — the variant you would promote and why; "
    "(3) Caveats — what the artifacts do not tell you. Only use facts present in the "
    "artifacts; if a metric is missing, say so rather than inventing it."
)

# Training-step counts used as the deterministic sweep knob. ``lerobot train``
# exposes ``--steps`` (but no ``--seed``), so steps give a real, comparable axis.
DEFAULT_SWEEP_STEPS = (50, 100, 150, 200)


def default_sweep_run_id(now: datetime | None = None) -> str:
    return f"tf-sim-sweep-{utc_stamp(now)}"


def sweep_variants(
    num_variants: int,
    *,
    steps_grid: Iterable[int] = DEFAULT_SWEEP_STEPS,
) -> list[dict[str, Any]]:
    """Return a deterministic variant grid (one dict per GPU train Job).

    The grid varies ``--steps`` so the GPU stage is fully defined by code, never
    by model output. ``num_variants`` is clamped to ``[1, len(steps_grid)]``.
    """

    grid = [int(value) for value in steps_grid]
    if not grid:
        raise ValueError("steps_grid must be non-empty")
    count = max(1, min(int(num_variants), len(grid)))
    return [
        {"index": index, "id": f"v{index}-steps{grid[index]}", "steps": grid[index]}
        for index in range(count)
    ]


def sweep_variant_output_uri(sweep_root: str, variant_id: str) -> str:
    return join_uri(sweep_root, "variants", variant_id)


def build_sweep_design_prompt(
    *,
    objective: str,
    dataset: str,
    policy_type: str,
    variants: Iterable[dict[str, Any]],
) -> str:
    """Compose the prompt asking the text model to design the sweep rationale."""

    lines = [
        "Design the rationale for this robot-policy training sweep.",
        f"- Objective: {objective.strip() or '(none given)'}",
        f"- Dataset: {dataset}",
        f"- Policy type: {policy_type}",
        "- Variant grid (these exact runs will launch on Nebius GPUs):",
    ]
    for variant in variants:
        extras = ", ".join(
            f"{key}={value}" for key, value in variant.items() if key not in {"index", "id"}
        )
        lines.append(f"  - {variant['id']}: {extras}" if extras else f"  - {variant['id']}")
    lines.append("")
    lines.append("Write the per-variant hypotheses described in your instructions.")
    return "\n".join(lines) + "\n"


def build_ranking_prompt(
    *,
    objective: str,
    runs: Iterable[dict[str, str]],
) -> str:
    """Compose the prompt that ranks completed sweep runs from their artifacts.

    Each run is ``{"id", "uri", "digest"}`` where ``digest`` is the bounded
    textual artifact digest produced by :func:`summarize_run_artifacts`.
    """

    header = [
        "Rank the completed training-sweep variants below.",
        f"- Objective: {objective.strip() or '(none given)'}",
        "",
        "Each variant's artifacts (configs and logs) follow. Base your report only",
        "on these.",
        "",
    ]
    blocks: list[str] = []
    for run in runs:
        blocks.append(
            f"## Variant {run['id']}\n- Artifacts location: {run.get('uri', '')}\n\n{run.get('digest', '')}"
        )
    if not blocks:
        blocks.append("(no completed variants were provided)")
    return "\n".join(header) + "\n\n".join(blocks) + "\n"
