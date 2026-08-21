"""Grade a Physical AI Data Factory run with the Cosmos Evaluator checks.

The blueprint's augment stage publishes one directory per scenario variant::

    <augment_uri>/<clip>/augmented_video.mp4
    <augment_uri>/<clip>/frame-*.png
    <augment_uri>/<clip>/metadata.json      # sampled `variables`, conditioning info

:func:`evaluate_run` walks those directories and, per variant, runs

- the **attribute verification** check against the variant's sampled appearance
  values, with the config manifest's variable table as the option set, and
- the **hallucination** check against the run's source clip, when one is
  resolvable, and
- an NPA **source-relative temporal consistency** companion diagnostic that
  measures both excess frame-to-frame variation and collapsed source motion,
  and
- an NPA **source-relative protected-appearance fidelity** companion check that
  distinguishes bounded global photometric changes from excessive or localized
  material recolouring.

It writes a single ``npa.cosmos_evaluator.report.v1`` document. The blueprint's
quality gate requires both its aggregate ``score`` and explicit ``passed``
disposition. Temporal consistency is advisory by default; deployments may make
it a hard check only after calibrating the noise floor for their capture path.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from npa.workbench.cosmos_evaluator.attribute_verification import (
    AttributeVerificationResult,
    verify_attributes,
)
from npa.workbench.cosmos_evaluator.appearance_fidelity import (
    DEFAULT_BLUR_KSIZE as APPEARANCE_BLUR_KSIZE,
    DEFAULT_CHROMA_INSTABILITY_TOLERANCE as APPEARANCE_CHROMA_INSTABILITY_TOLERANCE,
    DEFAULT_GLOBAL_CHROMA_TOLERANCE as APPEARANCE_GLOBAL_CHROMA_TOLERANCE,
    DEFAULT_LOCAL_CHROMA_TOLERANCE as APPEARANCE_LOCAL_CHROMA_TOLERANCE,
    DEFAULT_LUMINANCE_TOLERANCE as APPEARANCE_LUMINANCE_TOLERANCE,
    DEFAULT_MAX_DIMENSION as APPEARANCE_MAX_DIMENSION,
    DEFAULT_THRESHOLD as APPEARANCE_FIDELITY_THRESHOLD,
    AppearanceFidelityResult,
    check_appearance_fidelity,
)
from npa.workbench.cosmos_evaluator.hallucination import (
    DEFAULT_THRESHOLD as HALLUCINATION_THRESHOLD,
    HallucinationResult,
    check_hallucination,
)
from npa.workbench.cosmos_evaluator.temporal_consistency import (
    DEFAULT_BLUR_KSIZE as TEMPORAL_BLUR_KSIZE,
    DEFAULT_NOISE_FLOOR as TEMPORAL_NOISE_FLOOR,
    DEFAULT_THRESHOLD as TEMPORAL_CONSISTENCY_THRESHOLD,
    TemporalConsistencyResult,
    check_temporal_consistency,
)
from npa.workbench.cosmos_evaluator.upstream import (
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosEvaluatorError,
    CosmosEvaluatorStorageError,
    upstream_source_dir,
)

_log = logging.getLogger(__name__)

RESULT_FILENAME = "cosmos_evaluator.json"
REPORT_SCHEMA = "npa.cosmos_evaluator.report.v1"
VIDEO_NAME = "augmented_video.mp4"
METADATA_NAME = "metadata.json"
CONFIG_MANIFEST_NAME = "manifest.json"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv", ".avi"})
TEMPORAL_MODES = frozenset({"advisory", "required"})
APPEARANCE_MODES = frozenset({"advisory", "required"})

# Weight on the hallucination score when the variant really is a re-render of the
# run's own footage. Without input conditioning the two clips show different
# scenes, so hallucinated-motion counts say nothing and the aggregate score falls
# back to the attribute pass rate alone.
DEFAULT_HALLUCINATION_WEIGHT = 0.5

# Sampled appearance combos carry a `prompt` alongside the attributes; it is an
# instruction, not a visual attribute, so it is never turned into a question.
NON_ATTRIBUTE_KEYS = frozenset({"prompt", "inference_seed"})


@dataclass(frozen=True)
class ClipEvaluation:
    """Quality-check results for one augmented variant."""

    clip_id: str
    score: float
    passed: bool
    input_conditioned: bool
    status: str = "completed"
    temporal_enforced: bool = False
    appearance_enforced: bool = False
    variables: dict[str, str] = field(default_factory=dict)
    attribute_verification: dict[str, Any] | None = None
    hallucination: dict[str, Any] | None = None
    temporal_consistency: dict[str, Any] | None = None
    appearance_fidelity: dict[str, Any] | None = None
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluateRunResult:
    """Run-level evaluator report."""

    status: str
    score: float
    passed: bool
    augment_uri: str
    output_uri: str
    result_uri: str
    clip_count: int
    passed_clips: int
    threshold: float
    generated_at: str
    batch_policy: str = "all-variants"
    temporal_mode: str = "advisory"
    appearance_mode: str = "advisory"
    attribute_sample_policy: str = "ranking"
    engines: list[str] = field(default_factory=list)
    clips: list[ClipEvaluation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clips"] = [clip.to_dict() for clip in self.clips]
        payload["schema"] = REPORT_SCHEMA
        payload["upstream"] = {"repo": UPSTREAM_REPO, "license": UPSTREAM_LICENSE}
        return payload


def report_uri_for(output_uri: str) -> str:
    if output_uri.endswith(".json"):
        return output_uri
    return output_uri.rstrip("/") + f"/{RESULT_FILENAME}"


def evaluate_run(
    *,
    augment_uri: str,
    output_uri: str,
    input_uri: str = "",
    configs_uri: str = "",
    original_video: str = "",
    threshold: float = HALLUCINATION_THRESHOLD,
    hallucination_weight: float = DEFAULT_HALLUCINATION_WEIGHT,
    temporal_threshold: float = TEMPORAL_CONSISTENCY_THRESHOLD,
    temporal_regions_json: str = "",
    temporal_mode: str = "advisory",
    temporal_noise_floor: float = TEMPORAL_NOISE_FLOOR,
    temporal_blur_ksize: int = TEMPORAL_BLUR_KSIZE,
    appearance_threshold: float = APPEARANCE_FIDELITY_THRESHOLD,
    appearance_regions_json: str = "",
    appearance_mode: str = "advisory",
    appearance_luminance_tolerance: float = APPEARANCE_LUMINANCE_TOLERANCE,
    appearance_global_chroma_tolerance: float = APPEARANCE_GLOBAL_CHROMA_TOLERANCE,
    appearance_local_chroma_tolerance: float = APPEARANCE_LOCAL_CHROMA_TOLERANCE,
    appearance_chroma_instability_tolerance: float = APPEARANCE_CHROMA_INSTABILITY_TOLERANCE,
    appearance_blur_ksize: int = APPEARANCE_BLUR_KSIZE,
    appearance_max_dimension: int = APPEARANCE_MAX_DIMENSION,
    question_model: str = "",
    vlm_model: str = "",
    max_clips: int = 0,
    attribute_sample_policy: str = "ranking",
    client: Any | None = None,
    storage: Any | None = None,
) -> EvaluateRunResult:
    """Run the configured Cosmos Evaluator checks over every augmented variant."""

    if not augment_uri:
        raise CosmosEvaluatorError("--augment-uri is required")
    if not output_uri:
        raise CosmosEvaluatorError("--output-uri is required")
    if attribute_sample_policy not in {"ranking", "holdout"}:
        raise CosmosEvaluatorError(
            "--attribute-sample-policy must be ranking or holdout"
        )
    if not 0.0 <= hallucination_weight <= 1.0:
        raise CosmosEvaluatorError("--hallucination-weight must be between 0.0 and 1.0")
    if not 0.0 < temporal_threshold <= 1.0:
        raise CosmosEvaluatorError(
            "--temporal-threshold must be greater than 0.0 and at most 1.0"
        )
    if temporal_mode not in TEMPORAL_MODES:
        raise CosmosEvaluatorError("--temporal-mode must be advisory or required")
    if temporal_noise_floor <= 0.0:
        raise CosmosEvaluatorError("--temporal-noise-floor must be greater than 0.0")
    if temporal_blur_ksize < 1 or temporal_blur_ksize % 2 == 0:
        raise CosmosEvaluatorError(
            "--temporal-blur-ksize must be a positive odd integer"
        )
    if not 0.0 < appearance_threshold <= 1.0:
        raise CosmosEvaluatorError(
            "--appearance-threshold must be greater than 0.0 and at most 1.0"
        )
    if appearance_mode not in APPEARANCE_MODES:
        raise CosmosEvaluatorError("--appearance-mode must be advisory or required")
    for option, value in (
        ("--appearance-luminance-tolerance", appearance_luminance_tolerance),
        ("--appearance-global-chroma-tolerance", appearance_global_chroma_tolerance),
        ("--appearance-local-chroma-tolerance", appearance_local_chroma_tolerance),
        (
            "--appearance-chroma-instability-tolerance",
            appearance_chroma_instability_tolerance,
        ),
    ):
        if value <= 0.0:
            raise CosmosEvaluatorError(f"{option} must be greater than 0.0")
    if appearance_blur_ksize < 1 or appearance_blur_ksize % 2 == 0:
        raise CosmosEvaluatorError(
            "--appearance-blur-ksize must be a positive odd integer"
        )
    if appearance_max_dimension < 16:
        raise CosmosEvaluatorError("--appearance-max-dimension must be at least 16")

    store = storage if storage is not None else _storage()
    warnings: list[str] = []
    option_table = _load_option_table(configs_uri, store=store, warnings=warnings)

    with tempfile.TemporaryDirectory(prefix="npa-cosmos-eval-run-") as tmp:
        workdir = Path(tmp)
        clip_targets = _list_clip_targets(augment_uri, store=store)
        if not clip_targets:
            selection = _selection_manifest(augment_uri, store=store)
            if selection is None:
                raise CosmosEvaluatorError(
                    f"no augmented variant directories found under {augment_uri}"
                )
            return EvaluateRunResult(
                status="completed",
                score=0.0,
                passed=False,
                augment_uri=augment_uri,
                output_uri=output_uri,
                result_uri=report_uri_for(output_uri),
                clip_count=0,
                passed_clips=0,
                threshold=threshold,
                generated_at=datetime.now(timezone.utc).isoformat(),
                batch_policy="independent-hard-pass-selection",
                temporal_mode=temporal_mode,
                appearance_mode=appearance_mode,
                attribute_sample_policy=attribute_sample_policy,
                warnings=["ranking produced no independently hard-passing candidate"],
            )
        if max_clips and max_clips > 0:
            clip_targets = clip_targets[:max_clips]

        source_clips = _resolve_source_clips(
            original_video=original_video,
            input_uri=input_uri,
            store=store,
            workdir=workdir / "source",
            warnings=warnings,
        )

        evaluations: list[ClipEvaluation] = []
        status = "completed"
        for clip_id, clip_uri in clip_targets:
            try:
                evaluations.append(
                    _evaluate_clip(
                        clip_id=clip_id,
                        clip_uri=clip_uri,
                        workdir=workdir / clip_id,
                        store=store,
                        client=client,
                        option_table=option_table,
                        source_clips=source_clips,
                        threshold=threshold,
                        hallucination_weight=hallucination_weight,
                        temporal_threshold=temporal_threshold,
                        temporal_regions_json=temporal_regions_json,
                        temporal_mode=temporal_mode,
                        temporal_noise_floor=temporal_noise_floor,
                        temporal_blur_ksize=temporal_blur_ksize,
                        appearance_threshold=appearance_threshold,
                        appearance_regions_json=appearance_regions_json,
                        appearance_mode=appearance_mode,
                        appearance_luminance_tolerance=appearance_luminance_tolerance,
                        appearance_global_chroma_tolerance=appearance_global_chroma_tolerance,
                        appearance_local_chroma_tolerance=appearance_local_chroma_tolerance,
                        appearance_chroma_instability_tolerance=appearance_chroma_instability_tolerance,
                        appearance_blur_ksize=appearance_blur_ksize,
                        appearance_max_dimension=appearance_max_dimension,
                        question_model=question_model,
                        vlm_model=vlm_model,
                        attribute_sample_policy=attribute_sample_policy,
                        warnings=warnings,
                    )
                )
            except CosmosEvaluatorStorageError as exc:
                # Storage stopped answering, so the remaining variants would skip for
                # the same reason and average into a score that describes the outage
                # rather than the run. Report what was actually graded, marked
                # degraded, instead of a full batch of zeros that reads as real.
                status = "degraded"
                warnings.append(str(exc)[:300])
                break

        if status == "completed" and any(
            clip.status != "completed" for clip in evaluations
        ):
            status = "degraded"

    scores = [clip.score for clip in evaluations]
    run_score = round(sum(scores) / len(scores), 6) if scores else 0.0
    passed_clips = sum(1 for clip in evaluations if clip.passed)
    engines = sorted(
        {
            str(result.get("engine", ""))
            for clip in evaluations
            for result in (
                clip.hallucination,
                clip.temporal_consistency,
                clip.appearance_fidelity,
            )
            if result
        }
        - {""}
    )
    result_uri = report_uri_for(output_uri)
    return EvaluateRunResult(
        status=status,
        score=run_score,
        # A degraded run never learned enough to promote anything.
        passed=(
            status == "completed"
            and bool(evaluations)
            and passed_clips == len(evaluations)
            and run_score >= threshold
        ),
        augment_uri=augment_uri,
        output_uri=output_uri,
        result_uri=result_uri,
        clip_count=len(evaluations),
        passed_clips=passed_clips,
        threshold=threshold,
        generated_at=datetime.now(timezone.utc).isoformat(),
        batch_policy=(
            "independent-hard-pass-final-validation"
            if attribute_sample_policy == "holdout"
            else "all-variants"
        ),
        temporal_mode=temporal_mode,
        appearance_mode=appearance_mode,
        attribute_sample_policy=attribute_sample_policy,
        engines=engines,
        clips=evaluations,
        warnings=warnings,
    )


def _evaluate_clip(
    *,
    clip_id: str,
    clip_uri: str,
    workdir: Path,
    store: Any,
    client: Any | None,
    option_table: dict[str, list[str]],
    source_clips: Sequence[Path],
    threshold: float,
    hallucination_weight: float,
    temporal_threshold: float,
    temporal_regions_json: str,
    temporal_mode: str,
    temporal_noise_floor: float,
    temporal_blur_ksize: int,
    appearance_threshold: float,
    appearance_regions_json: str,
    appearance_mode: str,
    appearance_luminance_tolerance: float,
    appearance_global_chroma_tolerance: float,
    appearance_local_chroma_tolerance: float,
    appearance_chroma_instability_tolerance: float,
    appearance_blur_ksize: int,
    appearance_max_dimension: int,
    question_model: str,
    vlm_model: str,
    attribute_sample_policy: str,
    warnings: list[str],
) -> ClipEvaluation:
    workdir.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    degraded = False

    raw_metadata = _download_json(clip_uri + METADATA_NAME, store=store)
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}
        degraded = True
        warnings.append(f"variant metadata is not an object for {clip_id}")
        skipped.append("variant metadata is malformed")
    raw_variables = metadata.get("variables")
    if raw_variables is not None and not isinstance(raw_variables, dict):
        raw_variables = {}
        degraded = True
        warnings.append(f"variant variables are not an object for {clip_id}")
        skipped.append("variant variables are malformed")
    variables = {
        str(key): str(value)
        for key, value in (raw_variables or {}).items()
        if key not in NON_ATTRIBUTE_KEYS and str(value).strip()
    }
    input_conditioned = bool(metadata.get("input_conditioned"))
    source_clip = _select_source_clip(
        clip_id=clip_id,
        metadata=metadata,
        source_clips=source_clips,
        warnings=warnings,
    )

    video = _download_file(clip_uri + VIDEO_NAME, workdir, store=store)
    frame = None
    if video is None:
        frame = _download_first_frame(clip_uri, workdir, store=store)
        if frame is None:
            skipped.append("no augmented video or frame to evaluate")

    attribute_result: AttributeVerificationResult | None = None
    if not variables:
        skipped.append("variant metadata carries no appearance variables")
    elif video is None and frame is None:
        skipped.append("attribute verification needs a video or frame")
    else:
        try:
            attribute_result = verify_attributes(
                clip_id=clip_id,
                video=str(video) if video is not None else None,
                frame=str(frame) if video is None and frame is not None else None,
                selected_variables=variables,
                variable_options=option_table,
                question_model=question_model,
                vlm_model=vlm_model,
                client=client,
                sample_policy=attribute_sample_policy,
            )
        except Exception as exc:  # noqa: BLE001 - keep grading the remaining variants
            message = f"attribute verification failed for {clip_id}: {exc}"[:300]
            # exc_info so a defect in this code is distinguishable from a flaky endpoint.
            _log.warning(message, exc_info=True)
            warnings.append(message)
            skipped.append("attribute verification failed")
            degraded = True
        else:
            # Upstream attribute verification records per-question endpoint
            # failures instead of raising. Preserve that distinction at run level
            # so a transient VLM outage is not reported as measured bad quality.
            if any(check.error for check in attribute_result.checks):
                degraded = True

    hallucination_result: HallucinationResult | None = None
    if video is None:
        skipped.append("hallucination check needs the augmented video")
    elif source_clip is None:
        skipped.append(
            "hallucination check needs a source clip (pass --original-video or --input-uri)"
        )
    else:
        try:
            hallucination_result = check_hallucination(
                clip_id=clip_id,
                original_video=source_clip,
                augmented_video=video,
                threshold=threshold,
            )
        except Exception as exc:  # noqa: BLE001 - keep grading the remaining variants
            message = f"hallucination check failed for {clip_id}: {exc}"[:300]
            _log.warning(message, exc_info=True)
            warnings.append(message)
            skipped.append("hallucination check failed")
            if input_conditioned:
                degraded = True

    temporal_result: TemporalConsistencyResult | None = None
    if not input_conditioned:
        skipped.append(
            "temporal consistency only applies to input-conditioned variants"
        )
    elif video is None:
        skipped.append("temporal consistency needs the augmented video")
    elif source_clip is None:
        skipped.append("temporal consistency needs a source clip")
    else:
        try:
            temporal_result = check_temporal_consistency(
                clip_id=clip_id,
                original_video=source_clip,
                augmented_video=video,
                threshold=temporal_threshold,
                regions=temporal_regions_json,
                noise_floor=temporal_noise_floor,
                blur_ksize=temporal_blur_ksize,
            )
        except Exception as exc:  # noqa: BLE001 - keep grading the remaining variants
            message = f"temporal consistency check failed for {clip_id}: {exc}"[:300]
            _log.warning(message, exc_info=True)
            warnings.append(message)
            skipped.append("temporal consistency check failed")
            if temporal_mode == "required":
                degraded = True

    appearance_result: AppearanceFidelityResult | None = None
    if not input_conditioned:
        skipped.append(
            "appearance fidelity only applies to input-conditioned variants"
        )
    elif video is None:
        skipped.append("appearance fidelity needs the augmented video")
    elif source_clip is None:
        skipped.append("appearance fidelity needs a source clip")
    else:
        try:
            appearance_result = check_appearance_fidelity(
                clip_id=clip_id,
                original_video=source_clip,
                augmented_video=video,
                threshold=appearance_threshold,
                regions=appearance_regions_json,
                luminance_tolerance=appearance_luminance_tolerance,
                global_chroma_tolerance=appearance_global_chroma_tolerance,
                local_chroma_tolerance=appearance_local_chroma_tolerance,
                chroma_instability_tolerance=appearance_chroma_instability_tolerance,
                blur_ksize=appearance_blur_ksize,
                max_dimension=appearance_max_dimension,
            )
        except Exception as exc:  # noqa: BLE001 - keep grading remaining variants
            message = f"appearance fidelity check failed for {clip_id}: {exc}"[:300]
            _log.warning(message, exc_info=True)
            warnings.append(message)
            skipped.append("appearance fidelity check failed")
            if appearance_mode == "required":
                degraded = True

    score, passed = _combine_scores(
        attribute_result=attribute_result,
        hallucination_result=hallucination_result,
        input_conditioned=input_conditioned,
        hallucination_weight=hallucination_weight,
        temporal_result=temporal_result,
        temporal_required=input_conditioned and temporal_mode == "required",
        appearance_result=appearance_result,
        appearance_required=input_conditioned and appearance_mode == "required",
        score_threshold=threshold,
    )
    return ClipEvaluation(
        clip_id=clip_id,
        score=score,
        passed=passed,
        input_conditioned=input_conditioned,
        status="degraded" if degraded else "completed",
        temporal_enforced=input_conditioned and temporal_mode == "required",
        appearance_enforced=input_conditioned and appearance_mode == "required",
        variables=variables,
        attribute_verification=attribute_result.to_dict() if attribute_result else None,
        hallucination=hallucination_result.to_dict() if hallucination_result else None,
        temporal_consistency=temporal_result.to_dict() if temporal_result else None,
        appearance_fidelity=(
            appearance_result.to_dict() if appearance_result else None
        ),
        skipped=skipped,
    )


def _combine_scores(
    *,
    attribute_result: AttributeVerificationResult | None,
    hallucination_result: HallucinationResult | None,
    input_conditioned: bool,
    hallucination_weight: float,
    temporal_result: TemporalConsistencyResult | None = None,
    temporal_required: bool = False,
    appearance_result: AppearanceFidelityResult | None = None,
    appearance_required: bool = False,
    score_threshold: float = 0.0,
) -> tuple[float, bool]:
    """Blend the two check scores into the number the quality gate reads.

    The hallucination score only counts when the variant re-rendered the run's own
    footage; otherwise the clips show different scenes and its motion comparison
    carries no signal, so it stays informational.
    """

    attribute_score = attribute_result.score if attribute_result else None
    hallucination_score = hallucination_result.score if hallucination_result else None
    # A conditioned augmentation must be judged against its source.  Falling
    # back to an attribute-only pass when the source/hallucination result is
    # missing silently weakens the advertised quality gate.
    if input_conditioned and hallucination_score is None:
        return 0.0, False
    use_hallucination = input_conditioned and hallucination_score is not None

    if attribute_score is None:
        return 0.0, False
    if not use_hallucination:
        if input_conditioned:
            return 0.0, False
        score = round(attribute_score, 6)
        return score, bool(
            attribute_result and attribute_result.passed
        ) and score >= score_threshold

    assert hallucination_score is not None
    blended = (
        hallucination_weight * hallucination_score
        + (1.0 - hallucination_weight) * attribute_score
    )
    passed = bool(attribute_result and attribute_result.passed) and bool(
        hallucination_result and hallucination_result.passed
    )
    if temporal_required:
        if temporal_result is None:
            return 0.0, False
        blended = min(blended, temporal_result.score)
        passed = passed and temporal_result.passed
    if appearance_required:
        if appearance_result is None:
            return 0.0, False
        blended = min(blended, appearance_result.score)
        passed = passed and appearance_result.passed
    score = round(blended, 6)
    return score, passed and score >= score_threshold


# ---------------------------------------------------------------------------
# Artifact plumbing
# ---------------------------------------------------------------------------


def _storage() -> Any:
    # Deferred: a run over local paths must not need object-storage credentials.
    from npa.clients.storage import LazyStorageClient

    return LazyStorageClient()


def _split(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


def _list_keys(uri: str, *, store: Any) -> list[str]:
    prefixed = uri if uri.endswith("/") else uri + "/"
    bucket, prefix = _split(prefixed)
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = store.s3.list_objects_v2(**kwargs)
        keys.extend(
            entry["Key"] for entry in page.get("Contents", []) if entry.get("Key")
        )
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _list_clip_dirs(augment_uri: str, *, store: Any) -> list[str]:
    """Legacy variant directory names directly under ``augment_uri``."""

    prefixed = augment_uri if augment_uri.endswith("/") else augment_uri + "/"
    if not _is_remote(prefixed):
        root = Path(_local_path(prefixed))
        if not root.is_dir():
            return []
        return sorted(
            child.name
            for child in root.iterdir()
            if child.is_dir() and child.name != "_attempts"
        )
    _, prefix = _split(prefixed)
    names: set[str] = set()
    for key in _list_keys(prefixed, store=store):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        head, _, tail = rest.partition("/")
        if tail and head and head != "_attempts":
            names.add(head)
    return sorted(names)


def _selection_manifest(augment_uri: str, *, store: Any) -> dict[str, Any] | None:
    """Return a truthful empty hard-pass selection, never an absent augment."""

    manifest = _download_json(
        augment_uri.rstrip("/") + "/manifest.json", store=store
    )
    if not isinstance(manifest, dict):
        return None
    if (
        manifest.get("selection_policy")
        != "independent-hard-pass-only"
        or manifest.get("variant_count") != 0
        or manifest.get("variants") != []
    ):
        return None
    from npa.workbench.cosmos.transfer import validate_committed_run_manifest

    try:
        validate_committed_run_manifest(manifest, augment_uri)
    except (TypeError, ValueError) as exc:
        raise CosmosEvaluatorError(str(exc)) from exc
    return manifest


def _list_clip_targets(augment_uri: str, *, store: Any) -> list[tuple[str, str]]:
    """Resolve only variants committed by the canonical transfer manifest.

    New multi-node runs keep every recovery generation in ``_attempts/``. Direct
    prefix enumeration would mix old and current clips, so the executed manifest
    is the sole authority. Legacy manifests without ``variants`` retain the old
    direct-directory fallback.
    """

    manifest_uri = augment_uri.rstrip("/") + "/manifest.json"
    listed_keys: list[str] = []
    if _is_remote(manifest_uri):
        _bucket, manifest_key = _split(manifest_uri)
        listed_keys = _list_keys(augment_uri, store=store)
        manifest_present = manifest_key in listed_keys
        _unused_bucket, prefix = _split(
            augment_uri if augment_uri.endswith("/") else augment_uri + "/"
        )
        has_attempts = any(
            key.startswith(prefix + "_attempts/") for key in listed_keys
        )
    else:
        manifest_present = Path(_local_path(manifest_uri)).is_file()
        has_attempts = (Path(_local_path(augment_uri)) / "_attempts").exists()
    manifest = (
        _download_json(manifest_uri, store=store) if manifest_present else None
    )
    if manifest_present and not isinstance(manifest, dict):
        raise CosmosEvaluatorError("canonical augment manifest is not an object")
    if isinstance(manifest, dict):
        from npa.workbench.cosmos.transfer import validate_committed_run_manifest

        try:
            variants = validate_committed_run_manifest(manifest, augment_uri)
        except (TypeError, ValueError) as exc:
            raise CosmosEvaluatorError(str(exc)) from exc
        if variants:
            targets: list[tuple[str, str]] = []
            for item in variants:
                if not isinstance(item, dict):
                    raise CosmosEvaluatorError("augment manifest has an invalid variant")
                clip = str(item.get("clip") or "").strip()
                video_uri = str(item.get("augmented_video_uri") or "").strip()
                if not clip or not video_uri or "/" not in video_uri:
                    raise CosmosEvaluatorError(
                        "augment manifest variant is missing its clip or generated video URI"
                    )
                targets.append((clip, video_uri.rsplit("/", 1)[0] + "/"))
            return targets
    if has_attempts:
        raise CosmosEvaluatorError(
            "augment attempt objects exist without a valid canonical manifest; "
            "refusing to infer a recovery generation"
        )
    return [
        (clip, augment_uri.rstrip("/") + f"/{clip}/")
        for clip in _list_clip_dirs(augment_uri, store=store)
    ]


def _local_path(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def _is_object_absent(exc: BaseException) -> bool:
    """True when object storage said "no such object", rather than failing to answer.

    The two cases lead to opposite conclusions: an absent variant is a legitimate
    skip, while a credential or endpoint failure means the run learned nothing and
    must not be reported as a batch of clips that all scored zero.
    """

    from botocore.exceptions import ClientError

    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}
    return False


def _download_file(uri: str, dest_dir: Path, *, store: Any) -> Path | None:
    if not _is_remote(uri):
        local = Path(_local_path(uri))
        return local if local.is_file() else None
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        local = store.download_path(uri, str(dest_dir))
    except Exception as exc:  # noqa: BLE001 - a missing artifact is not fatal
        if _is_object_absent(exc):
            _log.info("no object at %s: %s", uri, exc)
            return None
        # Anything else — bad endpoint, expired credentials, a transient outage —
        # means the scores that follow would describe our own storage access, not the
        # variants. Raising keeps that from being reported as a real batch of zeros.
        _log.warning("could not read %s: %s", uri, exc, exc_info=True)
        raise CosmosEvaluatorStorageError(f"could not read {uri}: {exc}") from exc
    path = Path(local)
    if path.is_dir():
        wanted = uri.rstrip("/").split("/")[-1]
        matches = sorted(path.rglob(wanted))
        return matches[0] if matches else None
    return path if path.is_file() else None


def _download_json(uri: str, *, store: Any) -> Any | None:
    if not _is_remote(uri):
        local = Path(_local_path(uri))
        if not local.is_file():
            return None
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    with tempfile.TemporaryDirectory(prefix="npa-cosmos-eval-json-") as tmp:
        path = _download_file(uri, Path(tmp), store=store)
        if path is None:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _download_first_frame(clip_uri: str, workdir: Path, *, store: Any) -> Path | None:
    """Fall back to a published PNG frame when the variant has no mp4."""

    if not _is_remote(clip_uri):
        root = Path(_local_path(clip_uri))
        frames = sorted(root.glob("frame-*.png")) if root.is_dir() else []
        return frames[0] if frames else None
    _, prefix = _split(clip_uri if clip_uri.endswith("/") else clip_uri + "/")
    keys = [
        key for key in _list_keys(clip_uri, store=store) if key.lower().endswith(".png")
    ]
    if not keys:
        return None
    bucket, _ = _split(clip_uri)
    return _download_file(f"s3://{bucket}/{sorted(keys)[0]}", workdir, store=store)


def _load_option_table(
    configs_uri: str,
    *,
    store: Any,
    warnings: list[str],
) -> dict[str, list[str]]:
    """Read the sampler's full option table from the config manifest.

    Falls back to the blueprint's own appearance table so a run without a config
    manifest still gets multi-option questions instead of single-option ones.
    """

    manifest: dict[str, Any] | None = None
    if configs_uri:
        uri = (
            configs_uri
            if configs_uri.endswith(".json")
            else configs_uri.rstrip("/") + f"/{CONFIG_MANIFEST_NAME}"
        )
        manifest = _download_json(uri, store=store)
        if manifest is None:
            warnings.append(
                f"config manifest not found at {uri}; using the blueprint's default option table"
            )
    variables = manifest.get("variables") if isinstance(manifest, dict) else None
    if isinstance(variables, dict) and variables:
        return {
            str(key): [str(option) for option in values]
            for key, values in variables.items()
            if isinstance(values, (list, tuple)) and values
        }
    from npa.workflows.data_factory_stages import APPEARANCE_VARIABLES

    return {key: list(values) for key, values in APPEARANCE_VARIABLES.items()}


def _resolve_source_clips(
    *,
    original_video: str,
    input_uri: str,
    store: Any,
    workdir: Path,
    warnings: list[str],
) -> list[Path]:
    """Materialize every candidate source clip without choosing one globally."""

    if original_video:
        path = _download_file(original_video, workdir, store=store)
        if path is None:
            warnings.append(f"--original-video {original_video} could not be read")
        return [path] if path is not None else []
    if not input_uri:
        return []
    if not _is_remote(input_uri):
        root = Path(_local_path(input_uri))
        if root.is_file():
            return [root] if root.suffix.lower() in VIDEO_SUFFIXES else []
        return (
            sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            )
            if root.is_dir()
            else []
        )

    prefixed = input_uri if input_uri.endswith("/") else input_uri + "/"
    keys = [
        key
        for key in _list_keys(prefixed, store=store)
        if Path(key).suffix.lower() in VIDEO_SUFFIXES
    ]
    bucket, _ = _split(prefixed)
    sources: list[Path] = []
    for index, key in enumerate(sorted(keys)):
        path = _download_file(
            f"s3://{bucket}/{key}", workdir / f"source-{index:04d}", store=store
        )
        if path is None:
            warnings.append(f"source clip {Path(key).name} could not be read")
        else:
            sources.append(path)
    return sources


def _select_source_clip(
    *,
    clip_id: str,
    metadata: dict[str, Any],
    source_clips: Sequence[Path],
    warnings: list[str],
) -> Path | None:
    """Match one variant to its recorded conditioned input without first-key fallback."""

    if not source_clips:
        return None
    conditioned_input = Path(str(metadata.get("conditioned_input") or "")).name
    if conditioned_input:
        matches = [path for path in source_clips if path.name == conditioned_input]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            warnings.append(
                f"source clip is ambiguous for {clip_id}: multiple inputs are named {conditioned_input}"
            )
            return None
    if len(source_clips) == 1:
        return source_clips[0]
    detail = (
        f"conditioned input {conditioned_input!r} was not found"
        if conditioned_input
        else "variant metadata has no conditioned_input"
    )
    warnings.append(f"source clip unresolved for {clip_id}: {detail}")
    return None


def write_report(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage: Any | None = None,
) -> str:
    """Write the evaluator report to S3 or a local path."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if _is_remote(result_uri):
        store = storage if storage is not None else _storage()
        with tempfile.TemporaryDirectory(prefix="npa-cosmos-eval-out-") as tmp:
            local = Path(tmp) / RESULT_FILENAME
            local.write_text(body, encoding="utf-8")
            return store.upload_file(str(local), result_uri)
    path = Path(_local_path(result_uri))
    if path.suffix != ".json":
        path = path / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def evaluator_engine_summary(
    *, environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Describe which evaluator engine this environment will use."""

    root = upstream_source_dir(environ=environ)
    return {
        "upstream_repo": UPSTREAM_REPO,
        "upstream_license": UPSTREAM_LICENSE,
        "upstream_source": str(root) if root else "",
        "engine": "cosmos-evaluator-upstream" if root else "cosmos-evaluator-npa-port",
    }


def summarize_scores(clips: Sequence[ClipEvaluation]) -> dict[str, Any]:
    """Min / max / mean of per-variant scores, for report consumers."""

    scores = [clip.score for clip in clips]
    if not scores:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(scores),
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(sum(scores) / len(scores), 6),
    }
