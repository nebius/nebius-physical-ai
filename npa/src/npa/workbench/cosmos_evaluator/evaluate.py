"""Grade a Physical AI Data Factory run with the Cosmos Evaluator checks.

The blueprint's augment stage publishes one directory per scenario variant::

    <augment_uri>/<clip>/augmented_video.mp4
    <augment_uri>/<clip>/frame-*.png
    <augment_uri>/<clip>/metadata.json      # sampled `variables`, conditioning info

:func:`evaluate_run` walks those directories and, per variant, runs

- the **attribute verification** check against the variant's sampled appearance
  values, with the config manifest's variable table as the option set, and
- the **hallucination** check against the run's source clip, when one is
  resolvable.

It writes a single ``npa.cosmos_evaluator.report.v1`` document. Its ``score``
field is what the blueprint's quality gate thresholds on, so the gate needs no
knowledge of which checks ran.
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
from npa.workbench.cosmos_evaluator.hallucination import (
    DEFAULT_THRESHOLD as HALLUCINATION_THRESHOLD,
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

_log = logging.getLogger(__name__)

RESULT_FILENAME = "cosmos_evaluator.json"
REPORT_SCHEMA = "npa.cosmos_evaluator.report.v1"
VIDEO_NAME = "augmented_video.mp4"
METADATA_NAME = "metadata.json"
CONFIG_MANIFEST_NAME = "manifest.json"

# Weight on the hallucination score when the variant really is a re-render of the
# run's own footage. Without input conditioning the two clips show different
# scenes, so hallucinated-motion counts say nothing and the aggregate score falls
# back to the attribute pass rate alone.
DEFAULT_HALLUCINATION_WEIGHT = 0.5

# Sampled appearance combos carry a `prompt` alongside the attributes; it is an
# instruction, not a visual attribute, so it is never turned into a question.
NON_ATTRIBUTE_KEYS = frozenset({"prompt"})


@dataclass(frozen=True)
class ClipEvaluation:
    """Both checks' results for one augmented variant."""

    clip_id: str
    score: float
    passed: bool
    input_conditioned: bool
    variables: dict[str, str] = field(default_factory=dict)
    attribute_verification: dict[str, Any] | None = None
    hallucination: dict[str, Any] | None = None
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
    question_model: str = "",
    vlm_model: str = "",
    max_clips: int = 0,
    client: Any | None = None,
    storage: Any | None = None,
) -> EvaluateRunResult:
    """Run both Cosmos Evaluator checks over every augmented variant."""

    if not augment_uri:
        raise CosmosEvaluatorError("--augment-uri is required")
    if not output_uri:
        raise CosmosEvaluatorError("--output-uri is required")
    if not 0.0 <= hallucination_weight <= 1.0:
        raise CosmosEvaluatorError("--hallucination-weight must be between 0.0 and 1.0")

    store = storage if storage is not None else _storage()
    warnings: list[str] = []
    option_table = _load_option_table(configs_uri, store=store, warnings=warnings)

    with tempfile.TemporaryDirectory(prefix="npa-cosmos-eval-run-") as tmp:
        workdir = Path(tmp)
        clip_dirs = _list_clip_dirs(augment_uri, store=store)
        if not clip_dirs:
            raise CosmosEvaluatorError(f"no augmented variant directories found under {augment_uri}")
        if max_clips and max_clips > 0:
            clip_dirs = clip_dirs[:max_clips]

        source_clip = _resolve_source_clip(
            original_video=original_video,
            input_uri=input_uri,
            store=store,
            workdir=workdir / "source",
            warnings=warnings,
        )

        evaluations: list[ClipEvaluation] = []
        status = "completed"
        for clip_id in clip_dirs:
            try:
                evaluations.append(
                    _evaluate_clip(
                        clip_id=clip_id,
                        clip_uri=augment_uri.rstrip("/") + f"/{clip_id}/",
                        workdir=workdir / clip_id,
                        store=store,
                        client=client,
                        option_table=option_table,
                        source_clip=source_clip,
                        threshold=threshold,
                        hallucination_weight=hallucination_weight,
                        question_model=question_model,
                        vlm_model=vlm_model,
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

    scores = [clip.score for clip in evaluations]
    run_score = round(sum(scores) / len(scores), 6) if scores else 0.0
    passed_clips = sum(1 for clip in evaluations if clip.passed)
    engines = sorted(
        {
            str((clip.hallucination or {}).get("engine", ""))
            for clip in evaluations
            if clip.hallucination
        }
        - {""}
    )
    result_uri = report_uri_for(output_uri)
    return EvaluateRunResult(
        status=status,
        score=run_score,
        # A degraded run never learned enough to promote anything.
        passed=status == "completed" and run_score >= threshold,
        augment_uri=augment_uri,
        output_uri=output_uri,
        result_uri=result_uri,
        clip_count=len(evaluations),
        passed_clips=passed_clips,
        threshold=threshold,
        generated_at=datetime.now(timezone.utc).isoformat(),
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
    source_clip: Path | None,
    threshold: float,
    hallucination_weight: float,
    question_model: str,
    vlm_model: str,
    warnings: list[str],
) -> ClipEvaluation:
    workdir.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []

    metadata = _download_json(clip_uri + METADATA_NAME, store=store) or {}
    raw_variables = metadata.get("variables") if isinstance(metadata, dict) else {}
    variables = {
        str(key): str(value)
        for key, value in (raw_variables or {}).items()
        if key not in NON_ATTRIBUTE_KEYS and str(value).strip()
    }
    input_conditioned = bool(metadata.get("input_conditioned"))

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
            )
        except Exception as exc:  # noqa: BLE001 - keep grading the remaining variants
            message = f"attribute verification failed for {clip_id}: {exc}"[:300]
            # exc_info so a defect in this code is distinguishable from a flaky endpoint.
            _log.warning(message, exc_info=True)
            warnings.append(message)
            skipped.append("attribute verification failed")

    hallucination_result: HallucinationResult | None = None
    if video is None:
        skipped.append("hallucination check needs the augmented video")
    elif source_clip is None:
        skipped.append("hallucination check needs a source clip (pass --original-video or --input-uri)")
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

    score, passed = _combine_scores(
        attribute_result=attribute_result,
        hallucination_result=hallucination_result,
        input_conditioned=input_conditioned,
        hallucination_weight=hallucination_weight,
    )
    return ClipEvaluation(
        clip_id=clip_id,
        score=score,
        passed=passed,
        input_conditioned=input_conditioned,
        variables=variables,
        attribute_verification=attribute_result.to_dict() if attribute_result else None,
        hallucination=hallucination_result.to_dict() if hallucination_result else None,
        skipped=skipped,
    )


def _combine_scores(
    *,
    attribute_result: AttributeVerificationResult | None,
    hallucination_result: HallucinationResult | None,
    input_conditioned: bool,
    hallucination_weight: float,
) -> tuple[float, bool]:
    """Blend the two check scores into the number the quality gate reads.

    The hallucination score only counts when the variant re-rendered the run's own
    footage; otherwise the clips show different scenes and its motion comparison
    carries no signal, so it stays informational.
    """

    attribute_score = attribute_result.score if attribute_result else None
    hallucination_score = hallucination_result.score if hallucination_result else None
    use_hallucination = input_conditioned and hallucination_score is not None

    if attribute_score is None and not use_hallucination:
        return 0.0, False
    if attribute_score is None:
        assert hallucination_score is not None
        return round(hallucination_score, 6), bool(hallucination_result and hallucination_result.passed)
    if not use_hallucination:
        return round(attribute_score, 6), bool(attribute_result and attribute_result.passed)

    assert hallucination_score is not None
    blended = hallucination_weight * hallucination_score + (1.0 - hallucination_weight) * attribute_score
    passed = bool(attribute_result and attribute_result.passed) and bool(
        hallucination_result and hallucination_result.passed
    )
    return round(blended, 6), passed


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
        keys.extend(entry["Key"] for entry in page.get("Contents", []) if entry.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _list_clip_dirs(augment_uri: str, *, store: Any) -> list[str]:
    """Variant directory names directly under ``augment_uri``."""

    prefixed = augment_uri if augment_uri.endswith("/") else augment_uri + "/"
    if not _is_remote(prefixed):
        root = Path(_local_path(prefixed))
        if not root.is_dir():
            return []
        return sorted(child.name for child in root.iterdir() if child.is_dir())
    _, prefix = _split(prefixed)
    names: set[str] = set()
    for key in _list_keys(prefixed, store=store):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        head, _, tail = rest.partition("/")
        if tail and head:
            names.add(head)
    return sorted(names)


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


def _download_json(uri: str, *, store: Any) -> dict[str, Any] | None:
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
    keys = [key for key in _list_keys(clip_uri, store=store) if key.lower().endswith(".png")]
    if not keys:
        return None
    bucket, _ = _split(clip_uri)
    return _download_file(
        f"s3://{bucket}/{sorted(keys)[0]}", workdir, store=store
    )


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
        uri = configs_uri if configs_uri.endswith(".json") else configs_uri.rstrip("/") + f"/{CONFIG_MANIFEST_NAME}"
        manifest = _download_json(uri, store=store)
        if manifest is None:
            warnings.append(f"config manifest not found at {uri}; using the blueprint's default option table")
    variables = manifest.get("variables") if isinstance(manifest, dict) else None
    if isinstance(variables, dict) and variables:
        return {
            str(key): [str(option) for option in values]
            for key, values in variables.items()
            if isinstance(values, (list, tuple)) and values
        }
    from npa.workflows.data_factory_stages import APPEARANCE_VARIABLES

    return {key: list(values) for key, values in APPEARANCE_VARIABLES.items()}


def _resolve_source_clip(
    *,
    original_video: str,
    input_uri: str,
    store: Any,
    workdir: Path,
    warnings: list[str],
) -> Path | None:
    """Materialize the clip the augmented variants are compared against."""

    if original_video:
        path = _download_file(original_video, workdir, store=store)
        if path is None:
            warnings.append(f"--original-video {original_video} could not be read")
        return path
    if not input_uri:
        return None
    prefixed = input_uri if input_uri.endswith("/") else input_uri + "/"
    if not _is_remote(prefixed):
        root = Path(_local_path(prefixed))
        videos = sorted(root.rglob("*.mp4")) if root.is_dir() else []
        return videos[0] if videos else None
    keys = [key for key in _list_keys(prefixed, store=store) if key.lower().endswith(".mp4")]
    if not keys:
        return None
    bucket, _ = _split(prefixed)
    return _download_file(
        f"s3://{bucket}/{sorted(keys)[0]}", workdir, store=store
    )


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


def evaluator_engine_summary(*, environ: dict[str, str] | None = None) -> dict[str, Any]:
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
