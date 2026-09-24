"""Project PAIDF producer reports into factual Rerun review documents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from npa.verification import _SECRET_ASSIGNMENT
from npa.workflows.artifacts import _redact_json_value, redact_artifact_text


_SEMANTIC_IDENTIFIERS = frozenset(
    "run_id candidate_id clip_id clip_ids clip model question_model vlm_model "
    "region_id output_media_entity".split()
)


def _fields(names: str, **children: Any) -> dict[str, Any]:
    selected = {
        name: "identity" if name in _SEMANTIC_IDENTIFIERS else None
        for name in names.split()
    }
    return {**selected, **children}


_COMMON = "schema schema_version status run_id engine mode model revision sha256"
_MEDIA = _fields(
    "decoded_frames duration_seconds full_decode_passed height width sha256 fps "
    "frame_count generated_sha256 source_sha256 max_timestamp_error_seconds "
    "schema status visual_quality_evaluated codec container pixel_format frame_rate"
)
_REGION = _fields(
    "region_id bounds passed score chroma_delta_p95 chroma_instability_p95 "
    "local_chroma_residual_p95 luminance_delta_p95 acceleration_ratio "
    "augmented_mean_acceleration residual_mean_acceleration source_mean_acceleration noise_floor"
)
_METRIC = _fields(
    "clip_id engine passed score threshold total_frames aggregation blur_ksize "
    "frame_counts_match max_dimension chroma_instability_tolerance global_chroma_tolerance "
    "local_chroma_tolerance luminance_tolerance total_augmented_dynamic_pixels "
    "total_hallucinated_dynamic_pixels noise_floor",
    regions=[_REGION],
)
_CHECK = _fields(
    "variable value question expected_answer vlm_answer passed", options="choices"
)
_ATTRIBUTES = _fields(
    "clip_id passed score threshold total_checks passed_checks failed_checks "
    "question_model vlm_model",
    checks=[_CHECK],
)
_CLIP = _fields(
    "clip_id passed score status input_conditioned appearance_enforced temporal_enforced",
    variables="variables",
    attribute_verification=_ATTRIBUTES,
    hallucination=_METRIC,
    temporal_consistency=_METRIC,
    appearance_fidelity=_METRIC,
    temporal_alignment=_MEDIA,
)
_QUALITY = _fields(
    "schema decision evaluator_status hard_checks_passed quality_status reasons score threshold",
    evaluator_report_uri="reference",
)
_TOOL = _fields("name version arguments")
_DERIVATION = _fields(
    "kind sha256 derived_from_sha256 width height fps frame_count duration_seconds "
    "source_duration_seconds output_duration_seconds byte_size operations",
    staged_uri="reference",
    tool=_TOOL,
    media=_MEDIA,
    frame_derivation=_fields(
        "kind count",
        tool=_TOOL,
        staged_uri_pattern="reference",
        items=[_fields("name sha256 frame_index", staged_uri="reference")],
    ),
)
_INPUT = _fields(
    _COMMON + " source_kind kind input_origin input_origin_label conditioned_input "
    "conditioning_fps camera episode lerobot_version_family frame_count video_bytes byte_size "
    "immutable_revision source_asset_path asset_license asset_attribution "
    "redistribution_permitted hosted_service_use_permitted field_of_use_restrictions "
    "acceptance_required authentication_required delivery_mode transport",
    media=_MEDIA,
    derivation=_DERIVATION,
    authenticity=_fields("classification evidence"),
    cosmos_conditioning=_fields(
        "enabled mode environment_equivalent cli_equivalent", staged_uri="reference"
    ),
    **dict.fromkeys(
        (
            "original_video_uri",
            "staged_video_uri",
            "timeline_uri",
            "source_ref",
            "staged_canonical_s3_uri",
            "staged_source_uri",
            "provenance_uri",
            "authoritative_upstream_url",
            "source_asset_url",
            "episode_metadata_url",
            "asset_license_url",
        ),
        "reference",
    ),
)
_MOTION = _fields("enabled cosmos_weight source_weight raw_cosmos_outputs_preserved")
_VARIANT = _fields(
    "clip variant_index frame_count guidance seed steps video_bytes",
    augmented_video_uri="reference",
    control_uris="references",
    variables="variables",
    motion_preservation=_MOTION,
    temporal_alignment=_MEDIA,
)
_MANIFEST = _fields(
    _COMMON + " attempt conditioned_input frame_count guardrails input_conditioned "
    "input_conditioning structural_control variant_count variant_parallelism video_bytes "
    "weights_baked node_count control control_prompt mask_prompt",
    variants=[_VARIANT],
    clips="identity",
    motion_preservation=_MOTION,
    lineage=_fields("", captions_uri="reference", input_provenance_uri="reference"),
)
_EVALUATOR = _fields(
    _COMMON + " score passed threshold clip_count passed_clips engines generated_at "
    "alignment_mode appearance_mode attribute_evidence_mode attribute_sample_policy attribute_threshold "
    "batch_policy temporal_mode",
    clips=[_CLIP],
    upstream=_fields("license", repo="reference"),
    augment_uri="reference",
    output_uri="reference",
    result_uri="reference",
)
_CURATION = _fields(
    _COMMON + " clip_count augmented_clips clip_ids video_count frame_count "
    "curation_engine curated_kept curated_dropped artifact_count variant_count "
    "encoder input_videos clips_written clips_filtered motion_filter stages fiftyone_version "
    "candidate_count promotion_eligible_count archive_object_count source_inventory_object_count "
    "source_inventory_sha256 source_inventory_unchanged_after_publication",
    multiply=_fields("mode variant_count note"),
    input_source=_INPUT,
    fiftyone=_fields("fiftyone_version uniqueness_mean dedup_threshold"),
    upstream=_fields("license revision", repo="reference"),
    dataset_groups=[_fields("name label role")],
    candidate_results=[
        _fields(
            "candidate_id iteration clip_id quality_status score passed promotion_eligible"
        )
    ],
    dataset_uri="reference",
    written_uri="reference",
)
_FINAL = _fields(
    _COMMON
    + " artifact_count has_rrd multiply_mode variant_count augmentation_engine input_conditioned stages",
    input_source=_INPUT,
    written_uri="reference",
)
_CANDIDATE = _fields(
    "candidate_id iteration clip_id run_disposition candidate_disposition candidate_passed promotion_eligible score "
    "failed_attributes hallucination_status source_comparison_entity output_media_entity",
    attribute_results=[_CHECK],
    hallucination=_METRIC,
    temporal_consistency=_METRIC,
    appearance_fidelity=_METRIC,
    final_disposition=_QUALITY,
)
_SCHEMAS = {
    "input": _INPUT,
    "augment": _MANIFEST,
    "evaluator": _EVALUATOR,
    "quality": _QUALITY,
    "curation": _CURATION,
    "final": _FINAL,
    "candidate": _CANDIDATE,
    "config": _fields("scene", augmentations=["variables"]),
    "metadata": _fields("", variables="variables"),
    "captions": _fields("model", captions=[_fields("caption", image="reference")]),
}
# These are operator fields, not guesses based on values or an arbitrary *_id suffix.
_CREDENTIAL_FIELDS = frozenset(
    "authorization cookie password passwd private_key secret token access_key "
    "api_key client_secret secret_key aws_access_key_id aws_secret_access_key "
    "aws_session_token hf_token ngc_api_key nebius_iam_token".split()
)
_PRIVATE_FIELDS = (
    frozenset(
        "bucket bucket_name endpoint endpoint_url hostname host pod pod_name pod_id "
        "node node_name node_id cluster cluster_name cluster_id project project_id "
        "tenant tenant_id cwd working_directory local_path credential_path".split()
    )
    | _CREDENTIAL_FIELDS
)
_REFERENCE_FIELDS = frozenset(
    "original_video_uri staged_video_uri timeline_uri staged_canonical_s3_uri "
    "staged_source_uri provenance_uri staged_uri source_ref captions_uri "
    "input_provenance_uri augmented_video_uri evaluator_report_uri augment_uri "
    "output_uri result_uri dataset_uri written_uri".split()
)
_LOCATION_TOKEN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"'`\)\]\}]+|"
    r"(?<![\w:/])(?:/[\w.~%-]+){2,}|\b[A-Za-z]:[\\/][^\s<>\"'`]+"
)
_WITHHELD = "[private location omitted]"


def _public_references() -> frozenset[str]:
    from npa.workflows.data_factory_input import load_starter_contract
    from npa.workbench.cosmos_evaluator.upstream import UPSTREAM_REPO
    from npa.workbench.cosmos_curate.upstream import UPSTREAM_REPO as CURATOR_REPO

    contract = load_starter_contract()
    source = contract["source"]
    return frozenset(
        [
            UPSTREAM_REPO,
            CURATOR_REPO,
            contract["license"]["url"],
            *(
                source[key]
                for key in ("authoritative_url", "asset_url", "episode_metadata_url")
            ),
        ]
    )


class _PaidfReview:
    def __init__(self, local: Path, run_uri: str = "") -> None:
        self.local = local.resolve()
        self.run_uri = run_uri.rstrip("/")
        self.public = _public_references()
        self.private: set[str] = set()
        self.credentials: set[str] = set()
        self.sources: dict[Path, tuple[bytes, Any]] = {}
        self._remember_location(self.run_uri)
        self.private.add(str(self.local))
        for path in local.rglob("*.json"):
            if path.is_symlink():
                continue
            try:
                raw = path.read_bytes()
                payload = json.loads(raw)
            except (OSError, ValueError):
                continue
            self.sources[path.resolve()] = (raw, payload)
            self._collect_private(payload)

    def _remember_location(self, value: str) -> None:
        if not value or value in self.public:
            return
        try:
            parsed = urlsplit(value)
        except ValueError:
            return
        if parsed.hostname:
            self.private.add(parsed.hostname)

    def _collect_private(self, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in _PRIVATE_FIELDS and isinstance(value, str) and value:
                    self.private.add(value)
                    self._remember_location(value)
                    if key in _CREDENTIAL_FIELDS:
                        self.credentials.add(value)
                if key in _REFERENCE_FIELDS and isinstance(value, str):
                    self._remember_location(value)
                self._collect_private(value)
        elif isinstance(payload, list):
            for value in payload:
                self._collect_private(value)

    def reference(self, value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        if value in self.public:
            return value
        try:
            parsed, root = urlsplit(value), urlsplit(self.run_uri)
            if parsed.query or parsed.fragment or parsed.username or parsed.password:
                return _WITHHELD
            if parsed.scheme:
                if parsed.scheme != "s3" or (parsed.scheme, parsed.netloc) != (
                    root.scheme,
                    root.netloc,
                ):
                    return _WITHHELD
                prefix = root.path.rstrip("/") + "/"
                if not parsed.path.startswith(prefix):
                    return _WITHHELD
                relative = parsed.path[len(prefix) :]
            elif Path(value).is_absolute():
                relative = Path(value).resolve().relative_to(self.local).as_posix()
            else:
                relative = value
            parts = PurePosixPath(unquote(relative)).parts
            if (
                not parts
                or ".." in parts
                or "\\" in unquote(relative)
                or ":" in relative
                or parts[0] == "/"
            ):
                return _WITHHELD
            return relative
        except (ValueError, OSError):
            return _WITHHELD

    def text(self, value: str, *, semantic_identity: bool = False) -> str:
        text = _LOCATION_TOKEN.sub(lambda match: self.reference(match.group()), value)
        text, _ = redact_artifact_text(text)
        text = _SECRET_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}=[REDACTED]", text
        )
        # A model/clip identifier can equal a hostname without describing that
        # host. Credentials remain private even when placed in an identifier.
        values = self.credentials if semantic_identity else self.private
        for private in sorted(values, key=len, reverse=True):
            text = re.sub(
                r"(?<![\w.-])" + re.escape(private) + r"(?![\w-]|\.[\w-])",
                "[private identity omitted]",
                text,
            )
        return text

    def identity(self, value: Any) -> Any:
        if isinstance(value, list):
            return [
                self.identity(item)
                for item in value
                if not isinstance(item, (dict, list))
            ]
        return (
            self.text(value, semantic_identity=True)
            if isinstance(value, str)
            else self._scalar(value)
        )

    def _scalar(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, list):
            return [
                self._scalar(item)
                for item in value
                if not isinstance(item, (dict, list))
            ]
        return None

    def _project(self, value: Any, shape: Any) -> Any:
        if shape == "reference":
            return self.reference(value)
        if shape == "identity":
            return self.identity(value)
        if shape in ("variables", "choices", "references"):
            return self._named_values(value, shape)
        if shape is None:
            return self._scalar(value)
        if isinstance(shape, list):
            return (
                [self._project(item, shape[0]) for item in value]
                if isinstance(value, list)
                else []
            )
        if not isinstance(value, dict):
            return {}
        return {
            key: self._project(value[key], child)
            for key, child in shape.items()
            if key in value
        }

    def _named_values(self, value: Any, shape: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        selected = {}
        for key, item in value.items():
            if key in _PRIVATE_FIELDS or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_ -]*", key
            ):
                continue
            if shape == "choices" and (
                len(key) != 1 or key not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ):
                continue
            if not isinstance(item, (str, int, float, bool)) and item is not None:
                continue
            selected[self.text(key)] = (
                self.reference(item) if shape == "references" else self._scalar(item)
            )
        return selected

    def payload(self, payload: Any, kind: str) -> dict[str, Any]:
        projected, _ = _redact_json_value(self._project(payload, _SCHEMAS[kind]))
        return projected

    def read(self, path: Path, kind: str) -> dict[str, Any] | None:
        source = self.sources.get(path.resolve())
        if source is None or not isinstance(source[1], dict):
            return None
        projected = self.payload(source[1], kind)
        projected["source_report"] = self._source_report(path, source[0])
        return projected

    def candidate_evaluation(self, path: Path, clip: str) -> dict[str, Any]:
        source = self.sources.get(path.resolve())
        if source is None or not isinstance(source[1], dict):
            return {}
        for item in source[1].get("clips", []):
            if isinstance(item, dict) and str(item.get("clip_id") or "") == clip:
                # Lookup uses the original producer key, even if its displayed
                # value must be redacted because it also contains a credential.
                projected, _ = _redact_json_value(self._project(item, _CLIP))
                return {
                    **projected,
                    "source_report": self._source_report(path, source[0]),
                }
        return {}

    def _source_report(self, path: Path, raw: bytes) -> dict[str, str]:
        return {
            "artifact": path.resolve().relative_to(self.local).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "view": "selected producer facts; artifact references are relative to the run root",
        }
