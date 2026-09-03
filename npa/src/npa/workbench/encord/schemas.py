"""Validated durable contracts for the Encord transport."""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PUSH_RECEIPT_SCHEMA = "npa.encord.push_receipt.v1"
PULL_MANIFEST_SCHEMA = "npa.encord.pull_manifest.v1"
ROUNDTRIP_REPORT_SCHEMA = "npa.encord.roundtrip_report.v1"
IDENTITY_SIDECAR_SCHEMA = "npa.encord.identity_sidecar.v1"

PUSH_RECEIPT_FILENAME = "push_receipt.json"
PULL_MANIFEST_FILENAME = "manifest.json"
ROUNDTRIP_REPORT_FILENAME = "roundtrip_report.json"

DEFAULT_MEDIA_FILTER = "videos-images"
DEFAULT_TRANSFER = "register"
DEFAULT_POLL_TIMEOUT_SECONDS = 1800
DURABILITY_WARNING = (
    "If the artifact store fails after an external mutation, the newest mutation "
    "may not be represented by the last durable checkpoint. The tool stops further "
    "mutation and exits nonzero."
)

ArtifactPhase = Literal["provisional", "checkpoint", "final"]
RunStatus = Literal["running", "completed", "partial", "failed"]
ItemOutcome = Literal["successful", "unresolved", "failed", "unattempted"]
TransferMode = Literal["register", "upload"]
IdentityState = Literal["resolved", "unresolved"]
RegistrationState = Literal[
    "not_applicable",
    "unattempted",
    "submitted",
    "registered",
    "existing",
    "failed",
    "unresolved",
]
LinkState = Literal[
    "not_requested", "unattempted", "linked", "already_linked", "failed"
]
ChecksumKind = Literal[
    "sha256", "s3_checksum_sha256", "md5", "etag_opaque", "none"
]


class EncordToolError(RuntimeError):
    """Raised when an Encord operation cannot truthfully report completion."""


class EncordAuthError(EncordToolError):
    """Raised when no usable Encord credential can be resolved."""


def _nonempty(value: str, info: object) -> str:
    del info
    value = value.strip()
    if not value:
        raise ValueError("value must not be empty")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, revalidate_instances="always"
    )


class OutcomeCounts(StrictModel):
    discovered: int = Field(ge=0)
    attempted: int = Field(ge=0)
    successful: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    failed: int = Field(ge=0)
    unattempted: int = Field(ge=0)

    @model_validator(mode="after")
    def reconcile(self) -> "OutcomeCounts":
        if self.discovered != (
            self.successful + self.unresolved + self.failed + self.unattempted
        ):
            raise ValueError("discovered count does not reconcile with outcomes")
        if self.attempted != self.successful + self.unresolved + self.failed:
            raise ValueError("attempted count does not reconcile with outcomes")
        return self

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[ItemOutcome]) -> "OutcomeCounts":
        return cls(
            discovered=len(outcomes),
            attempted=sum(outcome != "unattempted" for outcome in outcomes),
            successful=outcomes.count("successful"),
            unresolved=outcomes.count("unresolved"),
            failed=outcomes.count("failed"),
            unattempted=outcomes.count("unattempted"),
        )


class IdentitySidecarRow(StrictModel):
    source_uri: str
    record_id: str = ""
    item_uuid: str = ""

    _source_uri = field_validator("source_uri")(_nonempty)

    @model_validator(mode="after")
    def require_assertion(self) -> "IdentitySidecarRow":
        self.record_id = self.record_id.strip()
        self.item_uuid = self.item_uuid.strip()
        if not (self.record_id or self.item_uuid):
            raise ValueError("sidecar row requires record_id or item_uuid")
        return self


class IdentitySidecar(StrictModel):
    schema_: Literal["npa.encord.identity_sidecar.v1"] = Field(
        default=IDENTITY_SIDECAR_SCHEMA, alias="schema"
    )
    items: list[IdentitySidecarRow]

    @model_validator(mode="after")
    def unique_rows(self) -> "IdentitySidecar":
        _reject_duplicates("source_uri", [row.source_uri for row in self.items])
        _reject_duplicates(
            "record_id", [row.record_id for row in self.items if row.record_id]
        )
        _reject_duplicates(
            "item_uuid", [row.item_uuid for row in self.items if row.item_uuid]
        )
        return self


class PushItem(StrictModel):
    source_uri: str
    bucket: str
    object_key: str
    category: str
    submitted_object_url: str = ""
    record_id: str = ""
    source_size: int = Field(default=0, ge=0)
    source_etag: str = ""
    source_etag_kind: ChecksumKind = "none"
    source_checksum: str = ""
    source_checksum_kind: ChecksumKind = "none"
    item_uuid: str = ""
    transfer_mode: TransferMode = DEFAULT_TRANSFER
    registration_state: RegistrationState = "unattempted"
    link_state: LinkState = "not_requested"
    identity_state: IdentityState = "unresolved"
    outcome: ItemOutcome = "unattempted"
    error_code: str = ""
    error: str = ""

    _required = field_validator("source_uri", "bucket", "object_key", "category")(
        _nonempty
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> "PushItem":
        from npa.workbench.encord.identity import canonical_s3_uri, normalize_object_url

        canonical = canonical_s3_uri(self.bucket, self.object_key)
        if self.source_uri != canonical:
            raise ValueError("push row source URI does not match its bucket and object key")
        if self.submitted_object_url:
            normalized_path = normalize_object_url(self.submitted_object_url).split(
                "://", 1
            )[1].partition("/")[2]
            expected_path = canonical.split("/", 3)[3]
            expected_suffix = f"{self.bucket}/{expected_path}"
            if not (
                normalized_path == expected_suffix
                or normalized_path.endswith(f"/{expected_suffix}")
            ):
                raise ValueError(
                    "push row submitted object URL does not match its bucket and object key"
                )
        if self.outcome == "successful":
            if self.identity_state != "resolved" or not self.item_uuid.strip():
                raise ValueError("successful push row requires exact Encord identity")
            if self.transfer_mode == "register":
                if self.registration_state not in {"registered", "existing"}:
                    raise ValueError("successful register row has invalid registration state")
                if not self.submitted_object_url.strip():
                    raise ValueError("successful register row requires submitted object URL")
            elif self.registration_state != "not_applicable":
                raise ValueError("successful upload row must use not_applicable registration")
            if self.transfer_mode == "upload" and self.source_checksum_kind != "sha256":
                raise ValueError("successful upload row requires a SHA-256 checksum")
            if self.error_code or self.error:
                raise ValueError("successful push row cannot contain an error")
        if self.outcome == "unattempted" and self.registration_state not in {
            "unattempted",
            "not_applicable",
        }:
            raise ValueError("unattempted push row has an attempted registration state")
        return self


class PushReceipt(StrictModel):
    schema_: Literal["npa.encord.push_receipt.v1"] = Field(
        default=PUSH_RECEIPT_SCHEMA, alias="schema"
    )
    tool: str = "encord"
    stage: str = "push"
    phase: ArtifactPhase
    status: RunStatus
    revision: int = Field(ge=0)
    generated_at: str
    updated_at: str
    workflow_run: str = ""
    input_uri: str
    endpoint_url: str = ""
    encord_domain: str
    transfer_mode: TransferMode = DEFAULT_TRANSFER
    idempotency: Literal["exact_identity", "not_guaranteed"] = "exact_identity"
    integration_id: str = ""
    integration_title: str = ""
    folder_uuid: str = ""
    folder_name: str
    folder_created: bool = False
    dataset_hash: str = ""
    dataset_title: str = ""
    dataset_created: bool = False
    dataset_requested: bool = False
    linked_count: int = Field(default=0, ge=0)
    media_filter: str
    counts: OutcomeCounts
    receipt_uri: str
    receipt_store_kind: Literal["s3", "local"]
    durability_warning: str = DURABILITY_WARNING
    identity_sidecar_uri: str = ""
    error_code: str = ""
    error: str = ""
    skipped_unsupported: list[str] = Field(default_factory=list)
    items: list[PushItem] = Field(default_factory=list)

    _required = field_validator(
        "generated_at",
        "updated_at",
        "input_uri",
        "encord_domain",
        "folder_name",
        "media_filter",
        "receipt_uri",
    )(_nonempty)

    @model_validator(mode="after")
    def validate_receipt(self) -> "PushReceipt":
        _validate_run_phase(self.phase, self.status)
        _validate_counts(self.counts, [item.outcome for item in self.items])
        _reject_duplicates("source_uri", [item.source_uri for item in self.items])
        _reject_duplicates(
            "bucket/object_key",
            [f"{item.bucket}\0{item.object_key}" for item in self.items],
        )
        _reject_duplicates(
            "record_id", [item.record_id for item in self.items if item.record_id]
        )
        _reject_duplicates(
            "item_uuid", [item.item_uuid for item in self.items if item.item_uuid]
        )
        if self.transfer_mode == "register" and self.idempotency != "exact_identity":
            raise ValueError("register receipt must declare exact_identity")
        if self.transfer_mode == "upload" and self.idempotency != "not_guaranteed":
            raise ValueError("upload receipt must declare not_guaranteed")
        if self.phase == "final":
            _validate_final_status(self.status, self.counts)
            if self.status == "completed":
                if not self.items:
                    raise ValueError("completed push receipt cannot be empty")
                if self.dataset_requested and any(
                    item.link_state not in {"linked", "already_linked"}
                    for item in self.items
                ):
                    raise ValueError("completed push receipt has an unlinked dataset row")
        return self


class PullItem(StrictModel):
    item_uuid: str = ""
    record_id: str = ""
    source_uri: str = ""
    name: str = ""
    item_type: str = ""
    mime_type: str = ""
    source_size: int = Field(default=0, ge=0)
    destination_uri: str = ""
    transfer: Literal["copy", "download", "unattempted"] = "unattempted"
    copy_attempted: bool = False
    copy_failure: str = ""
    outcome: ItemOutcome = "unattempted"
    destination_exists: bool = False
    destination_size: int = Field(default=0, ge=0)
    source_checksum: str = ""
    source_checksum_kind: ChecksumKind = "none"
    destination_checksum: str = ""
    destination_checksum_kind: ChecksumKind = "none"
    metadata_uri: str = ""
    metadata_state: Literal["unattempted", "written", "failed"] = "unattempted"
    error_code: str = ""
    error: str = ""

    @model_validator(mode="after")
    def validate_outcome(self) -> "PullItem":
        if self.outcome == "successful":
            if (
                not self.item_uuid.strip()
                or not self.source_uri.strip()
                or not self.destination_uri.strip()
            ):
                raise ValueError(
                    "successful pull row requires source, item, and destination identity"
                )
            if not self.destination_exists or self.destination_size <= 0:
                raise ValueError("successful pull row requires a nonempty destination")
            if self.source_size and self.source_size != self.destination_size:
                raise ValueError("successful pull row has a size mismatch")
            if self.metadata_state != "written" or not self.metadata_uri.strip():
                raise ValueError("successful pull row requires item metadata")
            if self.error_code or self.error:
                raise ValueError("successful pull row cannot contain an error")
        return self


class LabelArtifact(StrictModel):
    label_hash: str = ""
    data_hash: str = ""
    item_uuid: str = ""
    artifact_uri: str = ""
    outcome: ItemOutcome = "unattempted"
    error_code: str = ""
    error: str = ""

    @model_validator(mode="after")
    def validate_identity(self) -> "LabelArtifact":
        if not (self.label_hash.strip() or self.data_hash.strip() or self.item_uuid.strip()):
            raise ValueError("label artifact requires a stable identity")
        if self.outcome == "successful" and not self.artifact_uri.strip():
            raise ValueError("successful label artifact requires an artifact URI")
        return self


class PullManifest(StrictModel):
    schema_: Literal["npa.encord.pull_manifest.v1"] = Field(
        default=PULL_MANIFEST_SCHEMA, alias="schema"
    )
    tool: str = "encord"
    stage: str = "pull"
    phase: ArtifactPhase
    status: RunStatus
    revision: int = Field(ge=0)
    generated_at: str
    updated_at: str
    workflow_run: str = ""
    encord_domain: str
    source_kind: Literal["collection", "dataset", "project"]
    source_id: str
    source_name: str = ""
    label_export: Literal["none", "initialize"] = "none"
    label_export_remote_mutation: bool = False
    label_export_posture: str = "Labels were not initialized or exported."
    output_uri: str
    manifest_uri: str
    counts: OutcomeCounts
    label_counts: OutcomeCounts
    media_copied: int = Field(ge=0)
    media_downloaded: int = Field(ge=0)
    media_bytes: int = Field(ge=0)
    error_code: str = ""
    error: str = ""
    items: list[PullItem] = Field(default_factory=list)
    label_artifacts: list[LabelArtifact] = Field(default_factory=list)

    _required = field_validator(
        "generated_at", "updated_at", "encord_domain", "source_id", "output_uri", "manifest_uri"
    )(_nonempty)

    @model_validator(mode="after")
    def validate_manifest(self) -> "PullManifest":
        _validate_run_phase(self.phase, self.status)
        _validate_counts(self.counts, [item.outcome for item in self.items])
        _validate_counts(
            self.label_counts, [item.outcome for item in self.label_artifacts]
        )
        _reject_duplicates(
            "item_uuid", [item.item_uuid for item in self.items if item.item_uuid]
        )
        if self.label_export == "none":
            if self.label_export_remote_mutation or self.label_artifacts:
                raise ValueError("label_export none cannot contain label mutation artifacts")
        elif not self.label_export_remote_mutation:
            raise ValueError("label initialization must disclose its remote mutation posture")
        copied = sum(
            item.outcome == "successful" and item.transfer == "copy" for item in self.items
        )
        downloaded = sum(
            item.outcome == "successful" and item.transfer == "download"
            for item in self.items
        )
        byte_count = sum(
            item.destination_size for item in self.items if item.outcome == "successful"
        )
        if (self.media_copied, self.media_downloaded, self.media_bytes) != (
            copied,
            downloaded,
            byte_count,
        ):
            raise ValueError("pull convenience totals do not reconcile with rows")
        if self.phase == "final":
            outcomes = [item.outcome for item in self.items]
            if self.label_export == "initialize":
                outcomes.extend(item.outcome for item in self.label_artifacts)
            combined = OutcomeCounts.from_outcomes(outcomes)
            _validate_final_status(self.status, combined)
            if self.status == "completed" and not self.items:
                raise ValueError("completed pull manifest cannot have zero media rows")
        return self


class RoundtripItem(StrictModel):
    source_uri: str = ""
    record_id: str = ""
    item_uuid: str = ""
    destination_uri: str = ""
    expected_size: int = Field(default=0, ge=0)
    observed_size: int = Field(default=0, ge=0)
    expected_checksum: str = ""
    expected_checksum_kind: ChecksumKind = "none"
    observed_checksum: str = ""
    observed_checksum_kind: ChecksumKind = "none"
    integrity_state: Literal[
        "matched", "not_comparable", "missing", "size_mismatch", "checksum_mismatch"
    ] = "missing"
    relation: Literal["matched", "missing", "unexpected", "unresolved", "integrity_failed"]
    reasons: list[str] = Field(default_factory=list)


class RoundtripReport(StrictModel):
    schema_: Literal["npa.encord.roundtrip_report.v1"] = Field(
        default=ROUNDTRIP_REPORT_SCHEMA, alias="schema"
    )
    tool: str = "encord"
    stage: str = "verify"
    generated_at: str
    workflow_run: str = ""
    receipt_uri: str
    manifest_uri: str
    report_uri: str
    status: Literal["completed", "failed"]
    passed: bool
    expected: int = Field(ge=0)
    matched: int = Field(ge=0)
    missing: int = Field(ge=0)
    unexpected: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    destination_missing: int = Field(ge=0)
    size_mismatched: int = Field(ge=0)
    checksum_mismatched: int = Field(ge=0)
    checksum_unavailable: int = Field(ge=0)
    items: list[RoundtripItem] = Field(default_factory=list)

    _required = field_validator("generated_at", "receipt_uri", "manifest_uri", "report_uri")(
        _nonempty
    )

    @model_validator(mode="after")
    def validate_report(self) -> "RoundtripReport":
        expected_rows = [item for item in self.items if item.relation != "unexpected"]
        if self.expected != len(expected_rows):
            raise ValueError("roundtrip expected count does not reconcile")
        derived = {
            relation: sum(item.relation == relation for item in self.items)
            for relation in ("matched", "missing", "unexpected", "unresolved")
        }
        if (self.matched, self.missing, self.unexpected, self.unresolved) != (
            derived["matched"],
            derived["missing"],
            derived["unexpected"],
            derived["unresolved"],
        ):
            raise ValueError("roundtrip relation counts do not reconcile")
        integrity_counts = (
            sum(
                item.relation == "integrity_failed" and item.integrity_state == "missing"
                for item in self.items
            ),
            sum(item.integrity_state == "size_mismatch" for item in self.items),
            sum(item.integrity_state == "checksum_mismatch" for item in self.items),
            sum(item.integrity_state == "not_comparable" for item in self.items),
        )
        if (
            self.destination_missing,
            self.size_mismatched,
            self.checksum_mismatched,
            self.checksum_unavailable,
        ) != integrity_counts:
            raise ValueError("roundtrip integrity counts do not reconcile")
        if self.passed != (self.status == "completed"):
            raise ValueError("roundtrip passed and status disagree")
        if self.passed and any(item.relation != "matched" for item in self.items):
            raise ValueError("completed roundtrip report contains a discrepancy")
        return self


def _reject_duplicates(label: str, values: Sequence[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _validate_counts(counts: OutcomeCounts, outcomes: Sequence[ItemOutcome]) -> None:
    if counts != OutcomeCounts.from_outcomes(outcomes):
        raise ValueError("aggregate counts do not match item outcomes")


def _validate_run_phase(phase: ArtifactPhase, status: RunStatus) -> None:
    if phase in {"provisional", "checkpoint"} and status != "running":
        raise ValueError("non-final artifacts must have running status")
    if phase == "final" and status == "running":
        raise ValueError("final artifact cannot have running status")


def _validate_final_status(status: RunStatus, counts: OutcomeCounts) -> None:
    complete = (
        counts.discovered > 0
        and counts.successful == counts.discovered
        and counts.unresolved == counts.failed == counts.unattempted == 0
    )
    if status == "completed" and not complete:
        raise ValueError("completed artifact contains incomplete rows")
    has_progress = counts.successful > 0 or counts.unresolved > 0
    if status == "partial" and (complete or not has_progress):
        raise ValueError("partial artifact status is not supported by its rows")
    if status == "failed" and has_progress:
        raise ValueError("failed artifact must not hide successful or unresolved progress")
