"""Schemas and constants for the Encord workbench tool.

The Encord tool registers Nebius object-store media in place (bytes stay in the
bucket; Encord references them through an S3-compatible cloud integration) and
pulls curated data plus labels back to S3. These models are the durable receipt
and manifest contracts other stages consume.

Every enumerated field is a ``Literal`` so the vocabulary each artifact may
carry is typed in one place rather than enumerated in comments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PUSH_RECEIPT_SCHEMA = "npa.encord.push_receipt.v1"
PULL_MANIFEST_SCHEMA = "npa.encord.pull_manifest.v1"
CURATE_RECEIPT_SCHEMA = "npa.encord.curate_receipt.v1"
ROUNDTRIP_REPORT_SCHEMA = "npa.encord.roundtrip_report.v1"
PUSH_RECEIPT_FILENAME = "push_receipt.json"
PULL_MANIFEST_FILENAME = "manifest.json"
CURATE_RECEIPT_FILENAME = "curate_receipt.json"
ROUNDTRIP_REPORT_FILENAME = "roundtrip_report.json"

# How a recorded checksum was produced. A single-part S3 ETag is an MD5 digest;
# a multipart ETag is not a content digest and is recorded as "none".
ChecksumKind = Literal["none", "md5", "sha256"]
# Push --transfer modes: register keeps the bytes in the bucket and Encord
# references objectUrls; upload copies the bytes into Encord-hosted storage.
TransferMode = Literal["register", "upload"]
TRANSFER_MODES: tuple[TransferMode, ...] = ("register", "upload")
# Push --media filters. mcap/all expose the experimental MCAP path.
MediaFilter = Literal["videos-images", "mcap", "all"]
# Pull --source containers.
PullSourceKind = Literal["collection", "dataset", "project"]
PULL_SOURCES: tuple[PullSourceKind, ...] = ("collection", "dataset", "project")
# Per-item push outcome. experimental_error marks a discovered-but-unsupported
# input (MCAP) that is recorded in the receipt rather than sent with a guessed
# schema.
PushedItemStatus = Literal["registered", "uploaded", "error", "experimental_error"]
# The exact-identity signal that attributed an Encord uuid to a pushed item;
# empty while unresolved.
IdentitySignal = Literal["", "metadata", "object_url", "uploaded"]
# Receipt lifecycle. planned is the write-ahead copy that lands before the
# first Encord mutation; done/failed/timeout are terminal.
PushStatus = Literal["planned", "done", "failed", "timeout"]
# empty is a zero selection, timeout a selection still changing at the deadline.
CurateStatus = Literal["planned", "done", "empty", "timeout", "failed"]
# How a pulled item's bytes reached the output prefix; empty until attempted.
PullTransfer = Literal["", "copy", "download", "error"]
RoundtripRelation = Literal["matched", "missing", "unexpected"]
# unavailable: the two checksum kinds cannot be compared (e.g. multipart ETag).
ChecksumState = Literal["verified", "mismatched", "unavailable"]
ReportStatus = Literal["passed", "failed"]

DEFAULT_MEDIA_FILTER: MediaFilter = "videos-images"
DEFAULT_TRANSFER: TransferMode = "register"
DEFAULT_POLL_TIMEOUT_SECONDS = 1800
# add_preset_items is async server-side with no job handle; curate polls the
# collection until its item count is stable. Small folders settle in seconds.
DEFAULT_CURATE_POLL_SECONDS = 300


class EncordToolError(RuntimeError):
    """Raised when an Encord push/curate/pull/verify operation fails."""


class EncordAuthError(EncordToolError):
    """Raised when no usable Encord credential can be resolved."""


class EncordSdkMissingError(EncordToolError):
    """Raised when the optional ``encord`` package is not installed."""


class PushedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    # Exact identity: the canonical s3:// URI of the source object. This — via
    # the namespaced npa.source_uri clientMetadata registered with Encord —
    # and the normalized objectUrl are the only identity signals; display
    # names are never identity (adopted from PR #363).
    source_uri: str = ""
    object_url: str = ""
    category: str
    item_uuid: str = ""
    identity_signal: IdentitySignal = ""
    source_size: int = 0
    # The raw S3 ETag at push time (verbatim), so overwrites between push and
    # pull are detectable even when the ETag is not a content digest.
    source_etag: str = ""
    source_checksum: str = ""
    source_checksum_kind: ChecksumKind = "none"
    status: PushedItemStatus = "registered"
    error: str = ""


class PushReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=PUSH_RECEIPT_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "push"
    generated_at: str
    workflow_run: str = ""
    input_uri: str
    endpoint_url: str
    encord_domain: str
    transfer: TransferMode = "register"
    integration_id: str = ""
    integration_title: str = ""
    # Empty in the write-ahead ("planned") receipt, resolved before mutation.
    folder_uuid: str = ""
    folder_name: str
    folder_created: bool = False
    dataset_hash: str = ""
    dataset_title: str = ""
    dataset_created: bool = False
    linked_count: int = 0
    media_filter: MediaFilter
    status: PushStatus
    files_discovered: int = 0
    units_done: int = 0
    units_error: int = 0
    # Populated when the run failed partway: the exception that ended it, so the
    # receipt still explains a crash that happened after Encord was mutated.
    error: str = ""
    receipt_uri: str = ""
    items: list[PushedItem] = Field(default_factory=list)
    skipped_unsupported: list[str] = Field(default_factory=list)


class CurateFilter(BaseModel):
    """One workbench-declared quality filter, as parsed from metric:min:max."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    encord_metric: str
    min: float
    max: float
    # Computed metrics (brightness, sharpness, ...) require Encord's one-time
    # per-folder quality-metric computation; intrinsic ones (width, area, ...)
    # evaluate immediately. Recorded so a zero-selection diagnosis is honest.
    computed: bool = False


class CurateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=CURATE_RECEIPT_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "curate"
    generated_at: str
    workflow_run: str = ""
    encord_domain: str
    folder_uuid: str = ""
    folder_name: str = ""
    collection_uuid: str = ""
    collection_name: str = ""
    collection_created: bool = False
    preset_uuid: str = ""
    preset_name: str = ""
    # The transient preset is deleted once the selection lands; False after a
    # successful run means the delete failed and cleanup by prefix is owed.
    preset_deleted: bool = False
    filters: list[CurateFilter] = Field(default_factory=list)
    # The exact payload sent to create_preset, for reproducibility in Encord.
    filter_preset_json: dict = Field(default_factory=dict)
    # Storage items in the folder the filters were evaluated over, and how many
    # of them the selection kept.
    items_total: int = 0
    items_selected: int = 0
    status: CurateStatus
    # Populated when the run failed partway (see PushReceipt.error).
    error: str = ""
    receipt_uri: str = ""


class PulledItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_uuid: str
    name: str
    # Exact identity carried back from the item's npa.source_uri metadata.
    source_uri: str = ""
    item_type: str = ""
    mime_type: str = ""
    file_size: int = 0
    media_uri: str = ""
    transfer: PullTransfer = ""
    # Why the zero-egress server-side copy was not used for a same-endpoint
    # item (the run then fell back to a signed-URL download).
    copy_error: str = ""
    # Content evidence for the transferred bytes: sha256 of the streamed
    # download, or the destination ETag (md5 when single-part) for a
    # server-side copy.
    observed_size: int = 0
    checksum: str = ""
    checksum_kind: ChecksumKind = "none"
    error: str = ""


class RoundtripItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_uri: str = ""
    item_uuid: str = ""
    relation: RoundtripRelation
    expected_size: int = 0
    observed_size: int = 0
    expected_checksum: str = ""
    expected_checksum_kind: ChecksumKind = "none"
    observed_checksum: str = ""
    observed_checksum_kind: ChecksumKind = "none"
    checksum_state: ChecksumState = "unavailable"
    reasons: list[str] = Field(default_factory=list)


class RoundtripReport(BaseModel):
    """Terminal verifier joining a push receipt to a pull manifest by uuid.

    This makes checksum claims machine-checkable: a green report is the
    evidence that every pushed item came back with matching size and (where a
    content digest exists on both sides) matching checksum.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=ROUNDTRIP_REPORT_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "verify"
    generated_at: str
    workflow_run: str = ""
    receipt_uri: str
    manifest_uri: str
    report_uri: str = ""
    status: ReportStatus
    expected: int = 0
    matched: int = 0
    missing: int = 0
    unexpected: int = 0
    size_mismatched: int = 0
    checksum_verified: int = 0
    checksum_mismatched: int = 0
    checksum_unavailable: int = 0
    # Matched items that carried neither a comparable checksum nor a size on
    # both sides: nothing verifies them, so they fail the roundtrip.
    unverifiable: int = 0
    # Report-level reasons the roundtrip failed that no single item explains:
    # a receipt that never reached ``done``, or one with nothing to verify.
    defects: list[str] = Field(default_factory=list)
    error: str = ""
    items: list[RoundtripItem] = Field(default_factory=list)


class PullManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(default=PULL_MANIFEST_SCHEMA, alias="schema")
    tool: str = "encord"
    stage: str = "pull"
    generated_at: str
    workflow_run: str = ""
    encord_domain: str
    source_kind: PullSourceKind
    source_id: str
    source_name: str = ""
    output_uri: str
    manifest_uri: str = ""
    items_total: int = 0
    media_copied: int = 0
    media_downloaded: int = 0
    media_failed: int = 0
    label_rows: int = 0
    media_bytes: int = 0
    # Populated when the run failed partway (see PushReceipt.error).
    error: str = ""
    label_uris: list[str] = Field(default_factory=list)
    items: list[PulledItem] = Field(default_factory=list)
