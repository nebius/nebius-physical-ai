"""Exact workflow-run resolution shared by status, logs, artifacts, and cancel."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Mapping, cast
from urllib.parse import urlparse

from npa.orchestration.npa_workflow.run_state import (
    PAIDF_WORKFLOW_NAME,
    paidf_artifact_prefix,
    paidf_workflow_prefix,
)
from npa.orchestration.npa_workflow.submission_state import (
    inspect_submission_state,
    submission_proves_never_launched,
)
from npa.orchestration.skypilot.workflow import ManagedJobEvidence, lookup_managed_job
from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    WorkflowStateError,
    is_durable_workflow_manifest,
    parse_s3_uri,
    read_manifest,
    redact_text,
    resolve_workflow_s3_config,
    workflow_state_error_is_missing,
)


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
ResolutionOutcome = Literal["found", "absent", "unavailable", "not_supplied", "skipped"]


@dataclass(frozen=True)
class ResolutionCheck:
    source: str
    outcome: ResolutionOutcome
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"source": self.source, "outcome": self.outcome}
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class RunResolution:
    run_id: str
    project: str
    found: bool = False
    conclusively_absent: bool = False
    verification_unavailable: bool = False
    not_submitted: bool = False
    source: str = ""
    manifest_pending: bool = False
    manifest: dict[str, Any] | None = None
    runtime_state: dict[str, Any] = field(default_factory=dict)
    runtime_state_error: str = ""
    state: WorkflowS3Config | None = None
    receipt: dict[str, Any] = field(default_factory=dict)
    job_id: str = ""
    job_name: str = ""
    managed_job: ManagedJobEvidence | None = None
    run_prefix_uri: str = ""
    manifest_uri: str = ""
    workflow_name: str = ""
    durable_terminal_state: str = ""
    checks: list[ResolutionCheck] = field(default_factory=list)

    def checks_payload(self) -> list[dict[str, str]]:
        return [item.to_dict() for item in self.checks]


def validate_run_id(run_id: str) -> str:
    """Require a single safe path component for every exact lookup."""

    value = str(run_id or "").strip()
    if not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise WorkflowStateError(
            "run ID must be one 1-96 character path component containing only "
            "letters, digits, '.', '_', or '-'"
        )
    return value


def run_id_from_locator(run_id: str, workflow_s3_uri: str = "") -> str:
    locator = workflow_s3_uri or (run_id if str(run_id).startswith("s3://") else "")
    if not locator:
        return validate_run_id(run_id)
    _validate_s3_locator(locator)
    _bucket, prefix = parse_s3_uri(locator)
    parts = prefix.rstrip("/").split("/")
    if parts and parts[-1] == "manifest.json":
        parts.pop()
    if parts and parts[-1] == "npa-workflow":
        parts.pop()
    if not parts:
        raise WorkflowStateError("workflow S3 URI does not identify a run")
    if workflow_s3_uri and not str(run_id).startswith("s3://"):
        supplied = validate_run_id(run_id)
        # Supported locators end in either ``.../<run>/npa-workflow`` or
        # ``.../<run>/<workflow>/npa-workflow``. Only those trailing positions can
        # identify the supplied run. Accepting an earlier component lets the exact-
        # prefix fallback misreport an unrelated populated prefix as this run while
        # its manifest is still absent.
        if supplied not in parts[-2:]:
            raise WorkflowStateError(
                f"run ID {supplied!r} is not in the trailing run/workflow layout "
                "of --workflow-s3-uri"
            )
        return supplied
    resolved = validate_run_id(parts[-1])
    return resolved


def _validate_s3_locator(value: str) -> None:
    parsed = urlparse(str(value).strip())
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise WorkflowStateError(
            "workflow locator must be an exact s3://bucket/prefix URI"
        )
    parts = [part for part in parsed.path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise WorkflowStateError(
            "workflow S3 URI contains an unsafe or empty run prefix"
        )


def _validate_prefix(prefix: str) -> None:
    if any(part in {".", ".."} for part in str(prefix).split("/") if part):
        raise WorkflowStateError(
            "workflow S3 prefix contains an unsafe traversal component"
        )


def _state_from_uri(
    uri: str,
    *,
    run_id: str,
    project: str,
    s3_endpoint: str,
) -> WorkflowS3Config:
    _validate_s3_locator(uri)
    normalized = uri.rstrip("/")
    if normalized.endswith("/manifest.json"):
        normalized = normalized.removesuffix("/manifest.json")
    _bucket, prefix = parse_s3_uri(normalized)
    if PAIDF_WORKFLOW_NAME in prefix.split("/") and prefix.rstrip("/").endswith(
        f"/{run_id}"
    ):
        normalized = f"{normalized}/npa-workflow"
    return resolve_workflow_s3_config(
        run_id=run_id,
        project=project or None,
        workflow_s3_uri=normalized,
        s3_endpoint=s3_endpoint,
    )


def _parent_state(
    *,
    run_id: str,
    project: str,
    workflow_s3_prefix: str,
    s3_bucket: str,
    s3_endpoint: str,
) -> WorkflowS3Config:
    sentinel = "__npa_exact_run_parent__"
    child = resolve_workflow_s3_config(
        run_id=sentinel,
        project=project or None,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    _validate_prefix(child.prefix)
    prefix = child.prefix.removesuffix("/" + sentinel)
    if child.prefix == sentinel:
        prefix = ""
    return WorkflowS3Config(
        bucket=child.bucket,
        prefix=prefix,
        endpoint_url=child.endpoint_url,
        aws_access_key_id=child.aws_access_key_id,
        aws_secret_access_key=child.aws_secret_access_key,
        project=child.project,
    )


def _child_state(parent: WorkflowS3Config, prefix: str) -> WorkflowS3Config:
    return WorkflowS3Config(
        bucket=parent.bucket,
        prefix=prefix.strip("/"),
        endpoint_url=parent.endpoint_url,
        aws_access_key_id=parent.aws_access_key_id,
        aws_secret_access_key=parent.aws_secret_access_key,
        project=parent.project,
    )


def _probe_manifest(
    state: WorkflowS3Config, run_id: str
) -> tuple[ResolutionOutcome, dict[str, Any] | None, str]:
    try:
        manifest = read_manifest(state)
    except WorkflowStateError as exc:
        if workflow_state_error_is_missing(exc):
            return "absent", None, f"{state.uri.rstrip('/')}/manifest.json"
        return "unavailable", None, redact_text(str(exc))
    legacy_exact_manifest = bool(
        str(manifest.get("run_id") or "").strip()
        and str(manifest.get("workflow_name") or "").strip()
        and isinstance(manifest.get("stages"), dict)
    )
    if not is_durable_workflow_manifest(manifest) and not legacy_exact_manifest:
        return "unavailable", None, "object is not an NPA durable workflow manifest"
    if str(manifest.get("run_id") or "") != run_id:
        return (
            "unavailable",
            None,
            "manifest run id does not match the exact requested run",
        )
    return "found", manifest, f"{state.uri.rstrip('/')}/manifest.json"


def _probe_exact_prefix(state: WorkflowS3Config) -> tuple[ResolutionOutcome, str]:
    """Check one exact PAIDF run prefix with a bounded, paginated query."""

    run_prefix = state.prefix.rstrip("/").removesuffix("/npa-workflow") + "/"
    try:
        paginator = state.client().get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=state.bucket,
            Prefix=run_prefix,
            PaginationConfig={"MaxItems": 1, "PageSize": 1},
        )
        for page in pages:
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if key.startswith(run_prefix) and key != run_prefix:
                    return "found", f"partial artifact: s3://{state.bucket}/{key}"
            break
    except Exception as exc:  # noqa: BLE001 - provider errors are not absence
        return "unavailable", redact_text(str(exc))
    return "absent", f"no objects below s3://{state.bucket}/{run_prefix}"


def _attach_runtime_state(result: RunResolution, state: WorkflowS3Config) -> None:
    """Recover the exact active wave identity from a runtime ledger, if present."""

    from npa.orchestration.skypilot.workflow_state import get_json

    try:
        payload = get_json(state, "runtime.json")
    except WorkflowStateError as exc:
        if not workflow_state_error_is_missing(exc):
            result.runtime_state_error = redact_text(str(exc))
        return
    if str(payload.get("run_id") or "") != result.run_id:
        result.runtime_state_error = (
            "runtime ledger run id does not match the exact requested run"
        )
        return
    if not isinstance(payload.get("waves"), list):
        result.runtime_state_error = "runtime ledger waves field is not a list"
        return
    result.runtime_state = payload
    result.workflow_name = str(payload.get("workflow") or result.workflow_name)
    waves = [item for item in payload.get("waves") or [] if isinstance(item, dict)]
    active = next(
        (
            item
            for item in reversed(waves)
            if str(item.get("status") or "").lower()
            not in {"succeeded", "failed", "cancelled"}
        ),
        waves[-1] if waves else {},
    )
    if active:
        result.job_id = str(active.get("job_id") or result.job_id)
        result.job_name = str(active.get("job_name") or result.job_name)


def _receipt_workflow(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = receipt.get("workflow")
    return dict(value) if isinstance(value, Mapping) else {}


def _receipt_launch(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = receipt.get("launch")
    return dict(value) if isinstance(value, Mapping) else {}


def _mark_manifest(
    result: RunResolution,
    *,
    state: WorkflowS3Config,
    manifest: dict[str, Any],
    source: str,
) -> RunResolution:
    result.found = True
    result.source = source
    result.manifest = manifest
    result.state = state
    result.manifest_pending = False
    result.job_id = str(manifest.get("sky_job_id") or result.job_id)
    result.workflow_name = str(
        manifest.get("workflow")
        or manifest.get("workflow_name")
        or result.workflow_name
    )
    result.run_prefix_uri = str(
        manifest.get("run_prefix_uri") or ""
    ) or state.uri.removesuffix("/npa-workflow")
    result.manifest_uri = f"{state.uri.rstrip('/')}/manifest.json"
    _attach_runtime_state(result, state)
    return result


def resolve_run(
    run_id: str,
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
    sky_bin: str = "",
    exact_job_id: str = "",
    allow_local_not_submitted: bool = False,
) -> RunResolution:
    """Resolve exactly one run using the documented deterministic precedence."""

    resolved_id = run_id_from_locator(run_id, workflow_s3_uri)
    ledger_project = project or "default"
    result = RunResolution(
        run_id=resolved_id, project=project, job_id=exact_job_id.strip()
    )
    explicit_uri = workflow_s3_uri or (
        run_id if str(run_id).startswith("s3://") else ""
    )
    planned_only_candidate = False

    if explicit_uri:
        try:
            explicit_state = _state_from_uri(
                explicit_uri,
                run_id=resolved_id,
                project=project,
                s3_endpoint=s3_endpoint,
            )
            outcome, manifest, detail = _probe_manifest(explicit_state, resolved_id)
            if outcome == "absent":
                prefix_outcome, prefix_detail = _probe_exact_prefix(explicit_state)
                if prefix_outcome == "found":
                    outcome, detail = "found", prefix_detail
                    result.found = True
                    result.source = "explicit_workflow_s3_uri"
                    result.manifest_pending = True
                    _attach_runtime_state(result, explicit_state)
                elif prefix_outcome == "unavailable":
                    outcome, detail = "unavailable", prefix_detail
            result.checks.append(
                ResolutionCheck("explicit_workflow_s3_uri", outcome, detail)
            )
            result.state = explicit_state
            result.run_prefix_uri = explicit_state.uri.removesuffix("/npa-workflow")
            result.manifest_uri = f"{explicit_state.uri.rstrip('/')}/manifest.json"
            if outcome == "found" and manifest is not None:
                return _mark_manifest(
                    result,
                    state=explicit_state,
                    manifest=manifest,
                    source="explicit_workflow_s3_uri",
                )
            if outcome == "unavailable":
                result.verification_unavailable = True
        except Exception as exc:  # noqa: BLE001
            result.checks.append(
                ResolutionCheck(
                    "explicit_workflow_s3_uri", "unavailable", redact_text(str(exc))
                )
            )
            result.verification_unavailable = True
    else:
        result.checks.append(
            ResolutionCheck(
                "explicit_workflow_s3_uri", "not_supplied", "no explicit URI"
            )
        )

    receipt_read = inspect_submission_state(ledger_project, resolved_id)
    result.checks.append(
        ResolutionCheck(
            "durable_submission_receipt", receipt_read.outcome, receipt_read.error
        )
    )
    if receipt_read.outcome == "unavailable":
        result.verification_unavailable = True
    elif receipt_read.outcome == "found":
        result.receipt = receipt_read.payload
        workflow = _receipt_workflow(result.receipt)
        launch = _receipt_launch(result.receipt)
        if (
            allow_local_not_submitted
            and not exact_job_id
            and submission_proves_never_launched(
                result.receipt, project=ledger_project, run_id=resolved_id
            )
        ):
            planned_only_candidate = True
            result.checks[-1] = ResolutionCheck(
                "durable_submission_receipt",
                "found",
                "durable local ledger has no launch transition; checking later durable sources",
            )
        launch_status = str(launch.get("status") or "").lower()
        receipt_proves_run = bool(
            workflow
            or launch.get("sky_job_id")
            or launch_status in {"launching", "submitted", "completed"}
        )
        if not receipt_proves_run:
            result.checks[-1] = ResolutionCheck(
                "durable_submission_receipt",
                "absent",
                "local state contains no workflow launch receipt",
            )
            result.receipt = {}
            workflow = {}
            launch = {}
        result.job_id = result.job_id or str(launch.get("sky_job_id") or "")
        result.workflow_name = result.workflow_name or str(workflow.get("name") or "")
        receipt_uri = str(workflow.get("run_prefix_uri") or "")
        if receipt_uri and not explicit_uri:
            try:
                receipt_state = _state_from_uri(
                    f"{receipt_uri.rstrip('/')}/npa-workflow",
                    run_id=resolved_id,
                    project=project,
                    s3_endpoint=s3_endpoint,
                )
                outcome, manifest, detail = _probe_manifest(receipt_state, resolved_id)
                result.state = receipt_state
                result.run_prefix_uri = receipt_uri.rstrip("/")
                result.manifest_uri = f"{receipt_state.uri.rstrip('/')}/manifest.json"
                if outcome == "found" and manifest is not None:
                    return _mark_manifest(
                        result,
                        state=receipt_state,
                        manifest=manifest,
                        source="durable_submission_receipt",
                    )
                if outcome == "absent":
                    prefix_outcome, prefix_detail = _probe_exact_prefix(receipt_state)
                    if prefix_outcome == "found":
                        _attach_runtime_state(result, receipt_state)
                        result.checks[-1] = ResolutionCheck(
                            "durable_submission_receipt",
                            "found",
                            f"receipt found; {prefix_detail}",
                        )
                    elif prefix_outcome == "unavailable":
                        result.verification_unavailable = True
                if outcome == "unavailable":
                    result.verification_unavailable = True
                    result.checks[-1] = ResolutionCheck(
                        "durable_submission_receipt",
                        "found",
                        f"receipt found; manifest verification unavailable: {detail}",
                    )
            except Exception as exc:  # noqa: BLE001
                result.verification_unavailable = True
                result.checks[-1] = ResolutionCheck(
                    "durable_submission_receipt",
                    "found",
                    f"receipt found; location verification unavailable: {redact_text(str(exc))}",
                )
        if result.job_id or receipt_uri:
            result.found = True
            result.source = result.source or "durable_submission_receipt"
            result.manifest_pending = True

    from npa.orchestration.npa_workflow.first_run_state import terminal_run_evidence

    terminal_evidence = terminal_run_evidence(project=ledger_project, run_id=resolved_id)
    if terminal_evidence:
        result.found = True
        result.source = "project_run_terminal_ledger"
        result.durable_terminal_state = str(
            terminal_evidence.get("last_known_state") or ""
        ).upper()
        result.workflow_name = str(
            terminal_evidence.get("workflow_identity") or result.workflow_name
        )
        result.checks.append(
            ResolutionCheck(
                "project_run_terminal_ledger",
                "found",
                f"verified durable terminal state {result.durable_terminal_state}",
            )
        )

    parent: WorkflowS3Config | None = None
    if not explicit_uri and not result.found:
        try:
            parent = _parent_state(
                run_id=resolved_id,
                project=project,
                workflow_s3_prefix=workflow_s3_prefix,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
            )
            canonical_state = _child_state(parent, paidf_workflow_prefix(resolved_id))
            outcome, manifest, detail = _probe_manifest(canonical_state, resolved_id)
            if outcome == "found" and manifest is not None:
                result.checks.append(
                    ResolutionCheck("canonical_paidf_s3_prefix", "found", detail)
                )
                return _mark_manifest(
                    result,
                    state=canonical_state,
                    manifest=manifest,
                    source="canonical_paidf_s3_prefix",
                )
            prefix_outcome, prefix_detail = _probe_exact_prefix(canonical_state)
            combined: ResolutionOutcome = (
                "found"
                if prefix_outcome == "found"
                else (
                    "unavailable"
                    if "unavailable" in {outcome, prefix_outcome}
                    else "absent"
                )
            )
            combined_detail = detail if outcome == "unavailable" else prefix_detail
            result.checks.append(
                ResolutionCheck("canonical_paidf_s3_prefix", combined, combined_detail)
            )
            result.state = canonical_state
            result.run_prefix_uri = (
                f"s3://{canonical_state.bucket}/{paidf_artifact_prefix(resolved_id)}"
            )
            result.manifest_uri = f"{canonical_state.uri}/manifest.json"
            result.workflow_name = PAIDF_WORKFLOW_NAME
            if combined == "found":
                result.found = True
                result.source = "canonical_paidf_s3_prefix"
                result.manifest_pending = True
                _attach_runtime_state(result, canonical_state)
            elif combined == "unavailable":
                result.verification_unavailable = True
        except Exception as exc:  # noqa: BLE001
            result.checks.append(
                ResolutionCheck(
                    "canonical_paidf_s3_prefix", "unavailable", redact_text(str(exc))
                )
            )
            result.verification_unavailable = True
    elif explicit_uri:
        result.checks.append(
            ResolutionCheck(
                "canonical_paidf_s3_prefix", "skipped", "explicit URI has precedence"
            )
        )
    else:
        result.checks.append(
            ResolutionCheck(
                "canonical_paidf_s3_prefix", "skipped", "receipt has precedence"
            )
        )

    managed = lookup_managed_job(
        result.job_name or resolved_id,
        job_id=result.job_id,
        sky_bin=sky_bin or None,
    )
    result.managed_job = managed
    managed_outcome: ResolutionOutcome = (
        cast(ResolutionOutcome, managed.outcome)
        if managed.outcome in {"found", "absent", "unavailable", "not_supplied", "skipped"}
        else "unavailable"
    )
    detail = managed.error
    if result.job_id and managed_outcome != "found":
        recorded_by = "runtime ledger" if result.runtime_state else "submission receipt"
        detail = (
            f"job {result.job_id} recorded in {recorded_by}; live verification "
            f"{managed_outcome}" + (f": {managed.error}" if managed.error else "")
        )
    result.checks.append(ResolutionCheck("managed_job", managed_outcome, detail))
    if managed_outcome == "found":
        result.found = True
        result.source = result.source or "managed_job"
        result.manifest_pending = result.manifest is None
        result.job_id = managed.job_id
        result.workflow_name = result.workflow_name or PAIDF_WORKFLOW_NAME
    elif managed_outcome == "unavailable":
        result.verification_unavailable = True

    if result.found:
        result.checks.append(
            ResolutionCheck(
                "ordinary_workflow", "skipped", f"resolved via {result.source}"
            )
        )
        return result

    if explicit_uri:
        result.checks.append(
            ResolutionCheck(
                "ordinary_workflow", "skipped", "explicit URI was checked exactly"
            )
        )
    else:
        try:
            if parent is None:
                parent = _parent_state(
                    run_id=resolved_id,
                    project=project,
                    workflow_s3_prefix=workflow_s3_prefix,
                    s3_bucket=s3_bucket,
                    s3_endpoint=s3_endpoint,
                )
            base = parent.prefix.strip("/")
            ordinary_prefix = "/".join(part for part in (base, resolved_id) if part)
            ordinary_state = _child_state(parent, ordinary_prefix)
            outcome, manifest, detail = _probe_manifest(ordinary_state, resolved_id)
            result.checks.append(ResolutionCheck("ordinary_workflow", outcome, detail))
            if outcome == "found" and manifest is not None:
                return _mark_manifest(
                    result,
                    state=ordinary_state,
                    manifest=manifest,
                    source="ordinary_workflow",
                )
            if outcome == "unavailable":
                result.verification_unavailable = True
        except Exception as exc:  # noqa: BLE001
            result.checks.append(
                ResolutionCheck(
                    "ordinary_workflow", "unavailable", redact_text(str(exc))
                )
            )
            result.verification_unavailable = True

    if planned_only_candidate and not result.found and not result.verification_unavailable:
        later_checks = [
            check
            for check in result.checks
            if check.source != "durable_submission_receipt"
        ]
        if all(
            check.outcome in {"absent", "not_supplied", "skipped"}
            for check in later_checks
        ):
            result.not_submitted = True
            result.source = "durable_submission_receipt"
            return result
    result.conclusively_absent = not result.verification_unavailable and all(
        check.outcome in {"absent", "not_supplied", "skipped"}
        for check in result.checks
    )
    return result


def resolution_diagnostics(resolution: RunResolution) -> list[str]:
    lines = [
        f"{item.source}: {item.outcome}" + (f" ({item.detail})" if item.detail else "")
        for item in resolution.checks
    ]
    if resolution.conclusively_absent:
        lines.insert(
            0, "Run not found after all applicable exact sources were checked."
        )
    elif resolution.verification_unavailable and not resolution.found:
        lines.insert(
            0, "Run verification is unavailable; provider/auth failure is not absence."
        )
    return lines


def require_resolved_run(resolution: RunResolution) -> RunResolution:
    if resolution.found:
        return resolution
    raise WorkflowStateError("\n".join(resolution_diagnostics(resolution)))


def list_resolved_artifacts(
    resolution: RunResolution,
    *,
    stage: str = "",
    limit: int = 10_000,
) -> list[str]:
    """List artifacts below the resolver's exact location, never a guessed run."""

    require_resolved_run(resolution)
    if limit <= 0:
        raise WorkflowStateError("artifact list limit must be positive")
    if stage:
        validate_run_id(stage)
    state = resolution.state
    if state is None:
        raise WorkflowStateError(
            "run was found via managed-job evidence but its artifact location is unavailable"
        )
    is_paidf = (
        resolution.workflow_name == PAIDF_WORKFLOW_NAME
        or PAIDF_WORKFLOW_NAME in state.prefix.split("/")
        or resolution.manifest_pending
    )
    if not is_paidf:
        from npa.orchestration.skypilot.workflow_state import list_artifacts

        return list_artifacts(state, stage or None)

    run_prefix = state.prefix.rstrip("/").removesuffix("/npa-workflow") + "/"
    objects: list[str] = []
    try:
        paginator = state.client().get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=state.bucket,
            Prefix=run_prefix,
            PaginationConfig={"MaxItems": limit, "PageSize": min(limit, 1000)},
        )
        for page in pages:
            for item in page.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if not key.startswith(run_prefix) or key.endswith("/"):
                    continue
                relative = key[len(run_prefix) :]
                if relative.startswith("npa-workflow/"):
                    continue
                if stage and stage not in relative.split("/"):
                    continue
                objects.append(f"s3://{state.bucket}/{key}")
                if len(objects) >= limit:
                    return sorted(set(objects))
    except Exception as exc:  # noqa: BLE001
        raise WorkflowStateError(
            "artifact verification unavailable below "
            f"s3://{state.bucket}/{run_prefix}: {redact_text(str(exc))}"
        ) from exc
    return sorted(set(objects))
