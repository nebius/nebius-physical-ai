"""Runtime adapters for evidence-backed agent stage responses.

This source is embedded into the generated agent backend after
``agent_stages.py``.  The evidence semantics stay pure in ``agent_stages``;
this module performs the run-scoped S3 reads and state integration.
"""

from __future__ import annotations

# This source is embedded into backend.py, where these adapter dependencies are
# defined by the surrounding generated module.
import hashlib
import json
import re
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

try:
    from agent_backend.publication_reader import (
        PublicationConflict,
        REPORT_SUFFIX,
        RESERVED_SUFFIXES,
        resolve_committed_publication,
    )
except ModuleNotFoundError:
    from npa.agent_backend.publication_reader import (
        PublicationConflict,
        REPORT_SUFFIX,
        RESERVED_SUFFIXES,
        resolve_committed_publication,
    )

# NPA_EMBED_STANDALONE_START
# These adapter globals are intentionally supplied by the rendered backend. Use
# explicit standalone sentinels for direct helper tests instead of suppressing
# F821 for the entire embedded module.
if __name__ == "npa.cli.agent_stage_runtime":
    (
        ArtifactDiscoveryError,
        HTTPException,
        _agent_access_report,
        _agent_artifact_list_scope,
        _agent_s3_buckets,
        _agent_artifact_s3_client,
        _artifact_discovery_prefix,
        _discovery_exclude_roots,
        _load_selected_run_artifacts,
        _merge_sim2real_run_details,
        _now_iso,
        _slug,
        _validated_resolved_prefix,
        _workflow_draft_from_state,
        Artifact,
        artifact_role_for_relative_key,
        artifact_bucket_projects,
        build_artifact_backed_stages,
        coerce_authoritative_stage_evidence,
        find_run_artifacts,
        find_run_artifacts_across_buckets,
        is_inline_render,
        list_artifacts,
        local_demo_run_details,
        merge_stage_evidence,
        parse_stage_evidence_documents,
        render_hint_for_object,
        resolve_run_source,
        run_owns_workflow_stage_overlay,
        select_preferred_artifact,
        summarize_stage_evidence,
        validate_run_id,
    ) = (None,) * 32
# NPA_EMBED_STANDALONE_END


_MAX_STAGE_EVIDENCE_DOCUMENTS = 8
_MAX_STAGE_EVIDENCE_BYTES = 65_536
_MAX_PUBLICATION_JOURNAL_BYTES = 1024 * 1024


def _read_bounded_json_object(
    s3,
    bucket: str,
    key: str,
    *,
    max_bytes: int = _MAX_STAGE_EVIDENCE_BYTES,
    expected_sha256: str = "",
    expected_size: int | None = None,
):
    """Read one JSON object with a hard byte bound and deterministic cleanup."""
    if expected_size is not None and expected_size > max_bytes:
        raise PublicationConflict("committed report exceeds its bounded read limit")
    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        raw = body.read(max_bytes + 1)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if expected_size is not None and len(encoded) != expected_size:
            raise PublicationConflict(
                "committed report size changed between verification and read"
            )
        if expected_sha256 and hashlib.sha256(encoded).hexdigest() != expected_sha256:
            raise PublicationConflict(
                "committed report bytes changed between verification and read"
            )
        if len(encoded) > max_bytes:
            return None
        payload = json.loads(encoded)
        return payload if isinstance(payload, dict) else None
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            try:
                close()
            except (OSError, RuntimeError, ValueError):
                # Cleanup must not turn an otherwise safely bounded read into a
                # request failure. StreamingBody.close() is best-effort here.
                pass


def _read_publication_journal(s3, bucket: str, uri: str) -> bytes | None:
    prefix = f"s3://{bucket}/"
    if not uri.startswith(prefix):
        raise PublicationConflict(
            "publication journal resolved outside the selected artifact bucket"
        )
    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=uri.removeprefix(prefix))
        body = response["Body"]
        raw = body.read(_MAX_PUBLICATION_JOURNAL_BYTES + 1)
        payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    except KeyError:
        return None
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            try:
                close()
            except (OSError, RuntimeError, ValueError):
                pass
    if len(payload) > _MAX_PUBLICATION_JOURNAL_BYTES:
        raise PublicationConflict("publication journal exceeds its bounded read limit")
    return payload


def _verify_publication_target(s3, bucket: str, target) -> dict:
    """Bind one immutable object to the size and digest in its journal target."""

    uri = str(target.immutable_uri or "")
    prefix = f"s3://{bucket}/"
    if not uri.startswith(prefix):
        raise PublicationConflict(
            "committed publication object is outside the selected artifact bucket"
        )
    key = uri.removeprefix(prefix)
    head_object = getattr(s3, "head_object", None)
    head = {}
    if callable(head_object):
        head = head_object(Bucket=bucket, Key=key)
        if int(head.get("ContentLength") or -1) != int(target.size_bytes):
            raise PublicationConflict(
                "committed publication object size disagrees with its journal"
            )
        metadata_digest = str((head.get("Metadata") or {}).get("npa-sha256") or "")
        if metadata_digest and metadata_digest != target.sha256:
            raise PublicationConflict(
                "committed publication object digest disagrees with its journal"
            )
    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        while size <= target.size_bytes:
            chunk = body.read(min(1024 * 1024, target.size_bytes + 1 - size))
            if not chunk:
                break
            material = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
            digest.update(material)
            size += len(material)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if size != target.size_bytes or digest.hexdigest() != target.sha256:
        raise PublicationConflict(
            "committed publication object bytes disagree with its journal"
        )
    return head


def _run_root_key(artifacts: list, run_id: str) -> str:
    roots: set[str] = set()
    for item in artifacts:
        key = str(item.key or "")
        relative = str(getattr(item, "relative_key", "") or "").lstrip("/")
        suffix = f"/{relative}" if relative else ""
        if suffix and key.endswith(suffix):
            root = key[: -len(suffix)]
            if root == run_id or root.endswith(f"/{run_id}"):
                roots.add(root)
    if len(roots) == 1:
        return roots.pop()
    if roots:
        raise PublicationConflict("artifact inventory spans multiple run roots")

    marker = f"/{run_id}/"
    for item in artifacts:
        key = str(item.key or "")
        if marker in key:
            return key.rsplit(marker, 1)[0] + f"/{run_id}"
        if key.startswith(f"{run_id}/"):
            return run_id
    raise PublicationConflict("artifact inventory does not identify one run root")


def _resolve_committed_artifact_key(
    s3,
    bucket: str,
    run_root_key: str,
    key: str,
) -> str:
    """Resolve reserved aliases and reject stale immutable generations."""

    root_key = str(run_root_key or "").strip("/")
    normalized_key = str(key or "").strip("/")
    root_prefix = f"{root_key}/"
    if not root_key or not normalized_key.startswith(root_prefix):
        raise PublicationConflict("artifact key is outside the selected run root")
    relative = normalized_key[len(root_prefix) :]
    canonical_relatives = tuple(suffix.lstrip("/") for suffix in RESERVED_SUFFIXES)
    immutable = relative.startswith("reports/generations/") or relative.startswith(
        "components/history/stage_14/"
    )
    if relative not in canonical_relatives and not immutable:
        return normalized_key

    publication = resolve_committed_publication(
        lambda uri: _read_publication_journal(s3, bucket, uri),
        f"s3://{bucket}/{root_key}{REPORT_SUFFIX}",
    )
    if not publication.journaled:
        if immutable:
            raise PublicationConflict(
                "immutable publication artifact has no committed journal authority"
            )
        return normalized_key

    requested_uri = f"s3://{bucket}/{normalized_key}"
    if relative in canonical_relatives:
        target = publication.target(requested_uri)
        resolved_uri = publication.resolve(requested_uri)
        if resolved_uri is None:
            raise PublicationConflict(
                "the committed publication marks this artifact absent"
            )
    else:
        target = next(
            (
                candidate
                for candidate in publication.objects.values()
                if candidate.immutable_uri == requested_uri
            ),
            None,
        )
        if target is None:
            raise PublicationConflict(
                "artifact does not belong to the committed publication generation"
            )
        resolved_uri = requested_uri
    _verify_publication_target(s3, bucket, target)

    bucket_prefix = f"s3://{bucket}/"
    if not str(resolved_uri).startswith(bucket_prefix):
        raise PublicationConflict(
            "committed publication artifact resolved outside its selected bucket"
        )
    resolved_key = str(resolved_uri).removeprefix(bucket_prefix)
    if not resolved_key.startswith(root_prefix):
        raise PublicationConflict(
            "committed publication artifact resolved outside its selected run"
        )
    return resolved_key


def _committed_publication_artifacts(
    s3,
    bucket: str,
    run_id: str,
    artifacts: list,
    *,
    require_complete: bool = True,
    include_snapshot: bool = False,
) -> tuple:
    keys = [str(item.key or "") for item in artifacts]
    root_key = _run_root_key(artifacts, run_id)
    canonical_report_uri = f"s3://{bucket}/{root_key}/reports/sim2real-report.json"
    publication = resolve_committed_publication(
        lambda uri: _read_publication_journal(s3, bucket, uri),
        canonical_report_uri,
    )
    if not publication.journaled:
        report = next(
            (
                item
                for item in artifacts
                if str(item.key).endswith("/reports/sim2real-report.json")
            ),
            None,
        )
        result = (artifacts, report, select_preferred_artifact(artifacts), keys)
        return (*result, publication) if include_snapshot else result

    by_uri = {str(item.s3_uri or ""): item for item in artifacts}
    committed_uris = {
        target.immutable_uri
        for target in publication.objects.values()
        if target.immutable_uri is not None
    }
    missing = sorted(uri for uri in committed_uris if uri not in by_uri)
    if missing and require_complete:
        raise PublicationConflict(
            "committed publication objects are absent from the artifact inventory"
        )
    verified_heads = {}
    for target in publication.objects.values():
        if target.immutable_uri is not None:
            verified_heads[str(target.immutable_uri)] = _verify_publication_target(
                s3, bucket, target
            )
    if not require_complete:
        run_scope = f"{root_key}/"
        namespace = (
            root_key[: -len(f"/{run_id}")] if root_key.endswith(f"/{run_id}") else ""
        )
        for target in publication.objects.values():
            uri = str(target.immutable_uri or "")
            if (
                not uri
                or uri in by_uri
                or not str(target.canonical_uri).endswith(
                    ("/reports/sim2real.rrd", "/reports/sim2real.mcap")
                )
            ):
                continue
            key = uri.removeprefix(f"s3://{bucket}/")
            if not key.startswith(run_scope):
                raise PublicationConflict(
                    "committed publication artifact resolved outside its selected run"
                )
            relative_key = key[len(run_scope) :]
            head = verified_heads.get(uri) or {}
            modified = head.get("LastModified")
            if hasattr(modified, "isoformat"):
                modified = modified.isoformat()
            render = render_hint_for_object(key=key)
            by_uri[uri] = Artifact(
                run_id=run_id,
                key=key,
                s3_uri=uri,
                size=int(target.size_bytes),
                last_modified=str(modified or ""),
                render=render,
                inline=is_inline_render(render),
                role=artifact_role_for_relative_key(relative_key),
                namespace=namespace or "<bucket-root>",
                relative_key=relative_key,
            )
    report_uri = publication.resolve(canonical_report_uri)
    report = by_uri.get(str(report_uri or ""))
    canonical_rrd_uri = f"s3://{bucket}/{root_key}/reports/sim2real.rrd"
    canonical_mcap_uri = f"s3://{bucket}/{root_key}/reports/sim2real.mcap"
    viewable_uris = {
        publication.resolve(canonical_rrd_uri),
        publication.resolve(canonical_mcap_uri),
    }
    preferred = select_preferred_artifact(
        [by_uri[uri] for uri in viewable_uris if isinstance(uri, str) and uri in by_uri]
    )
    reserved_alias_keys = {
        target.canonical_uri.removeprefix(f"s3://{bucket}/")
        for target in publication.objects.values()
    }
    visible = [
        item
        for item in artifacts
        if str(item.key) not in reserved_alias_keys
        and (
            "/reports/generations/" not in str(item.key)
            and "/components/history/stage_14/" not in str(item.key)
            or str(item.s3_uri) in committed_uris
        )
    ]
    logical_keys = [str(item.key or "") for item in visible]
    logical_keys.extend(
        target.canonical_uri.removeprefix(f"s3://{bucket}/")
        for target in publication.objects.values()
        if target.immutable_uri is not None
    )
    result = (visible, report, preferred, logical_keys)
    return (*result, publication) if include_snapshot else result


def _assert_legacy_publication_snapshot_still_unjournaled(
    s3,
    bucket: str,
    publication,
) -> None:
    """Fence mutable-alias reads against the first journal publication."""

    if publication.journaled:
        return
    if _read_publication_journal(s3, bucket, publication.journal_uri) is not None:
        raise PublicationConflict(
            "publication journal appeared during a legacy artifact read"
        )


def _stage_evidence_candidate_rank(key: str) -> int | None:
    lower = str(key or "").lower()
    leaf = Path(lower).name
    if lower.endswith("/npa-workflow/status.json") or (
        "/logs/" in lower and lower.endswith("/status.json")
    ):
        return 0
    if lower.endswith("/npa-workflow/manifest.json"):
        return 1
    if leaf == "manifest.json":
        return 2
    if leaf.endswith("report.json"):
        return 3
    return None


def _workflow_stage_defs_from_state(state: dict) -> list[tuple[str, str, list[str]]]:
    draft = _workflow_draft_from_state(state)
    stages: list[tuple[str, str, list[str]]] = []
    plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {}
    for source in (plan.get("steps"), plan.get("states"), draft.get("states")):
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict):
                raw_id = str(
                    item.get("state") or item.get("id") or item.get("name") or ""
                ).strip()
                label = (
                    str(item.get("label") or item.get("description") or raw_id).strip()
                    or raw_id
                )
            else:
                raw_id = str(item or "").strip()
                label = raw_id
            if not raw_id:
                continue
            stage_id = _slug(raw_id, fallback="stage")
            patterns = [raw_id, raw_id.replace("_", "-"), raw_id.replace("-", "_")]
            if (stage_id, label, patterns) not in stages:
                stages.append((stage_id, label, patterns))
        if stages:
            break
    return stages


def _stage_evidence_documents(s3, bucket: str, artifacts: list) -> list:
    # Select the most authoritative typed candidates before any S3 GET. Reads are
    # bounded by both object count and bytes; parser order remains low-to-high
    # authority so status documents deterministically override snapshots.
    candidates = []
    for artifact in artifacts:
        key = str(getattr(artifact, "key", "") or "")
        rank = _stage_evidence_candidate_rank(key)
        if rank is None:
            continue
        size = int(getattr(artifact, "size", 0) or 0)
        if size > _MAX_STAGE_EVIDENCE_BYTES:
            continue
        candidates.append((rank, key))
    candidates.sort(key=lambda item: (item[0], item[1]))

    loaded = []
    for rank, key in candidates[:_MAX_STAGE_EVIDENCE_DOCUMENTS]:
        try:
            payload = _read_bounded_json_object(s3, bucket, key)
        except (
            ClientError,
            BotoCoreError,
            OSError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue
        if isinstance(payload, dict):
            loaded.append((rank, key, {"key": key, "payload": payload}))
    loaded.sort(key=lambda item: (-item[0], item[1]))
    return [document for _rank, _key, document in loaded]


_SENSITIVE_PUBLIC_MARKER = (
    r"(?:authorization|credentials?|password|passwd|private[_-]?key|"
    r"secret(?:[_-]?access)?(?:[_-]?key)?(?:[_-]?(?:id|value))?|"
    r"tokens?(?:[_-]?(?:id|value))?|api[_-]?key(?:[_-]?id)?|"
    r"access[_-]?key(?:[_-]?id)?)"
)
_SENSITIVE_PUBLIC_NAME = re.compile(
    rf"(?i)(?<![A-Za-z0-9]){_SENSITIVE_PUBLIC_MARKER}(?![A-Za-z0-9])"
)
_SENSITIVE_PUBLIC_VALUE = re.compile(
    rf"(?i)(?:authorization\s*:|bearer\s+|{_SENSITIVE_PUBLIC_MARKER}\s*(?:=|:|\s))"
)
_SENSITIVE_PUBLIC_NAME_TOKEN = (
    rf"(?:--?)?(?:[A-Za-z0-9]+[_.-])*{_SENSITIVE_PUBLIC_MARKER}"
)
_SENSITIVE_INLINE_ASSIGNMENT = re.compile(
    rf"(?i)(?P<name>(?<![A-Za-z0-9]){_SENSITIVE_PUBLIC_NAME_TOKEN})"
    r"(?P<separator>\s*(?:=|:)\s*|\s+)"
    r"(?P<secret>(?:bearer\s+)?[^\s]+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s]+")
_BARE_SENSITIVE_ARG = re.compile(rf"(?i)^{_SENSITIVE_PUBLIC_NAME_TOKEN}$")
_EMPTY_SENSITIVE_SEPARATOR = re.compile(
    rf"(?i)^(?P<name>{_SENSITIVE_PUBLIC_NAME_TOKEN})(?P<separator>\s*(?:=|:)\s*)$"
)


def _redact_inline_workflow_secret(value: str) -> str:
    def replace_assignment(match: re.Match) -> str:
        secret = str(match.group("secret") or "")
        replacement = (
            "Bearer <redacted>"
            if secret.lower().startswith("bearer ")
            else "<redacted>"
        )
        return str(match.group("name")) + str(match.group("separator")) + replacement

    redacted = _SENSITIVE_INLINE_ASSIGNMENT.sub(replace_assignment, value)
    return _BEARER_SECRET.sub("Bearer <redacted>", redacted)


def _public_workflow_command(argv) -> str:
    # Render manifest argv without reflecting embedded or following secrets.
    # A bare sensitive option consumes its next argv item even when the secret
    # begins with "-"; completed inline assignments never create pending state.
    values = argv if isinstance(argv, list) else [argv]
    public = []
    pending = ""
    for raw in values:
        value = str(raw or "")
        if pending == "authorization" and value.lower() == "bearer":
            public.append("Bearer")
            pending = "secret"
            continue
        elif pending:
            public.append("<redacted>")
            pending = ""
            continue
        redacted = _redact_inline_workflow_secret(value)
        if redacted != value:
            public.append(redacted)
            continue
        empty_separator = _EMPTY_SENSITIVE_SEPARATOR.fullmatch(value)
        if empty_separator:
            public.append(value)
            marker = str(empty_separator.group("name")).lower().lstrip("-")
            separator = str(empty_separator.group("separator"))
            # A standalone name followed by ':' is commonly split from its
            # value by argv construction. An empty '=' assignment is already a
            # complete (empty) value and must not consume an unrelated positional.
            pending = (
                "authorization"
                if marker == "authorization" and ":" in separator
                else "secret"
                if ":" in separator
                else ""
            )
            continue
        if _BARE_SENSITIVE_ARG.fullmatch(value):
            public.append(value)
            pending = (
                "authorization"
                if value.lower().lstrip("-") == "authorization"
                else "secret"
            )
            continue
        if value.lower() == "bearer":
            public.append("Bearer")
            pending = "secret"
            continue
        public.append(_public_workflow_output_uri(value) if "://" in value else value)
    return " ".join(public)[:2000]


def _public_url_without_credentials(value: str) -> str:
    return re.sub(
        r"(?i)(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@",
        r"\g<scheme><redacted>@",
        str(value or ""),
    )


def _public_workflow_output_uri(value: str) -> str:
    """Remove URL credentials and secret-bearing suffixes from public evidence."""
    public = _public_url_without_credentials(value).split("?", 1)[0].split("#", 1)[0]
    return _redact_inline_workflow_secret(public)


def _workflow_run_steps(documents: list) -> list:
    # Project the npa.workflow run manifest into a backward-compatible execution
    # log surface. Stage cards themselves come from parse_stage_evidence_documents.
    manifest = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        if str(document.get("key") or "").endswith("/npa-workflow/manifest.json"):
            payload = document.get("payload")
            if isinstance(payload, dict):
                manifest = payload
                break
    if not manifest:
        return []
    out = []
    for step in manifest.get("steps", []) if isinstance(manifest, dict) else []:
        if not isinstance(step, dict):
            continue
        command = _public_workflow_command(step.get("argv") or [])
        outputs = step.get("outputs") or []
        output_uri = ""
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            output_uri = _public_workflow_output_uri(outputs[0].get("uri") or "")
        out.append(
            {
                "stage": str(step.get("state") or ""),
                "status": str(step.get("status") or ""),
                "returncode": step.get("returncode"),
                "iteration": step.get("iteration"),
                "command": command,
                "output_uri": output_uri,
            }
        )
    return out


def _artifact_backed_run_details(
    state: dict,
    run_id: str,
    prefix: str = "",
    *,
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
    source_selected: bool = False,
) -> dict | None:
    if not run_id:
        return None
    exact_prefix = _validated_resolved_prefix(resolved_prefix)
    try:
        s3, settings = _agent_artifact_s3_client()
        artifacts = []
        run_bucket = settings["bucket"]
        access_report = _agent_access_report()
        bucket_projects = artifact_bucket_projects(access_report)
        if resource_bucket:
            run_bucket, selected_project, exact_prefix, artifacts = (
                _load_selected_run_artifacts(
                    s3=s3,
                    settings=settings,
                    run_id=run_id,
                    resource_bucket=resource_bucket,
                    project_id=project_id,
                    resolved_prefix=exact_prefix,
                    source_selected=source_selected,
                    exclude=_discovery_exclude_roots(),
                )
            )
            if selected_project:
                bucket_projects[run_bucket] = selected_project
        elif prefix:
            effective_prefix = _artifact_discovery_prefix(settings, prefix)
            artifacts = list_artifacts(
                settings["bucket"],
                validate_run_id(run_id),
                prefix=effective_prefix,
                s3=s3,
            )
        if not artifacts and not resource_bucket:
            run_bucket, artifacts = find_run_artifacts_across_buckets(
                _agent_s3_buckets(s3, settings),
                base_prefix=settings.get("prefix", ""),
                run_id=validate_run_id(run_id),
                s3=s3,
            )
    except HTTPException:
        raise
    except (
        ArtifactDiscoveryError,
        ClientError,
        BotoCoreError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None
    if not artifacts:
        return None
    run_root_key = _run_root_key(artifacts, run_id)
    run_suffix = f"/{run_id}"
    derived_prefix = (
        run_root_key[: -len(run_suffix)] if run_root_key.endswith(run_suffix) else ""
    )
    effective_prefix = (
        exact_prefix
        if resource_bucket
        else derived_prefix or settings.get("prefix", "")
    )
    try:
        (
            visible_artifacts,
            report_artifact,
            preferred,
            authority_keys,
            publication_snapshot,
        ) = _committed_publication_artifacts(
            s3,
            run_bucket,
            run_id,
            artifacts,
            include_snapshot=True,
        )
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc
    evidence_documents = _stage_evidence_documents(s3, run_bucket, visible_artifacts)
    parsed_evidence = parse_stage_evidence_documents(evidence_documents)
    workflow_steps = _workflow_run_steps(evidence_documents)
    stages = build_artifact_backed_stages(
        authority_keys,
        run_id=run_id,
        prefix=effective_prefix,
        workflow_stage_defs=_workflow_stage_defs_from_state(state),
        overlay_unmatched=run_owns_workflow_stage_overlay(state, run_id),
        authoritative_stages=parsed_evidence.get("stages", []),
        evidence_keys=[str(key) for key in parsed_evidence.get("consumed_sources", [])],
    )
    report_note = ""
    if report_artifact:
        read_identity: dict[str, object] = {}
        if publication_snapshot.journaled:
            canonical_report_uri = (
                f"s3://{run_bucket}/{run_root_key}/reports/sim2real-report.json"
            )
            report_target = publication_snapshot.target(canonical_report_uri)
            if report_target.immutable_uri != str(report_artifact.s3_uri or ""):
                raise HTTPException(
                    status_code=409,
                    detail="the selected publication generation is not committed",
                )
            read_identity = {
                "expected_sha256": report_target.sha256,
                "expected_size": report_target.size_bytes,
            }
        try:
            report = _read_bounded_json_object(
                s3,
                run_bucket,
                report_artifact.key,
                **read_identity,
            )
        except PublicationConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="the selected publication generation is not committed",
            ) from exc
        except (ClientError, BotoCoreError, OSError, KeyError, TypeError, ValueError):
            report = None
        if report:
            viz = report.get("visualization")
            viz = viz if isinstance(viz, dict) else {}
            outer_loop = report.get("outer_loop", {})
            decision = (
                outer_loop.get("latest_decision", {})
                if isinstance(outer_loop, dict)
                else {}
            )
            source = str(viz.get("source") or "").strip()
            success_rate = decision.get("success_rate")
            if source or success_rate is not None:
                report_note = (
                    "Report summary: visualization source="
                    + (source or "unknown")
                    + (
                        f", success_rate={success_rate}"
                        if success_rate is not None
                        else ""
                    )
                    + "."
                )
    try:
        _assert_legacy_publication_snapshot_still_unjournaled(
            s3,
            run_bucket,
            publication_snapshot,
        )
    except PublicationConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="the selected publication generation is not committed",
        ) from exc
    stage_summary = summarize_stage_evidence(stages)
    authoritative_run_status = str(parsed_evidence.get("run_status") or "").strip()
    return {
        "run_id": run_id,
        "source_type": "artifact_storage",
        "source_label": "S3 artifacts",
        "project_id": str(bucket_projects.get(run_bucket) or project_id or ""),
        "bucket": run_bucket,
        "resolved_prefix": effective_prefix,
        "workflow_name": str(parsed_evidence.get("workflow_name") or ""),
        "workflow_graph_source": str(parsed_evidence.get("graph_source") or ""),
        "status": authoritative_run_status or "status_unavailable",
        "status_label": str(
            parsed_evidence.get("run_status_label") or "Status unavailable"
        ),
        "status_source": str(parsed_evidence.get("run_status_source") or ""),
        "result": "artifacts_available",
        "submitted_at": "",
        "updated_at": str(parsed_evidence.get("updated_at") or "")
        or max(
            (str(item.last_modified or "") for item in artifacts), default=_now_iso()
        ),
        "selection": {},
        "stages": stages,
        "stage_summary": stage_summary,
        "artifact_count": len(visible_artifacts),
        "workflow_steps": workflow_steps,
        "logs": [
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": (
                    f"Observed {len(visible_artifacts)} S3 artifacts across "
                    f"{stage_summary.get('observed_stage_count', 0)} logical groups; "
                    "artifact presence does not establish execution success."
                ),
            },
            *[
                {
                    "timestamp": _now_iso(),
                    "level": "info"
                    if str(step.get("status") or "") in ("ok", "succeeded", "")
                    and step.get("returncode") in (0, None)
                    else "error",
                    "message": (
                        f"[{step.get('stage') or '?'}"
                        + (
                            f" #{step.get('iteration')}"
                            if step.get("iteration") not in (None, "")
                            else ""
                        )
                        + f"] rc={step.get('returncode')} ({step.get('status') or 'n/a'}) "
                        + f"$ {step.get('command') or ''}"
                    ),
                }
                for step in workflow_steps
            ],
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": (
                    f"Preferred viewable artifact: {preferred.key}"
                    if preferred
                    else "No preferred viewable artifact was observed."
                ),
            },
            {
                "timestamp": _now_iso(),
                "level": "info",
                "message": report_note
                or "No structured run report summary was available.",
            },
        ],
        "artifacts": [item.to_dict() for item in visible_artifacts[:25]],
    }


def _sim2real_run_details(
    state: dict,
    run_id: str = "",
    prefix: str = "",
    *,
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
    source_selected: bool = False,
) -> dict:
    latest = state.get("latest_submit", {})
    if not isinstance(latest, dict):
        latest = {}
    sim_viz = state.get("sim_viz", {})
    if not isinstance(sim_viz, dict):
        sim_viz = {}
    resolved_run_id = str(
        run_id
        or latest.get("run_id")
        or sim_viz.get("run_id")
        or state.get("active_run_id")
        or ""
    ).strip()
    history = (
        state.get("sim_viz_runs") if isinstance(state.get("sim_viz_runs"), dict) else {}
    )
    recorded = (
        history.get(resolved_run_id)
        if isinstance(history.get(resolved_run_id), dict)
        else {}
    )
    run_viz = dict(recorded)
    if not run_viz and str(sim_viz.get("run_id") or "").strip() == resolved_run_id:
        run_viz = dict(sim_viz)
    local_demo = local_demo_run_details(state, resolved_run_id, run_viz, _now_iso())
    if local_demo:
        return local_demo
    details_map = state.get("sim2real_runs")
    if not isinstance(details_map, dict):
        details_map = {}
    existing = details_map.get(resolved_run_id, {}) if resolved_run_id else {}
    details = dict(existing) if isinstance(existing, dict) else {}
    if details and isinstance(details.get("stages"), list):
        details["stages"] = coerce_authoritative_stage_evidence(
            details["stages"], source="agent_session_workflow_status"
        )
        details["stage_summary"] = summarize_stage_evidence(details["stages"])
    # A session-owned run already has its authoritative stage graph in local
    # state. Do not turn the default status poll into an all-bucket S3 search for
    # a just-submitted run that cannot have artifacts yet. Explicit source
    # selection still asks for artifact evidence, and artifact-only run IDs (no
    # local graph) continue through bounded discovery below.
    has_session_graph = bool(
        details and isinstance(details.get("stages"), list) and details.get("stages")
    )
    explicit_artifact_source = bool(
        prefix or resource_bucket or project_id or resolved_prefix or source_selected
    )
    artifact_details = None
    if explicit_artifact_source or not has_session_graph:
        artifact_details = _artifact_backed_run_details(
            state,
            resolved_run_id,
            prefix=prefix,
            resource_bucket=resource_bucket,
            project_id=project_id,
            resolved_prefix=resolved_prefix,
            source_selected=source_selected,
        )
    if artifact_details:
        authoritative_existing = [
            item
            for item in details.get("stages", [])
            if isinstance(item, dict)
            and str(
                item.get("authority")
                or (item.get("evidence") or {}).get("authority")
                or ""
            )
            == "authoritative"
        ]
        artifact_stages = [
            item
            for item in artifact_details.get("stages", [])
            if isinstance(item, dict)
        ]
        if authoritative_existing:
            artifact_details["stages"] = merge_stage_evidence(
                authoritative_existing, artifact_stages
            )
            artifact_details["stage_summary"] = summarize_stage_evidence(
                artifact_details["stages"]
            )
            if str(artifact_details.get("status") or "") == "status_unavailable":
                artifact_details["status"] = str(
                    details.get("status") or "status_unavailable"
                )
                artifact_details["status_label"] = str(
                    details.get("status_label")
                    or artifact_details.get("status_label")
                    or "Status unavailable"
                )
        details = _merge_sim2real_run_details(details, artifact_details)
    if not details:
        source_type, source_label = resolve_run_source(recorded, {}, resolved_run_id)
        details = {
            "run_id": resolved_run_id,
            "source_type": source_type,
            "source_label": source_label,
            "status": "status_unavailable",
            "status_label": "Status unavailable",
            "result": "unavailable",
            "submitted_at": "",
            "updated_at": str(recorded.get("rrd_updated_at") or ""),
            "selection": {},
            "stages": [],
            "stage_summary": summarize_stage_evidence([]),
            "logs": [
                {
                    "timestamp": _now_iso(),
                    "level": "info",
                    "message": (
                        "No authoritative workflow status or artifact-stage evidence "
                        "is available for this run."
                    ),
                }
            ],
            "artifacts": [],
        }
    details["run_id"] = resolved_run_id
    if run_viz.get("rrd_uri"):
        if str(details.get("result") or "") in {
            "",
            "unavailable",
            "recorded_not_launched",
        }:
            details["result"] = "recording_observed"
        for item in details.get("stages", []):
            if isinstance(item, dict) and item.get("id") == "stage_14_rerun_viz":
                item["artifact_count"] = max(1, int(item.get("artifact_count") or 0))
                observation = {
                    "type": "artifact_observation",
                    "source": "sim_viz_recording",
                    "authority": "observed",
                    "confidence": "high",
                    "reason": (
                        "A Rerun recording was observed; this alone does not establish "
                        "stage success."
                    ),
                    "observed_at": str(run_viz.get("rrd_updated_at") or ""),
                }
                item["observations"] = [
                    *[
                        entry
                        for entry in item.get("observations", [])
                        if isinstance(entry, dict)
                    ],
                    observation,
                ]
                if str(item.get("authority") or "") != "authoritative":
                    item["status"] = "observed_output"
                    item["status_label"] = "Observed output"
                    item["evidence"] = observation
                    item["evidence_type"] = observation["type"]
                    item["evidence_source"] = observation["source"]
                    item["authority"] = observation["authority"]
                    item["confidence"] = observation["confidence"]
                    item["diagnostic_reason"] = observation["reason"]
                    item["summary"] = observation["reason"]
    details["stage_summary"] = summarize_stage_evidence(
        [item for item in details.get("stages", []) if isinstance(item, dict)]
    )
    return details
