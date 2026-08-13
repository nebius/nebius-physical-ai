"""Typed, secret-safe capability validation for S3-compatible storage."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4


class StorageAction(str, Enum):
    HEAD = "head"
    GET = "get"
    PUT = "put"
    LIST = "list"
    DELETE = "delete"


class StorageCapabilityProfile(str, Enum):
    """Consumer contracts; cleanup is never an implicit required permission."""

    STANDARD = "standard"
    TERRAFORM_BACKEND = "terraform_backend"
    WORKFLOW_SUBMISSION = "workflow_submission"
    LEGACY_WRITE_DELETE = "legacy_write_delete"


class StorageCredentialContext(str, Enum):
    CONFIGURED = "configured"
    NEWLY_CREATED = "newly_created"
    KNOWN_INVALID_DIAGNOSTIC = "known_invalid_diagnostic"


PROFILE_ACTIONS: dict[StorageCapabilityProfile, tuple[StorageAction, ...]] = {
    StorageCapabilityProfile.STANDARD: (StorageAction.PUT, StorageAction.GET),
    StorageCapabilityProfile.TERRAFORM_BACKEND: (
        StorageAction.HEAD,
        StorageAction.GET,
        StorageAction.PUT,
        StorageAction.LIST,
    ),
    StorageCapabilityProfile.WORKFLOW_SUBMISSION: (StorageAction.PUT,),
    # Kept for callers that explicitly depend on the historical contract.
    StorageCapabilityProfile.LEGACY_WRITE_DELETE: (
        StorageAction.PUT,
        StorageAction.DELETE,
    ),
}


class StoragePhase(str, Enum):
    CONFIGURATION = "configuration"
    CLIENT_SETUP = "client_setup"
    HEAD = "head"
    WRITE = "write"
    LIST = "list"
    READ = "read"
    DELETE = "delete"
    VERIFY_DELETE = "verify_delete"


class StorageFailureKind(str, Enum):
    MISSING_CONFIGURATION = "missing_configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    ENDPOINT = "endpoint"
    SIGNING = "signing"
    CONNECTIVITY = "connectivity"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    THROTTLED = "throttled"
    SERVER = "server_error"
    MALFORMED_REQUEST = "malformed_request"
    INVALID_DATA = "invalid_data"
    UNKNOWN = "unknown"


class StorageRetryability(str, Enum):
    NEVER = "never"
    PROPAGATION = "propagation"
    TRANSIENT = "transient"


@dataclass(frozen=True)
class StorageProbeError:
    phase: StoragePhase
    kind: StorageFailureKind
    provider_code: str = ""
    status_code: int | None = None
    retryability: StorageRetryability = StorageRetryability.NEVER
    message: str = ""
    cause: BaseException | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StorageProbeResult:
    # The first six fields preserve the public constructor used by older callers.
    ok: bool
    code: str
    summary: str
    probe_key: str = ""
    cleanup_attempted: bool = False
    cleanup_succeeded: bool = False
    profile: StorageCapabilityProfile = StorageCapabilityProfile.STANDARD
    required_actions: tuple[StorageAction, ...] = ()
    attempted_actions: tuple[StorageAction, ...] = ()
    error: StorageProbeError | None = None
    cleanup_error: StorageProbeError | None = None
    retained_object: bool = False
    attempts: int = 1

    @property
    def phase(self) -> str:
        return self.error.phase.value if self.error else "complete"

    @property
    def provider_code(self) -> str:
        return self.error.provider_code if self.error else ""

    @property
    def status_code(self) -> int | None:
        return self.error.status_code if self.error else None

    @property
    def retryability(self) -> str:
        return (
            self.error.retryability.value
            if self.error
            else StorageRetryability.NEVER.value
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the stable, credential-free JSON contract."""

        def error_dict(error: StorageProbeError | None) -> dict[str, Any] | None:
            if error is None:
                return None
            return {
                "phase": error.phase.value,
                "kind": error.kind.value,
                "provider_code": error.provider_code,
                "status_code": error.status_code,
                "retryability": error.retryability.value,
                "message": error.message,
            }

        return {
            "ok": self.ok,
            "code": self.code,
            "summary": self.summary,
            "profile": self.profile.value,
            "required_actions": [action.value for action in self.required_actions],
            "attempted_actions": [action.value for action in self.attempted_actions],
            "probe_key": self.probe_key,
            "cleanup_attempted": self.cleanup_attempted,
            "cleanup_succeeded": self.cleanup_succeeded,
            "retained_object": self.retained_object,
            "attempts": self.attempts,
            "error": error_dict(self.error),
            "cleanup_error": error_dict(self.cleanup_error),
        }


@dataclass(frozen=True)
class StorageConvergencePolicy:
    max_attempts: int = 6
    initial_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 8.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if min(self.initial_delay_seconds, self.maximum_delay_seconds) < 0:
            raise ValueError("storage convergence delays cannot be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between zero and one")


def terraform_state_key(project: str, name: str) -> str:
    return f"npa/terraform-state/{project}/{name}/terraform.tfstate"


def terraform_backend_fingerprint(
    *,
    bucket: str,
    state_key: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str = "",
    region: str = "",
    addressing_style: str = "path",
) -> str:
    material = "\0".join(
        (
            bucket_name(bucket),
            str(state_key or "").lstrip("/"),
            str(endpoint_url or "").strip(),
            str(region or "").strip(),
            str(addressing_style or "path").strip(),
            str(access_key_id or ""),
            str(secret_access_key or ""),
            str(session_token or ""),
        )
    )
    return "sha256:" + hashlib.sha256(material.encode()).hexdigest()


def bucket_name(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("s3://"):
        return urlparse(cleaned).netloc
    return cleaned.split("/", 1)[0]


def _provider_context(exc: BaseException) -> tuple[str, int | None]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return "", None
    raw_error = response.get("Error")
    raw_metadata = response.get("ResponseMetadata")
    error = raw_error if isinstance(raw_error, dict) else {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    provider_code = str(error.get("Code", "") or "").strip()
    raw_status = metadata.get("HTTPStatusCode")
    try:
        status = int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        status = None
    return provider_code, status


def classify_storage_failure(
    exc: BaseException, *, phase: StoragePhase
) -> StorageProbeError:
    """Classify without copying provider messages, request IDs, or credentials."""

    provider_code, status = _provider_context(exc)
    normalized = provider_code.lower().replace("_", "").replace("-", "")
    type_name = type(exc).__name__
    kind = StorageFailureKind.UNKNOWN
    retryability = StorageRetryability.NEVER

    if normalized in {"signaturedoesnotmatch", "authorizationheadermalformed"}:
        kind = StorageFailureKind.SIGNING
    elif normalized in {
        "invalidaccesskeyid",
        "invalidtoken",
        "expiredtoken",
        "unauthenticated",
        "authfailure",
    } or status == 401:
        kind = StorageFailureKind.AUTHENTICATION
        retryability = StorageRetryability.PROPAGATION
    elif normalized in {"accessdenied", "forbidden", "permissiondenied"} or status == 403:
        kind = StorageFailureKind.AUTHORIZATION
        retryability = StorageRetryability.PROPAGATION
    elif normalized in {
        "notimplemented",
        "unsupportedheader",
        "unsupportedargument",
        "invalidrequestfeature",
    } or status == 501:
        kind = StorageFailureKind.UNSUPPORTED_CAPABILITY
    elif normalized in {"nosuchbucket", "nosuchkey", "notfound", "404"} or status == 404:
        kind = StorageFailureKind.NOT_FOUND
    elif normalized in {"slowdown", "throttling", "toomanyrequests"} or status == 429:
        kind = StorageFailureKind.THROTTLED
        retryability = StorageRetryability.TRANSIENT
    elif status is not None and 500 <= status <= 599:
        kind = StorageFailureKind.SERVER
        retryability = StorageRetryability.TRANSIENT
    elif status == 409:
        kind = StorageFailureKind.CONFLICT
    elif normalized in {"invalidargument", "invalidrequest", "malformedxml"} or status == 400:
        kind = StorageFailureKind.MALFORMED_REQUEST
    elif type_name in {"EndpointConnectionError", "InvalidEndpointURL"}:
        kind = StorageFailureKind.ENDPOINT
        retryability = StorageRetryability.TRANSIENT
    elif type_name in {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "HTTPClientError",
        "ReadTimeoutError",
        "ConnectionError",
        "TimeoutError",
    }:
        kind = StorageFailureKind.CONNECTIVITY
        retryability = StorageRetryability.TRANSIENT

    labels = {
        StorageFailureKind.AUTHENTICATION: "authentication failed",
        StorageFailureKind.AUTHORIZATION: "permission was denied",
        StorageFailureKind.UNSUPPORTED_CAPABILITY: "request capability is unsupported",
        StorageFailureKind.ENDPOINT: "endpoint configuration failed",
        StorageFailureKind.SIGNING: "request signing failed",
        StorageFailureKind.CONNECTIVITY: "endpoint could not be reached",
        StorageFailureKind.NOT_FOUND: "object or bucket was not found",
        StorageFailureKind.CONFLICT: "request conflicted with provider state",
        StorageFailureKind.THROTTLED: "request was throttled",
        StorageFailureKind.SERVER: "provider returned a server error",
        StorageFailureKind.MALFORMED_REQUEST: "request was malformed",
        StorageFailureKind.INVALID_DATA: "returned data was invalid",
        StorageFailureKind.UNKNOWN: "request failed",
        StorageFailureKind.MISSING_CONFIGURATION: "configuration is incomplete",
    }
    return StorageProbeError(
        phase=phase,
        kind=kind,
        provider_code=provider_code,
        status_code=status,
        retryability=retryability,
        message=f"S3 {phase.value} {labels[kind]}.",
        cause=exc,
    )


def _failure_result(
    error: StorageProbeError,
    *,
    profile: StorageCapabilityProfile,
    key: str = "",
    actions: tuple[StorageAction, ...] = (),
) -> StorageProbeResult:
    legacy_code = {
        StorageFailureKind.AUTHENTICATION: "forbidden",
        StorageFailureKind.AUTHORIZATION: "forbidden",
        StorageFailureKind.ENDPOINT: "endpoint_unreachable",
        StorageFailureKind.CONNECTIVITY: "endpoint_unreachable",
        StorageFailureKind.NOT_FOUND: "bucket_unreachable",
    }.get(error.kind, error.kind.value)
    return StorageProbeResult(
        False,
        legacy_code,
        error.message,
        probe_key=key,
        profile=profile,
        required_actions=PROFILE_ACTIONS[profile],
        attempted_actions=actions,
        error=error,
    )


def _apply_credential_context(
    error: StorageProbeError, context: StorageCredentialContext
) -> StorageProbeError:
    """Use caller-proven request context when a provider returns ambiguous 403."""

    if (
        context is StorageCredentialContext.KNOWN_INVALID_DIAGNOSTIC
        and error.kind
        in {StorageFailureKind.AUTHENTICATION, StorageFailureKind.AUTHORIZATION}
    ):
        return replace(
            error,
            kind=StorageFailureKind.AUTHENTICATION,
            retryability=StorageRetryability.NEVER,
            message="S3 authentication failed for the intentionally invalid diagnostic credential.",
        )
    return error


def _missing_result(
    missing: list[str], *, profile: StorageCapabilityProfile, backend: bool = False
) -> StorageProbeResult:
    message = ("Terraform backend" if backend else "S3 storage") + " is not configured; missing " + ", ".join(missing) + "."
    error = StorageProbeError(
        StoragePhase.CONFIGURATION,
        StorageFailureKind.MISSING_CONFIGURATION,
        message=message,
    )
    return _failure_result(error, profile=profile)


def _is_not_found(exc: BaseException) -> bool:
    return classify_storage_failure(exc, phase=StoragePhase.HEAD).kind is StorageFailureKind.NOT_FOUND


def _storage_client(
    *, endpoint: str, access: str, secret: str, session_token: str, region: str,
    addressing_style: str,
) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=session_token or None,
        region_name=region or None,
        config=Config(s3={"addressing_style": addressing_style or "path"}),
    )


def _cleanup_probe(
    client: Any, *, bucket: str, key: str
) -> tuple[bool, StorageProbeError | None]:
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - provider SDK exception families vary
        return False, classify_storage_failure(exc, phase=StoragePhase.DELETE)
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - expected NotFound
        error = classify_storage_failure(exc, phase=StoragePhase.VERIFY_DELETE)
        if error.kind is StorageFailureKind.NOT_FOUND:
            return True, None
        return False, error
    error = StorageProbeError(
        StoragePhase.VERIFY_DELETE,
        StorageFailureKind.CONFLICT,
        message="S3 delete returned success but the probe object remains.",
    )
    return False, error


def _with_cleanup(
    primary: StorageProbeResult,
    *, client: Any,
    bucket: str,
    key: str,
) -> StorageProbeResult:
    succeeded, cleanup_error = _cleanup_probe(client, bucket=bucket, key=key)
    # Cleanup is independent. Only the explicit legacy contract makes deletion a
    # required capability; otherwise retain the primary write/read diagnosis.
    ok = primary.ok
    code = primary.code
    summary = primary.summary
    error = primary.error
    if (
        primary.profile is StorageCapabilityProfile.LEGACY_WRITE_DELETE
        and not succeeded
        and primary.ok
    ):
        ok = False
        code = "cleanup_failed"
        summary = cleanup_error.message if cleanup_error else "S3 cleanup failed."
        error = cleanup_error
    elif not succeeded:
        summary = (
            primary.summary
            + f" Probe cleanup was not authorized or could not be verified; the "
            f"unique object remains at s3://{bucket}/{key}. DeleteObject is not a "
            "required capability for this consumer."
        )
    return replace(
        primary,
        ok=ok,
        code=code,
        summary=summary,
        error=error,
        cleanup_attempted=True,
        cleanup_succeeded=succeeded,
        cleanup_error=cleanup_error,
        retained_object=not succeeded,
        attempted_actions=primary.attempted_actions + (StorageAction.DELETE,),
    )


def probe_terraform_backend(
    *,
    bucket: str,
    state_key: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str = "",
    region: str = "",
    addressing_style: str = "path",
    client: Any | None = None,
    key_factory: Callable[[], str] | None = None,
) -> StorageProbeResult:
    """Verify Terraform's exact state prefix using an unconditional unique object."""

    profile = StorageCapabilityProfile.TERRAFORM_BACKEND
    name = bucket_name(bucket)
    key = str(state_key or "").strip().lstrip("/")
    endpoint = str(endpoint_url or "").strip()
    access = str(access_key_id or "").strip()
    secret = str(secret_access_key or "").strip()
    missing = [label for label, value in (("bucket", name), ("state_key", key), ("endpoint", endpoint), ("AWS_ACCESS_KEY_ID", access), ("AWS_SECRET_ACCESS_KEY", secret)) if not value]
    if missing:
        return _missing_result(missing, profile=profile, backend=True)
    if client is None:
        try:
            client = _storage_client(endpoint=endpoint, access=access, secret=secret, session_token=str(session_token or "").strip(), region=str(region or "").strip(), addressing_style=str(addressing_style or "path").strip())
        except Exception as exc:  # noqa: BLE001
            return _failure_result(classify_storage_failure(exc, phase=StoragePhase.CLIENT_SETUP), profile=profile)

    actions: tuple[StorageAction, ...] = (StorageAction.HEAD,)
    existing_state = False
    try:
        client.head_object(Bucket=name, Key=key)
    except Exception as exc:  # noqa: BLE001
        if not _is_not_found(exc):
            return _failure_result(classify_storage_failure(exc, phase=StoragePhase.HEAD), profile=profile, actions=actions)
    else:
        existing_state = True
        actions += (StorageAction.GET,)
        try:
            response = client.get_object(Bucket=name, Key=key)
            body = response.get("Body") if isinstance(response, dict) else None
            read = getattr(body, "read", None)
            parsed = json.loads((read() if callable(read) else body) or b"")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("version"), int):
                raise ValueError("invalid Terraform state shape")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            error = StorageProbeError(StoragePhase.READ, StorageFailureKind.INVALID_DATA, message="Terraform backend state exists but is not a readable Terraform JSON document.", cause=exc)
            return _failure_result(error, profile=profile, actions=actions)
        except Exception as exc:  # noqa: BLE001
            return _failure_result(classify_storage_failure(exc, phase=StoragePhase.READ), profile=profile, actions=actions)

    prefix = key.rsplit("/", 1)[0] if "/" in key else ""
    token = (key_factory or (lambda: uuid4().hex))()
    probe_key = "/".join(part for part in (prefix, ".npa-probes", f"backend-{token}.tmp") if part)
    created = False
    primary: StorageProbeResult
    try:
        actions += (StorageAction.PUT,)
        # UUID ownership makes collisions negligible and avoids relying on the
        # provider-specific If-None-Match implementation.
        client.put_object(Bucket=name, Key=probe_key, Body=b"npa-backend-probe-v2")
        created = True
        actions += (StorageAction.LIST,)
        listing = client.list_objects_v2(Bucket=name, Prefix=probe_key, MaxKeys=1)
        listed = {str(item.get("Key") or "") for item in (listing.get("Contents") or []) if isinstance(item, dict)}
        if probe_key not in listed:
            error = StorageProbeError(StoragePhase.LIST, StorageFailureKind.INVALID_DATA, message="Terraform backend listing did not return its unique probe object.")
            primary = _failure_result(error, profile=profile, key=probe_key, actions=actions)
        else:
            actions += (StorageAction.GET,)
            response = client.get_object(Bucket=name, Key=probe_key)
            body = response.get("Body") if isinstance(response, dict) else None
            read = getattr(body, "read", None)
            raw = read() if callable(read) else body
            if raw != b"npa-backend-probe-v2":
                error = StorageProbeError(StoragePhase.READ, StorageFailureKind.INVALID_DATA, message="Terraform backend probe did not round-trip exactly.")
                primary = _failure_result(error, profile=profile, key=probe_key, actions=actions)
            else:
                code = "existing_state_valid" if existing_state else "new_state_prefix_valid"
                primary = StorageProbeResult(True, code, "Terraform backend create/list/read capabilities are verified.", probe_key=probe_key, profile=profile, required_actions=PROFILE_ACTIONS[profile], attempted_actions=actions)
    except Exception as exc:  # noqa: BLE001
        phase = StoragePhase.WRITE if not created else (StoragePhase.LIST if StorageAction.LIST in actions and StorageAction.GET not in actions[-1:] else StoragePhase.READ)
        primary = _failure_result(classify_storage_failure(exc, phase=phase), profile=profile, key=probe_key, actions=actions)
    if created:
        return _with_cleanup(primary, client=client, bucket=name, key=probe_key)
    return primary


def probe_storage_write(
    *,
    bucket: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region: str = "",
    prefix: str = "",
    client: Any | None = None,
    profile: StorageCapabilityProfile = StorageCapabilityProfile.STANDARD,
    credential_context: StorageCredentialContext = StorageCredentialContext.CONFIGURED,
    key_factory: Callable[[], str] | None = None,
) -> StorageProbeResult:
    """Probe only a consumer profile's declared actions."""

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("probe_storage_write")

    name = bucket_name(bucket)
    endpoint = str(endpoint_url or "").strip()
    access = str(access_key_id or "").strip()
    secret = str(secret_access_key or "").strip()
    missing = [label for label, value in (("bucket", name), ("endpoint", endpoint), ("AWS_ACCESS_KEY_ID", access), ("AWS_SECRET_ACCESS_KEY", secret)) if not value]
    if missing:
        return _missing_result(missing, profile=profile)
    if client is None:
        try:
            client = _storage_client(endpoint=endpoint, access=access, secret=secret, session_token="", region=str(region or "").strip(), addressing_style="path")
        except Exception as exc:  # noqa: BLE001
            return _failure_result(classify_storage_failure(exc, phase=StoragePhase.CLIENT_SETUP), profile=profile)

    clean_prefix = str(prefix or "").strip().strip("/")
    token = (key_factory or (lambda: uuid4().hex))()
    key = "/".join(part for part in (clean_prefix, ".npa-probes", f"write-{token}.tmp") if part)
    payload = b"npa-storage-probe-v2"
    actions: tuple[StorageAction, ...] = (StorageAction.PUT,)
    try:
        client.put_object(Bucket=name, Key=key, Body=payload)
    except Exception as exc:  # noqa: BLE001
        error = classify_storage_failure(exc, phase=StoragePhase.WRITE)
        error = _apply_credential_context(error, credential_context)
        return _failure_result(error, profile=profile, key=key, actions=actions)

    primary = StorageProbeResult(True, "ok", "Declared S3 capabilities are verified.", probe_key=key, profile=profile, required_actions=PROFILE_ACTIONS[profile], attempted_actions=actions)
    if StorageAction.GET in PROFILE_ACTIONS[profile]:
        actions += (StorageAction.GET,)
        try:
            response = client.get_object(Bucket=name, Key=key)
            body = response.get("Body") if isinstance(response, dict) else None
            read = getattr(body, "read", None)
            raw = read() if callable(read) else body
            if raw != payload:
                raise ValueError("probe payload mismatch")
        except (ValueError, TypeError) as exc:
            error = StorageProbeError(StoragePhase.READ, StorageFailureKind.INVALID_DATA, message="S3 probe did not round-trip exactly.", cause=exc)
            primary = _failure_result(error, profile=profile, key=key, actions=actions)
        except Exception as exc:  # noqa: BLE001
            primary = _failure_result(classify_storage_failure(exc, phase=StoragePhase.READ), profile=profile, key=key, actions=actions)
        else:
            primary = replace(primary, attempted_actions=actions)

    if profile is StorageCapabilityProfile.WORKFLOW_SUBMISSION:
        return replace(primary, retained_object=True, summary=primary.summary + " Append-only profile leaves its uniquely named probe object intact.")
    return _with_cleanup(primary, client=client, bucket=name, key=key)


def converge_storage_probe(
    probe: Callable[[], StorageProbeResult],
    *,
    propagation_context: bool,
    policy: StorageConvergencePolicy = StorageConvergencePolicy(),
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    on_retry: Callable[[StorageProbeResult, float], None] | None = None,
) -> StorageProbeResult:
    """Retry only typed propagation/transient failures with capped jitter."""

    result = probe()
    for attempt in range(1, policy.max_attempts + 1):
        result = replace(result, attempts=attempt)
        if result.ok:
            return result
        error = result.error
        if error is None:
            return result
        allowed = error.retryability is StorageRetryability.TRANSIENT or (
            propagation_context
            and error.retryability is StorageRetryability.PROPAGATION
        )
        if not allowed or attempt >= policy.max_attempts:
            if (
                propagation_context
                and error.retryability is StorageRetryability.PROPAGATION
            ):
                terminal = replace(
                    error,
                    retryability=StorageRetryability.NEVER,
                    message=(
                        "S3 credential/IAM propagation did not converge; verify the "
                        f"required {error.phase.value} action and assigned role, then resume."
                    ),
                )
                result = replace(result, error=terminal, summary=terminal.message)
            elif error.retryability is StorageRetryability.PROPAGATION:
                # A configured/established identity has no fresh-grant evidence.
                # Preserve the authentication/authorization diagnosis and make
                # its terminal routing explicit instead of inventing propagation.
                result = replace(
                    result,
                    error=replace(error, retryability=StorageRetryability.NEVER),
                )
            return result
        base = min(
            policy.maximum_delay_seconds,
            policy.initial_delay_seconds * (2 ** (attempt - 1)),
        )
        jitter = base * policy.jitter_ratio * ((2 * random_value()) - 1)
        delay = max(0.0, base + jitter)
        if on_retry:
            on_retry(result, delay)
        sleep(delay)
        result = probe()
    return result
