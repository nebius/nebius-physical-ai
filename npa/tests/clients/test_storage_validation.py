from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from npa.clients.storage_validation import (
    StorageCapabilityProfile,
    StorageConvergencePolicy,
    StorageCredentialContext,
    StorageFailureKind,
    StoragePhase,
    StorageRetryability,
    classify_storage_failure,
    converge_storage_probe,
    probe_storage_write,
    probe_terraform_backend,
)


class ProviderError(Exception):
    def __init__(self, code: str, status: int, message: str = "") -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class EndpointConnectionError(Exception):
    pass


class ReadTimeoutError(Exception):
    pass


class FakeS3:
    def __init__(self, *, errors: dict[str, list[Exception]] | None = None) -> None:
        self.errors = {name: list(values) for name, values in (errors or {}).items()}
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict]] = []

    def _call(self, name: str, kwargs: dict) -> None:
        self.calls.append((name, kwargs))
        failures = self.errors.get(name, [])
        if failures:
            raise failures.pop(0)

    def put_object(self, **kwargs):
        self._call("put", kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def get_object(self, **kwargs):
        self._call("get", kwargs)
        key = kwargs["Key"]
        if key not in self.objects:
            raise ProviderError("NoSuchKey", 404)
        return {"Body": io.BytesIO(self.objects[key])}

    def list_objects_v2(self, **kwargs):
        self._call("list", kwargs)
        prefix = kwargs["Prefix"]
        return {"Contents": [{"Key": key} for key in self.objects if key.startswith(prefix)]}

    def delete_object(self, **kwargs):
        self._call("delete", kwargs)
        self.objects.pop(kwargs["Key"], None)

    def head_object(self, **kwargs):
        self._call("head", kwargs)
        if kwargs["Key"] not in self.objects:
            raise ProviderError("NoSuchKey", 404)
        return {}


def _probe(client: FakeS3, **kwargs):
    return probe_storage_write(
        bucket="bucket",
        endpoint_url="https://storage.example",
        access_key_id="access",
        secret_access_key="secret",
        client=client,
        **kwargs,
    )


def test_probe_requires_each_storage_prerequisite_without_using_a_client() -> None:
    result = probe_storage_write(
        bucket="", endpoint_url="", access_key_id="", secret_access_key=""
    )
    assert not result.ok
    assert result.error.kind is StorageFailureKind.MISSING_CONFIGURATION
    assert "bucket, endpoint, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY" in result.summary


def test_standard_profile_round_trips_and_reports_optional_cleanup() -> None:
    client = FakeS3()
    result = _probe(client, prefix="checks", key_factory=lambda: "unique")

    assert result.ok and result.cleanup_succeeded
    assert [name for name, _ in client.calls] == ["put", "get", "delete", "head"]
    assert result.required_actions == tuple(
        map(type(result.required_actions[0]), ("put", "get"))
    )
    assert result.probe_key == "checks/.npa-probes/write-unique.tmp"
    assert json.loads(json.dumps(result.as_dict()))["error"] is None


@pytest.mark.parametrize(
    ("code", "status", "kind", "retryability"),
    [
        ("InvalidRequest", 400, StorageFailureKind.MALFORMED_REQUEST, StorageRetryability.NEVER),
        ("InvalidAccessKeyId", 401, StorageFailureKind.AUTHENTICATION, StorageRetryability.PROPAGATION),
        ("AccessDenied", 403, StorageFailureKind.AUTHORIZATION, StorageRetryability.PROPAGATION),
        ("NoSuchBucket", 404, StorageFailureKind.NOT_FOUND, StorageRetryability.NEVER),
        ("Conflict", 409, StorageFailureKind.CONFLICT, StorageRetryability.NEVER),
        ("TooManyRequests", 429, StorageFailureKind.THROTTLED, StorageRetryability.TRANSIENT),
        ("InternalError", 500, StorageFailureKind.SERVER, StorageRetryability.TRANSIENT),
        ("ServiceUnavailable", 503, StorageFailureKind.SERVER, StorageRetryability.TRANSIENT),
        ("UnsupportedHeader", 400, StorageFailureKind.UNSUPPORTED_CAPABILITY, StorageRetryability.NEVER),
        ("SignatureDoesNotMatch", 403, StorageFailureKind.SIGNING, StorageRetryability.NEVER),
    ],
)
def test_typed_provider_classification_matrix(code, status, kind, retryability) -> None:
    error = classify_storage_failure(
        ProviderError(code, status, "NPA_SECRET_PROVIDER_CANARY"),
        phase=StoragePhase.WRITE,
    )
    assert error.kind is kind
    assert error.retryability is retryability
    assert error.status_code == status
    assert error.provider_code == code
    assert "CANARY" not in error.message
    assert "CANARY" not in repr(error)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (EndpointConnectionError("secret"), StorageFailureKind.ENDPOINT),
        (ReadTimeoutError("secret"), StorageFailureKind.CONNECTIVITY),
    ],
)
def test_endpoint_and_network_exceptions_are_typed_and_redacted(error, kind) -> None:
    result = _probe(FakeS3(errors={"put": [error]}))
    assert result.error.kind is kind
    assert result.error.retryability is StorageRetryability.TRANSIENT
    assert "secret" not in json.dumps(result.as_dict())


def test_delete_denial_is_independent_and_does_not_erase_primary_success() -> None:
    client = FakeS3(errors={"delete": [ProviderError("AccessDenied", 403)]})
    result = _probe(client)

    assert result.ok
    assert result.error is None
    assert result.cleanup_error.kind is StorageFailureKind.AUTHORIZATION
    assert result.cleanup_error.phase is StoragePhase.DELETE
    assert result.retained_object and not result.cleanup_succeeded


def test_primary_read_diagnosis_survives_cleanup_failure() -> None:
    client = FakeS3(
        errors={
            "get": [ProviderError("AccessDenied", 403)],
            "delete": [ProviderError("AccessDenied", 403)],
        }
    )
    result = _probe(client)

    assert not result.ok
    assert result.error.phase is StoragePhase.READ
    assert result.cleanup_error.phase is StoragePhase.DELETE
    assert result.summary.startswith(result.error.message)
    assert "Probe cleanup" in result.summary


def test_explicit_legacy_profile_preserves_required_delete_contract() -> None:
    client = FakeS3(errors={"delete": [ProviderError("AccessDenied", 403)]})
    result = _probe(client, profile=StorageCapabilityProfile.LEGACY_WRITE_DELETE)
    assert not result.ok
    assert result.code == "cleanup_failed"
    assert result.error.phase is StoragePhase.DELETE


def test_workflow_append_only_profile_calls_only_put() -> None:
    client = FakeS3()
    result = _probe(client, profile=StorageCapabilityProfile.WORKFLOW_SUBMISSION)
    assert result.ok and result.retained_object
    assert [name for name, _ in client.calls] == ["put"]
    assert [action.value for action in result.required_actions] == ["put"]


def test_new_credential_transient_403_converges_without_real_sleep() -> None:
    client = FakeS3(errors={"put": [ProviderError("AccessDenied", 403)]})
    delays: list[float] = []
    result = converge_storage_probe(
        lambda: _probe(client, key_factory=lambda: "stable-probe"),
        propagation_context=True,
        policy=StorageConvergencePolicy(max_attempts=3, initial_delay_seconds=1, jitter_ratio=0),
        sleep=delays.append,
    )
    assert result.ok and result.attempts == 2
    assert delays == [1]


def test_terminal_signature_failure_is_never_retried() -> None:
    client = FakeS3(errors={"put": [ProviderError("SignatureDoesNotMatch", 403)]})
    sleeps: list[float] = []
    result = converge_storage_probe(
        lambda: _probe(client), propagation_context=True, sleep=sleeps.append
    )
    assert not result.ok and result.error.kind is StorageFailureKind.SIGNING
    assert result.attempts == 1 and sleeps == []


@pytest.mark.parametrize(
    ("provider_code", "status"),
    [("AccessDenied", 403), ("InvalidAccessKeyId", 401)],
)
def test_known_invalid_request_context_is_terminal_authentication(
    provider_code: str, status: int
) -> None:
    result = _probe(
        FakeS3(errors={"put": [ProviderError(provider_code, status)]}),
        credential_context=StorageCredentialContext.KNOWN_INVALID_DIAGNOSTIC,
    )
    assert result.error.kind is StorageFailureKind.AUTHENTICATION
    assert result.error.retryability is StorageRetryability.NEVER
    assert "intentionally invalid diagnostic credential" in result.summary


def test_propagation_exhaustion_routes_to_actionable_typed_guidance() -> None:
    client = FakeS3(errors={"put": [ProviderError("AccessDenied", 403)] * 2})
    result = converge_storage_probe(
        lambda: _probe(client),
        propagation_context=True,
        policy=StorageConvergencePolicy(max_attempts=2, initial_delay_seconds=0),
        sleep=lambda _delay: None,
    )
    assert result.error.kind is StorageFailureKind.AUTHORIZATION
    assert result.error.retryability is StorageRetryability.NEVER
    assert "required write action" in result.summary


def test_configured_terminal_deny_does_not_invent_propagation_guidance() -> None:
    result = converge_storage_probe(
        lambda: _probe(FakeS3(errors={"put": [ProviderError("AccessDenied", 403)]})),
        propagation_context=False,
        sleep=lambda _delay: pytest.fail("terminal deny must not sleep"),
    )

    assert result.error.kind is StorageFailureKind.AUTHORIZATION
    assert result.error.retryability is StorageRetryability.NEVER
    assert result.summary == "S3 write permission was denied."


def test_terraform_probe_uses_unique_unconditional_objects_concurrently() -> None:
    client = FakeS3()

    def run(token: str):
        return probe_terraform_backend(
            bucket="bucket",
            state_key="npa/terraform-state/demo/default/terraform.tfstate",
            endpoint_url="https://storage.example",
            access_key_id="access",
            secret_access_key="secret",
            client=client,
            key_factory=lambda: token,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("one", "two")))

    assert all(result.ok and result.cleanup_succeeded for result in results)
    assert len({result.probe_key for result in results}) == 2
    put_calls = [kwargs for name, kwargs in client.calls if name == "put"]
    assert all("IfNoneMatch" not in call for call in put_calls)
    assert client.objects == {}
