from __future__ import annotations

from dataclasses import dataclass

import pytest

from npa.workflows.credential_preflight import (
    CREDENTIAL_CHECKS,
    CredentialProbes,
    check_hf,
    check_ngc,
    check_s3,
    check_token_factory,
    has_failure,
    run_credential_preflight,
)
from npa.workflows.sim2real_health import FAIL, PASS, WARN


@dataclass
class _Creds:
    hf_token: str = ""
    ngc_api_key: str = ""
    token_factory_api_key: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_endpoint: str = ""
    s3_bucket: str = ""


@dataclass
class _HFResult:
    ok: bool
    status_code: int | None = None
    error: str = ""


def test_hf_warns_when_missing() -> None:
    result = check_hf(_Creds(), CredentialProbes())
    assert result.status == WARN
    assert "HF_TOKEN" in result.summary


def test_hf_present_unverified_when_no_probe() -> None:
    result = check_hf(_Creds(hf_token="hf_x"), CredentialProbes())
    assert result.status == PASS
    assert "not verified" in result.summary


def test_hf_pass_when_validator_ok() -> None:
    probes = CredentialProbes(hf_validator=lambda token, repo: _HFResult(ok=True))
    result = check_hf(_Creds(hf_token="hf_x"), probes)
    assert result.status == PASS


def test_hf_fail_on_auth_rejection() -> None:
    probes = CredentialProbes(
        hf_validator=lambda token, repo: _HFResult(ok=False, status_code=401, error="denied")
    )
    result = check_hf(_Creds(hf_token="hf_bad"), probes)
    assert result.status == FAIL
    assert "denied" in result.details[0]


def test_hf_warn_on_network_error_not_fail() -> None:
    probes = CredentialProbes(
        hf_validator=lambda token, repo: _HFResult(ok=False, status_code=None, error="conn reset")
    )
    result = check_hf(_Creds(hf_token="hf_x"), probes)
    assert result.status == WARN


def test_ngc_warns_when_missing() -> None:
    assert check_ngc(_Creds(), CredentialProbes()).status == WARN


def test_ngc_warns_on_bad_prefix() -> None:
    result = check_ngc(_Creds(ngc_api_key="not-a-key"), CredentialProbes())
    assert result.status == WARN
    assert "nvapi-" in result.remedy


def test_ngc_pass_with_hyphen_key() -> None:
    # Real personal NGC keys are prefixed 'nvapi-'.
    assert check_ngc(_Creds(ngc_api_key="nvapi-abc123"), CredentialProbes()).status == PASS


def test_ngc_pass_with_underscore_key() -> None:
    # Older docs sometimes show 'nvapi_'; accept it too.
    assert check_ngc(_Creds(ngc_api_key="nvapi_abc"), CredentialProbes()).status == PASS


def test_s3_warns_without_keys() -> None:
    assert check_s3(_Creds(), CredentialProbes()).status == WARN


def test_s3_present_unverified_without_probe() -> None:
    creds = _Creds(
        s3_access_key_id="AK",
        s3_secret_access_key="SK",
        s3_endpoint="https://storage.eu-north1.nebius.cloud",
        s3_bucket="s3://bkt/",
    )
    result = check_s3(creds, CredentialProbes())
    assert result.status == PASS
    assert "not probed" in result.summary


def test_s3_pass_when_reachable() -> None:
    class _Client:
        def list_checkpoints(self, uri):
            return []

    creds = _Creds(
        s3_access_key_id="AK",
        s3_secret_access_key="SK",
        s3_endpoint="https://storage.eu-north1.nebius.cloud",
        s3_bucket="s3://bkt/",
    )
    probes = CredentialProbes(s3_client_factory=lambda: _Client())
    assert check_s3(creds, probes).status == PASS


def test_s3_fail_on_auth_error() -> None:
    class _Client:
        def list_checkpoints(self, uri):
            raise RuntimeError("403 Forbidden AccessDenied")

    creds = _Creds(
        s3_access_key_id="AK",
        s3_secret_access_key="SK",
        s3_endpoint="https://storage.eu-north1.nebius.cloud",
        s3_bucket="s3://bkt/",
    )
    probes = CredentialProbes(s3_client_factory=lambda: _Client())
    result = check_s3(creds, probes)
    assert result.status == FAIL
    assert "access key" in result.remedy.lower()


def test_token_factory_warns_when_missing() -> None:
    result = check_token_factory(_Creds(), CredentialProbes())
    assert result.status == WARN
    assert "TOKEN_FACTORY" in result.summary


def test_token_factory_pass_when_authenticated() -> None:
    probes = CredentialProbes(token_factory_verifier=lambda: ["m1", "m2"])
    result = check_token_factory(_Creds(token_factory_api_key="v1.abc"), probes)
    assert result.status == PASS
    assert "2 models" in result.summary


def test_token_factory_fail_when_verifier_raises() -> None:
    def _boom() -> list[str]:
        raise RuntimeError("401 unauthorized")

    probes = CredentialProbes(token_factory_verifier=_boom)
    result = check_token_factory(_Creds(token_factory_api_key="v1.bad"), probes)
    assert result.status == FAIL


def test_run_credential_preflight_default_order() -> None:
    results = run_credential_preflight(_Creds(), probes=CredentialProbes())
    assert [r.name for r in results] == list(CREDENTIAL_CHECKS)


def test_run_credential_preflight_rejects_unknown_check() -> None:
    with pytest.raises(ValueError):
        run_credential_preflight(_Creds(), checks=["bogus"])


def test_has_failure_true_when_any_fail() -> None:
    class _Client:
        def list_checkpoints(self, uri):
            raise RuntimeError("403")

    creds = _Creds(
        s3_access_key_id="AK",
        s3_secret_access_key="SK",
        s3_endpoint="https://x",
        s3_bucket="s3://bkt/",
    )
    results = run_credential_preflight(
        creds,
        probes=CredentialProbes(s3_client_factory=lambda: _Client()),
        checks=["s3"],
    )
    assert has_failure(results) is True
