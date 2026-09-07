"""Unit tests for the import-light Cosmos3 native Ray Serve client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from npa.workbench.cosmos.ray_serve import (
    Cosmos3RayServeError,
    RayBatchRequest,
    load_batch_request,
    submit_batch,
)


def _batch(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "Cosmos3-Nano",
                "samples": [
                    {"name": "one", "model_mode": "text2image", "prompt": "red cube"},
                    {"name": "two", "model_mode": "text2image", "prompt": "blue cube"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_batch_requires_unique_named_samples() -> None:
    with pytest.raises(ValueError, match="duplicate sample"):
        RayBatchRequest(samples=[{"name": "same"}, {"name": "same"}])
    with pytest.raises(ValueError, match="name is required"):
        RayBatchRequest(samples=[{"prompt": "missing"}])
    with pytest.raises(ValueError, match="must use only"):
        RayBatchRequest(samples=[{"name": "../../escape"}])


@pytest.mark.parametrize("mode", ["", "none", "disabled", "false"])
def test_server_rejects_disabled_management_auth_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from npa.workbench.cosmos.ray_server import main

    monkeypatch.setenv("RAY_AUTH_MODE", mode)
    with pytest.raises(RuntimeError, match="requires RAY_AUTH_MODE=token"):
        main()


def test_server_enables_management_auth_without_reusing_application_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from npa.workbench.cosmos.ray_server import _require_ray_authentication

    monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    monkeypatch.delenv("RAY_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAY_AUTH_TOKEN_PATH", "operator-token-file")
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "application-test-credential")
    token_file = _require_ray_authentication()
    try:
        assert os.environ["RAY_AUTH_MODE"] == "token"
        assert "RAY_AUTH_TOKEN" not in os.environ
        assert os.environ["RAY_AUTH_TOKEN_PATH"] == str(token_file)
        assert token_file.stat().st_mode & 0o777 == 0o600
        assert len(token_file.read_bytes()) >= 32
        assert token_file.read_text() != "application-test-credential"
    finally:
        token_file.unlink()


def test_server_removes_management_credential_when_runtime_fails(monkeypatch, tmp_path):
    from npa.workbench.cosmos import ray_server

    credential = tmp_path / "credential"
    credential.write_text("private test credential")
    monkeypatch.setattr(ray_server, "_require_ray_authentication", lambda: credential)

    def fail():
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(ray_server, "_run_server", fail)
    with pytest.raises(RuntimeError, match="runtime failed"):
        ray_server.main()
    assert not credential.exists()


def test_load_batch_accepts_list_shorthand(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text('[{"name":"one","prompt":"cube"}]', encoding="utf-8")
    assert load_batch_request(str(path)).samples[0]["name"] == "one"


def test_dry_run_is_import_light_and_keeps_guardrail_posture_server_owned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    result = submit_batch(
        input_path=str(source),
        output_path=str(tmp_path / "out"),
        endpoint="http://service.invalid:8000",
        dry_run=True,
    )
    assert result["status"] == "planned"
    assert result["batch_size"] == 2
    assert result["backend"] == "cosmos-framework-native-ray-serve"
    assert result["weights_baked"] is False


def test_submit_downloads_hash_checks_and_persists_structured_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "batch.json"
    destination = tmp_path / "published"
    _batch(source)
    media = b"synthetic-image-bytes"
    digest = hashlib.sha256(media).hexdigest()

    def request(method: str, url: str, **_kwargs: object) -> httpx.Response:
        request_obj = httpx.Request(method, url)
        payload = {
            "schema_version": "npa.cosmos3.ray-serve.batch.v1",
            "request_id": "request-safe",
            "model": "Cosmos3-Nano",
            "batch_size": 2,
            "outputs": [
                {"args": {"name": "one"}, "status": "success", "outputs": []},
                {"args": {"name": "two"}, "status": "success", "outputs": []},
            ],
            "artifacts": [
                {
                    "sample": "one",
                    "path": "request-safe/one/image.jpg",
                    "bytes": len(media),
                    "sha256": digest,
                }
            ],
            "guardrails": True,
            "max_batch_size": 4,
            "framework_revision": "5e67049cd94acb667786f1e6dd0dab821cb90c97",
            "server_source_revision": "a" * 40,
        }
        return httpx.Response(200, request=request_obj, json=payload)

    def get(url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request("GET", url), content=media)

    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "secret-for-test")
    result = submit_batch(
        input_path=str(source),
        output_path=str(destination),
        endpoint="http://service.invalid:8000",
    )
    assert result["status"] == "completed"
    assert result["guardrails"] is True
    assert (destination / "artifacts/request-safe/one/image.jpg").read_bytes() == media
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["structured_outputs"][0]["args"]["name"] == "one"


def test_submit_rejects_artifact_integrity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)

    def request(method: str, url: str, **_kwargs: object) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request(method, url),
            json={
                "request_id": "request-safe",
                "model": "Cosmos3-Nano",
                "batch_size": 2,
                "outputs": [{"sample": "one"}, {"sample": "two"}],
                "artifacts": [
                    {"sample": "one", "path": "x.jpg", "bytes": 2, "sha256": "0" * 64}
                ],
                "guardrails": True,
                "max_batch_size": 4,
                "framework_revision": "5e67049cd94acb667786f1e6dd0dab821cb90c97",
                "server_source_revision": "a" * 40,
            },
        )

    monkeypatch.setattr(httpx, "request", request)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", url), content=b"bad"
        ),
    )
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "secret-for-test")
    with pytest.raises(Cosmos3RayServeError, match="integrity mismatch"):
        submit_batch(
            input_path=str(source),
            output_path=str(tmp_path / "out"),
            endpoint="http://service.invalid:8000",
        )
