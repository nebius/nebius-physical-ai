from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workbench.access_approval import (
    AccessStatus,
    approval_plan,
    exact_requirements,
    probe_requirements,
    requirements_for_tool_refs,
    safe_resume_command,
)
from npa.workbench.model_access import GatedAsset, HF, NGC


def _hf_result(*, ok: bool, status_code: int = 200):
    return SimpleNamespace(ok=ok, status_code=status_code, error="synthetic")


def test_full_catalog_is_deduplicated_grouped_and_excludes_token_factory() -> None:
    requirements = exact_requirements()
    identities = {
        (item.provider, item.repo, item.repo_type, item.revision)
        for item in requirements
    }
    assert len(identities) == len(requirements)
    assert any(item.provider == HF for item in requirements)
    assert any(item.provider == NGC for item in requirements)
    assert all(set(item.capabilities) != {"token_factory"} for item in requirements)
    assert all(item.official_url.startswith("https://") for item in requirements)


def test_catalog_uses_existing_pinned_revisions() -> None:
    from npa.cli.groot import COSMOS_REASON_MODEL, COSMOS_REASON_REVISION
    from npa.workbench.alpamayo2_super.runtime import (
        DEFAULT_DATASET_REPO,
        DEFAULT_DATASET_REVISION,
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
    )
    from npa.workbench.cosmos.control_contract import COSMOS_TRANSFER_CHECKPOINTS

    revisions = {
        item.repo: item.revision for item in exact_requirements(gated_only=False)
    }
    assert revisions[DEFAULT_MODEL_ID] == DEFAULT_MODEL_REVISION
    assert revisions[DEFAULT_DATASET_REPO] == DEFAULT_DATASET_REVISION
    assert revisions[COSMOS_REASON_MODEL] == COSMOS_REASON_REVISION
    transfer_revisions = {
        item.revision
        for item in exact_requirements(["cosmos2"])
        if item.repo == "nvidia/Cosmos-Transfer2.5-2B"
    }
    assert transfer_revisions == {
        checkpoint.revision for checkpoint in COSMOS_TRANSFER_CHECKPOINTS.values()
    }


def test_approval_layer_has_no_fake_provider_acceptance_flag() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "npa"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            source_root / "workbench" / "access_approval.py",
            source_root / "agent_backend" / "access_approval.py",
        )
    )
    assert "ACCEPT_HF" not in text
    assert "ACCEPT_NGC" not in text
    assert "acceptance_flag" not in text


def test_toolref_closure_comes_from_catalog_metadata() -> None:
    requirements = requirements_for_tool_refs(
        ["workbench.nurec.check", "workbench.nurec.reconstruct"]
    )
    assert {(item.provider, item.repo) for item in requirements} == {
        (NGC, "nvcr.io/nvidia/nre/nre-ga:26.04")
    }


def test_hf_ready_pending_denied_unavailable_and_public_anonymous(tmp_path: Path) -> None:
    gated = GatedAsset(
        "vendor/gated",
        HF,
        ("demo",),
        True,
        revision="rev-gated",
        probe_path="weights/model.safetensors",
        official_url="https://huggingface.co/vendor/gated",
        terms_revision="v1",
    )
    public = GatedAsset(
        "vendor/public",
        HF,
        ("demo",),
        False,
        official_url="https://huggingface.co/vendor/public",
    )

    def probe(token: str, repo: str, repo_type: str, revision: str, probe_path: str):
        del repo_type
        if repo.endswith("public"):
            assert token == ""
            return _hf_result(ok=True)
        assert (revision, probe_path) == ("rev-gated", "weights/model.safetensors")
        return _hf_result(ok=token == "ready", status_code=403 if token else 401)

    ready = probe_requirements(
        [gated],
        hf_token="ready",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=tmp_path / "ready.json",
    )
    assert ready[0].status == AccessStatus.READY
    pending = probe_requirements(
        [gated],
        hf_token="pending",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=tmp_path / "pending.json",
    )
    assert pending[0].status == AccessStatus.PENDING
    missing = probe_requirements(
        [gated],
        hf_token="",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=tmp_path / "missing.json",
    )
    assert missing[0].reason == "missing_credentials"
    unavailable = probe_requirements(
        [gated],
        hf_token="token",
        ngc_key="",
        hf_validator=None,
        ngc_validator=None,
        state_path=tmp_path / "unavailable.json",
    )
    assert unavailable[0].status == AccessStatus.UNAVAILABLE
    anonymous = probe_requirements(
        [public],
        hf_token="",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=tmp_path / "public.json",
    )
    assert anonymous[0].status == AccessStatus.READY


@pytest.mark.parametrize("payload_status", [401, 403])
def test_metadata_visibility_cannot_make_denied_payload_ready(
    tmp_path: Path, payload_status: int
) -> None:
    item = GatedAsset(
        "vendor/gated",
        HF,
        ("demo",),
        True,
        revision="rev-gated",
        probe_path="weights/model.safetensors",
        official_url="https://huggingface.co/vendor/gated",
        terms_revision="terms-a",
    )
    observed: list[tuple[str, str, str]] = []

    def payload_probe(_token, repo, _repo_type, revision, probe_path):
        observed.append((repo, revision, probe_path))
        return _hf_result(ok=False, status_code=payload_status)

    evidence = probe_requirements(
        [item],
        hf_token="identity-valid-token",
        ngc_key="",
        hf_validator=payload_probe,
        ngc_validator=None,
        state_path=tmp_path / "state.json",
    )

    assert evidence[0].status == AccessStatus.PENDING
    assert evidence[0].reason == "manual_approval_required_or_pending"
    assert observed == [
        ("vendor/gated", "rev-gated", "weights/model.safetensors")
    ]


@pytest.mark.parametrize("credential", ["nvapi-personal", "registry-credential"])
def test_ngc_credential_probe_is_exact_and_denials_are_typed(
    tmp_path: Path, credential: str
) -> None:
    item = GatedAsset(
        "nvcr.io/nvidia/nre/nre-ga:26.04",
        NGC,
        ("nurec",),
        True,
        repo_type="container",
        revision="26.04",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga",
        terms_revision="v1",
    )
    images: list[str] = []

    def reachable(key: str, *, image: str) -> str:
        assert key == credential
        images.append(image)
        return "reachable"

    state_path = tmp_path / "ngc.json"
    evidence = probe_requirements(
        [item],
        hf_token="",
        ngc_key=credential,
        hf_validator=None,
        ngc_validator=reachable,
        state_path=state_path,
    )
    assert evidence[0].status == AccessStatus.READY
    assert images == [item.repo]
    assert credential not in state_path.read_text(encoding="utf-8")
    denied = probe_requirements(
        [item],
        hf_token="",
        ngc_key=credential,
        hf_validator=None,
        ngc_validator=lambda *_args, **_kwargs: "entitlement-required",
        state_path=tmp_path / "ngc-denied.json",
    )
    assert denied[0].status == AccessStatus.DENIED


def test_ngc_nonempty_bad_credential_is_denied_only_by_provider(
    tmp_path: Path,
) -> None:
    secret = "registry-bad-credential"
    item = GatedAsset(
        "nvcr.io/nvidia/nre/nre-ga:26.04",
        NGC,
        ("nurec",),
        True,
        repo_type="container",
        revision="26.04",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga",
        terms_revision="v1",
    )
    state_path = tmp_path / "rejected.json"
    evidence = probe_requirements(
        [item],
        hf_token="",
        ngc_key=secret,
        hf_validator=None,
        ngc_validator=lambda *_args, **_kwargs: "auth-401",
        state_path=state_path,
        force=True,
    )

    assert evidence[0].status == AccessStatus.DENIED
    assert evidence[0].reason == "credential_denied"
    assert secret not in json.dumps(evidence[0].as_dict(), sort_keys=True)
    assert secret not in state_path.read_text(encoding="utf-8")


def test_ready_cache_reuses_only_unchanged_credential_revision_and_terms(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    calls: list[str] = []

    def probe(token: str, repo: str, repo_type: str, revision: str, probe_path: str):
        del repo, repo_type, revision, probe_path
        calls.append(token)
        return _hf_result(ok=True)

    base = GatedAsset(
        "vendor/model",
        HF,
        ("demo",),
        True,
        revision="rev-a",
        probe_path="weights/model.safetensors",
        official_url="https://huggingface.co/vendor/model",
        terms_revision="terms-a",
    )
    first = probe_requirements(
        [base],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    second = probe_requirements(
        [base],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    changed_revision = GatedAsset(
        **{**base.__dict__, "revision": "rev-b"}
    )
    probe_requirements(
        [changed_revision],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    changed_terms = GatedAsset(**{**base.__dict__, "terms_revision": "terms-b"})
    probe_requirements(
        [changed_terms],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    probe_requirements(
        [base],
        hf_token="token-b",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    assert first[0].cached is False
    assert second[0].cached is True
    assert calls == ["token-a", "token-a", "token-a", "token-b"]
    assert state_path.stat().st_mode & 0o777 == 0o600
    serialized = state_path.read_text(encoding="utf-8")
    assert "token-a" not in serialized and "token-b" not in serialized


def test_ready_cache_is_invalidated_when_exact_probe_path_changes(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    calls: list[tuple[str, str]] = []

    def probe(_token, _repo, _repo_type, revision, probe_path):
        calls.append((revision, probe_path))
        return _hf_result(ok=True)

    base = GatedAsset(
        "vendor/model",
        HF,
        ("demo",),
        True,
        revision="rev-a",
        probe_path="weights/first.safetensors",
        official_url="https://huggingface.co/vendor/model",
        terms_revision="terms-a",
    )
    first = probe_requirements(
        [base],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )
    changed = GatedAsset(
        **{**base.__dict__, "probe_path": "weights/second.safetensors"}
    )
    second = probe_requirements(
        [changed],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=state_path,
    )

    assert first[0].status == AccessStatus.READY
    assert second[0].status == AccessStatus.READY
    assert second[0].cached is False
    assert calls == [
        ("rev-a", "weights/first.safetensors"),
        ("rev-a", "weights/second.safetensors"),
    ]


@pytest.mark.parametrize(
    "probe_path",
    ["", "README.md", "LICENSE", "config.json", "../weights/model.safetensors"],
)
def test_gated_hf_without_usable_payload_probe_fails_closed(
    tmp_path: Path, probe_path: str
) -> None:
    item = GatedAsset(
        "vendor/model",
        HF,
        ("demo",),
        True,
        revision="rev-a",
        probe_path=probe_path,
        official_url="https://huggingface.co/vendor/model",
        terms_revision="terms-a",
    )
    called = False

    def probe(*_args):
        nonlocal called
        called = True
        return _hf_result(ok=True)

    evidence = probe_requirements(
        [item],
        hf_token="token-a",
        ngc_key="",
        hf_validator=probe,
        ngc_validator=None,
        state_path=tmp_path / "state.json",
    )

    assert evidence[0].status == AccessStatus.UNAVAILABLE
    assert evidence[0].reason == "exact_payload_probe_missing"
    assert called is False


def test_plan_and_resume_contract_never_claim_acceptance_or_expose_secrets(
    tmp_path: Path,
) -> None:
    item = exact_requirements(["groot"])[0]
    evidence = probe_requirements(
        [item],
        hf_token="hf-never-print",
        ngc_key="",
        hf_validator=lambda *_args: _hf_result(ok=False, status_code=403),
        ngc_validator=None,
        state_path=tmp_path / "state.json",
    )
    resume = safe_resume_command(
        [
            "/venv/bin/npa",
            "workbench",
            "workflow",
            "submit",
            "demo.yaml",
            "--registry-password",
            "never-print",
        ]
    )
    plan = approval_plan(evidence, resume_command=resume)
    encoded = json.dumps(plan)
    assert plan["legal_assent_performed"] is False
    assert plan["status"] == "blocked"
    assert "never-print" not in encoded
    assert "--registry-password" not in resume
