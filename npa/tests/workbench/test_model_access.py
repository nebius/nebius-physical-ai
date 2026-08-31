from __future__ import annotations

from dataclasses import dataclass

import pytest

from npa.workbench.model_access import (
    HF_GATING_LAST_VERIFIED,
    WORKBENCH_ASSETS,
    access_note,
    all_capabilities,
    assets_for,
    check_hf_asset,
    check_ngc_key,
    check_workbench_access,
    gated_hf_assets,
    gated_hf_repos,
    has_failure,
    hf_model_url,
)
from npa.workflows.sim2real_health import FAIL, PASS, WARN


@dataclass
class _HFResult:
    ok: bool
    status_code: int | None = None
    error: str = ""


def _gated_asset():
    return next(a for a in WORKBENCH_ASSETS if a.gated)


def _public_asset():
    return next(a for a in WORKBENCH_ASSETS if not a.gated)


def test_catalog_matches_current_nvidia_hf_gating() -> None:
    assert HF_GATING_LAST_VERIFIED == "2026-08-31"
    repos = {a.repo for a in WORKBENCH_ASSETS}
    assert "nvidia/GR00T-N1.7-3B" in repos
    assert "nvidia/Alpamayo2-Super" in repos
    assert "nvidia/Cosmos-Reason2-8B" in repos
    gated = {a.repo for a in WORKBENCH_ASSETS if a.gated}
    assert "nvidia/PhysicalAI-Autonomous-Vehicles" in gated
    assert "nvidia/GR00T-N1.7-3B" not in gated
    assert "nvidia/Cosmos-Reason1-7B" not in gated
    assert "nvidia/Cosmos3-Nano" not in gated
    assert "nvidia/PhysicalAI-NuRec-PPISP" not in gated
    assert "nvidia/Cosmos-Reason2-8B" in gated
    assert "nvidia/Cosmos-Guardrail1" in gated


def test_hf_model_url() -> None:
    assert (
        hf_model_url("nvidia/GR00T-N1.7-3B")
        == "https://huggingface.co/nvidia/GR00T-N1.7-3B"
    )


def test_all_capabilities_includes_core_tools() -> None:
    caps = all_capabilities()
    for expected in ("groot", "cosmos", "paidf", "sim2real", "vlm_eval"):
        assert expected in caps


def test_assets_for_filters_by_capability() -> None:
    groot_assets = assets_for(["groot"])
    assert groot_assets, "expected at least one groot asset"
    assert all("groot" in a.capabilities for a in groot_assets)
    # 'all' / None returns the full catalog.
    assert assets_for(None) == WORKBENCH_ASSETS
    assert assets_for([]) == WORKBENCH_ASSETS


def test_paidf_access_is_scoped_to_the_gated_transfer_model() -> None:
    from npa.workbench.cosmos.control_contract import COSMOS_TRANSFER_CHECKPOINTS

    assets = assets_for(["paidf"])
    assert {asset.repo for asset in assets} == {"nvidia/Cosmos-Transfer2.5-2B"}
    assert {asset.revision for asset in assets} == {
        checkpoint.revision for checkpoint in COSMOS_TRANSFER_CHECKPOINTS.values()
    }
    assert all(asset.gated for asset in assets)


def test_hf_gated_warns_without_token() -> None:
    result = check_hf_asset(_gated_asset(), "", hf_validator=None)
    assert result.status == WARN
    assert "Agree and access" in result.remedy or "Accept the license" in result.remedy


def test_hf_present_unverified_offline() -> None:
    result = check_hf_asset(_gated_asset(), "hf_x", hf_validator=None)
    assert result.status == PASS
    assert "not verified" in result.summary


def test_hf_pass_when_validator_ok() -> None:
    result = check_hf_asset(
        _gated_asset(), "hf_x", hf_validator=lambda *args: _HFResult(ok=True)
    )
    assert result.status == PASS
    assert "access ok" in result.summary.lower()


def test_hf_gated_fail_points_at_acceptance_url() -> None:
    asset = _gated_asset()
    result = check_hf_asset(
        asset,
        "hf_x",
        hf_validator=lambda *args: _HFResult(
            ok=False, status_code=403, error="no access"
        ),
    )
    assert result.status == FAIL
    assert hf_model_url(asset.repo) in result.remedy
    assert "Agree and access repository" in result.remedy


def test_hf_public_401_is_token_problem_not_gating() -> None:
    result = check_hf_asset(
        _public_asset(),
        "hf_bad",
        hf_validator=lambda *args: _HFResult(ok=False, status_code=401, error="bad"),
    )
    assert result.status == FAIL
    assert "settings/tokens" in result.remedy


def test_hf_transient_error_warns() -> None:
    result = check_hf_asset(
        _gated_asset(),
        "hf_x",
        hf_validator=lambda *args: _HFResult(
            ok=False, status_code=None, error="timeout"
        ),
    )
    assert result.status == WARN


def test_hf_validator_diagnostic_redacts_token() -> None:
    token = "hf_synthetic_secret"
    result = check_hf_asset(
        _gated_asset(),
        token,
        hf_validator=lambda token, *_args: _HFResult(
            ok=False, status_code=403, error=f"upstream echoed {token}"
        ),
    )

    assert token not in " ".join((*result.details, result.summary, result.remedy))
    assert "<redacted>" in result.details[0]


def test_hf_validator_exception_is_sanitized() -> None:
    token = "hf_synthetic_exception_secret"

    def _raise(token, *_args):
        raise RuntimeError(f"upstream echoed {token}")

    result = check_hf_asset(_gated_asset(), token, hf_validator=_raise)

    assert result.status == WARN
    assert token not in " ".join((*result.details, result.summary, result.remedy))
    assert result.details == ("probe failed (RuntimeError)",)


def test_ngc_warns_when_needed_and_missing() -> None:
    assert check_ngc_key("", needed=True).status == WARN


def test_ngc_skipped_when_not_needed() -> None:
    result = check_ngc_key("", needed=False)
    assert result.status == PASS
    assert "not required" in result.summary


@pytest.mark.parametrize("credential", ["nvapi-abc", "registry-credential"])
def test_ngc_online_accepts_provider_validated_credential_shapes(
    credential: str,
) -> None:
    observed: list[str] = []

    def validator(key: str) -> str:
        observed.append(key)
        return "reachable"

    result = check_ngc_key(credential, needed=True, ngc_validator=validator)
    assert result.status == PASS
    assert observed == [credential]
    assert credential not in " ".join((result.summary, result.remedy, *result.details))


def test_ngc_nonempty_credential_is_unverified_offline() -> None:
    result = check_ngc_key("registry-credential", needed=True)

    assert result.status == WARN
    assert "not probed in offline mode" in result.summary


def test_ngc_credential_does_not_masquerade_as_entitlement() -> None:
    result = check_ngc_key(
        "nvapi-abc",
        needed=True,
        ngc_validator=lambda key: "entitlement-required",
    )
    assert result.status == FAIL
    assert "entitlement" in result.summary


@pytest.mark.parametrize(
    "outcome",
    ["auth-no-token", "auth-401", "auth-403"],
)
def test_ngc_definitive_auth_rejection_fails(outcome: str) -> None:
    result = check_ngc_key(
        "nvapi-synthetic",
        needed=True,
        ngc_validator=lambda key: outcome,
    )

    assert result.status == FAIL
    assert "credential rejected" in result.summary
    assert "health access" in result.remedy


@pytest.mark.parametrize("outcome", ["entitlement-required", "tags-401", "tags-403"])
def test_ngc_definitive_entitlement_rejection_fails(outcome: str) -> None:
    result = check_ngc_key(
        "nvapi-synthetic",
        needed=True,
        ngc_validator=lambda key: outcome,
    )

    assert result.status == FAIL
    assert "entitlement denied" in result.summary
    assert "health access" in result.remedy


def test_ngc_transport_failure_warns() -> None:
    result = check_ngc_key(
        "nvapi-synthetic",
        needed=True,
        ngc_validator=lambda key: "unreachable",
    )

    assert result.status == WARN
    assert "reachable" in result.remedy


def test_ngc_validator_exception_is_sanitized() -> None:
    secret = "nvapi-synthetic-exception-secret"

    def _raise(key: str) -> str:
        raise RuntimeError(f"upstream echoed {key}")

    result = check_ngc_key(secret, needed=True, ngc_validator=_raise)

    assert result.status == WARN
    assert secret not in " ".join((*result.details, result.summary, result.remedy))
    assert result.details == ("probe failed (RuntimeError)",)


def test_ngc_offline_does_not_infer_validity_from_format() -> None:
    result = check_ngc_key("bogus", needed=True)
    assert result.status == WARN
    assert "not probed" in result.summary


def test_check_workbench_access_ngc_first_then_hf() -> None:
    results = check_workbench_access(
        hf_token="hf_x", ngc_key="nvapi-x", hf_validator=None
    )
    assert results[0].name == "ngc"
    hf_names = {r.name for r in results[1:]}
    assert "nvidia/GR00T-N1.7-3B" in hf_names


def test_check_workbench_access_capability_scope_drops_ngc_when_not_needed() -> None:
    results = check_workbench_access(
        hf_token="hf_x", ngc_key="", hf_validator=None, capabilities=["vlm_eval"]
    )
    ngc = next(r for r in results if r.name == "ngc")
    assert ngc.status == PASS
    assert "not required" in ngc.summary
    # Only vlm_eval assets present.
    repos = {r.name for r in results if r.name != "ngc"}
    assert repos <= {a.repo for a in assets_for(["vlm_eval"])}


def test_check_workbench_access_flags_failure_on_gated_denial() -> None:
    results = check_workbench_access(
        hf_token="hf_x",
        ngc_key="nvapi-x",
        hf_validator=lambda *args: _HFResult(
            ok=False, status_code=403, error="denied"
        ),
        capabilities=["groot"],
    )
    assert has_failure(results) is True


def test_gated_hf_repos_returns_only_gated_hf() -> None:
    repos = gated_hf_repos()
    assert "nvidia/GR00T-N1.7-3B" not in repos
    assert "nvidia/Cosmos-Reason2-2B" in repos
    public = {a.repo for a in WORKBENCH_ASSETS if not a.gated}
    assert set(repos).isdisjoint(public)
    # Scoped to a capability, only that capability's gated repos come back.
    groot = gated_hf_repos(["groot"])
    assert "nvidia/GR00T-N1.7-3B" not in groot
    assert "nvidia/Cosmos-Reason2-2B" in groot
    assert all("groot" in a.capabilities for a in WORKBENCH_ASSETS if a.repo in groot)


def test_gated_hf_assets_preserve_repository_types() -> None:
    assets = {asset.repo: asset for asset in gated_hf_assets()}

    assert assets["nvidia/PhysicalAI-Autonomous-Vehicles"].repo_type == "dataset"
    assert assets["nvidia/Cosmos-Reason2-2B"].repo_type == "model"


def test_check_workbench_access_gated_only_skips_public() -> None:
    results = check_workbench_access(
        hf_token="hf_x", ngc_key="nvapi-x", hf_validator=None, gated_only=True
    )
    repos = {r.name for r in results if r.name != "ngc"}
    public = {a.repo for a in WORKBENCH_ASSETS if not a.gated}
    assert repos.isdisjoint(public)
    assert "nvidia/GR00T-N1.7-3B" not in repos
    assert "nvidia/Cosmos-Reason2-2B" in repos


def test_access_note_all_ok_is_one_positive_line() -> None:
    results = check_workbench_access(
        hf_token="hf_x",
        ngc_key="nvapi-x",
        hf_validator=lambda *args: _HFResult(ok=True),
        ngc_validator=lambda key: "reachable",
        gated_only=True,
    )
    note = access_note(results)
    assert "\n" not in note
    assert note.startswith("[NOTE]")
    assert "can access all checked workbench models" in note


def test_access_note_lists_hf_failures_on_one_line() -> None:
    denied = {"nvidia/Cosmos-Reason2-2B"}

    def _validator(token, repo, repo_type, revision, probe_path):
        del token, repo_type, revision, probe_path
        return _HFResult(
            ok=repo not in denied, status_code=403 if repo in denied else 200
        )

    results = check_workbench_access(
        hf_token="hf_x", ngc_key="nvapi-x", hf_validator=_validator, gated_only=True
    )
    note = access_note(results)
    assert "\n" not in note
    assert "HF has no access to:" in note
    assert "nvidia/Cosmos-Reason2-2B" in note
    assert "huggingface.co" in note


def test_access_note_ngc_missing_names_capabilities() -> None:
    results = check_workbench_access(
        hf_token="hf_x",
        ngc_key="",
        hf_validator=lambda *args: _HFResult(ok=True),
        gated_only=True,
    )
    note = access_note(results)
    assert "NGC not configured" in note
    assert "nurec" in note
    # NGC line must not conflate HF repo IDs with NGC container access.
    assert "nvidia/" not in note


def test_access_note_distinguishes_ngc_credential_rejection() -> None:
    results = check_workbench_access(
        hf_token="hf_synthetic",
        ngc_key="nvapi-synthetic",
        hf_validator=lambda *args: _HFResult(ok=True),
        ngc_validator=lambda key: "auth-401",
        gated_only=True,
    )

    note = access_note(results)
    assert "NGC credential rejected for: nurec" in note
    assert "entitlement denied" not in note


def test_access_note_counts_unverified() -> None:
    # No token + gated => WARN (unverified) for each gated model, NGC present.
    results = check_workbench_access(
        hf_token="", ngc_key="nvapi-x", hf_validator=None, gated_only=True
    )
    note = access_note(results)
    assert "unverified" in note


# --- Drift guards: the catalog must stay in sync with the real tool defaults ---
# If a tool changes its default model constant, one of these fails until
# WORKBENCH_ASSETS is updated, so the access check never silently goes stale.


def _catalog_repos() -> set[str]:
    return {asset.repo for asset in WORKBENCH_ASSETS}


def test_catalog_covers_light_default_constants() -> None:
    tf = pytest.importorskip("npa.clients.token_factory")
    constants = pytest.importorskip("npa.workflows.sim2real.constants")
    repos = _catalog_repos()
    expected = {
        tf.DEFAULT_TEXT_MODEL,
        tf.DEFAULT_VISION_MODEL,
        constants.DEFAULT_REASON2_MODEL,
        constants.DEFAULT_COSMOS3_MODEL,
        constants.DEFAULT_REFERENCE_VLM_MODEL,
    }
    # DEFAULT_LEROBOT_DATASET_ID names the S3 task-seed contract, not a model
    # or Hugging Face repository; task/dataset compatibility has its own tests.
    missing = expected - repos
    assert not missing, (
        f"WORKBENCH_ASSETS is missing tool default models: {sorted(missing)}"
    )


def test_catalog_covers_groot_and_cosmos_cli_defaults() -> None:
    groot = pytest.importorskip("npa.cli.groot")
    cosmos = pytest.importorskip("npa.cli.cosmos")
    repos = _catalog_repos()
    for const in (groot.DEFAULT_MODEL, groot.COSMOS_REASON_MODEL, cosmos.DEFAULT_MODEL):
        assert const in repos, f"{const} missing from WORKBENCH_ASSETS"


def test_catalog_covers_vlm_eval_default_model() -> None:
    vlm_eval = pytest.importorskip("npa.workbench.vlm_eval")
    assert vlm_eval.DEFAULT_MODEL in _catalog_repos()


def test_known_public_nvidia_defaults_are_not_marked_gated() -> None:
    gated = {a.repo for a in WORKBENCH_ASSETS if a.gated}
    for repo in {
        "nvidia/GR00T-N1.7-3B",
        "nvidia/GEAR-SONIC",
        "nvidia/Cosmos-Reason1-7B",
        "nvidia/Cosmos3-Nano",
        "nvidia/PhysicalAI-NuRec-PPISP",
    }:
        assert repo not in gated, f"{repo} should be marked gated=False"
