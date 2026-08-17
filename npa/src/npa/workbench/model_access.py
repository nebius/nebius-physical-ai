"""Access preflight for the gated models the workbench capabilities depend on.

Given a Hugging Face token and an NVIDIA NGC API key, this module reports
whether the token already has access to every gated model the workbench uses and
points at the exact page to accept for anything still gated.

Important: Hugging Face gated repositories require *interactive* license
account authorization upstream. NPA checks repository access with the supplied
token and does not invent another acceptance flag. NGC access likewise requires
both a key and a successful repository entitlement probe.

Every check is a pure function that takes the resolved tokens plus an injectable
Hugging Face validator; the CLI wires the real probe, tests inject fakes.
Nothing here imports GPU-heavy packages or touches infrastructure at import
time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from npa.workflows.sim2real_health import FAIL, PASS, WARN, CheckResult, has_failure

HF = "huggingface"
NGC = "ngc"
HF_GATING_LAST_VERIFIED = "2026-08-17"


@dataclass(frozen=True)
class GatedAsset:
    """A model/asset a workbench capability needs and how it is gated."""

    repo: str
    provider: str
    capabilities: tuple[str, ...]
    gated: bool
    note: str = ""
    repo_type: str = "model"


# This tuple is the single source of truth for the models the access check
# covers. To support or remove a workbench HF/NGC model, edit this list only —
# `npa configure`'s model-access NOTE, `npa workbench health access`, and
# `gated_hf_repos()` / `check_workbench_access()` all derive from it, so no other
# file needs to change.
# Gating metadata was reverified against Hugging Face's authoritative model and
# dataset APIs on 2026-08-14; capability-default drift tests below keep membership
# current, while the online preflight remains the final access authority.
#
# The entries mirror the tool default-model constants in:
#   - npa/src/npa/cli/groot/__init__.py         DEFAULT_MODEL, COSMOS_REASON_MODEL
#   - npa/src/npa/cli/cosmos/__init__.py        DEFAULT_MODEL
#   - npa/src/npa/workflows/sim2real/constants.py
#   - npa/src/npa/clients/token_factory.py
#   - npa/src/npa/workbench/vlm_eval/__init__.py
# `tests/workbench/test_model_access.py` imports those real constants and asserts
# each still appears here, so a default-model change fails CI until this list is
# updated. `gated=True` marks repos that require accepting the license on Hugging
# Face before the token can download them.
WORKBENCH_ASSETS: tuple[GatedAsset, ...] = (
    GatedAsset("nvidia/Alpamayo2-Super", HF, ("alpamayo2-super",), False),
    GatedAsset(
        "nvidia/PhysicalAI-Autonomous-Vehicles",
        HF,
        ("alpamayo2-super",),
        True,
        note=(
            "Accept the NVIDIA Autonomous Vehicle Dataset License Agreement "
            "interactively; dataset bytes are non-transferable and runtime-only."
        ),
        repo_type="dataset",
    ),
    GatedAsset("nvidia/GR00T-N1.7-3B", HF, ("groot",), False),
    GatedAsset("nvidia/GEAR-SONIC", HF, ("sonic",), False),
    GatedAsset("nvidia/Cosmos-Transfer2.5-2B", HF, ("paidf", "sim2real"), True),
    GatedAsset("nvidia/Cosmos-Reason2-2B", HF, ("groot", "sim2real"), True),
    GatedAsset("nvidia/Cosmos-Reason2-8B", HF, ("sim2real",), True),
    GatedAsset("nvidia/Cosmos-Reason1-7B", HF, ("cosmos",), False),
    GatedAsset("nvidia/Cosmos3-Nano", HF, ("cosmos3",), False),
    GatedAsset("nvidia/Cosmos-Guardrail1", HF, ("cosmos3",), True),
    GatedAsset("nvidia/Cosmos-1.0-Guardrail", HF, ("cosmos3-serving",), True),
    GatedAsset("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", HF, ("cosmos",), True),
    GatedAsset(
        "nvidia/PhysicalAI-NuRec-PPISP",
        HF,
        ("nurec",),
        False,
        repo_type="dataset",
    ),
    GatedAsset(
        "meta-llama/Llama-3.3-70B-Instruct",
        HF,
        ("token_factory",),
        True,
        note="Only needed to self-host; Token Factory serves it hosted (no HF gating).",
    ),
    GatedAsset("Qwen/Qwen2-VL-7B-Instruct", HF, ("vlm_eval",), False),
    GatedAsset(
        "Qwen/Qwen2.5-VL-72B-Instruct", HF, ("vlm_eval", "token_factory"), False
    ),
    GatedAsset("lerobot/pusht", HF, ("lerobot", "sim2real"), False),
)

# Capabilities whose NVIDIA containers/models are pulled from NGC and therefore
# need a valid NGC API key in addition to Hugging Face access.
NGC_CAPABILITIES: tuple[str, ...] = ("nurec",)


def hf_model_url(repo: str) -> str:
    """Return the Hugging Face page where a gated repo's license is accepted."""
    return f"https://huggingface.co/{repo}"


def all_capabilities() -> tuple[str, ...]:
    """Return every capability referenced by the catalog, sorted."""
    seen: set[str] = set()
    for asset in WORKBENCH_ASSETS:
        seen.update(asset.capabilities)
    seen.update(NGC_CAPABILITIES)
    return tuple(sorted(seen))


def assets_for(capabilities: Iterable[str] | None) -> tuple[GatedAsset, ...]:
    """Return the assets needed for *capabilities* (all when ``None``)."""
    if capabilities is None:
        return WORKBENCH_ASSETS
    wanted = {item.strip() for item in capabilities if item and item.strip()}
    if not wanted:
        return WORKBENCH_ASSETS
    return tuple(
        asset for asset in WORKBENCH_ASSETS if wanted.intersection(asset.capabilities)
    )


def gated_hf_repos(capabilities: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the license-gated Hugging Face repos for *capabilities*."""
    return tuple(
        asset.repo
        for asset in assets_for(capabilities)
        if asset.provider == HF and asset.gated
    )


def _ngc_needed(capabilities: Iterable[str] | None) -> bool:
    if capabilities is None:
        return True
    wanted = {item.strip() for item in capabilities if item and item.strip()}
    if not wanted:
        return True
    return bool(wanted.intersection(NGC_CAPABILITIES))


def _cap_suffix(asset: GatedAsset) -> str:
    return f" [{', '.join(asset.capabilities)}]"


def check_hf_asset(
    asset: GatedAsset,
    token: str,
    hf_validator: Callable[[str, str, str], Any] | None,
) -> CheckResult:
    """Check whether *token* can access one gated Hugging Face repo."""

    name = asset.repo
    caps = _cap_suffix(asset)
    if not token:
        if asset.gated:
            return CheckResult(
                name=name,
                status=WARN,
                summary=f"No HF token; cannot verify gated access to {asset.repo}.{caps}",
                remedy=(
                    f"Accept the license at {hf_model_url(asset.repo)} while signed in, "
                    "then export HF_TOKEN and run `npa configure --no-interactive "
                    "--save-env-credentials`."
                ),
            )
        if hf_validator is None:
            return CheckResult(
                name=name,
                status=PASS,
                summary=(
                    f"Public HF asset supports anonymous access; repository was not "
                    f"probed offline: {asset.repo}.{caps}"
                ),
                remedy="Set HF_TOKEN only for authenticated downloads or rate limits.",
            )
    if hf_validator is None:
        return CheckResult(
            name=name,
            status=PASS,
            summary=f"HF token present; {asset.repo} access not verified (offline).{caps}",
        )
    result = hf_validator(token, asset.repo, asset.repo_type)
    if getattr(result, "ok", False):
        return CheckResult(
            name=name,
            status=PASS,
            summary=(
                f"HF access ok ({'authenticated' if token else 'anonymous'}): "
                f"{asset.repo}.{caps}"
            ),
        )
    status_code = getattr(result, "status_code", None)
    error = getattr(result, "error", "") or "unknown error"
    if status_code in {401, 403}:
        if asset.gated:
            return CheckResult(
                name=name,
                status=FAIL,
                summary=f"HF token cannot access gated repo {asset.repo}.{caps}",
                remedy=(
                    f"Open {hf_model_url(asset.repo)} while signed in and click "
                    "'Agree and access repository', then re-run this check."
                ),
                details=(error,),
            )
        return CheckResult(
            name=name,
            status=FAIL,
            summary=(
                f"HF {'token rejected' if token else 'anonymous access rejected'} "
                f"for {asset.repo}.{caps}"
            ),
            remedy=(
                "Regenerate the token at https://huggingface.co/settings/tokens."
                if token
                else "The asset metadata may have changed; verify its Hugging Face page."
            ),
            details=(error,),
        )
    return CheckResult(
        name=name,
        status=WARN,
        summary=f"Could not verify HF access to {asset.repo} (transient).{caps}",
        remedy="Retry when the network is available.",
        details=(error,),
    )


def check_ngc_key(
    ngc_key: str,
    *,
    needed: bool,
    ngc_validator: Callable[[str], str] | None = None,
) -> CheckResult:
    """Check NGC credentials and, online, actual repository pull entitlement."""

    if not needed:
        return CheckResult(
            name="ngc",
            status=PASS,
            summary="NGC not required for the selected capabilities.",
        )
    key = ngc_key or ""
    if not key:
        return CheckResult(
            name="ngc",
            status=WARN,
            summary="NGC_API_KEY is not set (needed for the NuRec NRE image pull).",
            remedy=(
                "Create one at https://org.ngc.nvidia.com/setup/api-key and run "
                "`npa configure --no-interactive --save-env-credentials` with "
                "NGC_API_KEY set in the environment."
            ),
        )
    if not key.lower().startswith(("nvapi-", "nvapi_")):
        return CheckResult(
            name="ngc",
            status=WARN,
            summary="NGC_API_KEY is set but does not look like an NGC key.",
            remedy="NGC keys start with 'nvapi-'. Re-check the value.",
        )
    if ngc_validator is None:
        return CheckResult(
            name="ngc",
            status=WARN,
            summary=(
                "NGC_API_KEY is present and well-formed; repository entitlement "
                "was not probed in offline mode."
            ),
        )
    outcome = str(ngc_validator(key) or "unreachable")
    if outcome == "reachable":
        return CheckResult(
            name="ngc",
            status=PASS,
            summary="NGC_API_KEY can pull the selected NuRec NRE repository.",
        )
    return CheckResult(
        name="ngc",
        status=FAIL if outcome == "entitlement-required" else WARN,
        summary=f"NGC repository pull preflight failed: {outcome}.",
        remedy=(
            "Verify NGC_API_KEY and repository entitlement. The credential alone "
            "does not grant pull access."
        ),
    )


def check_workbench_access(
    *,
    hf_token: str,
    ngc_key: str,
    hf_validator: Callable[[str, str, str], Any] | None = None,
    ngc_validator: Callable[[str], str] | None = None,
    capabilities: Iterable[str] | None = None,
    gated_only: bool = False,
) -> list[CheckResult]:
    """Return access checks for every gated asset the capabilities require.

    The NGC key check comes first, then one result per Hugging Face asset in
    catalog order. When *hf_validator* is ``None`` the HF checks report presence
    only (offline mode). Pass ``gated_only=True`` to check just the license-gated
    repos (skips always-public repos) — useful to keep an interactive preflight
    fast.
    """

    selected = list(capabilities) if capabilities is not None else None
    results: list[CheckResult] = [
        check_ngc_key(
            ngc_key,
            needed=_ngc_needed(selected),
            ngc_validator=ngc_validator,
        )
    ]
    for asset in assets_for(selected):
        if asset.provider != HF:
            continue
        if gated_only and not asset.gated:
            continue
        results.append(check_hf_asset(asset, hf_token, hf_validator))
    return results


def access_note(results: list[CheckResult]) -> str:
    """Return a one-line ``[NOTE]`` naming the models HF/NGC cannot access.

    HF entries with a definitive rejection (401/403) are listed as "no access";
    entries that could not be reached are counted as "unverified". NGC access is
    a key presence/format check (there is no offline NGC probe), so a missing or
    malformed key lists the NVIDIA models its absence blocks.
    """

    hf_no = [r.name for r in results if r.name != "ngc" and r.status == FAIL]
    hf_unverified = [r.name for r in results if r.name != "ngc" and r.status == WARN]
    ngc = next((r for r in results if r.name == "ngc"), None)
    ngc_ok = ngc is None or ngc.status == PASS
    ngc_missing = bool(
        ngc is not None
        and ngc.status == WARN
        and ("not set" in ngc.summary or "does not look" in ngc.summary)
    )
    ngc_unverified = bool(ngc is not None and ngc.status == WARN and not ngc_missing)

    if not hf_no and ngc_ok and not hf_unverified:
        return "[NOTE] HF and NGC tokens can access all checked workbench models."

    parts: list[str] = []
    if hf_no:
        parts.append("HF has no access to: " + ", ".join(hf_no))
    if ngc_missing:
        # NGC gates NVIDIA *container/model pulls* for whole capabilities, not
        # individual HF repos — name the affected capabilities, not repo IDs.
        parts.append(
            "NGC not configured (blocks NVIDIA pulls for: "
            + ", ".join(NGC_CAPABILITIES)
            + ")"
        )
    elif ngc_unverified:
        parts.append(
            "NGC repository entitlement unverified for: "
            + ", ".join(NGC_CAPABILITIES)
        )
    elif not ngc_ok:
        parts.append(
            "NGC repository entitlement denied for: "
            + ", ".join(NGC_CAPABILITIES)
        )
    if hf_unverified:
        parts.append(f"{len(hf_unverified)} model(s) unverified")

    note = "[NOTE] " + "; ".join(parts) + "."
    if hf_no:
        note += " Accept gated licenses at https://huggingface.co/<model>."
    return note


__all__ = [
    "GatedAsset",
    "HF",
    "HF_GATING_LAST_VERIFIED",
    "NGC",
    "NGC_CAPABILITIES",
    "WORKBENCH_ASSETS",
    "access_note",
    "all_capabilities",
    "assets_for",
    "check_hf_asset",
    "check_ngc_key",
    "check_workbench_access",
    "gated_hf_repos",
    "has_failure",
    "hf_model_url",
]
