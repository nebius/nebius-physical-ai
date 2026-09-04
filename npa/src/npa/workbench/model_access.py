"""Access preflight for the gated models the workbench capabilities depend on.

Given a Hugging Face token and an NVIDIA NGC API key, this module reports
whether the token already has access to every gated model the workbench uses and
points at the exact page to accept for anything still gated.

Important: Hugging Face gated repositories require *interactive* license
account authorization upstream. NPA checks repository access with the supplied
token and does not invent another acceptance flag. NGC access likewise requires
both a key and a successful repository entitlement probe.

Every check is a pure function that takes the resolved tokens plus injectable
Hugging Face and NGC validators; the CLI wires the real probes, tests inject
fakes. Nothing here imports GPU-heavy packages or touches infrastructure at
import time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Iterable

from npa.workflows.sim2real_health import FAIL, PASS, WARN, CheckResult, has_failure

HF = "huggingface"
NGC = "ngc"
TOKEN_FACTORY = "token_factory"
HF_GATING_LAST_VERIFIED = "2026-09-04"

_HF_PAYLOAD_SUFFIXES = (
    ".arrow",
    ".bin",
    ".ckpt",
    ".jsonl",
    ".mp4",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".zip",
)
_HF_METADATA_FILENAMES = {
    ".gitattributes",
    "config.json",
    "license",
    "license.md",
    "model_card.md",
    "readme",
    "readme.md",
    "tokenizer.json",
    "tokenizer_config.json",
}


@dataclass(frozen=True)
class GatedAsset:
    """A model/asset a workbench capability needs and how it is gated."""

    repo: str
    provider: str
    capabilities: tuple[str, ...]
    gated: bool
    note: str = ""
    repo_type: str = "model"
    revision: str = ""
    official_url: str = ""
    terms_revision: str = ""
    probe_path: str = ""


def usable_hf_payload_probe(asset: GatedAsset) -> bool:
    """Return whether a gated HF asset names a pinned, non-metadata payload."""

    if asset.provider != HF or not asset.gated:
        return True
    revision = asset.revision.strip()
    path = asset.probe_path.strip().strip("/")
    if not revision or not path or ".." in path.split("/"):
        return False
    basename = path.rsplit("/", 1)[-1].casefold()
    if basename in _HF_METADATA_FILENAMES or basename.startswith(("readme", "license")):
        return False
    return basename.endswith(_HF_PAYLOAD_SUFFIXES)


# This tuple is the single source of truth for the models the access check
# covers. To support or remove a workbench HF/NGC model, edit this list only —
# `npa configure`'s model-access NOTE, `npa workbench health access`, and
# `gated_hf_repos()` / `check_workbench_access()` all derive from it, so no other
# file needs to change.
# Gating metadata was reverified against Hugging Face's authoritative model and
# dataset APIs on 2026-09-04; capability-default drift tests below keep membership
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
    GatedAsset(
        "nvidia/Alpamayo2-Super",
        HF,
        ("alpamayo2-super",),
        False,
        revision="00554695e729a6ff0b6281fd2c81b18d06e33dbe",
    ),
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
        revision="b719eea7f0a63619ef51ec7f54178af0937ef050",
        probe_path=(
            "calibration/camera_intrinsics.offline/"
            "camera_intrinsics.offline.chunk_0000.parquet"
        ),
        official_url="https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles",
        terms_revision="nvidia-av-dataset-license-current",
    ),
    GatedAsset(
        "nvidia/GR00T-N1.7-3B",
        HF,
        ("groot",),
        False,
        revision="2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    ),
    GatedAsset("nvidia/GEAR-SONIC", HF, ("sonic",), False),
    GatedAsset(
        "nvidia/Cosmos-Transfer2.5-2B",
        HF,
        ("cosmos2", "paidf", "sim2real"),
        True,
        revision="b67b64abda3801a9aceddbff2bdb86126c06db74",
        probe_path="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-Transfer2.5-2B",
        HF,
        ("cosmos2", "paidf", "sim2real"),
        True,
        revision="eb5325b77d358944da58a690157dd2b8071bbf85",
        probe_path="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-Transfer2.5-2B",
        HF,
        ("cosmos2", "paidf", "sim2real"),
        True,
        revision="dea7737ca29dd8d9086413c6dc5724b8250a0bb4",
        probe_path="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-Transfer2.5-2B",
        HF,
        ("cosmos2", "paidf", "sim2real"),
        True,
        revision="23057a4167b89de89a4a397fdbf3887994d115eb",
        probe_path="auto/multiview/4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-Reason2-2B",
        HF,
        ("groot",),
        True,
        revision="9ce19a195e423419c349abfc86fd07178b230561",
        probe_path="model.safetensors",
        official_url="https://huggingface.co/nvidia/Cosmos-Reason2-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-Reason2-8B",
        HF,
        ("cosmos",),
        True,
        revision="a9fae2cf89dc64db96b12860417f0eb403013bb9",
        probe_path="model-00001-of-00004.safetensors",
        official_url="https://huggingface.co/nvidia/Cosmos-Reason2-8B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos3-Super-Reasoner",
        TOKEN_FACTORY,
        ("sim2real", "token_factory"),
        False,
        note=(
            "Hosted OpenMDW-1.1 model; verify key/project availability and balance "
            "through Token Factory, not Hugging Face."
        ),
    ),
    GatedAsset("nvidia/Cosmos-Reason1-7B", HF, ("cosmos",), False),
    GatedAsset(
        "nvidia/Cosmos3-Nano",
        HF,
        ("cosmos3", "paidf", "paidf-dig"),
        False,
        revision="411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
        probe_path="transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
        official_url="https://huggingface.co/nvidia/Cosmos3-Nano",
    ),
    GatedAsset(
        "nvidia/Cosmos3-Edge",
        HF,
        ("paidf-dig",),
        False,
        revision="a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba",
        probe_path="processor_config.json",
        official_url="https://huggingface.co/nvidia/Cosmos3-Edge",
    ),
    GatedAsset(
        "Qwen/Qwen-Image-Edit-2511",
        HF,
        ("paidf", "paidf-iaa"),
        False,
        revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        official_url="https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
    ),
    GatedAsset(
        "nvidia/Cosmos3-Super-Image2Video",
        HF,
        ("paidf", "paidf-evg"),
        False,
        revision="4f847566f3d3388fbf0ac07b99dd1a6432db9ecd",
        official_url="https://huggingface.co/nvidia/Cosmos3-Super-Image2Video",
    ),
    GatedAsset(
        "nvidia/Cosmos-Guardrail1",
        HF,
        ("cosmos3", "paidf", "paidf-dig", "sim2real"),
        True,
        revision="d6d4bfa899a71454a700907664f3e88f503950cf",
        probe_path="video_content_safety_filter/safety_filter.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-Guardrail1",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "facebook/dinov2-large",
        HF,
        ("paidf-dig",),
        False,
        revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        probe_path="pytorch_model.bin",
        official_url="https://huggingface.co/facebook/dinov2-large",
    ),
    GatedAsset(
        "nvidia/C-RADIOv3-B",
        HF,
        ("paidf-dig",),
        False,
        revision="44653a0482cf460bb4f12595fc3cc3dfecc403d1",
        probe_path="model.safetensors",
        official_url="https://huggingface.co/nvidia/C-RADIOv3-B",
    ),
    GatedAsset(
        "Wan-AI/Wan2.2-TI2V-5B",
        HF,
        ("paidf-dig",),
        False,
        revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        probe_path="Wan2.2_VAE.pth",
        official_url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B",
    ),
    GatedAsset(
        "facebook/sam2.1-hiera-large",
        HF,
        ("paidf-dig",),
        False,
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        probe_path="sam2.1_hiera_large.pt",
        official_url="https://huggingface.co/facebook/sam2.1-hiera-large",
    ),
    GatedAsset(
        "Qwen/Qwen3Guard-Gen-0.6B",
        HF,
        ("paidf-dig",),
        False,
        revision="fada3b2f655b89601929198343c94cd2f64d93cc",
        probe_path="model.safetensors",
        official_url="https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B",
    ),
    GatedAsset(
        "Qwen/Qwen3-VL-8B-Instruct",
        HF,
        ("paidf-dig",),
        False,
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        probe_path="tokenizer.json",
        official_url="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct",
    ),
    GatedAsset(
        "nvidia/Cosmos-Predict2.5-2B",
        HF,
        ("sim2real",),
        True,
        revision="85f8ae7bfe8f5525c8d103429524dcf12f98bf7b",
        probe_path="tokenizer.pth",
        official_url="https://huggingface.co/nvidia/Cosmos-Predict2.5-2B",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-1.0-Guardrail",
        HF,
        ("cosmos3-serving",),
        True,
        revision="cf03c0395fac8c4de386c0bdab12cc4fc8d66362",
        probe_path="video_content_safety_filter/safety_filter.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-1.0-Guardrail",
        terms_revision="huggingface-gated-repository-current",
    ),
    GatedAsset(
        "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
        HF,
        ("cosmos",),
        True,
        revision="749ad047f60de0ab405ed078fd050ab2d35856f7",
        probe_path="model.pt",
        official_url="https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
        terms_revision="huggingface-gated-repository-current",
    ),
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
        official_url="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
        terms_revision="llama-3.3-community-license-current",
        revision="6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        probe_path="model-00001-of-00030.safetensors",
    ),
    GatedAsset("Qwen/Qwen2-VL-7B-Instruct", HF, ("vlm_eval",), False),
    GatedAsset(
        "Qwen/Qwen2.5-VL-72B-Instruct", HF, ("vlm_eval", "token_factory"), False
    ),
    GatedAsset("lerobot/pusht", HF, ("lerobot", "sim2real"), False),
    GatedAsset(
        "nvcr.io/nvidia/nre/nre-ga:26.04",
        NGC,
        ("nurec",),
        True,
        note=(
            "A free individual NGC account and personal API key are supported; "
            "no enterprise organization administrator or service key is required."
        ),
        repo_type="container",
        revision="26.04",
        official_url=(
            "https://catalog.ngc.nvidia.com/orgs/nvidia/nre/containers/nre-ga"
        ),
        terms_revision="nvidia-nurec-ngc-governing-terms-current",
    ),
    GatedAsset(
        "nvcr.io/nvidia/paidf-detection-and-tracking-rfdetr-service@sha256:6b35e63b95cab7cd772906bcb08be978de7526427f0d1925ab84439dd4a9561e",
        NGC,
        ("paidf-label-detection",),
        True,
        repo_type="container",
        revision="sha256:6b35e63b95cab7cd772906bcb08be978de7526427f0d1925ab84439dd4a9561e",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-detection-and-tracking-rfdetr-service",
        terms_revision="nvidia-ngc-governing-terms-current",
    ),
    GatedAsset(
        "nvcr.io/nvidia/paidf-captioning-service@sha256:17e1e3f53cc66342183f7d0b6eed76907993bb325a13db90c46d9a8cf664d804",
        NGC,
        ("paidf-label-captioning",),
        True,
        repo_type="container",
        revision="sha256:17e1e3f53cc66342183f7d0b6eed76907993bb325a13db90c46d9a8cf664d804",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-captioning-service",
        terms_revision="nvidia-ngc-governing-terms-current",
    ),
    GatedAsset(
        "nvcr.io/nvidia/paidf-visual-qa-service@sha256:e681c8dee849c7ac9fc5b182f51e9efd0da460972b08850d40f00aa9d5e3c97c",
        NGC,
        ("paidf-label-visual-qa",),
        True,
        repo_type="container",
        revision="sha256:e681c8dee849c7ac9fc5b182f51e9efd0da460972b08850d40f00aa9d5e3c97c",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-visual-qa-service",
        terms_revision="nvidia-ngc-governing-terms-current",
    ),
    GatedAsset(
        "nvcr.io/nvidia/paidf-event-and-person-attribute-search-service@sha256:0f581ff6d92efd391281e5787a8b1fda76556443ade47c1f5d59d4c345a01f6a",
        NGC,
        ("paidf-label-attribute-search",),
        True,
        repo_type="container",
        revision="sha256:0f581ff6d92efd391281e5787a8b1fda76556443ade47c1f5d59d4c345a01f6a",
        official_url="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-event-and-person-attribute-search-service",
        terms_revision="nvidia-ngc-governing-terms-current",
    ),
)

# Capabilities whose NVIDIA containers/models are pulled from NGC and therefore
# need a valid NGC API key in addition to Hugging Face access.
NGC_CAPABILITIES: tuple[str, ...] = tuple(
    sorted(
        {
            capability
            for asset in WORKBENCH_ASSETS
            if asset.provider == NGC
            for capability in asset.capabilities
        }
    )
)


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
    return tuple(asset.repo for asset in gated_hf_assets(capabilities))


def gated_hf_assets(
    capabilities: Iterable[str] | None = None,
) -> tuple[GatedAsset, ...]:
    """Return gated HF assets with the repository type needed by live probes."""

    return tuple(
        asset
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


def _hf_access_page(asset: GatedAsset) -> str:
    official = asset.official_url or hf_model_url(asset.repo)
    canonical = hf_model_url(asset.repo)
    if official == canonical:
        return official
    return f"{official} (canonical repository URL: {canonical})"


def _redact_secret(text: Any, secret: str) -> str:
    """Keep validator diagnostics useful without ever returning a credential."""

    rendered = str(text or "")
    return rendered.replace(secret, "<redacted>") if secret else rendered


def check_hf_asset(
    asset: GatedAsset,
    token: str,
    hf_validator: Callable[..., Any] | None,
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
                    f"Accept the license at {_hf_access_page(asset)} while signed in, "
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
    if not usable_hf_payload_probe(asset):
        return CheckResult(
            name=name,
            status=FAIL,
            summary=(
                f"Gated HF asset {asset.repo} has no exact payload probe; access "
                f"cannot be verified.{caps}"
            ),
            remedy="Add a pinned revision and non-metadata probe_path to GatedAsset.",
        )
    try:
        result = hf_validator(
            token,
            asset.repo,
            asset.repo_type,
            asset.revision,
            asset.probe_path,
        )
    except Exception as exc:  # noqa: BLE001 - a probe exception is transient
        return CheckResult(
            name=name,
            status=WARN,
            summary=f"Could not verify HF access to {asset.repo} (transient).{caps}",
            remedy="Retry when the network is available.",
            details=(f"probe failed ({type(exc).__name__})",),
        )
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
    error = _redact_secret(
        getattr(result, "error", "") or "unknown error",
        token,
    )
    if status_code in {401, 403}:
        if asset.gated:
            return CheckResult(
                name=name,
                status=FAIL,
                summary=f"HF token cannot access gated repo {asset.repo}.{caps}",
                remedy=(
                    f"Open {_hf_access_page(asset)} while signed in and click "
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


def check_ngc_artifact_access(api_key: str, *, image: str) -> str:
    """Probe the selected NGC manifest using the worker's registry pull protocol."""

    from npa.orchestration.skypilot.registry_preflight import (
        check_image_pull,
        parse_image_reference,
    )

    try:
        reference = parse_image_reference(image)
        if reference.registry != "nvcr.io":
            return "unresolved"
        result = check_image_pull(image, username="$oauthtoken", password=api_key)
    except Exception:  # noqa: BLE001 - provider errors must not expose credentials
        return "unreachable"
    if result.ok:
        return "reachable"
    if result.http_status == 402 or result.status == "forbidden":
        return "entitlement-required"
    if result.status in {"unauthorized", "no_credentials"}:
        return (
            f"auth-{result.http_status}"
            if result.http_status in {401, 403}
            else "auth-no-token"
        )
    if result.status == "not_found":
        return "manifest-404"
    return "unreachable"


def check_ngc_key(
    ngc_key: str,
    *,
    needed: bool,
    ngc_validator: Callable[..., str] | None = None,
    images: Iterable[str] | None = None,
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
            summary="NGC_API_KEY is not set (needed for the selected NGC artifact pulls).",
            remedy=(
                "Sign in with an ordinary individual NGC account at "
                "https://ngc.nvidia.com/signin, create a personal API key, and run "
                "`npa configure --no-interactive --save-env-credentials` with "
                "NGC_API_KEY set in the environment."
            ),
        )
    if ngc_validator is None:
        return CheckResult(
            name="ngc",
            status=WARN,
            summary=(
                "NGC_API_KEY is present; validity and repository entitlement were "
                "not probed in offline mode."
            ),
        )
    if images is not None:
        image_results = [
            (
                image,
                check_ngc_key(
                    key,
                    needed=True,
                    ngc_validator=partial(ngc_validator, image=image),
                ),
            )
            for image in images
        ]
        if not image_results:
            return check_ngc_key(key, needed=False)
        # Preserve the aggregate ``ngc`` row consumed by configure and callers,
        # while recording every exact artifact and retaining the worst outcome.
        primary = max(
            image_results, key=lambda row: {PASS: 0, WARN: 1, FAIL: 2}[row[1].status]
        )[1]
        return replace(
            primary,
            summary=(
                f"NGC_API_KEY can pull all {len(image_results)} selected NGC artifact(s)."
                if primary.status == PASS
                else primary.summary
            ),
            details=tuple(
                f"{image}: {result.status}: {result.summary}"
                for image, result in image_results
            ),
        )
    try:
        outcome = _redact_secret(ngc_validator(key) or "unreachable", key)
    except Exception as exc:  # noqa: BLE001 - a probe exception is transient
        return CheckResult(
            name="ngc",
            status=WARN,
            summary="Could not verify NGC repository access (transient).",
            remedy="Retry `npa workbench health access` when the network is available.",
            details=(f"probe failed ({type(exc).__name__})",),
        )
    if outcome == "reachable":
        return CheckResult(
            name="ngc",
            status=PASS,
            summary="NGC_API_KEY can pull the selected NGC artifact.",
        )
    credential_rejected = outcome in {
        "auth-no-token",
        "auth-401",
        "auth-403",
    }
    entitlement_rejected = outcome in {
        "entitlement-required",
        "tags-401",
        "tags-403",
    }
    rejected = credential_rejected or entitlement_rejected or outcome == "manifest-404"
    if credential_rejected:
        summary = f"NGC credential rejected during repository auth: {outcome}."
    elif entitlement_rejected:
        summary = f"NGC repository entitlement denied: {outcome}."
    elif outcome == "manifest-404":
        summary = "The selected NGC image manifest does not exist: manifest-404."
    else:
        summary = f"NGC repository pull preflight failed: {outcome}."
    return CheckResult(
        name="ngc",
        status=FAIL if rejected else WARN,
        summary=summary,
        remedy=(
            "Verify the selected NGC repository and immutable image digest, "
            "then re-run `npa workbench health access`."
            if outcome == "manifest-404"
            else "Regenerate NGC_API_KEY with NGC Catalog access if needed, verify the "
            "repository entitlement, then re-run `npa workbench health access`."
            if rejected
            else "Retry when the NGC registry is reachable; credential presence "
            "alone does not prove pull access."
        ),
    )


def check_workbench_access(
    *,
    hf_token: str,
    ngc_key: str,
    hf_validator: Callable[..., Any] | None = None,
    ngc_validator: Callable[..., str] | None = None,
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
            images=tuple(
                asset.repo for asset in assets_for(selected) if asset.provider == NGC
            ),
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
    entries that could not be reached are counted as "unverified". NGC separates
    missing/malformed keys, transient probe failures, credential rejection, and
    repository-entitlement denial.
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
    ngc_credential_rejected = bool(
        ngc is not None and ngc.status == FAIL and "credential rejected" in ngc.summary
    )

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
            "NGC repository entitlement unverified for: " + ", ".join(NGC_CAPABILITIES)
        )
    elif ngc_credential_rejected:
        parts.append("NGC credential rejected for: " + ", ".join(NGC_CAPABILITIES))
    elif not ngc_ok:
        parts.append(
            "NGC repository entitlement denied for: " + ", ".join(NGC_CAPABILITIES)
        )
    if hf_unverified:
        parts.append(f"{len(hf_unverified)} model(s) unverified")

    note = "[NOTE] " + "; ".join(parts) + "."
    if hf_no:
        note += " Accept gated licenses at https://huggingface.co/<model>."
    note += " Re-run `npa workbench health access` for full remediation."
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
    "check_ngc_artifact_access",
    "check_ngc_key",
    "check_workbench_access",
    "gated_hf_assets",
    "gated_hf_repos",
    "has_failure",
    "hf_model_url",
    "usable_hf_payload_probe",
]
