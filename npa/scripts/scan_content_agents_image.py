#!/usr/bin/env python3
"""Byte-scan every layer of a public Content Agents image.

The image may contain the Apache-2.0 Content Agents source, NPA's downloader,
and the reviewed upstream runtime lock. It must contain no OVRTX/Omniverse
runtime, model weights, samples, runtime caches, credentials, customer data, or
NVIDIA graphics-driver userspace. Nested archives and deleted earlier layers
are scanned through the shared publication walker.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402


CONTENT_AGENTS_REVISION = "36dbf3f274f8e256637230a05a085853f65cc175"
IMAGE_VERSION = "0.5.2-npa2"
OVRTX_VERSION = "0.3.0.312915"

FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ovrtx_runtime",
        re.compile(
            r"(?:^|/|!/)ovrtx(?:/|-.*\.(?:dist-info|egg-info)/)|"
            r"(?:^|/)libovrtx[^/]*\.(?:so|a)(?:\.|$)",
            re.I,
        ),
    ),
    (
        "omniverse_kit_runtime",
        re.compile(
            r"(?:^|/)(?:extscache|kit/kernel|kit-sdk|omniverse-kit)(?:/|$)|"
            r"(?:^|/)lib(?:carb|omni\.|omni_|omnikit)[^/]*\.so(?:\.|$)",
            re.I,
        ),
    ),
    (
        "nvidia_graphics_driver_userspace",
        re.compile(
            r"(?:^|/)lib(?:GLX_nvidia|EGL_nvidia|nvidia-(?:glcore|eglcore|"
            r"rtcore|allocator|api)|nvidia-egl-wayland)[^/]*\.so(?:\.|$)",
            re.I,
        ),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|onnx|gguf|bin\.index\.json)$|"
            r"(?:^|/)(?:models?|weights?|checkpoints?)(?:/|[^/]*)[^/]*"
            r"\.(?:bin|pt|pth)$)",
            re.I,
        ),
    ),
    (
        "sample_or_customer_payload",
        re.compile(
            r"(?:^|/)opt/content-agents/(?:examples?|samples?|tests?|"
            r"apps/[^/]+/(?:data|examples?|tests?))(?:/|$)|"
            r"(?:^|/)opt/content-agents/[^!]*\.(?:usd|usda|usdc|usdz|png|jpe?g|"
            r"webp|mp4|mov|mcap|bag)$",
            re.I,
        ),
    ),
    (
        "runtime_or_package_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:uv|pip|huggingface|wu/ovrtx|ovrtx)|"
            r"npa-model-cache/runtimes/content-agents|\.ovrtx_venv|wheelhouse)"
            r"(?:/|$)",
            re.I,
        ),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|"
            r"kubeconfig|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
    ("customer_workspace_data", re.compile(r"(?:^|/)workspace/.+", re.I)),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ovrtx_install_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:pylock\.ovrtx-runtime|pypi\.nvidia\.com/ovrtx|"
            r"\bpip\s+install\b[^\n]*\bovrtx(?:==|\s))",
            re.I | re.S,
        ),
    ),
    (
        "ovrtx_bootstrap_at_build",
        re.compile(r"\bRUN\b[^\n]*render_ovrtx[^\n]*--provision-only", re.I | re.S),
    ),
    (
        "content_agents_acceptance_gate",
        re.compile(
            r"\b(?:NPA_CONTENT_AGENTS_ACCEPT_NVIDIA_OMNIVERSE_TERMS|"
            r"OMNI_KIT_ACCEPT_EULA|ACCEPT_EULA)\b",
            re.I,
        ),
    ),
    (
        "credential_in_image_config",
        re.compile(
            r"\b(?:HF_TOKEN|NGC_API_KEY|NVIDIA_API_KEY|NEBIUS_TOKEN_FACTORY_KEY|"
            r"AWS_SECRET_ACCESS_KEY)\s*=\s*[^$<\s][^\s,\"]{7,}",
            re.I,
        ),
    ),
)

SECRET_CONTENT: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"(?<![0-9A-Z])AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{34}(?![A-Za-z0-9_])"),
    re.compile(
        rb"(?<![A-Za-z0-9_-])nvapi-[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])",
        re.I,
    ),
    re.compile(
        rb"(?<![A-Za-z0-9_.-])v1\.[A-Za-z0-9_-]{100,}\."
        rb"[A-Za-z0-9_-]{80,}(?![A-Za-z0-9_.-])"
    ),
    re.compile(
        rb"(?i)aws_secret_access_key\s*[=:]\s*"
        rb"[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"
    ),
)

FORBIDDEN_ELF_DEPENDENCY = re.compile(
    rb"lib(?:ovrtx|carb|omni[A-Za-z0-9_.-]*|GLX_nvidia|EGL_nvidia|"
    rb"nvidia-(?:glcore|eglcore|rtcore|allocator|api))[^\x00/]*\.so",
    re.I,
)


class ImageAuditError(RuntimeError):
    """Raised when OCI metadata violates the public packaging contract."""


def _image_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("config")
    return nested if isinstance(nested, dict) else config


def audit_config(
    config: dict[str, Any], *, expected_npa_source_sha: str = ""
) -> dict[str, Any]:
    image_config = _image_config(config)
    labels = image_config.get("Labels") or image_config.get("labels") or {}
    expected = {
        "npa.tool": "content-agents",
        "npa.redistribution": "public",
        "npa.driver_provisioning": "gpu-operator-host-mounted",
        "npa.driver_capabilities": "compute,utility,graphics,display",
        "npa.ovrtx.delivery": "runtime-fetch-from-nvidia",
        "npa.ovrtx.version": OVRTX_VERSION,
        "npa.content_agents.version": "0.5.2",
        "org.opencontainers.image.revision": CONTENT_AGENTS_REVISION,
        "org.opencontainers.image.version": IMAGE_VERSION,
        "org.opencontainers.image.licenses": "Apache-2.0",
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise ImageAuditError(f"image label {key} is not {value!r}")
    source_sha = str(labels.get("npa.source_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ImageAuditError("npa.source_revision is not a full Git SHA")
    if expected_npa_source_sha and source_sha != expected_npa_source_sha:
        raise ImageAuditError(
            "npa.source_revision differs from the requested checkpoint"
        )
    if str(image_config.get("User") or image_config.get("user") or "") not in {
        "ubuntu",
        "1000",
        "1000:1000",
    }:
        raise ImageAuditError("final image user is not the reviewed non-root identity")
    return {"labels": expected, "npa_source_revision": source_sha}


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files={},
        audited_libraries={},
        secret_content=SECRET_CONTENT,
        forbidden_elf_dependency=FORBIDDEN_ELF_DEPENDENCY,
    ):
        return walker.scan_tars(tars, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-npa-source-sha", default="")
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.rootfs_tar):
        parser.error("provide exactly one IMAGE or --rootfs-tar")
    if args.expected_npa_source_sha and not re.fullmatch(
        r"[0-9a-f]{40}", args.expected_npa_source_sha
    ):
        parser.error("--expected-npa-source-sha must be a full lowercase Git SHA")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-content-agents-byte-scan-") as tmp:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            metadata = audit_config(
                config, expected_npa_source_sha=args.expected_npa_source_sha
            )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - a scanner must fail closed
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_content_agents_image_byte_scan_v2",
        "image": args.image or "offline-rootfs",
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "findings": [asdict(item) for item in findings],
        **metadata,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
