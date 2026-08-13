#!/usr/bin/env python3
"""Fail closed when a built npa-ltx2 image contains LTX-2.5 or CUDA bytes.

npa-ltx2's whole redistribution argument is that it ships none of LTX-2.5 — not
the weights, and (unusually) not the ``ltx-core`` / ``ltx-pipelines`` source
either, because the LTX-2.x Community License Agreement covers the code as well.
That claim is about bytes in layers, so reading the Dockerfile does not verify
it. This does.

The archive traversal is shared with ``scan_image_wan_payload`` under its
documented ``payload_policy`` extension point; only the pattern tables below are
LTX-specific.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_image_wan_payload as walker  # noqa: E402

# Bytes that must not exist in any layer.
FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ltx_python_distribution",
        # The LTX-specific finding: an installed ltx_core/ltx_pipelines/ltx_trainer
        # package, or its dist-info, means the image is redistributing licensed
        # LTX-2.x source. This is the one most likely to appear by accident,
        # because "bake the code, fetch the weights" is the habitual shape for
        # every other model in this workbench.
        re.compile(
            r"(?:^|/)(?:site-packages|dist-packages)/"
            r"(?:ltx_(?:core|pipelines|trainer|kernels)(?:/|-)|"
            r"ltx[-_](?:core|pipelines|trainer|kernels)[^/]*\.(?:dist-info|egg-info)/)",
            re.I,
        ),
    ),
    (
        "ltx_source_tree",
        re.compile(
            r"(?:^|/)(?:packages/ltx-(?:core|pipelines|trainer|kernels)/|"
            r"ltx_pipelines/(?:distilled|dfr_pipeline|ti2vid_two_stages)\.py$)",
            re.I,
        ),
    ),
    (
        "ltx_weight_file",
        re.compile(r"(?:^|/)(?:ltx-2\.[0-9]+-|gemma4-12b-with-proj-ltx)[^/]*", re.I),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|bin\.index\.json)$|"
            r"(?:^|/)(?:pytorch_model|diffusion_pytorch_model|model|weights?|checkpoint)"
            r"[^/]*\.(?:bin|pt|pth)$)",
            re.I,
        ),
    ),
    (
        "nvidia_python_distribution",
        re.compile(
            r"(?:^|/)site-packages/(?:nvidia(?:/|_)|nvidia_[^/]*\.dist-info/)", re.I
        ),
    ),
    (
        "cuda_library",
        re.compile(
            r"(?:^|/)(?:lib)?(?:(?:[a-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|"
            r"nccl|nvrtc|nvjitlink|nvtx|nvtoolsext|nvfatbin|nvptxcompiler|cupti|"
            r"cufile|cusparse|cusolver|curand|cufft|npp[a-z]*)"
            r"(?:[a-z0-9_-]*)|nvidia(?:-[a-z0-9_-]+)?|(?:glx|egl)_nvidia|"
            r"nv(?:cuvid|optix|encode|decode))(?:[^/]*)"
            r"\.(?:so(?:\.|$)|a$)",
            re.I,
        ),
    ),
    (
        "torch_distribution",
        # torch itself is only ever installed into the operator's runtime cache,
        # never a layer; a baked torch means the CUDA gate was bypassed at build.
        re.compile(
            r"(?:^|/)site-packages/(?:torch(?:/|audio/|vision/)|"
            r"torch[^/]*\.dist-info/)",
            re.I,
        ),
    ),
    (
        "package_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:pip|uv|huggingface)|pip-cache|wheelhouse)(?:/|$)",
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
)

# Build history that would mean the image performed, at build time, the fetch the
# runtime gate exists to force the operator to authorise.
FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "baked_ltx_acceptance",
        # Pre-granting acceptance removes the operator from the licensing
        # decision, and the refusal is the legal mechanism — so baking the answer
        # is itself the violation, exactly as with the Isaac EULA variables.
        re.compile(
            r"\b(?:NPA_LTX_ACCEPT_COMMUNITY_LICENSE|NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS)"
            r"\s*=\s*(?:YES|yes|Yes)\b"
        ),
    ),
    (
        "baked_ltx_declaration",
        re.compile(
            r"\b(?:ENV|ARG)\s+[^\n]*\b(?:NPA_LTX_ENTITY_CLASS|NPA_LTX_USE_CLASS|"
            r"NPA_LTX_COMMERCIAL_AGREEMENT_REF)\s*=\s*\S",
            re.I,
        ),
    ),
    (
        "runtime_bootstrap_at_build",
        # The distinction that matters is *performs the fetch* vs. *proves the
        # fetch is refused*. `ensure`/`warm`/`exec`/`fetch-weights` download;
        # `health`, `status`, `version`, `terms`, and `assert-refusal` cannot —
        # assert-refusal scrubs the acceptance variables from its own child
        # environment, which is why the Dockerfile is allowed to run it.
        re.compile(
            r"\bRUN\b[^\n]*\bltx-runtime\s+(?:ensure|warm|exec|fetch-weights)\b",
            re.I | re.S,
        ),
    ),
    (
        "ltx_install_at_build",
        # `uv sync` needs no LTX-shaped argument to bake LTX: run inside the
        # fetched source tree it installs the whole workspace. The image never
        # syncs at build, so any build-time sync is a finding on its own.
        re.compile(
            r"\bRUN\b[^\n]*(?:\buv\s+sync\b|"
            r"(?:pip|uv)\s+(?:pip\s+)?install"
            r"(?:(?!&&|\|\||;)[^\n])*(?:ltx[-_](?:core|pipelines|trainer)|"
            r"github\.com/Lightricks))",
            re.I,
        ),
    ),
    (
        "cuda_install_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:download\.pytorch\.org/whl/cu|"
            r"pip\s+install(?:(?!&&|\|\||;)[^\n])*(?:nvidia-|torch==[^\s;&|]*\+cu))",
            re.I,
        ),
    ),
    (
        "hf_download_at_build",
        re.compile(
            r"\bRUN\b[^\n]*\b(?:hf|huggingface-cli)\s+download\b", re.I | re.S
        ),
    ),
    ("cuda_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
)


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one tar plus image history under the LTX policy (used by tests)."""

    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan layer/rootfs tars plus image history under the LTX policy."""

    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        # npa-ltx2 has no audited secret-shaped literals: it installs no AWS or
        # crypto SDK into a layer. An empty allowlist means any secret-shaped
        # bytes fail, which is the correct default for a near-empty image.
        audited_secret_files={},
        audited_libraries={},
    ):
        return walker.scan_tars(tars, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.rootfs_tar):
        parser.error("provide exactly one IMAGE or --rootfs-tar")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-ltx-byte-scan-") as tmp:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - any failure must fail closed
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_ltx_image_byte_scan_v1",
        "image": args.image or "offline-rootfs",
        "status": "pass" if not findings else "fail",
        "archives_scanned": len(tars),
        "findings": [asdict(item) for item in findings],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
