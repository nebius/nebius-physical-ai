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
        # Ratchet, not a live guard: nothing reads these variables any more, so
        # this can only fire if a future change reintroduces the self-certified
        # declaration that was deliberately removed. Cheap to keep, and the one
        # place a reintroduction would be visible in an artifact.
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
        re.compile(r"\bRUN\b[^\n]*\b(?:hf|huggingface-cli)\s+download\b", re.I | re.S),
    ),
    ("cuda_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
)


# Stock Debian bookworm binaries that carry key-format literals or a CUDA/NVENC
# ELF reference, pinned by exact SHA-256.
#
# An earlier version of this scanner passed empty allowlists, reasoning that a
# near-empty image installs no crypto SDK. That was wrong about our own image:
# the Dockerfile installs `openssh-server` (the SkyPilot bootstrap contract needs
# it) and `ffmpeg` (the output validator decodes with it), and those packages
# bring binaries whose parsers contain "-----BEGIN OPENSSH PRIVATE KEY-----" and
# whose libav* objects reference CUDA for the NVENC path they never take here.
# The scan therefore failed on the image's own base rather than on a payload.
#
# These are audited by identity, not waved through by path: every entry is the
# exact byte sequence apt installed, `dpkg -V` reports no checksum mismatch for
# any of the owning packages, and the walker re-flags any drift from these hashes
# as `audited_literal_byte_drift`. The libav* and libgnutls hashes are also
# byte-identical to the entries independently audited for npa-wan2-2, which is
# corroboration that these are the distribution's bytes and not ours.
#
# A CUDA *reference* is not a CUDA payload: `libcuda.so.1` is provided by the
# host driver at run time and appears in no layer. The rules above still fail on
# any actual CUDA runtime, wheel, or library the image might bake.
AUDITED_LITERAL_LIBRARY_SHA256: dict[str, str] = {
    # ffmpeg 7:5.1.9-0+deb12u1
    "usr/lib/x86_64-linux-gnu/libavcodec.so.59.37.100": (
        "4af5d9cffe2721f5c2cabf35d63c5c6a039b400df5721aaedf69995e37bd2a0d"
    ),
    "usr/lib/x86_64-linux-gnu/libavutil.so.57.28.100": (
        "c46ee8987cacb9f9af711f676cc98bb6d465a340566dad9c678d21baa26e2c9d"
    ),
    # libgnutls30 3.7.9-2+deb12u7
    "usr/lib/x86_64-linux-gnu/libgnutls.so.30.34.3": (
        "779b25d20249988bea2c1aa6bbeb218f5ae7ea8a9d30ce4f54ea37372965cc4b"
    ),
}

AUDITED_SECRET_LITERAL_FILE_SHA256: dict[str, str] = {
    # openssh-client / openssh-server 1:9.2p1-2+deb12u10
    "usr/bin/ssh": "04f2ff5f506a3f332e7adeb1478a4c551ae74acdd328e6fb5c2495664d4064e6",
    "usr/bin/ssh-add": (
        "1e02d3fca3c8d72570c11ded9681dcacd4775a33560786a27c489b9e4388ec06"
    ),
    "usr/bin/ssh-agent": (
        "165e70f42cf6a147ae2101fae05a4503791d95ac9de7cca7974577316d963829"
    ),
    "usr/bin/ssh-keygen": (
        "2843cd46c617cf771c32e6f8a2a11585d83510ec528e3aeff8f7a7f0445420ba"
    ),
    "usr/bin/ssh-keyscan": (
        "9475d0851a26a4f494dc7d40240b68b04874543cbd30f54195ff9af055aaf848"
    ),
    "usr/lib/openssh/ssh-keysign": (
        "c4f5b14604acf3cd0111ca969a0975194e4219f818565df12a7f3ad4663ca93a"
    ),
    "usr/lib/openssh/ssh-pkcs11-helper": (
        "9951c90e5173921f2c3ae553d6b2ce5967862944c12eb63d635586130728bd07"
    ),
    "usr/lib/openssh/ssh-sk-helper": (
        "9caaf480bbd3bfb916195c8ac868e3a1e78968408500f090f037b4a186aac1df"
    ),
    "usr/sbin/sshd": (
        "9f6cdc787a2d5144f3189e850fc104aa7d8ab12593a3d4e902c692a38794716e"
    ),
    # libssh-4 / libssh2-1 / libmbedcrypto7 / libunistring2, pulled in by curl
    # and ffmpeg's network protocols.
    "usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.9.6": (
        "6733636aeb1c5d541aa06c578a95d12c2253c4e99fc85623595341905d3d6221"
    ),
    "usr/lib/x86_64-linux-gnu/libssh2.so.1.0.1": (
        "66f751ec9d3d5bff254a498e020d37f9bbde8e0193ba11d1b78f29153ffe694a"
    ),
    "usr/lib/x86_64-linux-gnu/libmbedcrypto.so.2.28.3": (
        "c04f91fdb172e17ddb21c9e0b75c04cb4f802bdfc40bb65484550746cd0019a8"
    ),
    "usr/lib/x86_64-linux-gnu/libunistring.so.2.2.0": (
        "bc5951aa3d6eaba20ff9688efa3420dc95785aae3709ec48ff6df46d6f409ee5"
    ),
}


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one tar plus image history under the LTX policy (used by tests)."""

    return scan_tars([rootfs_tar], config)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan layer/rootfs tars plus image history under the LTX policy."""

    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files=AUDITED_SECRET_LITERAL_FILE_SHA256,
        audited_libraries=AUDITED_LITERAL_LIBRARY_SHA256,
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
