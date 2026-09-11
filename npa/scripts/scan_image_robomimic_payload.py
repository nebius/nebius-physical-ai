#!/usr/bin/env python3
"""Fail closed on CUDA/runtime/data/cache/output bytes in a robomimic image."""

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


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "torch_or_triton_distribution",
        re.compile(
            r"(?:^|/)(?:site-packages|dist-packages)/(?:torch(?:/|vision/)|"
            r"torch(?:vision)?-[^/]*\.dist-info/|triton(?:/|-[^/]*\.dist-info/))",
            re.I,
        ),
    ),
    (
        "nvidia_python_distribution",
        re.compile(
            r"(?:^|/)(?:site-packages|dist-packages)/(?:nvidia(?:/|_)|"
            r"nvidia_[^/]*\.dist-info/)",
            re.I,
        ),
    ),
    (
        "cuda_library",
        re.compile(
            r"(?:^|/)(?:lib)?(?:(?:[a-z0-9]+_)*(?:cuda|cudart|cublas|cudnn|"
            r"nccl|nvrtc|nvjitlink|nvtx|nvtoolsext|nvfatbin|nvptxcompiler|cupti|"
            r"cufile|cusparse|cusolver|curand|cufft|npp[a-z]*)(?:[a-z0-9_-]*)|"
            r"nvidia(?:-[a-z0-9_-]+)?|(?:glx|egl)_nvidia|"
            r"nv(?:cuvid|optix|encode|decode))(?:[^/]*)\.(?:so(?:\.|$)|a$)",
            re.I,
        ),
    ),
    (
        "cuda_tool_or_header",
        re.compile(
            r"(?:^|/)(?:(?:usr/local|opt)/cuda(?:-[0-9.]+)?(?:/|$)|"
            r"usr/include/(?:cuda|cublas|cudnn|nccl|nvrtc|nvToolsExt|npp|cupti|"
            r"cufft|cusparse|cusolver|curand)[^/]*\.h$|(?:nvcc|ptxas|cuobjdump|"
            r"compute-sanitizer|nvidia-smi)(?:$|\.))",
            re.I,
        ),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|pth|pt|bin\.index\.json)$|"
            r"(?:^|/)(?:models?|weights?|checkpoints?)(?:/|$))",
            re.I,
        ),
    ),
    (
        "dataset_payload",
        re.compile(r"\.(?:hdf5|h5)$", re.I),
    ),
    (
        "populated_runtime_cache",
        re.compile(
            r"(?:^|/)(?:opt/npa-runtime/robomimic|\.cache/(?:pip|uv|huggingface)|"
            r"pip-cache|wheelhouse)/(?:.+)",
            re.I,
        ),
    ),
    (
        "run_output_or_proof",
        re.compile(
            r"(?:^|/)workspace/(?:byof-inputs|byof-runs)/.+|"
            r"(?:^|/)robomimic-smoke\.json$",
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

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cuda_or_pytorch_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
    (
        "cuda_install_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:download\.pytorch\.org/whl/cu|"
            r"(?:pip|uv)(?:\s+pip)?\s+install(?:(?!&&|\|\||;)[^\n])*"
            r"(?:nvidia-|torch==[^\s;&|]*\+cu|torchvision==[^\s;&|]*\+cu))",
            re.I,
        ),
    ),
    (
        "runtime_population_at_build",
        re.compile(
            r"\bRUN\b[^\n]*\brobomimic-runtime\s+(?:ensure|install|fetch|warm|exec)\b",
            re.I | re.S,
        ),
    ),
    (
        "dataset_fetch_at_build",
        re.compile(
            r"\bRUN\b[^\n]*(?:robomimic_datasets|low_dim_v15\.hdf5|huggingface-cli\s+download)",
            re.I | re.S,
        ),
    ),
    (
        "invented_acceptance_proxy",
        re.compile(
            r"\b(?:ENV|ARG)\s+[^\n]*\b(?:ACCEPT_EULA|CUDA_ACCEPT|CUDNN_ACCEPT|"
            r"NPA_ROBOMIMIC_ACCEPT)[A-Z0-9_]*\s*=\s*\S+",
            re.I,
        ),
    ),
)


def scan_tars(tars: list[Path], config: dict[str, Any]) -> list[walker.Finding]:
    """Scan every supplied layer and OCI config under the robomimic policy."""

    with walker.payload_policy(
        forbidden_paths=FORBIDDEN_PATHS,
        forbidden_history=FORBIDDEN_HISTORY,
        audited_secret_files={},
        audited_libraries={},
    ):
        return walker.scan_tars(tars, config)


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[walker.Finding]:
    """Scan one test rootfs plus an OCI config."""

    return scan_tars([rootfs_tar], config)


docker_save_material = walker.docker_save_material


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--rootfs-tar", type=Path)
    parser.add_argument("--docker-save", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        sum(bool(value) for value in (args.image, args.rootfs_tar, args.docker_save))
        != 1
    ):
        parser.error("provide exactly one IMAGE, --rootfs-tar, or --docker-save")
    if args.config_json and not args.rootfs_tar:
        parser.error("--config-json is valid only with --rootfs-tar")

    try:
        with tempfile.TemporaryDirectory(prefix="npa-robomimic-byte-scan-") as tmp:
            if args.image:
                tars, config = walker.remote_material(args.image, Path(tmp))
            elif args.docker_save:
                tars, config = docker_save_material(args.docker_save, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = scan_tars(tars, config)
    except Exception as exc:  # noqa: BLE001 - every unreadable artifact fails closed
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_robomimic_image_byte_scan_v1",
        "image": args.image
        or ("docker-save" if args.docker_save else "offline-rootfs"),
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
    raise SystemExit(main())
