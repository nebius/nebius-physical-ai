#!/usr/bin/env python3
"""Fail closed when a built Wan image contains forbidden shipped bytes/history."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    detail: str


FORBIDDEN_PATHS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bundled_ffmpeg_executable",
        re.compile(r"(?:^|/)site-packages/imageio_ffmpeg/binaries/ffmpeg[^/]*$", re.I),
    ),
    (
        "historical_lgpl_python_distribution",
        re.compile(
            r"(?:^|/)site-packages/(?:easydict|soxr|librosa)-[^/]*\.dist-info/",
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
            r"(?:^|/)(?:lib)?(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|cusparse|cusolver|curand|cufft)(?:[^/]*)\.(?:so(?:\.|$)|a$)",
            re.I,
        ),
    ),
    (
        "checkpoint_or_weight",
        re.compile(
            r"(?:\.(?:safetensors|ckpt|bin\.index\.json)$|"
            r"(?:^|/)(?:models?|weights?|checkpoints?)(?:/|[^/]*)[^/]*\.(?:bin|pt|pth)$)",
            re.I,
        ),
    ),
    (
        "package_cache",
        re.compile(
            r"(?:^|/)(?:\.cache/(?:pip|huggingface)|pip-cache|wheelhouse)(?:/|$)",
            re.I,
        ),
    ),
    (
        "credential_file",
        re.compile(
            r"(?:^|/)(?:\.aws/credentials|\.docker/config\.json|\.git-credentials|kubeconfig|etc/ssh/ssh_host_(?:rsa|ecdsa|ed25519)_key)$",
            re.I,
        ),
    ),
)

FORBIDDEN_HISTORY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cuda_base", re.compile(r"\b(?:nvidia/cuda|pytorch/pytorch):", re.I)),
    (
        "cuda_install_at_build",
        re.compile(
            r"\bRUN\b.*(?:download\.pytorch\.org/whl/cu|pip\s+install[^\n]*(?:nvidia-|torch==[^\n]*\+cu))",
            re.I | re.S,
        ),
    ),
    (
        "runtime_bootstrap_at_build",
        re.compile(r"\bRUN\b.*\bwan-runtime\s+(?:ensure|warm|exec)\b", re.I | re.S),
    ),
    (
        "baked_acceptance",
        re.compile(r"\bNPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS\s*=\s*YES\b", re.I),
    ),
)

SECRET_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i)(?:aws_secret_access_key|hf_token)\s*[=:]\s*[^$<\s][^\s]{7,}"),
)

SECRET_CONTENT_EXCLUSIONS = (
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/PIL/(?:ImageFont\.py|__pycache__/ImageFont\.)"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/accelerate/commands/config/sagemaker\.py$"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/boto3/(?:examples/cloudfront\.rst|session\.py)$"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/boto3-[^/]+\.dist-info/METADATA$"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/botocore/(?:credentials\.py|data/(?:iam/2010-05-08|sts/2011-06-15)/examples-1\.json)$"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/botocore-[^/]+\.dist-info/METADATA$"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/cryptography/hazmat/primitives/serialization/(?:ssh\.py|__pycache__/ssh\.)"
    ),
    re.compile(
        r"^opt/wan-base/lib/python3\.10/site-packages/diffusers/loaders/textual_inversion\.py$"
    ),
    re.compile(
        r"^usr/(?:bin/ssh(?:-add|-agent|-keygen|-keyscan)?|lib/openssh/ssh-(?:keysign|pkcs11-helper|sk-helper)|sbin/sshd)$"
    ),
    re.compile(
        r"^usr/lib/x86_64-linux-gnu/(?:libmbedcrypto|libssh-gcrypt|libssh2|libunistring)\.so\."
    ),
)


def _scan_secret_content(path: str) -> bool:
    """Scan mutable/application paths, not known OSS binaries and example source."""

    return not any(pattern.search(path) for pattern in SECRET_CONTENT_EXCLUSIONS)


def _history_text(config: dict[str, Any]) -> str:
    return (
        "\n".join(
            str(entry.get("created_by") or "")
            for entry in config.get("history") or []
            if isinstance(entry, dict)
        )
        + "\n"
        + "\n".join(
            str(item) for item in ((config.get("config") or {}).get("Env") or [])
        )
    )


def scan(rootfs_tar: Path, config: dict[str, Any]) -> list[Finding]:
    """Scan one tar plus image history (used directly by mutation tests)."""

    return _scan_tars([rootfs_tar], config)


def _scan_tars(tars: list[Path], config: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for tar_path in tars:
        with tarfile.open(tar_path, "r:*") as archive:
            for member in archive:
                path = member.name.lstrip("./")
                if not path:
                    continue
                for kind, pattern in FORBIDDEN_PATHS:
                    if pattern.search(path):
                        findings.append(
                            Finding(kind, path, f"forbidden bytes in {tar_path.name}")
                        )
                if (
                    not member.isfile()
                    or member.size > 2_000_000
                    or not _scan_secret_content(path)
                ):
                    continue
                fileobj = archive.extractfile(member)
                if fileobj is None:
                    continue
                payload = fileobj.read()
                for secret_pattern in SECRET_CONTENT:
                    if secret_pattern.search(payload):
                        findings.append(
                            Finding(
                                "credential_content",
                                path,
                                f"secret-like bytes in {tar_path.name}",
                            )
                        )
                        break

    history = _history_text(config)
    for kind, pattern in FORBIDDEN_HISTORY:
        if pattern.search(history):
            findings.append(
                Finding(
                    kind, "<image-history>", "forbidden build history or environment"
                )
            )
    return findings


def _repository(image: str) -> str:
    ref = image.removeprefix("docker:").split("@", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    return ref[:colon] if colon > slash else ref


def _remote_material(image: str, temp_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    if not shutil.which("crane"):
        raise RuntimeError("crane is required to scan a built image")
    rootfs = temp_dir / "rootfs.tar"
    with rootfs.open("wb") as stream:
        proc = subprocess.run(
            ["crane", "export", "--platform", "linux/amd64", image, "-"],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode:
        raise RuntimeError(
            f"crane export failed: {proc.stderr.decode(errors='replace')}"
        )
    config_proc = subprocess.run(
        ["crane", "config", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if config_proc.returncode:
        raise RuntimeError(f"crane config failed: {config_proc.stderr}")
    manifest_proc = subprocess.run(
        ["crane", "manifest", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if manifest_proc.returncode:
        raise RuntimeError(f"crane manifest failed: {manifest_proc.stderr}")
    manifest = json.loads(manifest_proc.stdout)
    tars = [rootfs]
    repository = _repository(image)
    for index, layer in enumerate(manifest.get("layers") or []):
        digest = str(layer.get("digest") or "")
        if not digest:
            raise RuntimeError("image manifest layer has no digest")
        layer_path = temp_dir / f"layer-{index:03d}.tar.gz"
        with layer_path.open("wb") as stream:
            layer_proc = subprocess.run(
                ["crane", "blob", f"{repository}@{digest}"],
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
            )
        if layer_proc.returncode:
            raise RuntimeError(
                f"crane blob failed for {digest}: {layer_proc.stderr.decode(errors='replace')}"
            )
        tars.append(layer_path)
    return tars, json.loads(config_proc.stdout)


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
        with tempfile.TemporaryDirectory(prefix="npa-wan-byte-scan-") as tmp:
            if args.image:
                tars, config = _remote_material(args.image, Path(tmp))
            else:
                tars = [args.rootfs_tar]
                config = (
                    json.loads(args.config_json.read_text()) if args.config_json else {}
                )
            findings = _scan_tars(tars, config)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    result = {
        "format": "npa_wan_image_byte_scan_v1",
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
