from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"
BUILD_SCRIPT = IMAGE_ROOT / "build.sh"
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_robomimic_image_contract", IMAGE_ROOT / "verify_image.py"
)
assert VERIFY_SPEC and VERIFY_SPEC.loader
VERIFIER = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFIER)


def test_neutral_image_is_pinned_non_root_and_contains_no_cuda_install() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "FROM --platform=linux/amd64 python:3.11.16-slim-bookworm@sha256:"
        "528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
    ) in text
    assert "USER ubuntu" in text
    assert 'ENTRYPOINT ["/opt/npa/robomimic/entrypoint.sh"]' in text
    assert "org.nebius.npa.skypilot-bootstrap-contract" in text
    normalized = re.sub(r"\\\n\s*", " ", text)
    assert re.search(r"\b(?:nvidia/cuda|pytorch/pytorch):", normalized, re.I) is None
    assert (
        re.search(r"\bpip\s+install\b[^\n]*(?:torch|nvidia-)", normalized, re.I) is None
    )
    assert "robomimic-runtime assert-refusal" in normalized
    assert "robomimic-runtime ensure" not in normalized
    assert "robomimic-runtime exec" not in normalized
    assert "apt-get" not in normalized
    assert "git -C /opt/robomimic" not in normalized
    assert "source=build-inputs" in normalized
    assert "dpkg --unpack /mnt/robomimic-build-inputs/debian/*.deb" in normalized
    assert "verify_image.py build-inputs" in normalized
    assert "verify_image.py debian-install" in normalized


def test_neutral_image_boundaries_and_locks_are_explicit() -> None:
    baked = (IMAGE_ROOT / "baked-requirements.lock").read_bytes()
    baked_names = {
        line.split(b"==", 1)[0].decode("ascii")
        for line in baked.splitlines()
        if re.match(rb"^[A-Za-z0-9_.-]+==", line)
    }
    assert baked_names == {
        "anyio",
        "boto3",
        "botocore",
        "certifi",
        "charset-normalizer",
        "contourpy",
        "cycler",
        "diffusers",
        "filelock",
        "fonttools",
        "fsspec",
        "h11",
        "h5py",
        "hf-xet",
        "httpcore",
        "httpx",
        "huggingface-hub",
        "idna",
        "imageio",
        "importlib-metadata",
        "jmespath",
        "kiwisolver",
        "matplotlib",
        "numpy",
        "packaging",
        "pillow",
        "psutil",
        "pyparsing",
        "python-dateutil",
        "pyyaml",
        "regex",
        "requests",
        "s3transfer",
        "safetensors",
        "six",
        "termcolor",
        "tqdm",
        "typing-extensions",
        "urllib3",
        "zipp",
    }
    assert set(
        VERIFIER._locked_baked_packages(IMAGE_ROOT / "baked-requirements.lock")
    ) == (baked_names)
    # Import reachability at the pinned source revision is broader than the BC
    # code path itself: algo registration imports Diffusion Policy, while the
    # observation stack imports vis_utils and therefore matplotlib eagerly.
    assert {
        "diffusers",
        "huggingface-hub",
        "imageio",
        "matplotlib",
    } <= baked_names
    # These upstream install requirements are lazy and are not exercised by the
    # headless, non-language, non-rendering gate.
    assert {
        "egl-probe",
        "imageio-ffmpeg",
        "tensorboard",
        "tensorboardx",
        "transformers",
    }.isdisjoint(baked_names)
    lowered = baked.lower()
    for forbidden in (b"torch==", b"torchvision==", b"triton==", b"nvidia-"):
        assert forbidden not in lowered
    runtime = json.loads((IMAGE_ROOT / "runtime-requirements.lock").read_text())
    assert runtime["schema"] == "npa.robomimic.runtime-lock.v1"
    assert runtime["packages"]["torch"] == "2.7.1+cu128"
    assert runtime["packages"]["nvidia-cudnn-cu12"] == "9.7.1.26"
    assert runtime["artifact_hash_closure"] == "required-in-runtime-inventory"
    for filename in (
        "debian-packages.lock",
        "REDISTRIBUTION.md",
        "source-manifest.json",
        "THIRD_PARTY_NOTICES.md",
        "runtime_bootstrap.sh",
        "verify_image.py",
        "smoke.py",
    ):
        assert (IMAGE_ROOT / filename).is_file()


def test_no_local_consent_proxy_exists() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in IMAGE_ROOT.iterdir()
        if path.is_file()
    )
    assert "NPA_ROBOMIMIC_ACCEPT" not in combined
    assert "CUDA_ACCEPT" not in combined
    assert "CUDNN_ACCEPT" not in combined


def test_build_helper_defaults_local_and_uses_only_committed_context() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"' in text
    assert "NPA_PUBLIC_REGISTRY" not in text
    assert 'archive "${revision}:npa/docker/workbench/robomimic"' in text
    assert '"${context}/verify_image.py" prepare-build-inputs' in text
    assert (
        'docker build --platform linux/amd64 --pull=false --tag "${image}" '
        '"${context}"'
    ) in text
    assert '"${repo_root}/npa/docker/workbench/robomimic"' not in text


def test_immutable_build_locks_bind_exact_source_and_debian_bytes() -> None:
    debian = VERIFIER._debian_lock(IMAGE_ROOT / "debian-packages.lock")
    source = VERIFIER._source_manifest(IMAGE_ROOT / "source-manifest.json")
    assert len(debian["packages"]) == 78
    assert len(debian["sources"]) == 58
    assert sum(item["size"] for item in debian["packages"]) == 29_807_672
    assert debian["roots"] == [
        "ca-certificates",
        "openssh-server",
        "procps",
        "rsync",
        "sudo",
    ]
    assert all(item["debian_license_labels"] for item in debian["sources"])
    assert source["git_tree_sha1"] == "4c8ebe35dbef16126dadf59cf8b771b9203753ab"
    assert source["archive"] == {
        "format": "git-archive-tar",
        "path": "source/robomimic.tar",
        "size": 57_907_200,
        "sha256": "8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b",
        "member_count": 226,
        "regular_file_count": 201,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lock: lock["packages"][0].update(url="http://snapshot.debian.org/a.deb"),
        lambda lock: lock["packages"][0]["depends"].append("missing-package"),
        lambda lock: lock["sources"][0].update(
            license_reference_url="https://example.invalid/copyright"
        ),
        lambda lock: lock["packages"][0].update(name="nvidia-cuda"),
        lambda lock: lock["packages"].pop(),
    ],
)
def test_debian_lock_parser_rejects_hostile_mutations(mutation) -> None:
    lock = json.loads((IMAGE_ROOT / "debian-packages.lock").read_text())
    mutation(lock)
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER._checked_debian_lock(lock)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "0" * 40),
        ("git_tree_sha1", "f" * 40),
        ("repository", "https://example.invalid/robomimic.git"),
    ],
)
def test_source_manifest_parser_rejects_hostile_mutations(field, value) -> None:
    manifest = json.loads((IMAGE_ROOT / "source-manifest.json").read_text())
    manifest[field] = value
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER._checked_source_manifest(manifest)


def _source_archive(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, dict]:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for name, raw in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    raw = archive_bytes.getvalue()
    path = tmp_path / "source.tar"
    path.write_bytes(raw)
    manifest = {
        "archive": {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "member_count": len(members),
            "regular_file_count": len(members),
        },
        "license": {
            "path": "LICENSE",
            "sha256": hashlib.sha256(members["LICENSE"]).hexdigest(),
        },
    }
    return path, manifest


def test_source_archive_parser_rejects_path_traversal(tmp_path: Path) -> None:
    path, manifest = _source_archive(
        tmp_path, {"LICENSE": b"MIT\n", "../outside": b"hostile\n"}
    )
    with pytest.raises(VERIFIER.VerificationError, match="unsafe object"):
        VERIFIER._verify_source_archive(path, manifest)


def test_source_archive_parser_rejects_byte_mutation(tmp_path: Path) -> None:
    path, manifest = _source_archive(tmp_path, {"LICENSE": b"MIT\n"})
    manifest = copy.deepcopy(manifest)
    manifest["archive"]["sha256"] = "0" * 64
    with pytest.raises(VERIFIER.VerificationError, match="identity mismatch"):
        VERIFIER._verify_source_archive(path, manifest)
