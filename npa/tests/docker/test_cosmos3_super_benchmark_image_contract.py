from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa/docker/workbench/cosmos3-super-benchmark"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
BUILD = IMAGE_DIR / "build.sh"
CONTRACT = ROOT / "npa/docker/workbench/packaging-contract.yaml"
UPSTREAM_DIGEST = (
    "sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587"
)


def test_wrapper_preserves_exact_upstream_and_adds_only_bootstrap_surface() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert f"docker.io/vllm/vllm-omni:cosmos3@{UPSTREAM_DIGEST}" in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert "openssh-server rsync sudo" in text
    assert "dpkg-query -W openssh-server rsync sudo" in text
    assert not re.search(r"(?i)(HF_TOKEN|NGC_API_KEY|ACCEPT.*=YES)", text)
    assert "USER ubuntu" in text
    assert "UV_CACHE_DIR=/home/ubuntu/.cache/uv" in text
    assert 'ENTRYPOINT ["/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint"]' in text


def test_wrapper_is_operator_private_and_never_publicly_publishable() -> None:
    from npa.deploy.images import (
        RESTRICTED_PUBLICATION_TOOLS,
        is_publicly_redistributable,
    )

    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["cosmos3-super-benchmark"]
    assert entry["redistribution"] == "restricted"
    assert "cosmos3-super-benchmark" in RESTRICTED_PUBLICATION_TOOLS
    assert is_publicly_redistributable("cosmos3-super-benchmark") is False
    assert (IMAGE_DIR / "REDISTRIBUTION.md").is_file()


def test_push_build_requires_clean_tree_and_full_sha_tag() -> None:
    text = BUILD.read_text(encoding="utf-8")

    assert "git -C \"$REPO_ROOT\" diff --quiet" in text
    assert "git -C \"$REPO_ROOT\" diff --cached --quiet" in text
    assert "^[0-9a-f]{40}$" in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert "env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN" in text
