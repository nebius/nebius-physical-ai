from pathlib import Path

from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "Dockerfile"


def test_isaac_arena_image_is_exact_source_and_payload_clean_by_construction() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ed0fd12be862078be316c73eb7cf423ba9b1c5cd" in text
    assert "4e62ddbd7edc40fb47e62a0d5ba523eebc129482b4ce083612f17c79f1fc40a8" in text
    assert (
        "npa-isaac-lab@sha256:e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6"
        in text
    )
    assert "--output /tmp/isaac-arena.tar.gz" in text
    assert "sha256sum -c -" in text
    assert "grep -q '^Apache License' /opt/isaac-arena/LICENSE.md" in text
    assert 'test -z "$(find /opt/isaac-arena' in text
    assert text.rstrip().endswith(
        'ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]'
    )
    assert "USER ubuntu" in text


def test_isaac_arena_image_catalog_identity() -> None:
    assert CONTAINER_IMAGE_NAMES["isaac-arena"] == "npa-isaac-arena"
    assert SUPPORTED_TOOL_VERSIONS["isaac-arena"] == "0.3.0-isaaclab3-unbuilt"
