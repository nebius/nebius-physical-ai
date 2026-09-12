from pathlib import Path

from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "Dockerfile"
RUNTIME_REQUIREMENTS = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "runtime-requirements.txt"
)


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
    assert "grep -q 'Apache License' /opt/isaac-arena/LICENSE.md" in text
    assert "grep -q 'Version 2.0, January 2004' /opt/isaac-arena/LICENSE.md" in text
    assert "NPA_LIGHT_WORKBENCH_TOOL=isaac-arena" in text
    assert "a96f7b2afe874037ad7738166f124b2eca6eae0e35daec377d7786c26274660c" in text
    assert "npa-cli-requirements.txt" in text
    assert "runtime-requirements.txt" in text
    assert "--require-hashes" in text
    assert "import onnxruntime, pinocchio, pink" in text
    assert "'pin':'4.0.0'" in text
    assert "'pin-pink':'3.3.0'" in text
    assert "'onnxruntime':'1.27.0'" in text
    assert "'daqp':'0.8.5'" in text
    assert "'qpsolvers':'4.12.0'" in text
    assert "npa workbench isaac-arena evaluate" in text
    assert "--dry-run" in text
    assert 'test -z "$(find /opt/isaac-arena' in text
    assert text.rstrip().endswith(
        'ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]'
    )
    assert "USER ubuntu" in text


def test_isaac_arena_runtime_dependency_closure_is_hash_locked() -> None:
    text = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    expected = {
        "daqp": "0.8.5",
        "onnxruntime": "1.27.0",
        "pin": "4.0.0",
        "pin-pink": "3.3.0",
        "qpsolvers": "4.12.0",
    }
    for distribution, version in expected.items():
        assert f"{distribution}=={version} \\" in text
    requirement_lines = [
        line for line in text.splitlines() if line and not line.startswith((" ", "#"))
    ]
    assert requirement_lines
    assert all(line.endswith(" \\") for line in requirement_lines)
    assert text.count("--hash=sha256:") >= len(requirement_lines)


def test_isaac_arena_image_catalog_identity() -> None:
    assert CONTAINER_IMAGE_NAMES["isaac-arena"] == "npa-isaac-arena"
    assert SUPPORTED_TOOL_VERSIONS["isaac-arena"] == "0.3.0-isaaclab3-unbuilt"
