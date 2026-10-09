"""Keep source rebuilds from inheriting the image bytes withdrawn by #807."""

from pathlib import Path
import importlib.util
import re

from packaging.version import Version
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKBENCH = ROOT / "npa/docker/workbench"
WITHDRAWN_PARENTS = {
    "isaac-lab": "e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6",
    "envgen": "08eb75118f5a04194d33a60308212db7706dd9c339d74afc5471a58608bf0422",
}


@pytest.mark.parametrize(
    "child,parent",
    [
        ("isaac-arena", "isaac-lab"),
        ("sim2real-reference-policy", "envgen"),
        ("sim2real-explore-policy", "envgen"),
        ("lerobot-vlm-rl", "envgen"),
        ("sim2real-eval", "envgen"),
    ],
)
def test_rebuilt_children_cannot_inherit_withdrawn_parent_layers(child, parent):
    text = (WORKBENCH / child / "Dockerfile").read_text()
    pin = re.search(
        rf"(?:^FROM |^ARG BASE_IMAGE=)(ghcr.io/nebius/nebius-physical-ai/"
        rf"npa-{parent}@sha256:([0-9a-f]{{64}}))$",
        text,
        re.MULTILINE,
    )
    assert pin is not None, "A repaired child still requires an immutable parent"
    assert pin.group(2) != WITHDRAWN_PARENTS[parent]


def test_arena_safety_catalog_tracks_the_actual_parent():
    text = (WORKBENCH / "isaac-arena/Dockerfile").read_text()
    parent = re.search(r"^FROM (\S+)$", text, re.MULTILINE).group(1)
    manifest = yaml.safe_load(
        (ROOT / "npa/src/npa/smoke/golden_evals.yaml").read_text()
    )
    assert manifest["containers"]["isaac-arena"]["safety"]["base_image"] == parent


@pytest.mark.parametrize(
    "child", ["sim2real-reference-policy", "sim2real-explore-policy"]
)
def test_policy_children_bind_the_actual_sdk_source_to_their_revision(child):
    text = (WORKBENCH / child / "Dockerfile").read_text()
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in text
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in text
    for source in ("pyproject.toml", "src/npa", "workflows"):
        assert f"COPY --chown=ubuntu:ubuntu {source} " in text
    assert text.index("RUN rm -rf /opt/npa/src/npa") < text.index(
        "COPY --chown=ubuntu:ubuntu src/npa "
    )
    assert text.index("COPY --chown=ubuntu:ubuntu src/npa ") < text.index(
        "&& env -u PYTHONPATH python -c"
    )
    assert "find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx" in text
    assert text.index("USER ubuntu") > text.index("python -m pip check")
    assert "verify-sim2real-sdk-source.py" in text
    assert "--source-root /opt/npa/src" in text


def test_thin_child_sdk_verification_preserves_image_independent_config(monkeypatch):
    from npa.deploy import images

    def fail_image_resolution(*args, **kwargs):
        pytest.fail("Artifact/diagnostic configuration attempted image planning")

    monkeypatch.setattr(images, "container_image_for_tool", fail_image_resolution)
    source = WORKBENCH / "common/verify_sim2real_sdk_source.py"
    spec = importlib.util.spec_from_file_location("verify_child_sdk", source)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(ROOT / "npa/src")
    with pytest.raises(RuntimeError, match="different SDK source"):
        verifier.verify(ROOT / "unrelated-source")


@pytest.mark.parametrize("name,floor", [("GitPython", "3.1.62"), ("Werkzeug", "3.1.9")])
def test_isaac_2_runtime_dependency_fixes_preserve_the_runtime_family(name, floor):
    requirements = (WORKBENCH / "common/isaac-oss-deps.txt").read_text()
    pin = re.search(rf"^{name}==(\S+)$", requirements, re.MULTILINE)
    assert pin and Version(pin.group(1)) >= Version(floor)
    assert "ARG ISAAC_LAB_VERSION=2.3.2" in (WORKBENCH / "sonic/Dockerfile").read_text()


def test_loop_eval_uses_the_fixed_kernel_header_snapshot():
    text = (WORKBENCH / "sim2real-eval/Dockerfile").read_text()
    assert "ARG UBUNTU_SNAPSHOT=20261001T000000Z" in text
    assert (
        'install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" 5.15.0-194.204' in text
    )
