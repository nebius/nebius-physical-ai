"""Guard the npa-cosmos3 image's registration and its no-baked-weights posture.

The image is redistributable only because it carries framework SOURCE and never
model weights: every gated Cosmos 3 checkpoint downloads at runtime under the
operator's own Hugging Face (and, when required, NGC) credentials. These tests
keep that property, and the tool's workbench registrations, from silently
regressing.
"""

from __future__ import annotations

import re
from pathlib import Path

try:  # tomllib is stdlib from 3.11; the repo still supports 3.10 via tomli.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib

import yaml

from npa.deploy.images import CONTAINER_IMAGE_NAMES, supported_tool_version
from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.smoke.manifest import container

REPO_ROOT = Path(__file__).resolve().parents[3]
NPA_ROOT = REPO_ROOT / "npa"
DOCKERFILE = NPA_ROOT / "docker/workbench/cosmos3/Dockerfile"
SMOKE_SCRIPT = NPA_ROOT / "docker/workbench/cosmos3/smoke_functional.sh"
CONTRACT = NPA_ROOT / "docker/workbench/packaging-contract.yaml"

# Anything that would pull weight bytes into a build layer.
WEIGHT_FETCH_PATTERNS = (
    r"hf\s+download",
    r"huggingface-cli\s+download",
    r"snapshot_download",
    r"git\s+lfs\s+pull",
)


def _dockerfile_instructions() -> str:
    lines = [
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def test_cosmos3_is_registered_as_a_container_tool() -> None:
    assert CONTAINER_IMAGE_NAMES["cosmos3"] == "npa-cosmos3"

    pinned = supported_tool_version("cosmos3")
    pyproject = tomllib.loads((NPA_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["npa"]["supported-tools"]["cosmos3"] == pinned


def test_cosmos3_has_a_real_capability_golden_eval() -> None:
    spec = container("cosmos3")

    assert spec.image == "npa-cosmos3"
    assert spec.dockerfile == "npa/docker/workbench/cosmos3/Dockerfile"
    assert spec.golden_eval.kind == "container-smoke"
    assert spec.golden_eval.module == "npa.smoke.test_cosmos3_generate_functional"
    assert spec.golden_eval.gpu == "required"
    assert "cosmos3" in GOLDEN_EVAL_CAPABILITIES


def test_cosmos3_is_declared_public_because_weights_stay_out() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    entry = contract["images"]["cosmos3"]

    assert entry["dockerfile"] == "cosmos3/Dockerfile"
    assert entry["tier"] == "job"
    assert entry["redistribution"] == "public"
    assert "no model weights" in entry["notes"].lower()


def test_dockerfile_never_downloads_model_weights() -> None:
    instructions = _dockerfile_instructions()

    for pattern in WEIGHT_FETCH_PATTERNS:
        assert not re.search(pattern, instructions, flags=re.IGNORECASE), (
            f"npa-cosmos3 Dockerfile matches {pattern!r}: weights must download at "
            "runtime with the operator's own credentials, never into a layer."
        )


def test_dockerfile_pins_the_framework_and_guards_against_baked_weights() -> None:
    instructions = _dockerfile_instructions()

    # A pinned 40-char commit keeps builds reproducible and the golden eval meaningful.
    assert re.search(r"ARG COSMOS3_REF=[0-9a-f]{40}", instructions)
    # The build-time guard that fails if a checkpoint file landed in a layer.
    assert "*.safetensors" in instructions
    assert "model weights baked into image" in instructions
    # Upstream attribution travels with the redistributed source.
    assert "/opt/cosmos3/licenses" in instructions


def test_smoke_script_requires_the_operator_token() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "HF_TOKEN" in text
    assert "npa.smoke.test_cosmos3_generate_functional" in text
