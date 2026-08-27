"""Guard the npa-cosmos3 image's registration and its no-baked-weights posture.

The image is redistributable only because it carries framework SOURCE and never
model weights: every gated Cosmos 3 checkpoint downloads at runtime under the
operator's own Hugging Face (and, when required, NGC) credentials. These tests
keep that property, and the tool's workbench registrations, from silently
regressing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess

try:  # tomllib is stdlib from 3.11; the repo still supports 3.10 via tomli.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib

import yaml
import pytest

from npa.deploy.images import CONTAINER_IMAGE_NAMES, supported_tool_version
from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.smoke.manifest import container

REPO_ROOT = Path(__file__).resolve().parents[3]
NPA_ROOT = REPO_ROOT / "npa"
DOCKERFILE = NPA_ROOT / "docker/workbench/cosmos3/Dockerfile"
BUILD_SCRIPT = NPA_ROOT / "docker/workbench/cosmos3/build.sh"
ENTRYPOINT = NPA_ROOT / "docker/workbench/cosmos3/entrypoint.sh"
SMOKE_SCRIPT = NPA_ROOT / "docker/workbench/cosmos3/smoke_functional.sh"
VERIFY_ENV = NPA_ROOT / "docker/workbench/cosmos3/verify_env.py"
CONTRACT = NPA_ROOT / "docker/workbench/packaging-contract.yaml"
# Deliberate tripwire: do not derive this from the production resolver. Keeping
# an independent literal makes a one-sided tag edit fail instead of teaching the
# test the same mistake; this is safer than coupling packaging tests to another
# mutable tag source.
COSMOS3_RELEASE_TAG = "1.2.2-cu130-r6"

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
    assert pinned == COSMOS3_RELEASE_TAG
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


def test_cosmos3_image_satisfies_the_skypilot_bootstrap_contract() -> None:
    instructions = _dockerfile_instructions()
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert "org.nebius.npa.skypilot-bootstrap-contract" in instructions
    assert "skypilot-0.12.2-v1" in instructions
    assert "rsync" in instructions
    assert "/usr/local/bin/entrypoint.sh" in instructions
    assert "ARG NPA_SOURCE_SHA" in instructions
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in instructions
    assert "NPA_BAKED_PYTHON=/opt/npa/.venv/bin/python" in instructions
    assert 'test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40' in instructions
    assert 'exec "$MODE" "$@"' in entrypoint
    assert "exec sleep infinity" in entrypoint
    assert "checkpoint-eval|generate|reason|text-to-image" in entrypoint
    assert 'CMD ["sleep", "infinity"]' in instructions
    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in instructions
    assert "imageio_ffmpeg/binaries/ffmpeg*" in instructions

    completed = subprocess.run(
        [
            "bash",
            str(ENTRYPOINT),
            "/bin/sh",
            "-c",
            "printf %s skypilot-forwarded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "skypilot-forwarded"


def test_cosmos3_has_one_canonical_build_source() -> None:
    assert not (DOCKERFILE.parent / "Dockerfile.k8s-prereqs").exists()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert contract["images"]["cosmos3"]["dockerfile"] == "cosmos3/Dockerfile"


def test_canonical_build_binds_source_sha_and_requires_registry_input() -> None:
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert BUILD_SCRIPT.stat().st_mode & 0o111
    assert 'NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C' in build_script
    assert '--build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}"' in build_script
    assert 'REGISTRY="${REGISTRY:-}"' in build_script
    assert "pass --registry or set REGISTRY" in build_script
    assert "nebius.cloud/" not in build_script


def _load_verify_env():
    spec = importlib.util.spec_from_file_location("cosmos3_verify_env", VERIFY_ENV)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_build_rejects_the_known_bad_xet_pair(monkeypatch) -> None:
    module = _load_verify_env()
    versions = {"huggingface_hub": "1.23.0", "hf-xet": "1.5.1"}
    monkeypatch.setattr(module.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match="known-bad gated-download pair"):
        module.check_hf_transfer_pair()


def test_image_build_accepts_the_measured_compatible_xet_pair(monkeypatch) -> None:
    module = _load_verify_env()
    versions = {"huggingface_hub": "0.36.2", "hf-xet": "1.3.2"}
    monkeypatch.setattr(module.metadata, "version", versions.__getitem__)

    assert module.check_hf_transfer_pair() == (
        "huggingface_hub=0.36.2 hf-xet=1.3.2 (Xet enabled)"
    )


def test_smoke_script_requires_the_operator_token() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "HF_TOKEN" in text
    assert "npa.smoke.test_cosmos3_generate_functional" in text
