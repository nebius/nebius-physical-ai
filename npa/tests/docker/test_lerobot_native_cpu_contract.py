"""Prevent publishing a LeRobot image with an incompatible decoder closure."""

from __future__ import annotations

import ast
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "npa/docker/workbench/lerobot"
# The explicitly versioned integration bounds; stock upstream bounds differ.
INTEGRATION = {
    "torch": ">=2.13.0,<2.14.0",
    "torchvision": ">=0.28.0,<0.29.0",
    "torchcodec": ">=0.16.0,<0.17.0",
    "diffusers": ">=0.38.0,<0.39.0",
    "wandb": ">=0.30.0,<0.31.0",
}


def _check_constraints(text):
    requirements = {
        requirement.name: requirement
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
        for requirement in [Requirement(line)]
    }
    assert set(requirements) == set(INTEGRATION)
    for name, requirement in requirements.items():
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1 and specifiers[0].operator == "=="
        version = Version(specifiers[0].version)
        assert version in Requirement(name + INTEGRATION[name]).specifier
    assert Version(list(requirements["torch"].specifier)[0].version) == Version(
        "2.13.0"
    )
    assert Version(list(requirements["torchcodec"].specifier)[0].version) == Version(
        "0.16.0"
    )


def test_default_dependencies_identify_secure_integration_and_decoder_abi():
    _check_constraints((RECIPE / "default-runtime-requirements.txt").read_text())
    recipe = (RECIPE / "Dockerfile").read_text()
    assert "--constraint /opt/lerobot/default-constraints.txt" in recipe
    assert "pip uninstall -y wandb" not in recipe
    assert '"torch==2.12.1"' not in recipe
    assert '"diffusers>=0.38.0"' not in recipe
    assert "pip check" in recipe
    assert "COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa" in recipe
    assert "prepare-secure-wheel.py" in recipe
    assert "lerobot-0.5.1+npa.secure1-py3-none-any.whl" in recipe
    assert (
        'ENTRYPOINT ["/opt/lerobot/venv/bin/python", "-m", "npa.server.app"]' in recipe
    )
    assert 'CMD /opt/lerobot/venv/bin/python -c "' in recipe
    assert recipe.index("ENV PATH=/usr/bin:$PATH") < recipe.index("RUN --network=none")


@pytest.mark.parametrize(
    "original,replacement",
    [("torch==2.13.0", "torch==2.12.1"), ("torchcodec==0.16.0", "torchcodec==0.10.0")],
)
def test_invalid_secure_or_decoder_constraints_are_rejected(original, replacement):
    text = (RECIPE / "default-runtime-requirements.txt").read_text()
    with pytest.raises(AssertionError):
        _check_constraints(text.replace(original, replacement))


def test_build_gate_uses_native_default_decoder_without_network_access():
    recipe = (RECIPE / "Dockerfile").read_text()
    assert (
        "RUN --network=none /opt/lerobot/venv/bin/python /opt/lerobot/smoke-native-cpu.py"
        in recipe
    )
    source = (RECIPE / "smoke_native_cpu.py").read_text()
    tree = ast.parse(source)
    dataset_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "LeRobotDataset"
    ]
    assert dataset_calls
    assert all(
        "video_backend" not in {arg.arg for arg in call.keywords}
        for call in dataset_calls
    )
    assert "loss.backward()" in source and "optimizer.step()" in source
    assert "changed > 0" in source
    assert "ACTPolicy.from_pretrained(checkpoint)" in source
    assert "pretrained_path=str(checkpoint)" in source
    assert "from npa.server.app import app" in source
    assert "from npa.server.app import PolicyState" in source
    assert "state.load(str(checkpoint))" in source
    assert "state.predict(" in source
    assert "torch.testing.assert_close(expected.cpu(), actual)" in source
    assert '[interpreter, "-m", "npa.server.app"]' in source
    http_calls = {
        (call.args[1].value, call.args[2].value)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_http_request"
    }
    assert {("GET", "/health"), ("POST", "/serve"), ("POST", "/infer")} <= http_calls
