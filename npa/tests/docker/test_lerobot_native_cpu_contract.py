"""Prevent publishing a LeRobot image with an incompatible decoder closure."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "npa/docker/workbench/lerobot"


def _integration_bounds() -> dict[str, str]:
    path = RECIPE / "prepare_secure_wheel.py"
    spec = importlib.util.spec_from_file_location("secure_lerobot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        Requirement(rewritten).name: rewritten
        for rewritten in module.DEPENDENCIES.values()
    }


# The wheel's rewritten bounds are the source of truth; stock upstream bounds
# differ. Reading them here prevents the test fixture from drifting from the
# wheel metadata it is meant to validate.
INTEGRATION = _integration_bounds()


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
        assert version in Requirement(INTEGRATION[name]).specifier
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
    optional_branch, default_branch = recipe.split(
        'elif [ "${LEROBOT_VERSION}" = "0.5.1" ]; then', 1
    )
    assert "pip uninstall -y wandb" not in optional_branch
    assert "pip uninstall -y wandb" not in default_branch
    assert '"torch==2.12.1"' not in recipe
    assert '"diffusers>=0.38.0"' not in recipe
    check = "/opt/lerobot/venv/bin/pip check"
    assert recipe.count(check) == 1
    assert recipe.index(check) > recipe.index("Unsupported LeRobot package version")
    assert "COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa" not in recipe
    expected_copies = (
        ("src/npa/__init__.py", "/opt/npa/src/npa/__init__.py"),
        ("src/npa/clients/__init__.py", "/opt/npa/src/npa/clients/__init__.py"),
        ("src/npa/clients/storage.py", "/opt/npa/src/npa/clients/storage.py"),
        ("src/npa/server", "/opt/npa/src/npa/server"),
        ("src/npa/smoke", "/opt/npa/src/npa/smoke"),
    )
    for source_path, target_path in expected_copies:
        assert f"COPY --chown=ubuntu:ubuntu {source_path} {target_path}" in recipe
    assert (
        "COPY --chown=ubuntu:ubuntu src/npa/clients /opt/npa/src/npa/clients"
        not in recipe
    )
    assert "stage's own NPA install" in recipe
    assert "shipping it here would make the repair's import" in recipe
    assert "prepare-secure-wheel.py" in recipe
    assert "lerobot-0.5.1+npa.secure1-py3-none-any.whl" in recipe
    assert (
        'ENTRYPOINT ["/opt/lerobot/venv/bin/python", "-m", "npa.server.app"]' in recipe
    )
    assert 'CMD /opt/lerobot/venv/bin/python -c "' in recipe
    assert recipe.index("ENV PATH=/usr/bin:$PATH") < recipe.index(
        'RUN if [ "${LEROBOT_VERSION}" = "0.5.1" ]; then'
    )


@pytest.mark.parametrize(
    "original,replacement",
    [("torch==2.13.0", "torch==2.12.1"), ("torchcodec==0.16.0", "torchcodec==0.10.0")],
)
def test_invalid_secure_or_decoder_constraints_are_rejected(original, replacement):
    text = (RECIPE / "default-runtime-requirements.txt").read_text()
    with pytest.raises(AssertionError):
        _check_constraints(text.replace(original, replacement))


def test_build_gate_uses_native_default_decoder():
    recipe = (RECIPE / "Dockerfile").read_text()
    assert 'RUN if [ "${LEROBOT_VERSION}" = "0.5.1" ]; then' in recipe
    assert "/opt/lerobot/venv/bin/python /opt/lerobot/smoke-native-cpu.py" in recipe
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
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    decoder_contract = functions["_assert_native_decoder_contract"]
    assertions = [
        ast.unparse(node.test)
        for node in decoder_contract.body
        if isinstance(node, ast.Assert)
    ]
    assert "codec == 'torchcodec'" in assertions
    assert "version == '0.16.0'" in assertions
    native_dataset = functions["_native_dataset"]
    contract_call = next(
        node
        for node in ast.walk(native_dataset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_assert_native_decoder_contract"
    )
    dataset_create = next(
        node
        for node in ast.walk(native_dataset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
    )
    assert contract_call.lineno < dataset_create.lineno
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


@pytest.mark.parametrize(
    "codec,version",
    [("pyav", "0.16.0"), ("torchcodec", "0.15.0")],
)
def test_native_decoder_contract_rejects_fallback_or_wrong_version(codec, version):
    """The offline contract must reject PyAV fallback and an ABI-drifted codec."""

    source = (RECIPE / "smoke_native_cpu.py").read_text()
    tree = ast.parse(source)
    decoder_contract = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_assert_native_decoder_contract"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[decoder_contract], type_ignores=[]), "<contract>", "exec"
        ),
        namespace,
    )
    with pytest.raises(AssertionError):
        namespace["_assert_native_decoder_contract"](codec, version)


def test_run_smoke_uses_explicit_lerobot_venv_interpreter():
    smoke = (RECIPE / "run_smoke.sh").read_text()
    assert smoke.count("/opt/lerobot/venv/bin/python -m npa.smoke.") == 2
    assert "\npython -m npa.smoke.test_lerobot_env" not in smoke
    assert "\npython -m npa.smoke.test_lerobot_functional" not in smoke


def test_native_server_startup_has_a_bounded_wait():
    source = (RECIPE / "smoke_native_cpu.py").read_text()
    assert "SERVER_START_TIMEOUT_SECONDS = 20" in source
    assert "deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS" in source
    assert "process.poll() is None and time.monotonic() < deadline" in source
    assert "except OSError:" in source
