"""Exercise model selection and old-image rejection without hosted inference."""

import sys
import subprocess

import pytest

from npa.orchestration.npa_workflow.sim2real_evaluator_probe import render_evaluator_probe


def _run_probe(model, *, preparation=""):
    script = preparation + "\nactual = 'a' * 40\n" + render_evaluator_probe(model)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("model", ["MiniMaxAI/MiniMax-M3", "nvidia/Cosmos3-Super-Reasoner"])
def test_current_baked_evaluator_accepts_explicit_model(model):
    result = _run_probe(model)
    assert result.returncode == 0, result.stderr
    assert f"baked Sim2Real evaluator verified {model}" in result.stdout


def test_old_image_fails_before_a_hosted_request():
    result = _run_probe("MiniMaxAI/MiniMax-M3", preparation=(
        "import sys\nfrom types import ModuleType\n"
        "sys.modules['npa.workbench.cosmos.reason'] = ModuleType('legacy')\n"
    ))
    assert result.returncode != 0
    assert "image is incompatible with the selected model MiniMaxAI/MiniMax-M3" in result.stderr


def test_probe_refuses_an_evaluator_that_accepts_an_empty_boundary():
    result = _run_probe("MiniMaxAI/MiniMax-M3", preparation=(
        "from npa.workflows.sim2real import stage9_evaluator\n"
        "stage9_evaluator.validate_hosted_evaluator = lambda **_: {}\n"
    ))
    assert result.returncode != 0
    assert "accepted an empty Stage 8 boundary" in result.stderr


def test_unsupported_model_is_not_substituted():
    result = _run_probe("unsupported/model")
    assert result.returncode != 0
    assert "image is incompatible" in result.stderr


def test_model_is_a_literal_in_generated_code(tmp_path):
    marker = tmp_path / "injected"
    model = f"'); __import__('pathlib').Path({str(marker)!r}).touch(); #"
    result = _run_probe(model)
    assert result.returncode != 0
    assert "image is incompatible" in result.stderr
    assert not marker.exists()
