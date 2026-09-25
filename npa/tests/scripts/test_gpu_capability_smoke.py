"""Exercise strict FA4 smoke failures, independent references and SASS coverage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py"
)


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gpu_capability_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (12, 0)])
@pytest.mark.parametrize(
    "failure",
    [
        AttributeError("'NoneType' object has no attribute '_trait'"),
        RuntimeError("CUDA illegal memory access"),
        AssertionError("dK does not match reference"),
        ImportError("FA4 unavailable"),
    ],
)
def test_runtime_failures_never_pass(
    smoke, monkeypatch, tmp_path, capsys, capability, failure
):
    def fail(expected, report):
        assert expected == f"{capability[0]}.{capability[1]}"
        report["environment"] = {"capability": list(capability)}
        raise failure

    monkeypatch.setattr(smoke, "_run_checks", fail)
    output = tmp_path / "failed.json"
    assert (
        smoke.main(
            [
                "--expect-capability",
                f"{capability[0]}.{capability[1]}",
                "--json-output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert str(failure) in report["error"]
    assert "GPU_CAPABILITY_SMOKE_OK" not in capsys.readouterr().out


def test_obsolete_failure_bypass_is_rejected(smoke):
    with pytest.raises(SystemExit) as error:
        smoke.main(["--allow-no-tma"])
    assert error.value.code == 2


def test_success_report_retains_measurements(smoke, monkeypatch, tmp_path, capsys):
    def succeed(expected, report):
        report["cases"] = [
            {"status": "passed", "metrics": {"dQ": {"max_abs_error": 0.001}}}
        ]

    monkeypatch.setattr(smoke, "_run_checks", succeed)
    output = tmp_path / "passed.json"
    assert smoke.main(["--json-output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["backend"] == "flash_attn.cute"
    assert report["cases"][0]["metrics"]["dQ"]["max_abs_error"] == 0.001
    assert "GPU_CAPABILITY_SMOKE_OK" in capsys.readouterr().out


def test_device_mismatch_stops_before_attention(smoke):
    cuda = SimpleNamespace(
        get_device_capability=lambda: (12, 0), get_arch_list=lambda: ["sm_120"]
    )
    with pytest.raises(ValueError, match="Expected capability"):
        smoke._check_device(SimpleNamespace(cuda=cuda), "10.0")


def test_ptx_only_wheel_cannot_pass_device_check(smoke):
    cuda = SimpleNamespace(
        get_device_capability=lambda: (12, 0), get_arch_list=lambda: ["compute_120"]
    )
    with pytest.raises(ValueError, match="No compatible native SASS"):
        smoke._check_device(SimpleNamespace(cuda=cuda), "12.0")


def test_b300_is_covered_by_sm100_sass(smoke):
    assert smoke.covering_sass_arch((10, 3), ["sm_90", "sm_100", "sm_120"]) == (10, 0)


def test_sass_coverage_never_crosses_a_cuda_major(smoke):
    assert smoke.covering_sass_arch((10, 3), ["sm_90", "sm_120"]) is None
    assert smoke.covering_sass_arch((12, 0), ["sm_100", "compute_120"]) is None


def test_explicit_fa4_import_does_not_select_fa2(smoke, monkeypatch):
    import sys

    dense, varlen = object(), object()
    root = ModuleType("flash_attn")
    root.flash_attn_func = object()
    cute = ModuleType("flash_attn.cute")
    cute.flash_attn_func, cute.flash_attn_varlen_func = dense, varlen
    monkeypatch.setitem(sys.modules, "flash_attn", root)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    assert smoke._attention_functions() == (dense, varlen)


def test_cross_attention_reference_uses_bottom_right_mask(smoke):
    torch = pytest.importorskip("torch")
    query = torch.zeros(1, 2, 2, 8, dtype=torch.float64, requires_grad=True)
    key = torch.zeros(1, 4, 1, 8, dtype=torch.float64, requires_grad=True)
    value = torch.arange(4, dtype=torch.float64).view(1, 4, 1, 1).expand(1, 4, 1, 8)
    actual = smoke._reference_attention(torch, query, key, value, causal=True)
    # With zero logits Q[0] averages V[0:3], Q[1] averages V[0:4].
    expected = (
        torch.tensor([1.0, 1.5], dtype=torch.float64).view(1, 2, 1, 1).expand_as(actual)
    )
    torch.testing.assert_close(actual, expected)


def test_varlen_reference_preserves_sequence_boundaries(smoke):
    torch = pytest.importorskip("torch")
    query = torch.zeros(3, 1, 8, dtype=torch.float64)
    key = torch.zeros(5, 1, 8, dtype=torch.float64)
    value = torch.tensor([1, 1, 7, 7, 7], dtype=torch.float64)[:, None, None].expand_as(
        key
    )
    actual = smoke._reference_varlen(
        torch, (query, key, value), ((1, 2), (2, 3)), False
    )
    expected = torch.tensor([1, 7, 7], dtype=torch.float64)[:, None, None].expand_as(
        query
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, 2.0])
def test_comparison_rejects_nonfinite_or_wrong_results(smoke, bad):
    torch = pytest.importorskip("torch")
    actual = torch.full((8,), bad, dtype=torch.float32)
    reference = torch.ones(8, dtype=torch.float64)
    with pytest.raises(AssertionError):
        smoke._compare_tensor(torch, actual, reference, "bfloat16")


def test_comparison_records_numerical_error(smoke):
    torch = pytest.importorskip("torch")
    metrics = smoke._compare_tensor(
        torch, torch.tensor([1.001]), torch.tensor([1.0]).double(), "bfloat16"
    )
    assert metrics["max_abs_error"] == pytest.approx(0.001, abs=1e-6)
    assert metrics["relative_l2_error"] == pytest.approx(0.001, abs=1e-6)


def test_golden_eval_runs_strict_smoke_on_selected_gpu():
    import yaml

    root = Path(__file__).resolve().parents[2]
    evals = yaml.safe_load((root / "src/npa/smoke/golden_evals.yaml").read_text())
    entry = evals["containers"]["base-cuda13-b300"]["golden_eval"]
    assert entry["command"] == "python /npa/gpu_capability_smoke.py"
    assert (
        entry["script"]
        == "npa/docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py"
    )
