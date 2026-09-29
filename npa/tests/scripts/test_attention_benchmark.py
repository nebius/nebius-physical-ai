"""Keep FA2/FA4 comparisons explicit, untuned by default and independently checked."""

import importlib
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("attention_benchmark_backend")


@pytest.mark.parametrize("requested,installed", [("fa2", "fa4"), ("fa4", "fa2")])
def test_wrong_image_cannot_become_a_baseline(
    backend, monkeypatch, requested, installed
):
    monkeypatch.setenv("NPA_ATTENTION_BACKEND", installed)
    with pytest.raises(RuntimeError, match="matching base image"):
        backend.attention_backend(requested)


def test_fa2_requires_real_extension_not_root_adapter(backend, monkeypatch):
    monkeypatch.setenv("NPA_ATTENTION_BACKEND", "fa2")
    monkeypatch.setattr(backend.metadata, "version", lambda name: "2.8.3")
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", None)
    with pytest.raises(ModuleNotFoundError):
        backend.attention_backend("fa2")


def test_fa4_calls_cute_and_unwraps_output(backend, monkeypatch):
    monkeypatch.setenv("NPA_ATTENTION_BACKEND", "fa4")
    cute = ModuleType("flash_attn.cute")
    calls = []

    def attention(*args, **kwargs):
        calls.append(kwargs)
        return "output", "lse"

    cute.flash_attn_func = attention
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    assert backend.attention_backend("fa4")(1, 2, 3, causal=True) == "output"
    assert calls == [{"causal": True, "pack_gqa": False, "num_splits": 1}]


def test_experimental_tiles_refuse_training(backend, monkeypatch):
    torch = pytest.importorskip("torch")
    interface = ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = lambda *a, **k: pytest.fail(
        "Cannot launch tiled training"
    )
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setenv("NPA_FLASH_ATTN_COMMIT", backend.FA_COMMIT)
    function = backend._tiled_inference((64, 64))
    query = torch.ones(1, 4, 2, 64, requires_grad=True)
    with pytest.raises(ValueError, match="inference-only"):
        function(query, query, query)


def test_tiles_refuse_an_unpinned_runtime(backend, monkeypatch):
    pytest.importorskip("torch")
    interface = ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = object()
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setenv("NPA_FLASH_ATTN_COMMIT", "another-revision")
    with pytest.raises(RuntimeError, match="exact pinned"):
        backend._tiled_inference((64, 64))


def test_fa2_cannot_accept_fa4_tile_argument(backend):
    with pytest.raises(ValueError, match="Tiles require FA4"):
        backend.attention_backend("fa2", (64, 64))


def test_math_reference_preserves_causal_gqa(backend):
    torch = pytest.importorskip("torch")
    benchmark = importlib.import_module("attention_kernel_benchmark")
    query = torch.zeros(1, 3, 2, 8)
    key = torch.zeros(1, 3, 1, 8)
    value = torch.arange(3).float()[None, :, None, None].expand_as(key)
    expected = torch.tensor([0, 0.5, 1])[None, :, None, None].expand_as(query)
    torch.testing.assert_close(
        benchmark._reference(torch, (query, key, value), True), expected
    )


def test_incorrect_kernel_cannot_pass_to_timing(backend):
    torch = pytest.importorskip("torch")
    benchmark = importlib.import_module("attention_kernel_benchmark")
    query = torch.ones(1, 3, 2, 8)
    with pytest.raises(AssertionError):
        benchmark._correctness(
            torch,
            lambda *a, **k: torch.zeros_like(query),
            (query, query, query),
            {"causal": False, "dtype": "float16"},
            False,
        )


def _reports():
    template = {
        "status": "passed",
        "instrumented_timing": False,
        "tile": None,
        "environment": {
            "gpu": "RTX PRO 6000",
            "image_id": "fa2-image",
            "packages": {"torch": "2.13.0", "flash-attn": "2.8.3"},
        },
        "model": "fixture",
        "model_revision": "fixed",
        "steps": 30,
        "repeats": 3,
        "cases": [{"name": "case", "samples": [{"seconds": 2}] * 3}],
        "backend": "fa2",
    }
    reports = [copy.deepcopy(template) for _ in range(4)]
    for report in reports[1:3]:
        report["backend"] = "fa4"
        report["environment"]["image_id"] = "fa4-image"
        report["environment"]["packages"] = {"torch": "2.13.0", "flash-attn-4": "4.0"}
        report["cases"][0]["samples"] = [{"seconds": 2.5}] * 3
    return reports


def test_comparison_retains_regressions_and_block_variance(backend):
    comparison = importlib.import_module("compare_attention_sdxl")
    result = comparison.comparison(_reports())
    assert result["cases"][0]["fa2_time_over_fa4_time"] == 0.8
    assert result["cases"][0]["fa4"]["block_medians"] == [2.5, 2.5]


@pytest.mark.parametrize(
    "mutation",
    [
        "torch",
        "steps",
        "tile",
        "image",
        "incomplete",
        "instrumentation",
        "nan",
        "samples",
    ],
)
def test_unmatched_or_incomplete_reports_cannot_claim_speedup(backend, mutation):
    comparison = importlib.import_module("compare_attention_sdxl")
    reports = _reports()
    report = reports[1]
    if mutation == "torch":
        report["environment"]["packages"]["torch"] = "different"
    elif mutation == "steps":
        report["steps"] = 1
    elif mutation == "tile":
        report["tile"] = [64, 64]
    elif mutation == "image":
        report["environment"]["image_id"] = "another-image"
    elif mutation == "incomplete":
        report["status"] = "incomplete"
    elif mutation == "instrumentation":
        report["instrumented_timing"] = True
    elif mutation == "nan":
        report["cases"][0]["samples"][0]["seconds"] = float("nan")
    else:
        report["cases"][0]["samples"].pop()
    with pytest.raises(ValueError):
        comparison.comparison(reports)


def test_provenance_records_all_packages_using_normalized_names(backend, monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": name})
        for name in ("flash_attn", "Torch", "Pillow")
    ]
    monkeypatch.setattr(backend.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(backend.metadata, "version", lambda name: name + "-version")
    assert backend._package_versions("fa2") == {
        "flash-attn": "flash-attn-version",
        "pillow": "pillow-version",
        "torch": "torch-version",
    }


def test_provenance_requires_the_requested_distribution(backend, monkeypatch):
    monkeypatch.setattr(backend.metadata, "distributions", lambda: [])
    with pytest.raises(RuntimeError, match="Missing distribution"):
        backend._package_versions("fa4")


def test_provenance_fingerprints_the_actual_benchmark_sources(backend):
    fingerprints = backend._benchmark_sources()
    assert "fa4_sdxl_model.json" in fingerprints
    assert "attention_benchmark_backend.py" in fingerprints
    assert all(len(value) == 64 for value in fingerprints.values())
