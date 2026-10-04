"""Keep RTX validation bound to real source and prohibit hidden cache misses."""

import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def validator(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("validate_rtx_attention")
    helper = tmp_path / "rtx.py"
    helper.write_text("# Synthetic launcher fixture.\n")
    rtx = ModuleType("flash_attn.rtx")
    rtx.__file__ = str(helper)
    rtx.make_inference_attention = Mock(return_value="bound-attention")
    rtx._compile_kernel = Mock()
    package = ModuleType("flash_attn")
    package.rtx = rtx
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.rtx", rtx)
    monkeypatch.setattr(module, "benchmark_environment", Mock(return_value={}))
    args = SimpleNamespace(
        expect_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        require_cache_hit=False,
    )
    return module, rtx, args


def test_mismatched_source_cannot_receive_a_validation_receipt(validator):
    module, rtx, args = validator
    args.expect_helper_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="expected source"):
        module._prepare(args)
    rtx.make_inference_attention.assert_not_called()
    module.benchmark_environment.assert_not_called()


def test_cache_reload_mode_cannot_compile_a_replacement(validator):
    module, rtx, args = validator
    args.require_cache_hit = True
    attention, report = module._prepare(args)
    assert attention == "bound-attention"
    assert report["persistent_cache_reload"] is True
    with pytest.raises(AssertionError, match="JIT compilation"):
        rtx._compile_kernel()


def test_fp64_oracle_uses_causal_gqa_prefixes(validator):
    module, _, _ = validator
    query = torch.zeros(1, 3, 2, 8, dtype=torch.float16)
    key = torch.zeros(1, 3, 1, 8, dtype=torch.float16)
    value = torch.arange(3, dtype=torch.float16)[None, :, None, None].expand_as(key)
    expected = torch.tensor([0, 0.5, 1], dtype=torch.float64)
    expected = expected[None, :, None, None].expand_as(query)
    actual = module._reference((query, key, value), True)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
