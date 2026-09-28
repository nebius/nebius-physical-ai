"""Reject FA2/FA4 API ambiguity before entering native GPU code."""

import importlib.util
from importlib import metadata
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/base/cuda13-b300/scripts/flash_attn_root.py"
)
SEQUENCES = dict(
    cu_seqlens_q=object(), cu_seqlens_k=object(), max_seqlen_q=129, max_seqlen_k=193
)


@pytest.fixture
def adapter(monkeypatch):
    cute = ModuleType("flash_attn.cute")
    cute.flash_attn_func = Mock()
    cute.flash_attn_varlen_func = Mock()
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setattr(metadata, "version", lambda name: "test-fa4")
    spec = importlib.util.spec_from_file_location("fa4_root", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["dense", "varlen"])
def route(adapter, request):
    if request.param == "varlen":
        return adapter.flash_attn_varlen_func, adapter._varlen, SEQUENCES
    return adapter.flash_attn_func, adapter._dense, {}


@pytest.mark.parametrize("tuple_return", [False, True])
def test_named_arguments_keep_meaning_and_tensor_identity(route, tuple_return):
    entry, kernel, sequences = route
    q, k, v, output = (object() for _ in range(4))
    kernel.return_value = (output, object()) if tuple_return else output
    assert (
        entry(q, k, v, **sequences, dropout_p=0.0, causal=True, softmax_scale=0.25)
        is output
    )
    kernel.assert_called_once_with(
        q,
        k,
        v,
        **sequences,
        causal=True,
        softmax_scale=0.25,
        pack_gqa=False,
        num_splits=1,
    )


@pytest.mark.parametrize("extra", [(0.0,), (None,), (0.0, 0.25, True)])
def test_ambiguous_positional_options_never_enter_kernel(route, extra):
    entry, kernel, sequences = route
    with pytest.raises(TypeError):
        entry(object(), object(), object(), *extra, **sequences)
    kernel.assert_not_called()


def test_fa2_positional_sequence_metadata_is_rejected(adapter):
    with pytest.raises(TypeError):
        adapter.flash_attn_varlen_func(*[object()] * 7)
    adapter._varlen.assert_not_called()


@pytest.mark.parametrize(
    "option",
    [
        {"dropout_p": 0.1},
        {"dropout_p": float("nan")},
        {"pack_gqa": True},
        {"num_splits": 2},
        {"num_splits": 0},
        {"alibi_slopes": None},
        {"window_size": (-1, -1)},
        {"deterministic": True},
        {"return_attn_probs": True},
        {"return_lse": True},
        {"qv": object()},
        {"softcap": 1.0},
        {"mask_mod": object()},
    ],
)
def test_unqualified_options_fail_without_discarding_semantics(route, option):
    entry, kernel, sequences = route
    with pytest.raises((TypeError, ValueError)):
        entry(object(), object(), object(), **sequences, **option)
    kernel.assert_not_called()


def test_kernel_failure_propagates_without_retry_or_backend_change(route):
    entry, kernel, sequences = route
    failure = RuntimeError("native kernel failed")
    kernel.side_effect = failure
    with pytest.raises(RuntimeError) as error:
        entry(object(), object(), object(), **sequences)
    assert error.value is failure
    kernel.assert_called_once()


def test_missing_sequence_metadata_never_enters_kernel(adapter):
    with pytest.raises(TypeError):
        adapter.flash_attn_varlen_func(object(), object(), object())
    adapter._varlen.assert_not_called()
