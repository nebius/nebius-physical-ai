"""Guard opt-in inference tuning against silent training and semantic changes."""

import importlib.util
from importlib import metadata
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/base/cuda13-blackwell/scripts/flash_attn_rtx.py"
)


@pytest.fixture
def tuned(monkeypatch):
    spec = importlib.util.spec_from_file_location("fa4_rtx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cute, interface = (
        ModuleType("flash_attn.cute"),
        ModuleType("flash_attn.cute.interface"),
    )
    cute.flash_attn_func = Mock(return_value=("native-output", None))
    interface._flash_attn_fwd = Mock(return_value=("tuned-output", None))
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    cache = ModuleType("flash_attn.cute.cache_utils")
    cache.get_jit_cache = lambda name: {}
    utilities = ModuleType("flash_attn.cute.utils")
    utilities.AuxData = lambda: (None, None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.cache_utils", cache)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.utils", utilities)
    monkeypatch.setenv("NPA_FLASH_ATTN_COMMIT", module._COMMIT)
    monkeypatch.setattr(metadata, "version", lambda name: module._VERSION)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    return module, cute.flash_attn_func, interface._flash_attn_fwd


def _tensor(
    sequence=4096,
    heads=10,
    dimension=64,
    dtype=torch.float16,
    *,
    contiguous=False,
    pointer=16,
):
    return SimpleNamespace(
        shape=(2, sequence, heads, dimension),
        ndim=4,
        dtype=dtype,
        device=torch.device("cuda:0"),
        requires_grad=False,
        is_contiguous=lambda: contiguous,
        data_ptr=lambda: pointer,
    )


def test_selected_inference_uses_measured_tile(tuned):
    module, native, kernel = tuned
    q = _tensor()
    with torch.inference_mode():
        assert module.make_inference_attention()(q, q, q) == "tuned-output"
    kernel.assert_called_once_with(
        q, q, q, tile_mn=(64, 128), causal=False, pack_gqa=False, num_splits=1
    )
    native.assert_not_called()


def test_direct_launch_reuses_code_but_not_tensors_or_output(tuned, monkeypatch):
    module, native, tiled = tuned
    launch = Mock()
    compile_kernel = Mock(return_value=launch)
    monkeypatch.setattr(module, "_compile_kernel", compile_kernel)
    outputs = [object(), object()]
    monkeypatch.setattr(torch, "empty_like", Mock(side_effect=outputs))
    first, second = (_tensor(contiguous=True, pointer=p) for p in (16, 32))
    function = module.make_inference_attention()
    with torch.inference_mode():
        assert function(first, first, first) is outputs[0]
        assert function(second, second, second) is outputs[1]
    assert compile_kernel.call_count == 1
    assert launch.call_args_list[0].args[:4] == (first, first, first, outputs[0])
    assert launch.call_args_list[1].args[:4] == (second, second, second, outputs[1])
    native.assert_not_called()
    tiled.assert_not_called()


def test_unaligned_storage_retains_native_validation(tuned, monkeypatch):
    module, native, tiled = tuned
    compile_kernel = Mock()
    monkeypatch.setattr(module, "_compile_kernel", compile_kernel)
    query = _tensor(contiguous=True, pointer=18)
    with torch.inference_mode():
        assert module.make_inference_attention()(query, query, query) == "tuned-output"
    tiled.assert_called_once()
    compile_kernel.assert_not_called()
    native.assert_not_called()


def test_direct_compile_failure_never_substitutes_a_backend(tuned, monkeypatch):
    module, native, tiled = tuned
    monkeypatch.setattr(torch, "empty_like", Mock(return_value=object()))
    monkeypatch.setattr(
        module, "_compile_kernel", Mock(side_effect=RuntimeError("compile failed"))
    )
    query = _tensor(contiguous=True)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="compile failed"):
        module.make_inference_attention()(query, query, query)
    native.assert_not_called()
    tiled.assert_not_called()


def test_unqualified_shape_keeps_native_fa4(tuned):
    module, native, kernel = tuned
    q = _tensor(sequence=2048)
    with torch.inference_mode():
        assert module.make_inference_attention()(q, q, q) == "native-output"
    native.assert_called_once_with(q, q, q, causal=False, pack_gqa=False, num_splits=1)
    kernel.assert_not_called()


@pytest.mark.parametrize("gradients", ["enabled", "query", "key", "value"])
def test_tuning_never_silently_discards_gradients(tuned, gradients):
    module, native, kernel = tuned
    tensors = [_tensor() for _ in range(3)]
    if gradients != "enabled":
        tensors[("query", "key", "value").index(gradients)].requires_grad = True
    with torch.set_grad_enabled(gradients == "enabled"):
        with pytest.raises(ValueError, match="inference-only"):
            module.make_inference_attention()(*tensors)
    native.assert_not_called()
    kernel.assert_not_called()


@pytest.mark.parametrize("mismatch", ["pin", "version", "architecture", "device"])
def test_revision_and_device_guards(tuned, monkeypatch, mismatch):
    module, native, kernel = tuned
    q, k, v = (_tensor() for _ in range(3))
    if mismatch == "pin":
        monkeypatch.setenv("NPA_FLASH_ATTN_COMMIT", "different")
    elif mismatch == "version":
        monkeypatch.setattr(metadata, "version", lambda name: "different")
    elif mismatch == "architecture":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    else:
        k.device = torch.device("cuda:1")
    with torch.inference_mode(), pytest.raises((ValueError, RuntimeError)):
        module.make_inference_attention()(q, k, v)
    native.assert_not_called()
    kernel.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"causal": "false"},
        {"dropout_p": 0.1},
        {"softmax_scale": 0.5},
        {"attention_mask": None},
    ],
)
def test_unsupported_semantics_fail_before_dispatch(tuned, options):
    module, native, kernel = tuned
    q = _tensor()
    with torch.inference_mode(), pytest.raises(TypeError):
        module.make_inference_attention()(q, q, q, **options)
    native.assert_not_called()
    kernel.assert_not_called()


def test_native_failure_propagates_without_fallback(tuned):
    module, native, kernel = tuned
    kernel.side_effect = RuntimeError("kernel failed")
    q = _tensor()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="kernel failed"):
        module.make_inference_attention()(q, q, q)
    native.assert_not_called()


@pytest.mark.parametrize(
    "dtype,causal", [(torch.bfloat16, False), (torch.float16, True)]
)
def test_unqualified_dtype_or_mask_uses_native(tuned, dtype, causal):
    module, native, kernel = tuned
    q = _tensor(dtype=dtype)
    with torch.inference_mode():
        assert (
            module.make_inference_attention()(q, q, q, causal=causal) == "native-output"
        )
    kernel.assert_not_called()
