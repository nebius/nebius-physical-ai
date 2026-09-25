"""Check SDXL attention layout and refusal of unsupported processor semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fa4_sdxl_attention.py"
spec = importlib.util.spec_from_file_location("fa4_sdxl_attention", SCRIPT)
assert spec and spec.loader
processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processor_module)


def _layer():
    return SimpleNamespace(
        heads=2,
        norm_cross=False,
        residual_connection=False,
        rescale_output_factor=1,
        spatial_norm=None,
        group_norm=None,
        norm_q=None,
        norm_k=None,
        to_q=torch.nn.Linear(8, 8, dtype=torch.float64),
        to_k=torch.nn.Linear(8, 8, dtype=torch.float64),
        to_v=torch.nn.Linear(8, 8, dtype=torch.float64),
        to_out=[torch.nn.Linear(8, 8, dtype=torch.float64), torch.nn.Identity()],
    )


@pytest.mark.parametrize("cross_attention", [False, True])
@pytest.mark.parametrize("tuple_result", [False, True])
def test_projection_layout_matches_independent_attention(cross_attention, tuple_result):
    torch.manual_seed(7)
    layer = _layer()
    hidden = torch.randn(2, 7, 8, dtype=torch.float64)
    context = torch.randn(2, 3, 8, dtype=torch.float64) if cross_attention else hidden

    def fake_fa4(query, key, value, **kwargs):
        assert kwargs == {"causal": False, "pack_gqa": False, "num_splits": 1}
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        return (output, None) if tuple_result else output

    processor = processor_module.FA4SDXLProcessor(fake_fa4)
    actual = processor(layer, hidden, context if cross_attention else None)
    query = layer.to_q(hidden).reshape(2, 7, 2, 4)
    key = layer.to_k(context).reshape(2, -1, 2, 4)
    value = layer.to_v(context).reshape(2, -1, 2, 4)
    probabilities = (torch.einsum("bqhd,bkhd->bhqk", query, key) / 2).softmax(-1)
    attended = torch.einsum("bhqk,bkhd->bqhd", probabilities, value).flatten(2)
    torch.testing.assert_close(actual, layer.to_out[0](attended))
    assert sum(processor.shapes.values()) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("norm_cross", True), ("residual_connection", True), ("rescale_output_factor", 2)]
    + [(name, object()) for name in ("spatial_norm", "group_norm", "norm_q", "norm_k")],
)
def test_unsupported_layer_never_calls_fa4(field, value):
    layer = _layer()
    setattr(layer, field, value)
    processor = processor_module.FA4SDXLProcessor(lambda *args, **kwargs: pytest.fail())
    with pytest.raises(ValueError, match="stock SDXL"):
        processor(layer, torch.zeros(2, 7, 8))


@pytest.mark.parametrize("kwargs", [{"attention_mask": True}, {"temb": True}])
def test_mask_and_temporal_embedding_are_not_silently_dropped(kwargs):
    processor = processor_module.FA4SDXLProcessor(lambda *args, **kwargs: pytest.fail())
    with pytest.raises(ValueError, match="stock SDXL"):
        processor(_layer(), torch.zeros(2, 7, 8), **kwargs)


def test_kernel_failure_propagates_without_fallback():
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA kernel failed")

    processor = processor_module.FA4SDXLProcessor(fail)
    with pytest.raises(RuntimeError, match="CUDA kernel failed"):
        processor(_layer(), torch.zeros(2, 7, 8, dtype=torch.float64))
