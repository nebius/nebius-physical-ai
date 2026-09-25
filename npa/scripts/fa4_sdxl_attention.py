"""Route the stock SDXL UNet's supported attention layers directly to FA4."""

from collections import Counter


def _validate_layer(attention, hidden_states, attention_mask, temb):
    unsupported = (
        attention_mask is not None
        or temb is not None
        or hidden_states.ndim != 3
        or attention.norm_cross
        or attention.residual_connection
        or attention.rescale_output_factor != 1
    )
    normalizers = ("spatial_norm", "group_norm", "norm_q", "norm_k")
    if unsupported or any(getattr(attention, name) is not None for name in normalizers):
        raise ValueError("FA4 validation processor supports only stock SDXL attention")


class FA4SDXLProcessor:
    """Execute SDXL self/cross attention without a fallback backend.

    Args:
        attention_function: Explicit ``flash_attn.cute.flash_attn_func`` callable.
    Returns:
        A callable Diffusers processor with measured call counts and shapes.
    Raises:
        ValueError: A layer requests behavior outside the qualified SDXL subset.
    """

    def __init__(self, attention_function):
        self.attention_function = attention_function
        self.shapes = Counter()
        self.profile_inputs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        _validate_layer(attn, hidden_states, attention_mask, temb)
        context = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(context).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(context).unflatten(-1, (attn.heads, -1))
        shape = (tuple(query.shape), tuple(key.shape), tuple(value.shape))
        self.shapes[shape] += 1
        if self.profile_inputs is None:
            self.profile_inputs = (query.detach(), key.detach(), value.detach())
        result = self.attention_function(
            query, key, value, causal=False, pack_gqa=False, num_splits=1
        )
        output = result[0] if isinstance(result, tuple) else result
        return attn.to_out[1](attn.to_out[0](output.flatten(2).to(query.dtype)))
