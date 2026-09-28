"""Select and verify the upstream activation recomputation policy."""

from npa.workbench.flex_pi.runtime import FlexPiError


def activation_checkpointing_overrides(mode):
    """Resolve one policy across both experts and their mixed attention.

    Args:
        mode: On to recompute activations or off to retain them for backward.
    Returns:
        Hydra overrides that participate in the recorded workload identity.
    Raises:
        FlexPiError: The requested policy is unsupported.
    """
    if mode not in {"on", "off"}:
        raise FlexPiError("activation-checkpointing must be on or off")
    enabled = str(mode == "on").lower()
    overrides = [
        f"model.video_dit_config.use_gradient_checkpointing={enabled}",
        f"model.action_dit_config.use_gradient_checkpointing={enabled}",
        f"model.mot_checkpoint_mixed_attn={enabled}",
    ]
    if mode == "off":
        overrides.append('+npa_activation_checkpointing="off"')
    return overrides


def activation_checkpointing_receipt(configuration, model):
    """Reject a loaded model whose recomputation flags differ from the request.

    Args:
        configuration: Resolved upstream configuration with the requested policy.
        model: Instantiated upstream model, before distributed wrapping.
    Returns:
        The actual video, action and mixed-attention recomputation settings.
    Raises:
        FlexPiError: Any observed setting differs from the requested policy.
    """
    mode = configuration.get("npa_activation_checkpointing", "on")
    activation_checkpointing_overrides(mode)
    observed = {
        "video": model.mot.mixtures["video"].use_gradient_checkpointing,
        "action": model.mot.mixtures["action"].use_gradient_checkpointing,
        "mixed_attention": model.mot.mot_checkpoint_mixed_attn,
    }
    if any(value is not (mode == "on") for value in observed.values()):
        raise FlexPiError("loaded model differs from activation-checkpointing policy")
    return {"policy": mode, **observed}
