"""Validate an explicit TRAIN-only Comet prompt override."""

from __future__ import annotations


def prompt_override(args, *, train_panel: bool) -> str | None:
    """Return a canonical literal override restricted to recorded TRAIN.

    Args:
        args: Campaign arguments containing policy and recording settings.
        train_panel: Whether the validated panel is non-reporting TRAIN.
    Returns:
        The exact literal override, or ``None`` when the default applies.
    Raises:
        ValueError: The override or its execution scope differs.
    """
    value = getattr(args, "policy_prompt_override", None)
    if value is None:
        return None
    if (
        getattr(args, "policy_kind", None) != "comet-trained"
        or not getattr(args, "train_experience", False)
        or not train_panel
    ):
        raise ValueError("Policy prompt override requires recorded comet-trained TRAIN")
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 512
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError("Policy prompt override must be one canonical literal")
    return value
