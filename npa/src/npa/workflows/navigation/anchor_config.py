"""Validate optional anchor intent before binding a real baseline checkpoint."""

import math

ANCHOR_CLASS = "npa.workflows.navigation.anchor_ppo:BaselineAnchoredPPO"


def validate_anchor_coefficient(value):
    """Validate an explicitly enabled policy-retention coefficient without torch.

    Args:
        value: Finite positive numeric coefficient; omit the field to disable.
    Returns:
        Validated coefficient as a float.
    Raises:
        ValueError: Value is boolean, nonnumeric, nonfinite or nonpositive.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("baseline_anchor_coefficient must be finite and positive")
    return float(value)


def anchor_enabled(recipe):
    """Identify explicitly enabled executable recipes without native imports.

    Args:
        recipe: Validated navigation recipe.
    Returns:
        Whether a positive, fully bound anchor was supplied.
    Raises:
        None.
    """
    anchor = recipe.baseline_anchor
    return anchor is not None and anchor.coefficient > 0


def configure_training_anchor(settings, recipe, *, training):
    """Select the extension only for an explicitly enabled native PPO training run.

    Args:
        settings: Mutable resolved registered-task configuration.
        recipe: Validated, fully bound navigation recipe.
        training: False for all policy inference and replay operations.
    Returns:
        None; disabled and inference settings remain unchanged.
    Raises:
        ValueError: The registered algorithm is not the supported native PPO.
    """
    if not training or not anchor_enabled(recipe):
        return
    if settings["algorithm"]["class_name"] != "PPO":
        raise ValueError("baseline anchor requires registered native PPO")
    settings["algorithm"]["class_name"] = ANCHOR_CLASS
    settings["algorithm"]["baseline_anchor"] = recipe.baseline_anchor.model_dump()


def inference_settings(settings, recipe):
    """Remove only the authenticated training-only anchor from learner comparison.

    Args:
        settings: Recorded, resolved training configuration.
        recipe: Same sealed executable recipe.
    Returns:
        Native inference settings; all other values remain exact.
    Raises:
        ValueError: Recorded anchor differs from the sealed recipe.
    """
    from copy import deepcopy

    if not anchor_enabled(recipe):
        return settings
    result = deepcopy(settings)
    anchor = result["algorithm"].pop("baseline_anchor", None)
    if anchor != recipe.baseline_anchor.model_dump():
        raise ValueError("recorded learner anchor differs from the sealed recipe")
    if "class_name" in result["algorithm"]:
        if result["algorithm"]["class_name"] != ANCHOR_CLASS:
            raise ValueError("recorded learner uses an unexpected anchor class")
        result["algorithm"]["class_name"] = "PPO"
    return result
