"""Select repaired, digest-pinned public candidates for PAIDF workflow tasks."""

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    container_image_for_tool,
)

# These exact public development builds passed the trusted publication workflow.
# They replace the quarantined release bytes only in workflow image selection;
# release promotion and the stale-release quarantine remain independent.
# Cosmos3: https://github.com/nebius/nebius-physical-ai/actions/runs/37378387786
# Evaluator/Curator: https://github.com/nebius/nebius-physical-ai/actions/runs/37271661779
_PAIDF_CANDIDATES = {
    "cosmos3": (
        "6462f27f98e1ded31d17b00944f943db3381ac35",
        "sha256:1aa6c473d95863709766f02d3e199cf8cf860adbe585b409a23a82bae4a2c37e",
    ),
    "cosmos-evaluator": (
        "bd6145947145f75f9065707e3f5c53f841fab9d3",
        "sha256:5d1335f58d5cc5e11cb0d4ccd0023cafe6e3b0bcc8af405f1f9d18013a17edfa",
    ),
    "cosmos-curate": (
        "bd6145947145f75f9065707e3f5c53f841fab9d3",
        "sha256:11e596bfb2cb46a6435dfe8013484882b1b0d6c7747ead1edb8c32c94a0581ad",
    ),
}
_PAIDF_TOOL_REFS = frozenset(
    {
        "workbench.cosmos3.prepare_video_input",
        "workbench.cosmos3.generate_variants",
        "workbench.cosmos_evaluator.evaluate",
        "workbench.cosmos_curate.curate",
    }
)


def public_workflow_image_default(
    tool: str, *, tool_ref: str, registry: str | None
) -> str:
    """Return a repaired public workflow candidate, retaining registry overrides.

    Args:
        tool: Canonical workbench image tool name.
        tool_ref: Workflow action whose runtime is being selected.
        registry: Explicit registry selection, or None for the official default.

    Returns:
        The immutable candidate reference, or an empty string for normal routing.

    Raises:
        ValueError: The image resolver rejects the selected development candidate.
    """
    if tool_ref not in _PAIDF_TOOL_REFS:
        return ""
    if registry and registry.rstrip("/") != DEFAULT_CONTAINER_REGISTRY:
        return ""
    candidate = _PAIDF_CANDIDATES.get(tool)
    if candidate is None:
        return ""
    source_sha, digest = candidate
    image = container_image_for_tool(
        tool, registry=DEFAULT_CONTAINER_REGISTRY, tag=f"dev-{source_sha}"
    )
    return f"{image}@{digest}"
