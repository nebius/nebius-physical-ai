"""Require the World API token for resolved Marble generation workflows."""


def marble_secret_names(spec):
    """Select the token only for generation or the manufacturing preflight.

    Args: A loaded workflow spec with config overrides already merged.
    Returns: Required secret variable names, never their values.
    Raises: None.
    """
    for state in spec.states.values():
        if state.tool_ref == "workbench.marble.pallet_preflight":
            return ("WLT_API_KEY",)
        if state.tool_ref == "workbench.marble.acquire":
            source = state.params.get(
                "world_source", spec.config.get("world_source", "generate")
            )
            if source != "sample-hobbit":
                return ("WLT_API_KEY",)
    return ()
