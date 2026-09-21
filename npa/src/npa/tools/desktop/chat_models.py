"""Resolve mobile model controls from the shared Codex runtime's live catalog."""


def available_models(rpc):
    """Read every visible model and its supported reasoning options.

    Args:
        rpc: Authenticated local Codex connection.
    Returns:
        The complete visible model catalog.
    Raises:
        RuntimeError: Codex cannot provide its catalog.
    """
    models = []
    cursor = None
    while True:
        page = rpc.call("model/list", {"includeHidden": False, "cursor": cursor})
        models.extend(page["data"])
        cursor = page.get("nextCursor")
        if not cursor:
            return models


def model_selection(rpc, body, thread):
    """Validate an explicit model change without accepting permission overrides.

    Args:
        rpc: Authenticated local Codex connection.
        body: Requested model and reasoning effort for one thread.
        thread: Current authoritative thread metadata.
    Returns:
        Parameters for thread/settings/update.
    Raises:
        ValueError: The requested model, effort, or field is unsupported.
        RuntimeError: Codex cannot provide its catalog.
    """
    if set(body) - {"id", "model", "effort"}:
        raise ValueError("Only model and reasoning settings can be changed here.")
    name = body.get("model", thread.get("model"))
    model = next((m for m in available_models(rpc) if m["model"] == name), None)
    if model is None:
        raise ValueError("Choose a model available in your Codex account.")
    supported = [o["reasoningEffort"] for o in model["supportedReasoningEfforts"]]
    effort = body.get("effort", thread.get("reasoningEffort"))
    if effort not in supported:
        if "effort" in body:
            raise ValueError("This model does not support that reasoning effort.")
        effort = model["defaultReasoningEffort"]
    return {"threadId": body["id"], "model": name, "effort": effort}
