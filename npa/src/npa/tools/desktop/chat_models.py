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
    if set(body) - {"id", "model", "effort", "serviceTier", "mode"}:
        raise ValueError(
            "Only model and reasoning settings, speed, and mode can be changed here."
        )
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
    result = {"threadId": body["id"], "model": name, "effort": effort}
    _speed(result, body, model)
    _mode(rpc, result, body, thread)
    return result


def _speed(result, body, model):
    if "serviceTier" not in body:
        return
    tier = body["serviceTier"]
    supported = {entry["id"] for entry in model.get("serviceTiers", [])}
    if tier not in {None, "default", *supported}:
        raise ValueError("This model does not support that speed.")
    result["serviceTier"] = tier


def _mode(rpc, result, body, thread):
    previous = thread.get("collaborationMode")
    mode = body.get("mode")
    if mode is None and previous is None:
        return
    if mode is not None:
        modes = rpc.call("collaborationMode/list", {})["data"]
        if mode not in {entry["mode"] for entry in modes}:
            raise ValueError("Choose an available chat mode.")
    chosen = mode or previous["mode"]
    settings = dict((previous or {}).get("settings", {}))
    if previous and chosen != previous["mode"]:
        settings["developer_instructions"] = None
    settings.update(model=result["model"], reasoning_effort=result["effort"])
    settings.setdefault("developer_instructions", None)
    result["collaborationMode"] = {"mode": chosen, "settings": settings}
