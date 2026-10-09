"""Translate explicit browser choices without guessing a thread's permissions."""


def permission_settings(mode):
    """Resolve a supported permission choice for subsequent turns.

    Args:
        mode: Explicit browser selector value.
    Returns:
        Codex approval and sandbox settings.
    Raises:
        ValueError: The choice is unsupported.
    """
    policies = {
        "read-only": ("on-request", {"type": "readOnly"}),
        "workspace": ("on-request", {"type": "workspaceWrite"}),
        "full": ("never", {"type": "dangerFullAccess"}),
    }
    if not isinstance(mode, str) or mode not in policies:
        raise ValueError("Choose Read only, Workspace, or Full permissions.")
    approval, sandbox = policies[mode]
    return {"approvalPolicy": approval, "sandboxPolicy": sandbox}


def settings_metadata(settings):
    """Project authoritative runtime settings onto browser thread metadata.

    Args:
        settings: Start, resume, or settings-notification payload.
    Returns:
        Present settings, with a selector value only for known policies.
    Raises:
        None.
    """
    fields = ("model", "serviceTier", "approvalPolicy", "collaborationMode")
    result = {key: settings[key] for key in fields if key in settings}
    if "effort" in settings or "reasoningEffort" in settings:
        result["reasoningEffort"] = settings.get(
            "effort", settings.get("reasoningEffort")
        )
    sandbox = settings.get("sandboxPolicy", settings.get("sandbox"))
    if isinstance(sandbox, dict):
        result["sandboxPolicy"] = sandbox
        result["permissionMode"] = _permission_mode(settings, sandbox)
    if settings.get("collaborationMode"):
        result["mode"] = settings["collaborationMode"]["mode"]
    return result


def _permission_mode(settings, sandbox):
    # Custom roots, networking, and approval policies must not look like presets.
    defaults = {
        "readOnly": {"networkAccess": False},
        "workspaceWrite": {
            "writableRoots": [],
            "networkAccess": False,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        },
    }.get(sandbox.get("type"), {})
    normalized = {
        key: value
        for key, value in sandbox.items()
        if key not in defaults or value != defaults[key]
    }
    for mode in ("read-only", "workspace", "full"):
        preset = permission_settings(mode)
        if (
            normalized == preset["sandboxPolicy"]
            and settings.get("approvalPolicy") == preset["approvalPolicy"]
        ):
            return mode
    return "custom"
