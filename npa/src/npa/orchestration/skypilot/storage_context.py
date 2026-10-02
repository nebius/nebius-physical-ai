"""Carry resolved workflow storage into scoped SkyPilot control-plane operations."""

from contextvars import ContextVar

from npa.orchestration.npa_workflow.submit_credentials import STORAGE_ENDPOINT_ENV_NAMES

_STORAGE_CONTEXT = ContextVar("npa_workflow_storage_context", default=None)


def call_with_workflow_storage(state, operation, *args, **kwargs):
    """Run an observation or recovery with the durable reader's storage identity.

    Args:
        state: Resolved WorkflowS3Config, or None when no durable identity exists.
        operation: Standard controller operation to invoke without changing its API.
        args: Positional arguments for the operation.
        kwargs: Keyword arguments for the operation.
    Returns:
        The operation's unchanged result; caller environment and outer scope persist.
    Raises:
        ValueError: The resolved storage credential pair is incomplete.
        Exception: The underlying operation fails; its original exception propagates.
    """
    overrides = _storage_overrides(state)
    if not overrides:
        return operation(*args, **kwargs)
    token = _STORAGE_CONTEXT.set(overrides)
    try:
        return operation(*args, **kwargs)
    finally:
        _STORAGE_CONTEXT.reset(token)


def _storage_overrides(state):
    if state is None:
        return {}
    access, secret = state.aws_access_key_id, state.aws_secret_access_key
    if bool(access) != bool(secret):
        raise ValueError(
            "Resolved workflow storage requires a complete credential pair"
        )
    if not access:
        return {}
    if not state.endpoint_url:
        raise ValueError("Resolved workflow storage requires its selected endpoint")
    # WorkflowS3Config's durable client uses this explicit pair without an ambient
    # session token. Controller recovery must use that same complete principal.
    return {
        **dict.fromkeys(STORAGE_ENDPOINT_ENV_NAMES, state.endpoint_url),
        "AWS_ACCESS_KEY_ID": access,
        "AWS_SECRET_ACCESS_KEY": secret,
        "AWS_SESSION_TOKEN": "",
        "AWS_SECURITY_TOKEN": "",
        "NPA_SKYPILOT_PROJECT": state.project or "",
    }


def _apply_storage_context(environment):
    return {**environment, **(_STORAGE_CONTEXT.get() or {})}
