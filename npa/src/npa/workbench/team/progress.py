"""Project recorded workflow progress without contacting cloud providers."""

from pathlib import Path

from .errors import TeamError
from .session_storage import read_private_json


def recorded_progress(state_dir: Path, run_id: str) -> dict | None:
    """Read stage names and counts from the server's authoritative runtime record.

    Args:
        state_dir, run_id: Trusted installation directory and owned run identity.
    Returns:
        Latest persisted progress, or None when a safe record is unavailable.
    Raises:
        None; optional progress never replaces the ledger's lifecycle status.
    """
    root = state_dir / "runs" / run_id / "records" / "npa-workflow"
    try:
        document = read_private_json(root, "runtime.json")
    except (FileNotFoundError, TeamError):
        return None
    if not isinstance(document, dict) or document.get("run_id") != run_id:
        return None
    waves = document.get("waves")
    if not isinstance(waves, list) or not waves:
        return None
    if any(not isinstance(wave, dict) for wave in waves):
        return None
    latest = waves[-1]
    states, status = latest.get("states"), latest.get("status")
    if (
        not isinstance(states, list)
        or not states
        or any(not isinstance(state, str) or not state for state in states)
        or not isinstance(status, str)
        or not status
    ):
        return None
    return {
        "states": states,
        "wave_status": status,
        "completed_waves": sum(wave.get("status") == "succeeded" for wave in waves),
        "recorded_waves": len(waves),
    }
