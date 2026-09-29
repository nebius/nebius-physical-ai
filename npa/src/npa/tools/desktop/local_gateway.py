"""Add a local chat route to an existing managed HTTPS gateway over SSH."""

import json
from pathlib import Path
import subprocess

from . import _validate_host


def gateway_operation(host, action, **options):
    """Inspect or configure an existing desktop or mobile gateway.

    Args:
        host: Existing SSH alias.
        action: inspect or configure.
        options: Selected origin and private routing options.
    Returns:
        Private gateway metadata; callers must not print credentials.
    Raises:
        ValueError: The host is invalid.
        RuntimeError: The gateway is unsupported or SSH fails.
    """
    _validate_host(host)
    source = (Path(__file__).parent / "gateway_remote.py").read_text()
    payload = source + "\nmain(" + repr({"action": action, **options}) + ")\n"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "--", host, "/usr/bin/python3", "-"],
        input=payload,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        log = Path.home() / ".local/share/nebius-desktop/local-chat/gateway-error.log"
        log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        log.touch(mode=0o600, exist_ok=True)
        log.chmod(0o600)
        log.write_text(result.stderr)
        raise RuntimeError(
            f"Gateway operation failed; inspect the private log at {log}."
        )
    return json.loads(result.stdout)
