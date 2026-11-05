"""Fault injector: truncate the first real object-store read, then get out of the way.

Loaded with `python -m` inside the built image so the `register` stage runs the
real Open3D pipeline against the real store, with exactly one transport failure
of the kind that actually killed a live stage. Nothing about the tool's own
retry logic is stubbed: the retry has to recover using real bytes.
"""

from __future__ import annotations

import json
import os
import sys

from npa.workbench.open3d import runtime

MODES = ("raise", "short", "none")


def required_env(name: str, allowed: tuple[str, ...] = ()) -> str:
    """Report a missing setting as usage rather than as a KeyError.

    This injector is driven entirely by the environment because it is loaded with
    `python -m` ahead of the real stage and has no argv of its own to parse. A bare
    `KeyError: 'INJECT_MODE'` reads as a broken harness rather than an unset variable,
    which is the wrong thing to tell someone trying to reproduce a result.
    """
    value = os.environ.get(name)
    if value is None:
        expected = f", one of {', '.join(allowed)}" if allowed else ""
        raise SystemExit(
            f"{name} is not set{expected}. This fault injector is configured through "
            "the environment: set INJECT_MODE and INJECT_LOG, optionally INJECT_CALL, "
            "then load it with `python -m` inside the built image."
        )
    if allowed and value not in allowed:
        raise SystemExit(f"{name}={value!r} is not one of {', '.join(allowed)}.")
    return value


MODE = required_env("INJECT_MODE", MODES)
LOG_PATH = required_env("INJECT_LOG")
# Which read to disturb: 1 is the manifest, 2 is the first fragment. The two are
# guarded differently (JSON well-formedness vs a recorded digest), so both need
# a real-bytes test.
TARGET = int(os.environ.get("INJECT_CALL", "1"))
real_read = runtime.read_bytes_uri
log: list[dict] = []


def injected(uri: str) -> bytes:
    call = len(log) + 1
    # Only the targeted read is disturbed; every other read is untouched.
    if call == TARGET and MODE == "raise":
        log.append(
            {"call": call, "uri_suffix": uri[-12:], "injected": "IncompleteRead"}
        )
        raise OSError("IncompleteRead(5848412 bytes read, 276537 more expected)")
    try:
        payload = real_read(uri)
    except Exception as exc:
        log.append(
            {"call": call, "uri_suffix": uri[-12:], "real_error": type(exc).__name__}
        )
        raise
    if call == TARGET and MODE == "short":
        cut = payload[: len(payload) // 2]
        log.append(
            {
                "call": call,
                "uri_suffix": uri[-12:],
                "injected": "silent truncation",
                "real_bytes": len(payload),
                "delivered_bytes": len(cut),
            }
        )
        return cut
    log.append(
        {"call": call, "uri_suffix": uri[-12:], "injected": None, "bytes": len(payload)}
    )
    return payload


runtime.read_bytes_uri = injected

from npa.cli.main import app  # noqa: E402  (must patch before the CLI imports run)

code = 0
try:
    app(args=sys.argv[1:], standalone_mode=False)
except SystemExit as exc:  # typer's own exit
    code = int(exc.code or 0)
except Exception as exc:
    code = 1
    log.append({"stage_error": f"{type(exc).__name__}: {exc}"})

with open(LOG_PATH, "w") as handle:
    json.dump({"mode": MODE, "reads": log, "exit_code": code}, handle, indent=2)
sys.exit(code)
