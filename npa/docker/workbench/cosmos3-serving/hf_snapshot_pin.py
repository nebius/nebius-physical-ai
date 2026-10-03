"""Pin guardrail snapshot downloads inside the runtime-only venv.

This module is installed as ``sitecustomize.py`` after the operator accepts the
runtime closure.  It changes no baked payload: it only makes every upstream
``snapshot_download`` for the separately gated guardrail repository resolve to
the revision whose entitlement was checked before service startup.
"""

from __future__ import annotations

import functools
import os

import huggingface_hub


_snapshot_download = huggingface_hub.snapshot_download


@functools.wraps(_snapshot_download)
def _pinned_snapshot_download(repo_id, *args, **kwargs):
    guardrail = os.environ.get(
        "NPA_COSMOS3_SERVE_GUARDRAIL_MODEL", "nvidia/Cosmos-1.0-Guardrail"
    )
    revision = os.environ.get("NPA_COSMOS3_SERVE_GUARDRAIL_REVISION")
    if repo_id == guardrail and revision:
        requested = kwargs.get("revision")
        if requested not in (None, "main", revision):
            raise RuntimeError(
                f"refusing unpinned guardrail revision {requested!r}; expected {revision}"
            )
        kwargs["revision"] = revision
    return _snapshot_download(repo_id, *args, **kwargs)


huggingface_hub.snapshot_download = _pinned_snapshot_download
