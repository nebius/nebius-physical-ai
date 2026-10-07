"""Retain private stage failures without exposing worker output to console logs."""

from __future__ import annotations

import sys
import traceback
import uuid

from npa.workbench.dataset.storage import write_json_uri


def _record(output_uri, payload):
    uri = output_uri.rsplit("/", 1)[0] + "/diagnostics/" + uuid.uuid4().hex + ".json"
    try:
        write_json_uri(uri, payload)
    except Exception:
        # Storage/authentication failures must not replace the original stage error.
        print(
            "Private diagnostic storage failed; original failure retained.",
            file=sys.stderr,
        )


def _failure(output_uri, error):
    _record(
        output_uri,
        {
            "error_type": type(error).__name__,
            "traceback": "".join(traceback.format_exception(error)),
        },
    )


def _cleanup(action, output_uri):
    try:
        action()
    except Exception as error:
        _failure(output_uri, error)
        print(
            "Cleanup failed; inspect private diagnostics for retained resources.",
            file=sys.stderr,
        )
