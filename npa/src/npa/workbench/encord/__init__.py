"""npa.workbench.encord - push to, curate in, pull from, and verify Encord.

Only schemas and pure string helpers are re-exported here; the optional
``encord`` SDK is imported lazily inside functions, so this package imports
cleanly without it.
"""

from __future__ import annotations

from npa.workbench.encord.curate import curate_receipt_uri_for
from npa.workbench.encord.pull import pull_manifest_uri_for
from npa.workbench.encord.push import push_receipt_uri_for
from npa.workbench.encord.schemas import (
    CURATE_RECEIPT_FILENAME,
    CURATE_RECEIPT_SCHEMA,
    PULL_MANIFEST_FILENAME,
    PULL_MANIFEST_SCHEMA,
    PUSH_RECEIPT_FILENAME,
    PUSH_RECEIPT_SCHEMA,
    ROUNDTRIP_REPORT_FILENAME,
    ROUNDTRIP_REPORT_SCHEMA,
    CurateReceipt,
    EncordAuthError,
    EncordToolError,
    PulledItem,
    PullManifest,
    PushedItem,
    PushReceipt,
    RoundtripReport,
)
from npa.workbench.encord.verify import roundtrip_report_uri_for

__all__ = [
    "CURATE_RECEIPT_FILENAME",
    "CURATE_RECEIPT_SCHEMA",
    "PULL_MANIFEST_FILENAME",
    "PULL_MANIFEST_SCHEMA",
    "PUSH_RECEIPT_FILENAME",
    "PUSH_RECEIPT_SCHEMA",
    "ROUNDTRIP_REPORT_FILENAME",
    "ROUNDTRIP_REPORT_SCHEMA",
    "CurateReceipt",
    "EncordAuthError",
    "EncordToolError",
    "PulledItem",
    "PullManifest",
    "PushedItem",
    "PushReceipt",
    "RoundtripReport",
    "curate_receipt_uri_for",
    "pull_manifest_uri_for",
    "push_receipt_uri_for",
    "roundtrip_report_uri_for",
]
