"""The `system-info` management verb: what this Encord tool is and can reach.

Reports the SDK pin, the API domain, which credential transports are configured
(names only — never values), and the contracts the tool writes, so an operator
or agent can check the setup without touching Encord.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from npa.workbench.encord.client import (
    configured_credential_transports,
    resolve_domain,
)
from npa.workbench.encord.curate import METRIC_FILTERS
from npa.workbench.encord.push import MEDIA_CATEGORIES
from npa.workbench.encord.schemas import (
    CURATE_RECEIPT_SCHEMA,
    PULL_MANIFEST_SCHEMA,
    PUSH_RECEIPT_SCHEMA,
    ROUNDTRIP_REPORT_SCHEMA,
    TRANSFER_MODES,
)


def _encord_sdk_version() -> str:
    try:
        return version("encord")
    except PackageNotFoundError:
        return ""


def system_info_payload(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Runtime information for the Encord tool; makes no Encord API call."""

    sdk_version = _encord_sdk_version()
    return {
        "status": "ok",
        "tool": "encord",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "encord_sdk": sdk_version or "not installed (pip install 'npa[encord]')",
        "encord_domain": resolve_domain(environ),
        "credential_transports": configured_credential_transports(environ),
        "transfer_modes": list(TRANSFER_MODES),
        "supported_media": sorted(MEDIA_CATEGORIES),
        "curate_metrics": sorted(METRIC_FILTERS),
        "schemas": {
            "push": PUSH_RECEIPT_SCHEMA,
            "curate": CURATE_RECEIPT_SCHEMA,
            "pull": PULL_MANIFEST_SCHEMA,
            "verify": ROUNDTRIP_REPORT_SCHEMA,
        },
    }
