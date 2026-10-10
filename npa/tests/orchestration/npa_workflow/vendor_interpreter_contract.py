"""Shared rendered-shell expectations for vendor interpreter tests."""

from __future__ import annotations

VENDOR_INTERPRETER_CLI_IMPORT = "npa.cli.main"
VENDOR_INTERPRETER_CLI_PROBE = f"-c 'import {VENDOR_INTERPRETER_CLI_IMPORT}'"
VENDOR_INTERPRETER_CLI_INSTALL_GUARD = (
    f'if ! "$npa_vendor_python" {VENDOR_INTERPRETER_CLI_PROBE}'
)
VENDOR_INTERPRETER_CLI_WARNING = (
    f"warning: {VENDOR_INTERPRETER_CLI_IMPORT} is not importable "
    "from $npa_vendor_python:"
)
VENDOR_INTERPRETER_CLI_DIAGNOSTIC = (
    f'"$npa_vendor_python" {VENDOR_INTERPRETER_CLI_PROBE} 2>&1 | tail -3 >&2'
)
