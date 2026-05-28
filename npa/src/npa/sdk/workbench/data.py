"""SDK wrappers for the Workbench data bridge CLI."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

sync = make_cli_wrapper("npa.cli.workbench.data", "sync_cmd", "Sync S3 data prefixes.")
status = make_cli_wrapper("npa.cli.workbench.data", "status_cmd", "Show data prefix status.")
list = make_cli_wrapper("npa.cli.workbench.data", "list_cmd", "List data prefix objects.")

__all__ = ["list", "status", "sync"]
