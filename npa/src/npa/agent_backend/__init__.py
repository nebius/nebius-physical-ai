"""Shipped agent-backend package (Phase G).

Modules here are *shipped* to the agent VM as importable files (uploaded next to
``backend.py`` and imported via ``sys.path``) rather than string-substituted into
the ``agent.py`` bootstrap f-string. This is the migration target for the
agentic logic that currently uses the embed mechanism; modules are moved here
incrementally, keeping the embed mechanism working for anything not yet migrated.

Behavior is byte-for-byte identical to the embedded version: the same module
source runs on the VM, just imported from a file instead of inlined. The
``npa/src/npa/cli/agent_*.py`` shims re-export from here so existing import paths
and tests are unchanged.
"""
