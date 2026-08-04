"""Shipped agent-backend package (Phase G).

Modules here are *shipped* to the agent VM as importable files (uploaded next to
``backend.py`` and imported via ``sys.path``) rather than string-substituted into
the ``agent.py`` bootstrap f-string. Actions, semantic routing, the Sim2Real
outer loop, and memory now use this package; the embed mechanism remains only for
older modules not included in that migration.

Public imports and deployment wiring remain compatible with the embedded
version: the same module source runs on the VM, just imported from a file instead
of inlined. Intentional behavior changes are documented and tested. The
``npa/src/npa/cli/agent_*.py`` shims re-export from here so existing import paths
and tests are unchanged.
"""
