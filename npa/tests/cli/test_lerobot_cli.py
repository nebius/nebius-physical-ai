"""Compatibility entrypoint for the LeRobot CLI tests.

The LeRobot workbench tests historically lived in test_workbench_cli.py. Keep
this file so targeted runs match the per-tool naming convention used by the
other workbench tools.
"""

from .test_workbench_cli import *  # noqa: F401,F403
