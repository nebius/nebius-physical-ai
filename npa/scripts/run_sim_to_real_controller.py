#!/usr/bin/env python3
"""Run the Sereact sim-to-real controller loop."""

from __future__ import annotations

import sys

from npa.workflows.sereact_sim_to_real import main


if __name__ == "__main__":
    sys.exit(main())
