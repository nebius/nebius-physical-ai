#!/usr/bin/env python3
"""Thin shim: the implementation moved into the package as ``npa.workflows.isaac_capture``.

A `toolRef` runs `python3 -m npa.workflows.isaac_capture` inside a pod that has npa installed
but no repo checkout, so the code cannot live under `npa/scripts/`. This shim keeps the path
documented in `docs/hackathon-isaac-token-factory.md` working for anyone with a checkout.
"""

from __future__ import annotations

import sys

from npa.workflows.isaac_capture import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
