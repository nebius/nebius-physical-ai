#!/usr/bin/env python3
"""Forward the scheduler command without adding a shell or hidden workload."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("openvla-oft bootstrap requires a command")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
