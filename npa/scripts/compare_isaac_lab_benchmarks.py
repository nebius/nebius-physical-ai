#!/usr/bin/env python3
"""Emit a sanitized matched Isaac Lab 3-vs-2 benchmark report."""

from __future__ import annotations

import argparse
import json

from npa.workflows.isaac_lab_benchmark import compare_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", help="JSON record files")
    args = parser.parse_args()
    print(json.dumps(compare_files(args.records), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
