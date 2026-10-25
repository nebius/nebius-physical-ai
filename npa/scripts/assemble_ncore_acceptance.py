#!/usr/bin/env python3
"""Assemble independently reviewed NCore publication acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from image_byte_scan import core as W
from ncore_publication.acceptance import build_statement, finalize_acceptance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    statement = subparsers.add_parser("statement")
    statement.add_argument("--analysis-root", type=Path, required=True)
    statement.add_argument("--gate-dir", type=Path, required=True)
    statement.add_argument("--evidence-root", type=Path, required=True)
    statement.add_argument("--proposed-manifest", type=Path, required=True)
    statement.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--analysis-root", type=Path, required=True)
    finalize.add_argument("--statement", type=Path, required=True)
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    os.umask(0o077)
    args = _parser().parse_args()
    with W.authorized_roots(Path.cwd(), args.analysis_root):
        if args.action == "statement":
            result = build_statement(
                analysis_root=args.analysis_root,
                gate_dir=args.gate_dir,
                evidence_root=args.evidence_root,
                proposed_manifest_path=args.proposed_manifest,
                output_path=args.output,
            )
        else:
            result = finalize_acceptance(
                analysis_root=args.analysis_root,
                statement_path=args.statement,
                review_path=args.review,
                output_path=args.output,
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "action": args.action,
                "candidate_commit": result.get(
                    "candidate_commit", result.get("development_sha")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
