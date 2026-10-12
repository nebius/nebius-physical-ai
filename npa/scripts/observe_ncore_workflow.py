#!/usr/bin/env python3
"""Retain live runtime identity and final status for NCore qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path

from image_byte_scan import core as W

from ncore_publication.process import ROOT, committed_npa_imports, committed_source
from ncore_publication.workflow_observer import observe_workflow


def _parser():
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-s3-uri", required=True)
    parser.add_argument("--project", default="")
    parser.add_argument("--sky-bin", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    return parser


def main(argv=None) -> int:
    os.umask(0o077)
    args = _parser().parse_args(argv)
    evidence_dir = args.evidence_dir.absolute()
    with (
        W.cancellation_scope(),
        W.authorized_roots(evidence_dir, ROOT),
        committed_npa_imports(args.source_sha),
    ):
        committed_source(args.source_sha)
        result = observe_workflow(
            source_sha=args.source_sha,
            run_id=args.run_id,
            workflow_s3_uri=args.workflow_s3_uri,
            project=args.project,
            sky_bin=args.sky_bin,
            context=args.context,
            namespace=args.namespace,
            expected_image=args.expected_image,
            evidence_dir=evidence_dir,
            poll_seconds=args.poll_seconds,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
