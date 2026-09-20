#!/usr/bin/env python3
"""Bind and run one checked local NCore candidate without registry access."""

from __future__ import annotations

import os
from pathlib import Path

from image_byte_scan import core as W
from ncore_publication.process import (
    ROOT,
    committed_npa_imports,
    committed_source,
)
from ncore_publication.qualification import (
    bind_candidate_image,
    run_candidate_qualification,
)


def _parser():
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--s3-env-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--control-output-path", required=True)
    parser.add_argument("--conversion-path", required=True)
    parser.add_argument("--audit-output-path", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    return parser


def _private_directory(path: Path) -> None:
    if path.exists():
        if (
            path.is_symlink()
            or not path.is_dir()
            or path.stat().st_uid != os.getuid()
            or path.stat().st_mode & 0o077
        ):
            raise ValueError("qualification directory is not private")
    else:
        path.mkdir(mode=0o700)


def main(argv=None) -> int:
    os.umask(0o077)
    try:
        args = _parser().parse_args(argv)
        root = args.analysis_root.absolute()
        with (
            W.cancellation_scope(),
            W.authorized_roots(root, ROOT),
            committed_npa_imports(args.source_sha),
        ):
            committed_source(args.source_sha)
            _private_directory(args.evidence_dir)
            _private_directory(args.cache_dir)
            candidate_path = args.evidence_dir / "candidate-image.json"
            bind_candidate_image(
                source_sha=args.source_sha,
                analysis_root=root,
                gate_dir=args.gate_dir,
                output_path=candidate_path,
            )
            run_candidate_qualification(
                run_id=args.run_id,
                candidate_path=candidate_path,
                evidence_dir=args.evidence_dir,
                cache_dir=args.cache_dir,
                env_file=args.s3_env_file,
                input_path=args.input_path,
                control_output_path=args.control_output_path,
                conversion_path=args.conversion_path,
                audit_output_path=args.audit_output_path,
                expected_archive_sha256=args.expected_archive_sha256,
            )
        print("NCore checked local candidate qualification passed")
        return 0
    except (Exception, KeyboardInterrupt):
        print(
            "NCore checked local candidate qualification failed; inspect private evidence"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
