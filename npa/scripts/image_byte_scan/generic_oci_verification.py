#!/usr/bin/env python3
"""Verify a generic Linux/amd64 OCI graph before complete-byte scanning.

This is deliberately product-neutral.  It verifies the exact archive identity,
descriptor graph, decoded layer ``diff_id`` values, and every historical regular
file before the shared scanner accounts for every archive and layer byte.  It
does not authorize publication, runtime use, redistribution, or any product
claim.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import (
        core as W,
        ncore_verification as _graph,
        prepare as P,
    )
else:
    from . import core as W, ncore_verification as _graph, prepare as P


SCHEMA = "npa.generic-oci-verification.v1"
PLATFORM = _graph.PLATFORM


def inspect(fd, length, expected_id):
    """Re-export the shared complete OCI graph inspection."""
    return _graph.inspect(fd, length, expected_id)


def bind(result, verification, expected_id):
    """Rebind a generic receipt to the exact archive graph before scanning."""
    return _graph.bind(result, verification, expected_id)


def verify(archive_binding, expected_id):
    """Return a product-neutral receipt for an exact OCI archive."""
    receipt = _graph.verify(archive_binding, expected_id)
    receipt["schema_version"] = SCHEMA
    return receipt


def main(argv=None):
    """Write an owner-only generic graph verification receipt."""
    os.umask(0o077)
    held = None
    try:
        with W.cancellation_scope():
            parser = W.SanitizedArgumentParser(description=__doc__)
            parser.add_argument("--analysis-root", type=Path, required=True)
            parser.add_argument("--trusted-root", type=Path, required=True)
            parser.add_argument("--archive", type=Path, required=True)
            parser.add_argument("--expected-image-id", required=True)
            parser.add_argument("--output-dir", type=Path, required=True)
            args = parser.parse_args(argv)
            with W.authorized_roots(args.analysis_root, args.trusted_root):
                result = verify(P.binding(args.archive), args.expected_image_id)
                directory, held = W.create_output(args.output_dir)
                W.write_private_json(directory, "verification.json", result)
        print("generic OCI graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("generic OCI graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
