#!/usr/bin/env python3
"""Prepare NCore OCI graph/count input for the complete-byte scanner.

This receipt verifies archive identity, graph closure, decoded layer digests and
the all-ancestor regular-file population. It does not authorize publication or
replace the mandatory byte scan, confidentiality policy, or other image gates.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W, oci_verification as O, prepare as P
else:
    from . import core as W, oci_verification as O, prepare as P

SCHEMA = "npa.ncore.oci-verification.v1"
PLATFORM = O.PLATFORM
inspect = O.inspect
bind = O.bind


def verify(archive_binding, expected_id):
    """Preserve NCore's report contract using shared structural OCI verification.

    Args:
        archive_binding: Owner-only archive path and SHA-256 binding.
        expected_id: Independently obtained image index digest.
    Returns:
        The existing NCore graph and ancestor-population report.
    Raises:
        W.ScanError: Archive, graph, layer, or population verification fails.
    """
    return {"schema_version": SCHEMA, **O.verify_graph(archive_binding, expected_id)}


def main(argv=None):
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
        print("NCore OCI graph verification completed")
        return 0
    except W.INPUT_ERRORS:
        print("NCore OCI graph verification failed")
        return 1
    finally:
        if held is not None:
            os.close(held)


if __name__ == "__main__":
    raise SystemExit(main())
