#!/usr/bin/env python3
"""In-image entry point for the LTX-2.5 operator licensing gate.

The gate logic itself is ``npa/src/npa/workbench/ltx2/licensing.py``, copied into
the image verbatim by the Dockerfile. It is deliberately stdlib-only so the same
tested module runs both in the repo's unit suite and inside the container, before
any dependency has been installed. Re-implementing the gate in shell would give
us two versions of a legal control and one of them would drift.

Exit codes follow the convention the workbench already uses for licence refusals:
78 (``EX_CONFIG``) means the operator has not made a declaration we may act on,
and nothing has been fetched.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import licensing  # noqa: E402  (path is set above so the copied module resolves)

EX_CONFIG = 78


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "check"

    if mode == "terms":
        # Printable summary of what the operator is being asked to accept. Safe
        # to run without a declaration, and never fetches anything.
        print(licensing.refusal_text("Declaration not yet made."))
        return 0

    try:
        declaration = licensing.declaration_from_env(os.environ)
    except licensing.LtxLicenseError as error:
        print(str(error), file=sys.stderr)
        return EX_CONFIG

    if mode == "check":
        print(
            "npa-ltx2: operator declaration accepted "
            f"(entity={declaration.entity_class}, use={declaration.use_class}, "
            f"derived_model_training={declaration.derived_model_training})",
            file=sys.stderr,
        )
        return 0

    if mode == "declaration":
        print(json.dumps(declaration.as_dict(), indent=2, sort_keys=True))
        return 0

    if mode == "provenance":
        run_id = os.environ.get("NPA_LTX_RUN_ID", "unknown")
        outputs = tuple(
            item for item in os.environ.get("NPA_LTX_OUTPUTS", "").split(",") if item
        )
        model_files = tuple(
            item for item in os.environ.get("NPA_LTX_MODEL_FILES", "").split(",") if item
        )
        record = licensing.ProvenanceRecord(
            declaration=declaration,
            run_id=run_id,
            outputs=outputs,
            model_files=model_files,
        )
        print(json.dumps(record.as_dict(), indent=2, sort_keys=True))
        return 0

    print(
        f"npa-ltx2: unknown gate mode {mode!r} "
        "(use check, terms, declaration, or provenance)",
        file=sys.stderr,
    )
    return EX_CONFIG


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
