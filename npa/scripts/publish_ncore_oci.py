#!/usr/bin/env python3
"""Run the NCore-only committed OCI build, prepublication gates and exact transfer."""

import sys
import tempfile


if __name__ == "__main__":
    # Start project imports with an empty private cache namespace. -B alone
    # disables cache writes but still permits reading existing .pyc files.
    with tempfile.TemporaryDirectory(prefix="npa-publisher-imports-") as cache:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = cache
        from ncore_publication.cli import main

        raise SystemExit(main())
