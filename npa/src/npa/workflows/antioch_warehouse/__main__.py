"""Run the Antioch-authored warehouse workflow stage module."""

import os
import sys
import traceback

from .runner import main

try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit's atexit fast shutdown can otherwise replace an unhandled error with 0.
    # This worker process has already attempted to publish its failure evidence.
    os._exit(1)
