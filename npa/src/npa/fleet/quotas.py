"""Compatibility module alias for backend-owned capacity/quota preflight."""

import sys

from npa.cluster_backends import quotas as _implementation

sys.modules[__name__] = _implementation
