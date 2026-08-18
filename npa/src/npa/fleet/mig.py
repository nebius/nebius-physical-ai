"""Compatibility module alias for backend-owned Managed Kubernetes MIG support."""

import sys

from npa.cluster_backends import mig as _implementation

sys.modules[__name__] = _implementation
