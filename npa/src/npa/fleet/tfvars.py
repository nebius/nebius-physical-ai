"""Compatibility module alias for the backend-owned mk8s renderer."""

import sys

from npa.cluster_backends import mk8s_render as _implementation

sys.modules[__name__] = _implementation
