"""Manifest runtime — shared invocation layer (MVP).

One execution path (Runtime.invoke); four thin waiters (cli, sdk, yaml, api).
"""

from .runtime import Runtime, InvocationResult
from .catalog import Catalog
from .schema import load_descriptor, parse_descriptor, DescriptorError
from .backends import LocalDockerBackend
from .nebius_backend import NebiusBackend

__all__ = [
    "Runtime",
    "InvocationResult",
    "Catalog",
    "load_descriptor",
    "parse_descriptor",
    "DescriptorError",
    "LocalDockerBackend",
    "NebiusBackend",
]
