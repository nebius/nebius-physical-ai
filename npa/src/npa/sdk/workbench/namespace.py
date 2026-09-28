"""Python access to native Kubernetes namespace management."""

from npa.workbench.namespaces import (
    apply_namespace,
    namespace_manifests,
    write_namespace_context,
)

__all__ = ["apply_namespace", "namespace_manifests", "write_namespace_context"]
