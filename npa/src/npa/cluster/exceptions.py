"""Exceptions for NPA-managed Kubernetes clusters."""

from __future__ import annotations


class ClusterError(RuntimeError):
    """Base error for cluster lifecycle operations."""


class ClusterConfigError(ClusterError, ValueError):
    """Invalid or incomplete cluster configuration."""


class ClusterNotFoundError(ClusterError):
    """Cluster was not found in Nebius or local state."""


class ClusterStateError(ClusterError):
    """Local cluster state is missing or malformed."""


class ClusterTimeoutError(ClusterError, TimeoutError):
    """Cluster did not reach the requested state before timeout."""


class NodeGroupError(ClusterError):
    """Base error for node-group lifecycle operations."""


class NodeGroupNotFoundError(NodeGroupError):
    """Node group was not found in Nebius or local state."""


class NodeGroupTimeoutError(NodeGroupError, TimeoutError):
    """Node group did not reach the requested state before timeout."""
