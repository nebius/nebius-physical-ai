"""Report fixed NCore publication phases without inspecting private evidence."""

from contextlib import contextmanager
import sys


_PHASES = frozenset({
    "prepare", "build", "check", "publish", "inputs", "build-receipt",
    "prepublication", "source-binding", "source-guards", "oci-graph", "provenance",
    "byte-scan", "inspection-archives", "shipped-source", "source-delivery",
    "payload", "payload-history", "image-security", "selected-base", "components",
    "bootstrap", "source-recheck", "registry-transfer", "registry-tag-lookup",
    "registry-copy", "registry-visibility", "anonymous-verification", "anonymous-copy",
    "anonymous-graph", "anonymous-tag-check",
})


@contextmanager
def phase(identifier):
    """Emit flushed begin/pass/failure markers for one allowlisted phase.

    Args:
        identifier: Fixed identifier from the NCore publication phase allowlist.
    Returns:
        Context manager; pass means the enclosed checks returned successfully.
    Raises:
        ValueError: The identifier is not an allowlisted plain string.
        BaseException: Any enclosed failure is re-raised without formatting it.
    """
    if type(identifier) is not str or identifier not in _PHASES:
        raise ValueError("invalid_ncore_publication_phase")
    print(f"NCore OCI phase={identifier} status=begin", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        print(f"NCore OCI phase={identifier} status=failure", file=sys.stderr, flush=True)
        raise
    print(f"NCore OCI phase={identifier} status=pass", file=sys.stderr, flush=True)


def run_phase(identifier, operation, *args):
    """Run an existing NCore check inside its fixed diagnostic phase.

    Args:
        identifier: Allowlisted phase identifier.
        operation: Existing check or action; its output stays private.
        *args: Arguments forwarded only to the operation.
    Returns:
        The operation's result, unchanged.
    Raises:
        BaseException: Phase validation or the operation failed.
    """
    with phase(identifier):
        return operation(*args)
