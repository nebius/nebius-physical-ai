"""All-replica least-outstanding request routing for Ray Serve 2.56.0."""

EXPECTED_RAY_VERSION = "2.56.0"


def _assert_ray_version() -> None:
    # This module subclasses private Ray Serve internals (FIFOMixin /
    # RequestRouter) whose behavior changed between releases. A rebuild against
    # a different Ray version would silently alter routing semantics, so fail
    # loudly instead. The check runs BEFORE the import below: importing first
    # would execute the risky import before the guard can reject a wrong
    # version. Skipped when ray is not installed at all (unit-test harnesses
    # fake ray.serve.request_router); the import below then decides what
    # happens next.
    try:
        import ray
    except ImportError:
        return
    actual = ray.__version__
    if actual != EXPECTED_RAY_VERSION:
        raise RuntimeError(
            f"nano_video_router requires ray=={EXPECTED_RAY_VERSION} "
            f"(it subclasses ray.serve.request_router internals); "
            f"found ray=={actual}. Rebuild the image against the pinned "
            "Ray version."
        )


_assert_ray_version()

# Deliberately after the version guard above (E402): the guard must reject a
# wrong Ray version before this import executes.
from ray.serve.request_router import FIFOMixin, RequestRouter  # noqa: E402


class LeastOutstandingRouter(FIFOMixin, RequestRouter):
    # FIFO fallback retains pending requests when an out-of-order assignment
    # retires the scheduler that previously claimed their routing metadata.
    # This governs request order; every selection still compares all replicas.
    def __init__(self, *args, **kwargs):
        # Ray 2.56's cached-success path can keep a routing task alive after
        # out-of-order assignments and spawn probes without yielding. Await a
        # fresh full-rank queue snapshot instead; strict rejection stays enabled.
        kwargs["use_replica_queue_len_cache"] = False
        super().__init__(*args, **kwargs)

    async def choose_replicas(self, candidate_replicas, pending_request=None):
        # There is no narrower locality rank to try before normal retry backoff.
        if pending_request is not None:
            pending_request.routing_context.should_backoff = True
        # Ray probes the entire rank, chooses its shortest available queue, and
        # retains capacity rejection for races between routing tasks.
        return [list(candidate_replicas)]
