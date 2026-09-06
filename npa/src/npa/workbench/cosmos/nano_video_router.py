"""All-replica least-outstanding request routing for Ray Serve 2.56.0."""

from ray.serve.request_router import FIFOMixin, RequestRouter


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
