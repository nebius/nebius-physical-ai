"""All-replica least-outstanding request routing for Ray Serve 2.56.0."""

from ray.serve.request_router import RequestRouter


class LeastOutstandingRouter(RequestRouter):
    async def choose_replicas(self, candidate_replicas, pending_request=None):
        # Ray chooses the shortest available queue in this single full rank.
        # It retains capacity rejection/backoff for races across proxy routers.
        # This deliberately does not sample two replicas or prioritize locality.
        return [list(candidate_replicas)]
