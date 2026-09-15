"""SkyPilot's supported server plugin hook for a task-owned request queue."""
from sky.server.plugins import BasePlugin
from sky.server.requests.queues.base import MultiprocessingQueueFactory


class IsolatedQueuePlugin(BasePlugin):
    """Avoid SkyPilot's process-global default queue-manager TCP port."""

    def __init__(self, port: int) -> None:
        if not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ValueError("isolated request queue requires a loopback port")
        self.port = port

    def install(self, extension_context):
        extension_context.register_queue_backend_factory(MultiprocessingQueueFactory(self.port))
