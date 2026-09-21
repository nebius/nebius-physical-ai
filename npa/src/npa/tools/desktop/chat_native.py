"""Connect the shared chat service to the existing Mac VS Code session owner."""

from pathlib import Path
import subprocess
import socket
import errno

try:
    from .chat_rpc import CodexConnection
except ImportError:
    from chat_rpc import CodexConnection


class NativeTransport:
    """Exchange private JSONL frames with the native Codex adapter.

    Args:
        config: Private local runtime configuration.
        config_path: Path passed to the child without embedding credentials.
    Returns:
        None.
    Raises:
        OSError: The configured Node runtime cannot start.
    """

    def __init__(self, config, config_path):
        self.child = None
        self.socket = None
        if config.get("native_socket"):
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(config["native_socket"])
            self.reader = self.socket.makefile("r")
            self.writer = self.socket.makefile("w")
            return
        adapter = Path(__file__).parent / "native" / "bridge.mjs"
        self.child = subprocess.Popen(
            [config["node"], str(adapter), str(config_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self.reader = self.child.stdout
        self.writer = self.child.stdin

    def send(self, message):
        """Write one JSONL frame to the private adapter.

        Args:
            message: Serialized protocol message.
        Returns:
            None.
        Raises:
            OSError: The adapter connection has closed.
        """
        self.writer.write(message + "\n")
        self.writer.flush()

    def __iter__(self):
        return iter(self.reader)

    def close(self):
        """Release this client without terminating the independent engine.

        Args:
            None.
        Returns:
            None.
        Raises:
            OSError: Closing the transport fails unexpectedly.
        """
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except OSError as error:
                if error.errno != errno.ENOTCONN:
                    raise
            self.writer.close()
            self.reader.close()
            self.socket.close()
        if self.child:
            self.writer.close()
            self.child.wait()


def native_connection(config, config_path, on_notification):
    """Create a connection that follows existing Mac chats without replacing them.

    Args:
        config: Private local runtime configuration.
        config_path: Configuration file for the Node adapter.
        on_notification: Callback used by the shared chat service.
    Returns:
        A CodexConnection with the same interface as the Linux connection.
    Raises:
        OSError: The adapter cannot start.
        RuntimeError: Protocol initialization fails.
    """
    transport = NativeTransport(config, config_path)
    return CodexConnection(None, on_notification, transport=transport)
