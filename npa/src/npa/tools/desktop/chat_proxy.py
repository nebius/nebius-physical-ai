"""Connect the VS Code Codex extension to the same private runtime as mobile chat."""

import json
import os
from pathlib import Path
import sys
import threading

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import unix_connect


def main(config_path, arguments):
    """Proxy IDE JSONL messages without taking ownership of the shared runtime.

    Args:
        config_path: Private desktop chat configuration.
        arguments: Original Codex command arguments from the IDE.
    Returns:
        None.
    Raises:
        OSError: The runtime or installed Codex executable is unavailable.
    """
    config = json.loads(Path(config_path).read_text())
    if "app-server" not in arguments:
        os.execv(config["binary"], [config["binary"], *arguments])
    with unix_connect(
        config["socket"], uri="ws://localhost", max_size=None, compression=None
    ) as connection:

        def forward_input():
            try:
                for line in sys.stdin:
                    if line.strip():
                        connection.send(line.strip())
            except (OSError, ConnectionClosed):
                pass
            finally:
                connection.close()

        threading.Thread(target=forward_input, daemon=True).start()
        try:
            for message in connection:
                print(message, flush=True)
        except (ConnectionClosed, BrokenPipeError):
            return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
