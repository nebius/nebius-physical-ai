"""Check that a background process retains its POSIX shared memory after logout."""

import argparse
import json
import os
from pathlib import Path
import time
import uuid


def _probe(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    path = Path("/dev/shm") / ("wam_ipc_probe_" + uuid.uuid4().hex)
    started = time.time()
    with output.open("x") as receipt:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.write(descriptor, b"workload-owned POSIX IPC probe\n")
            present = True
            for _ in range(300):
                time.sleep(0.1)
                if not path.exists():
                    present = False
                    break
            result = {
                "started_unix": started,
                "ended_unix": time.time(),
                "survived_logout": present,
                "observation_window_seconds": 30,
            }
            receipt.write(json.dumps(result, indent=2) + "\n")
        finally:
            os.close(descriptor)
            path.unlink(missing_ok=True)
    return 0 if present else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    os.umask(0o077)
    raise SystemExit(_probe(parser.parse_args().output_path))
