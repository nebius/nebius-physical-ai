"""Check that a background process retains its POSIX shared memory after logout."""

import argparse
import json
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import time


def _exists(name):
    try:
        view = SharedMemory(name=name)
    except FileNotFoundError:
        return False
    view.close()
    return True


def _probe(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with output.open("x") as receipt:
        memory = SharedMemory(create=True, size=64)
        try:
            memory.buf[:9] = b"wam-probe"
            present = True
            for _ in range(300):
                time.sleep(0.1)
                if not _exists(memory.name):
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
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                # The failed probe already recorded that logout removed the object.
                pass
    return 0 if present else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    os.umask(0o077)
    raise SystemExit(_probe(parser.parse_args().output_path))
