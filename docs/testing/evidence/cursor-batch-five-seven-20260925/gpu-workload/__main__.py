"""Dispatch the immutable zip payload to its training or verification program."""

import sys

if len(sys.argv) < 2 or sys.argv[1] not in {"train", "verify"}:
    raise SystemExit("usage: payload.pyz {train|verify} [arguments]")
command = sys.argv.pop(1)
if command == "train":
    from train import main
else:
    from verify import main
main()
