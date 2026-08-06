"""Bounded argv transport for generated Isaac Job shell programs."""

from __future__ import annotations

import base64
import gzip

_ARG_CHUNK_CHARS = 60_000
_SCRIPT_PATH = "/tmp/npa-isaac-job-script.sh"
_DECODE_STUB = (
    "set -euo pipefail; "
    f"printf '%s' \"$@\" | base64 --decode | gzip --decompress > {_SCRIPT_PATH}; "
    f"chmod 700 {_SCRIPT_PATH}; exec /bin/bash {_SCRIPT_PATH}"
)


def compressed_bash_launch(script: str) -> tuple[list[str], list[str]]:
    """Return a bash command/argv whose individual arguments stay below 128 KiB.

    Linux limits each individual argv/environment string to 128 KiB even when
    the process-wide ``ARG_MAX`` is larger. Generated Isaac programs include
    task source and curated scenarios, so passing the program directly to
    ``bash -lc`` can fail before the container starts. Deterministically gzip
    and base64 the bytes, split them into conservative chunks, and let a small
    login-shell stub reconstruct and execute the exact program in the pod.
    """

    encoded = base64.b64encode(gzip.compress(script.encode("utf-8"), mtime=0)).decode(
        "ascii"
    )
    chunks = [
        encoded[offset : offset + _ARG_CHUNK_CHARS]
        for offset in range(0, len(encoded), _ARG_CHUNK_CHARS)
    ]
    # With ``bash -c COMMAND ARG0 ARG1...``, the label becomes $0 and only the
    # payload chunks become "$@" inside the decoder stub.
    return ["/bin/bash", "-lc"], [_DECODE_STUB, "npa-isaac-payload", *chunks]


def decode_compressed_bash_args(args: list[str]) -> str:
    """Decode :func:`compressed_bash_launch` output for tests/audits."""

    if len(args) < 3 or args[0] != _DECODE_STUB or args[1] != "npa-isaac-payload":
        raise ValueError("not an NPA compressed Isaac bash payload")
    return gzip.decompress(base64.b64decode("".join(args[2:]))).decode("utf-8")
