# Legacy CUDA 13 base build entrypoint

Use [`../cuda13-blackwell`](../cuda13-blackwell/README.md) for new builds.
This directory retains `build.sh` as a forwarding wrapper for existing callers.
The Dockerfile and validation scripts live only in the canonical directory.
Both image tag prefixes refer to the same output of each base build.
