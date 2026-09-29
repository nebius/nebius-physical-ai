# Legacy CUDA 13 base build entrypoint

Use [`../cuda13-blackwell`](../cuda13-blackwell/README.md) for new builds.
This directory retains `build.sh` as a forwarding wrapper for existing callers.
The Dockerfile and validation scripts live only in the canonical directory.
The default FA4 build tags the same image with both prefixes. The new FA2
variant uses only its canonical `cuda13-blackwell-fa2` tag.
