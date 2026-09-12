# Third-party notices for the LIBERO neutral bootstrap

The neutral image contains no LIBERO, dataset, model, MuJoCo, PyTorch, CUDA,
cuDNN, NCCL, NVIDIA-wheel, robomimic, or render-asset payload. The following
notices cover only bytes intended for the neutral image.

- `python:3.10-slim-bookworm`, linux/amd64 manifest
  `sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496`,
  is produced by the Docker Official Images Python project. CPython 3.10.21 is
  available under the Python Software Foundation License. The base also ships
  pip (MIT), setuptools (MIT), wheel (MIT), and their recorded vendored OSS
  dependencies.
- Debian 12 packages are redistributed under their individual licenses. The
  authoritative package, source, hash, and corresponding-source locations are
  in `debian-packages.lock`; installed copyright texts remain under
  `/usr/share/doc/*/copyright`.
- The NPA bootstrap, scanner, workflow, and test code is Apache-2.0 under this
  repository's license.

Runtime-only references in `runtime-manifest.json` are identity and delivery
metadata, not embedded software or a permission claim. When separately
authorized, pinned LIBERO source is MIT, the selected demonstration is
CC-BY-4.0 with attribution to LIBERO / Lifelong Robot Learning, and
`google-bert/bert-base-cased` is Apache-2.0. Each GPU/runtime package remains
subject to its own upstream terms and the manager-issued use decision.
