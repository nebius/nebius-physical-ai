# Third-party notices for the LIBERO neutral bootstrap

The neutral image contains no LIBERO, dataset, model, MuJoCo, PyTorch, CUDA,
cuDNN, NCCL, NVIDIA-wheel, robomimic, or render-asset payload. The following
notices cover only bytes intended for the neutral image.

The OCI license label uses
`Apache-2.0 AND LicenseRef-NPA-LIBERO-Neutral-Third-Party`: Apache-2.0 covers
the NPA-authored bytes, while this file and the installed Debian copyright
records are authoritative for the separately licensed base and package bytes.

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
subject to its own upstream terms and the customer's personal acknowledgement. The
manifest's hash-bound official MIT, CC BY 4.0, Apache 2.0, PyTorch, CUDA,
NVIDIA software, and cuDNN terms sources are runtime refusal inputs only; none
of their fetched bytes is retained in this image or treated as acceptance.
All 135 runtime artifacts now have a reviewed positive size and license
classification, and the image manifest binds the owner review report. Runtime
use remains disabled until an authenticated NPA customer/control-plane surface
issues a short-lived authorization bound to the customer, run, exact terms,
runtime manifest, and immutable qualified candidate when one exists. HF/NGC
credentials prove upstream access only; metadata review and download success
are not consent or authorization to download.
