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
  `/usr/share/doc/*/copyright`. The pinned CMake, GNU Make, and GNU C++
  toolchain and their Debian dependencies build runtime-authorized source
  packages inside the existing network-isolated sandbox; runtime packages
  and their generated wheels remain absent from the image.
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
The NVIDIA software agreement uses the explicitly declared
`nvidia-navigation-uuid-v1` identity: only two equal generated navigation UUIDs
in fixed HTML and JavaScript contexts become a fixed zero UUID. The complete
remaining document and its byte length stay hash-bound, including all agreement
text. Runtime diagnostics retain the observed raw hash and canonical hash.
Every other terms source and runtime artifact uses its original raw-byte hash.
All 135 runtime artifacts now have a reviewed positive size and license
classification, and the image manifest binds the owner review report. Runtime
use remains disabled until the customer directly signs a short-lived
authorization with a customer-controlled key, bound to the customer, run,
exact terms, runtime manifest, and immutable qualified candidate when one
exists. NPA may authenticate the caller, transport the signed evidence, and
validate it, but neither the manager nor the control plane accepts, acknowledges,
signs, issues, or invents customer terms evidence. HF/NGC
credentials prove upstream access only; metadata review and download success
are not consent or authorization to download.
