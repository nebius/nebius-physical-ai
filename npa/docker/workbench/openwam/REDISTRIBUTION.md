# OpenWAM image redistribution record

Review date: 2026-10-03. This is an engineering record, not legal advice.

This Dockerfile is an **operator-private, unvalidated** runtime recipe. It
clones OpenWAM source at `48bd67b89d489b14d03b8d92bc66e65d306df32e`, retains
its Apache-2.0 license in `/opt/openwam/LICENSE`, and provisions its pinned
CUDA Python environment. It also installs the separate upstream-documented
LIBERO client environment from
`Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01`,
with OpenWAM's supplied PyTorch compatibility patch. The source tree carries
LIBERO's MIT license.

No OpenWAM-alpha checkpoint, Wan model/backbone, LIBERO benchmark dataset,
trained checkpoint, generated observation, cache, credential, or acceptance
record is copied into an image layer. The workflow downloads the exact public
revisions into run-scoped storage and labels them as private artifacts. This
does not imply a grant to redistribute those runtime artifacts or outputs.

The image has not yet passed an exact built-byte license/SBOM review, pushed
digest scan, anonymous-pull test, or GPU workflow qualification. It must stay
in an operator-controlled registry. Do not promote it to a public catalog or
public registry solely because the OpenWAM and model cards identify Apache-2.0.
Before any redistribution decision, inspect the actual built image and all
resolved OS, CUDA, PyTorch, Python-package, source, data, model, cache, and
output terms separately. No new NPA EULA, `ACCEPT_*` variable, or per-image
attestation is required by this recipe; any future upstream acceptance follows
only the upstream provider's documented mechanism.

Optional telemetry remains disabled: the image sets Weights & Biases offline
mode and invokes Rerun's documented `analytics disable` command for its final
user during build. Neither setting accepts an upstream term or grants an
upstream payload right.
