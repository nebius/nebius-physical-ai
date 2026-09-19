# SAM 3.1 redistribution boundary

The public image ships NPA-owned Apache-2.0 bootstrap/runner code, a pinned
official Python 3.12 Debian base, Debian packages and permissively licensed
Python bootstrap packages. Debian package copyright/license files and Python
distribution license files remain in the image. FFmpeg is the Debian package;
its license and corresponding Debian source are available through Debian's
package archive. No modified FFmpeg binary is shipped.

The [SAM License](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/LICENSE)
governs upstream SAM source and weights. Its permissions and use restrictions
must be reviewed for the operator's activity; the pyproject's MIT classifier
does not replace the actual license. This container redistributes none of that
source or model payload. Source is fetched directly from Meta's pinned Git
revision at runtime, preserving its LICENSE, and the checkpoint is fetched from
the pinned [gated model repository](https://huggingface.co/facebook/sam3.1).

PyTorch, CUDA dependencies and the remaining inference closure are also runtime
downloads into the operator's cache. Their upstream terms apply to those
downloads. A successful checkpoint read verifies exact payload access; it is
not a legal attestation or a grant to redistribute source, weights or caches.
Do not commit that populated cache into an image or publish it as an artifact.

Mandatory build gates inspect every layer for SAM source, weights, package
caches, CUDA payload and secrets. This classification is independent of GPU
acceptance: SAM 3.1 remains excluded from supported release promotion until the
exact public digest has passed a real video segmentation workload.
