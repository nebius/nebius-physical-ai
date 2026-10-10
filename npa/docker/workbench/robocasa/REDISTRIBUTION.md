# RoboCasa candidate runtime boundary

This recipe remains a validation candidate, not a supported or published image.
Source eligibility, private use, exact-image safety, GPU behavior, and public
release authorization are separate requirements. Older images retain their
original bytes and findings.

The replacement base is the anonymous NVIDIA CUDA 12.9.1 Ubuntu22.04 **base**
linux/amd64 manifest, pinned in the Dockerfile and base-security inventory. It
does not inherit the former CUDA12.4 cuDNN development layers. CUDA runtime
terms remain separate from cuDNN; the application uses the hash-locked CUDA12
Torch closure and does not claim a new GPU capability from the base name.

The exact `nvidia-cudnn-cu12==9.20.0.48` public wheel embeds the January2020
supplement, whose redistribution list includes headers. The
[current cuDNN supplement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
instead identifies runtime shared libraries. This recipe selects their narrower
intersection: eight byte-identical runtime ELF libraries and all package/license
metadata are retained; fourteen verified SDK headers are omitted in the **same
RUN** that installs the wheel. No cryptographic self-test is removed. The
installed RECORD is updated to describe the actual filtered population, and a
receipt records the original wheel hash and retained member hashes. The original
license remains unmodified; this does not claim NVIDIA endorsement.

`cudnn-runtime-boundary.json` binds every immutable member to the exact public
wheel. The build rejects unknown files, versions, changed hashes, missing
notices, malformed RECORD entries, and symlinks before deleting any header.
The filter is for an isolated build installation, not for repairing a shared
environment or a previously committed image layer. Ancestor bytes cannot be
sanitized by later deletion.

The NVSHMEM3.4.5 wheel's own notice is retained. Its complete official product
supplement and third-party notice are additionally hash-verified under
`/usr/share/doc/robocasa-npa-act/NVSHMEM-LICENSE.txt`. Existing RoboCasa,
robosuite, LeRobot, qpsolvers and quadprog notices and corresponding-source
archives remain required. Runtime-fetched kitchen assets and all model/data
payloads remain outside the image.

The recipe upgrades inherited Ubuntu packages and rejects versions older than
the GnuPG and OpenSSL fixes identified by
[USN-7946-1](https://ubuntu.com/security/notices/USN-7946-1) and
[CVE-2026-84782](https://ubuntu.com/security/CVE-2026-84782).

Upgrading the merged runtime does not remove vulnerable binaries from earlier
image layers. The final stage therefore starts from `scratch` and copies only
the fully prepared filesystem, after package upgrades, header cleanup and
runtime checks. No old CUDA-base or installer-builder layers are inherited.
The base's exact CUDA driver constraints, library path and visibility settings
are restored alongside the unchanged service configuration; their public
manifest/config identities are recorded in `cuda-base-runtime.json`.
The complete final package database, license notices and corresponding sources
remain in that filesystem. This changes the shipped layer population, not the
historical evidence: older images and their raw findings remain unchanged.
The new image must prove its single final root, actual configuration, complete
package/byte population and real workload behavior before acceptance. The
source contract and metadata tests alone do not establish image safety.

The installer is `pip==26.2.1+npa.1`, an explicitly identified NPA derivative,
not an upstream pip release. The shared `common/secure_pip` builder uses pinned
upstream vendoring machinery, preserves notices and records the three repaired
vendor sources. Its original bootstrap and build tools remain in an independent
build stage. Only the hash-verified derivative wheel and build receipt enter
the final image. Ubuntu pip/venv seed packages are not installed; the service
venv is created with `--without-pip` and bootstrapped from that wheel offline.
The wheel, receipt and `pip/NPA_VENDOR_REPAIR.json` remain available for audit.
This avoids retaining old seed bytes in earlier final-image layers, rather than
attempting to hide them with a later deletion. It does not prove an absence of
all vulnerabilities. Full exact-image vulnerability, secret, byte, license and capability
evidence remains required; this source document is not an acceptance receipt.
