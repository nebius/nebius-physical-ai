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
[CVE-2026-84782](https://ubuntu.com/security/CVE-2026-84782). Pip is hash-locked,
but a version pin is not proof that its vendored dependencies are vulnerability
free. Full exact-image vulnerability, secret, byte, license and capability
evidence remains required; this source document is not an acceptance receipt.
