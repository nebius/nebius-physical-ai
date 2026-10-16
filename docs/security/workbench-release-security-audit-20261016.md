# Workbench release security audit — October 16, 2026

**The zero-vulnerability objective is not achieved.** Every one of the 27
accepted public release digests has findings in the current vulnerability scan.
Source repairs do not change those immutable image bytes.

The [machine-readable inventory](workbench-release-security-audit-20261016.json)
binds each result to the accepted digest from
`npa/src/npa/deploy/public_release_manifest.json` at
`f331310cfb278294c864cb737c6a0c3205a00f1d`, Trivy 0.74.0, and its database
metadata. Full raw reports remain in access-controlled evidence. Scans used
the vulnerability scanner without severity or unfixed-vulnerability filtering.

These are **66,855 package/CVE occurrences**, including **248 critical** and
**6,172 high** occurrences; **20,149** have a reported fixed version. Repeated
matches across packages, targets, and images are counted separately. Version
matches require applicability review, including kernel-header findings and
packages with verified source backports. These counts do not establish that
each occurrence is exploitable.

| Accepted release | Critical | High | Total | Fixed version available |
| --- | ---: | ---: | ---: | ---: |
| alpamayo2-super | 5 | 213 | 5,441 | 1,992 |
| content-agents | 0 | 8 | 479 | 326 |
| cosmos | 5 | 54 | 1,491 | 638 |
| cosmos2-transfer | 2 | 79 | 1,014 | 345 |
| cosmos3-reason | 1 | 78 | 1,734 | 609 |
| cosmos3-serving | 12 | 589 | 4,905 | 708 |
| detection-training | 5 | 204 | 4,498 | 1,702 |
| diffusers | 23 | 359 | 1,324 | 128 |
| envgen | 8 | 345 | 5,856 | 1,514 |
| fiftyone | 4 | 271 | 1,019 | 98 |
| flex-pi | 7 | 358 | 5,032 | 1,659 |
| foxglove-embed | 0 | 44 | 143 | 142 |
| groot | 2 | 38 | 1,678 | 583 |
| lancedb | 21 | 374 | 5,829 | 3,033 |
| leisaac | 12 | 395 | 6,059 | 1,739 |
| lerobot-policy | 2 | 26 | 1,443 | 596 |
| lichtblick | 0 | 44 | 143 | 142 |
| lingbot-world | 23 | 359 | 1,324 | 128 |
| ltx2 | 23 | 354 | 1,316 | 123 |
| lyra2 | 5 | 154 | 4,565 | 1,100 |
| openarm | 7 | 352 | 5,955 | 1,635 |
| rerun-viewer | 10 | 341 | 1,191 | 276 |
| retargeting | 13 | 145 | 549 | 279 |
| sam2 | 23 | 361 | 1,326 | 130 |
| sim2real-control | 6 | 153 | 662 | 272 |
| sonic-mujoco | 6 | 115 | 555 | 124 |
| wan2-2 | 23 | 359 | 1,324 | 128 |

The requested scope covers every maintained Workbench runtime. This image
inventory covers accepted public releases. Other entries in the 63-entry
packaging contract, private/restricted images, and runtime-fetched payloads
still require their own byte-level coverage and compatible functional tests.
No zero claim applies to those unverified areas.

The repair candidate hardens authenticated LeIsaac/GR00T transport, YAML and
URDF parsing, immutable model downloads, and workflow token permissions. It
also rejects entity declarations in SONIC MuJoCo scenes and included robot XML
before simulator construction, while preserving ordinary scene loading. It
updates Isaac and Wan dependency closures and applies verified upstream NLTK
and constrained Setuptools fixes. Real CPU tests reproduce the Setuptools
manifest leak before the patch and deny it afterward, exercise NLTK sandbox
boundaries, encode/decode MoviePy output with patched Pillow, and validate
Wan's installed baked/hash-bound runtime dependency union.

Dependency scanning now includes the JSON package map and bootstrap installer
in the robomimic runtime lock. Its source candidate updates urllib3 and fsspec;
the remaining legacy Torch/CUDA packages still require remediation and a new
independently selected runtime inventory. Historical text-lock findings are
kept separate from the runtime that the bootstrap actually fetches.

Remaining work includes reconciliation of Isaac's exact vendor dependency
requirements with the security-upgraded OSS closure, dependency/API compatibility
for the legacy Torch/CUDA and vLLM families; exact rebuilt-image vulnerability, secret,
license, payload and complete-byte gates; compatible real-GPU workloads; and
digest-identical release promotion. A passing differential security gate or
unit suite does not replace those checks. No alert dismissal or scanner
suppression was used to reduce these counts.
