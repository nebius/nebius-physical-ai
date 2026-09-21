# Evo and AprilTag complete-byte review

Both scans completed with all declared archive, ancestor-layer, graph, input-binding and helper-join checks passing. The scanner returned **1 / valid:false** for both images, and those results remain unchanged.

Each graph carries confidential operational identifiers in its immutable provenance; the private export wrapper also contains registry references. AprilTag additionally contains the exact public CMake Windows default packaging certificate. Upstream CMake can copy this certificate into Windows package artifacts when no certificate was supplied, so it is not classified as an unused test fixture. Its frozen PKCS12 finding remains failed.

Every native finding was reviewed against its exact record kind, content hash, size and complete finding population. Other matches bind to public cryptographic self-tests, SSH parser constants, C++ symbols, source/API definitions, deterministic numeric test vectors, or directly verified package checksums. No operator credential was identified in those reviewed matches. This does not mean these graphs are confidential-data-free or generally production-safe.

The machine-readable summary binds the actual source, image graph, new OCI export and retained private raw reports. It preserves the earlier private CPU evidence as a separate scope. No public image promotion is accepted or requested here, and no rebuild is required merely to create an unrequested public release. A future clean public candidate would require a real producer remedy and fresh applicable gates; editing attestations or dropping them is not a valid repair.

The new exports preserve all 16 unique graph blobs, 12 ordered layer references, and the attestation. Original mixed-format export compatibility metadata was accounted for separately; this report does not relabel those original wrapper bytes as scanned. Scanned record bytes include stored and decoded representations and metadata, so they are not a count of unique image bytes. Embedded archives are inspected as stored, not recursively unpacked by this complete-byte core.

[Exact CMake certificate source](https://github.com/Kitware/CMake/blob/v3.22.1/Templates/Windows/Windows_TemporaryKey.pfx) and [its Windows generator behavior](https://github.com/Kitware/CMake/blob/v3.22.1/Source/cmVisualStudio10TargetGenerator.cxx#L4433) establish that finding’s narrow provenance. No raw private identifiers, credentials, paths, or archive contents are included in this proof summary.
