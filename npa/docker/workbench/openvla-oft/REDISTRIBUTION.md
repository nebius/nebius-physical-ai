# OpenVLA-OFT bootstrap distribution record

This proposed image is a neutral bootstrap and remains `unvalidated` in the
packaging contract: it is private/publication-quarantined until an exact built
image, SBOM, provenance, dependency inventory, and byte scan have been
reviewed. It contains the digest-pinned Python base, Debian build/runtime
prerequisites, and a minimal Apache-2.0 NPA artifact-control path with the
hash-pinned Boto3 S3 client closure. It does not contain OpenVLA-OFT source,
OpenVLA source or base weights, custom transformers, Torch/CUDA, OFT adapters,
LIBERO data/assets, credentials, cached downloads, or outputs.

After the unresolved direct-dependency terms are resolved, the operator-owned
writable cache may fetch upstream OFT source at
`e4287e94541f459edc4feabc4e181f537cd569a8` under its MIT license and LIBERO
source `8f1084e3132a39270c3a13ebe37270a43ece2a01` under its MIT license, then
resolve the base and adapter weights separately. Until an authoritative license
or permission covers the unpinned `moojink/dlimp_openvla` direct dependency, the
runtime fails before it fetches any of those upstream payloads. The OpenVLA
model card advertises MIT, while the current OpenVLA upstream README says its
Llama-2-derived pretrained models are subject to the Llama Community License.
That source-level qualification is not removed by this container. No image or
artifact produced here asserts redistribution rights for those runtime inputs.

No new EULA environment variable, acceptance checkbox, telemetry, or privacy
consent is defined by this image. A future upstream click-through applies only
to the concrete fetch that presents it and must use its documented mechanism.
