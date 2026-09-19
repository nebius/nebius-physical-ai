# robomimic neutral candidate redistribution boundary

This candidate contains the immutable MIT-licensed robomimic source, the neutral
Python base, and the exact 78-package Debian closure. No Python wheel closure is
baked into the public image: the historical `baked-requirements.lock` remains
source evidence only, while the complete public and restricted package map is
declared in `runtime-requirements.lock` and fetched into a customer-owned
runtime volume. The public image adds no solution/runtime Python wheel closure,
PyTorch, torchvision, Triton, NVIDIA CUDA, cuDNN, NCCL, model weights, dataset
bytes, populated runtime cache, credentials, or run output. Inherited Python
distributions and ancestor bytes belong to the pinned parent and remain outside
this absence claim until exact parent inspection is complete.
Its build path stages the exact Git tree and canonical archive plus every exact
Debian input outside the image, verifies them before use, and exposes those
payloads to the Dockerfile only through read-only build mounts. The Dockerfile
does not contact APT or Git and does not execute a Python wheel installer.

The source and Debian lock metadata are reproducibility inputs, not built-byte
or license acceptance. Debian package copyright notices must remain installed,
and all applicable source-conveyance duties must close before public
publication. The image is quarantined and must not be published merely because
its source is open source. Before any publication, an authorized transaction
must re-resolve the base manifest, build from the reviewed commit, inspect every
resulting layer, complete license/security/SBOM/provenance checks, validate the
private digest, and prove anonymous pull of that identical digest. None of
those claims is established by the checked-in candidate.

The CUDA-capable training runtime is a separate boundary. Before an
operator-supplied, pre-populated, read-only runtime volume is accessed, the
customer must explicitly accept the exact official CUDA/cuDNN terms for its
bounded run. The owner-only record binds the customer, run, runtime lock,
independently selected inventory digest, terms, and
expiry. A manager signature, runtime fetch, credential, private registry, image
selector, or environment flag does not grant that permission. The acceptance
action requires the exact digest emitted by the preceding notice action.

The runtime volume's own inventory is not self-attestation. The customer must
select its exact inventory SHA-256 independently, and the bootstrap must match
that hash and the customer entitlement. When explicitly enabled with
`NPA_ROBOMIMIC_RUNTIME_FETCH=1`, the bootstrap fetches the inventory's exact
wheel URLs from the pinned official hosts. Public PyPI/PyTorch hosts are always
anonymous; a bearer credential is sent only to its explicitly bound Hugging
Face or NVIDIA NGC origin. The customer's `HF_TOKEN` or `NGC_API_KEY` is
revalidated immediately before every request, installation, and publication,
and is never logged. Every wheel hash and size is checked, installation uses no
index or dependency resolution, and every installed `RECORD` is validated
before publishing the site-packages tree. `NPA_ROBOMIMIC_CUSTOMER_DENYLIST`
is an explicit customer runtime input; unset means the built-in safe path
denylist is used. The fetch phase is separate from the final read-only snapshot
and never places the credential or fetched bytes in the image.
Execution copies only declared
objects into a private staging tree, verifies the copy again, removes write
bits, and atomically publishes that run-local snapshot before invoking its
interpreter. Because the runtime UID owns this copy, absent write bits are
hygiene rather than an enforced read-only boundary. The independently selected
inventory hash and observed read-only source mount supply byte identity; the
private snapshot supplies point-in-time race resistance only. The customer
record authorizes only the bound use under the listed terms; none of these
mechanisms grants redistribution, publication, or broader service rights.
