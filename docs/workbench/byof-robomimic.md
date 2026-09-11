# robomimic neutral BYOF candidate

This is a quarantined Phase A packaging candidate, not a published image or a
live acceptance record. The intended image bakes immutable robomimic source and
non-CUDA low-dimensional dependencies on a neutral Python base. CUDA/PyTorch is
a separate, externally prepared runtime boundary. The planned workflow is
[`workflows/testing/byof-robomimic.yaml`](../../workflows/testing/byof-robomimic.yaml).

No image has been built. No source, dataset, CUDA runtime, or cache payload was
fetched while implementing this candidate. There is no accepted image digest,
anonymous pull proof, runtime-use approval, or B200 result.

## Six independent boundaries

| Boundary | Phase A contract | Status and gate |
| --- | --- | --- |
| Source | Bake only `ARISE-Initiative/robomimic@d309eaecc18acf4152a830a895a6984b8ac71b05`. Its MIT `LICENSE` SHA-256 is `7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556`. | Intended and statically checked; source bytes were not fetched or built in Phase A. A future build must record the observed commit and tree hash. |
| Baked runtime | Digest-pinned `python:3.11.16-slim-bookworm` plus exactly 48 hash-locked non-CUDA Python distributions. | Candidate only. It must contain no torch, torchvision, Triton, NVIDIA distribution, CUDA, cuDNN, or NCCL payload. Base and dependency licenses remain subject to built-byte review; PyTorch source licensing is not closure for wheel/base binary dependencies. |
| Weights | No pretrained weights are required or allowed in the image. | The four-step smoke produces its own run-scoped checkpoint only after authorization. Scanner rules reject common weight/checkpoint paths in every image layer. |
| Data and assets | Official Lift proficient-human low-dimensional HDF5 at `robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203`, path `v1.5/lift/ph/low_dim_v15.hdf5`. | Runtime fetch only after all gates. Accept only SHA-256 `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` and 21,084,088 bytes. Delete a mismatching partial before opening it. No simulator assets or rendering. |
| Runtime cache | A pre-populated operator volume mounted read-only at `/opt/npa-runtime/robomimic`. | The image cannot populate it. The verifier requires the exact lock, package map, ABI, source revision, complete regular-file/symlink inventory, hashes, sizes, and executable interpreter. Missing, corrupt, extra, escaping, or mismatched objects refuse with exit 78 without mutation. |
| Outputs | `/workspace/byof-runs/<run-id>` is separate from the input emptyDir and runtime PVC. | A later authorized run may write the checkpoint, config, logs, summary, and `robomimic-smoke.json` to its run-owned output prefix. Image and cache scans must prove output absence. Output rights remain a separate operator responsibility. |

Runtime fetching changes delivery, not permission. A credential, environment
flag, private registry, click-through proxy, or pre-populated cache is not
evidence that restricted bytes may be used, redistributed, or offered as a
service. Phase A deliberately adds no local consent proxy.

## Neutral candidate

The Dockerfile intends to use the exact parent index digest
`sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84`.
An authorized build must re-resolve its Linux/amd64 child and record the result;
historical resolution metadata is not byte-portable evidence.

`baked-requirements.lock` contains 48 exact wheel hashes and intentionally omits
torch, torchvision, Triton, and every `nvidia-*` distribution. The Dockerfile
installs the wheels with `--only-binary=:all: --no-deps --require-hashes`,
retains the immutable source without `.git`, removes SSH host keys, runs as
`ubuntu`, starts no SSH daemon by default, and leaves the runtime/input/output
mount points empty. Passwordless sudo exists solely for the repository's
SkyPilot Kubernetes bootstrap contract.

`runtime-requirements.lock` records the compatibility target used by the
deferred gate. It is not a downloader and its version list is not an artifact
hash closure. The external runtime inventory must supply that closure for every
installed file before consumption. `runtime_bootstrap.sh` exposes only
`verify`, `exec`, and `assert-refusal`; it has no ensure, fetch, install, sync,
warm, or network path.

`scan_image_robomimic_payload.py` is designed to inspect every layer and image history entry
plus OCI config, so deleting a prohibited object in a later layer cannot conceal
it. It rejects CUDA/PyTorch/NVIDIA payloads, tools and headers, HDF5 data,
weights, populated caches, credentials, run outputs, CUDA/vendor base history,
build-time runtime or dataset population, and invented acceptance variables.
Unreadable or unsupported images fail closed. This describes a future gate; it
is not a claim that built bytes were scanned in Phase A.

## CUDA and cuDNN decision

The dependent capability remains private and deferred. An authorized
Nebius/operator representative must record the exact CUDA/cuDNN distribution,
use, and service-rights decision. When authoritative NVIDIA interpretation is
needed, use NVIDIA's official license contact. Only then may the manager issue a
separate private staging/runtime transaction authorization.

The preferred sequence remains:

1. Independently redistributable exact CUDA closure, if component-by-component
   provenance, licenses, files, and service rights close.
2. Vendor-direct pinned operator-side installation under the documented
   acceptance/entitlement boundary, materialized outside this image.
3. Operator-private build-your-own image or runtime volume under the operator's
   controlled license and security process.

The first option is not currently closed. CUDA and cuDNN licenses identify some
redistributable runtime components, but that does not prove every byte in the
PyTorch/NVIDIA wheels is redistributable, excludes developer/static/header
payloads, satisfies notices/pass-through duties, or authorizes anonymous
distribution and downstream service use. The runner therefore retains an
unconditional Phase A refusal even when ordinary target selectors are present.

Official sources for the later human/vendor decision include the NVIDIA CUDA
Toolkit EULA, cuDNN Software License Agreement, CUDA container license, PyTorch
2.7.1 license/release material and CUDA 12.8 wheel index, and the official
Python image recipe/license. Review current authoritative copies immediately
before any transaction; repository prose is not the decision.

## Deferred exact-digest hard gate

After legal and transaction authorization, a future manager-approved stage must:

1. Build the reviewed neutral image under a full repository SHA without
   downloading runtime/data payloads into a layer.
2. Scan every built layer and OCI history, inventory all packages/licenses,
   scan vulnerabilities/secrets/malware, generate and verify SBOM/provenance,
   and prove the cache/input/output roots are empty.
3. Push only to authorized private staging, re-pull by exact digest, and repeat
   byte/security verification.
4. Prepare the CUDA runtime independently, record every file and symlink, hash
   its inventory, mount it read-only, and prove missing/corrupt/extra inventory
   refusal independently.
5. Bind the workload to exactly one STRICT-reserved B200; create a run-owned
   service account with only `get pods`; verify the executing pod's exact image
   digest, service account, one B200 model, and `sm_100` architecture before the
   dataset fetch.
6. Fetch and hash the official HDF5, produce nonempty disjoint train/held-out
   masks, run upstream `robomimic/scripts/train.py` for exactly four serialized
   Adam optimizer steps and two validation forward steps, and require finite
   train/validation loss.
7. Save and hash the genuine epoch checkpoint, reload it through
   `policy_from_checkpoint`, and infer exactly one held-out action of shape
   `[7]` whose values are finite and within `[-1, 1]`.
8. Write `$NPA_SMOKE_OUTPUT_DIR/robomimic-smoke.json` with the solution,
   capability list, immutable source/data/runtime identities, trajectory/sample
   and split proof, optimizer/loss proof, checkpoint/reload/action proof,
   observed B200 and pod digest, and exit status. Upload no HDF5 input.
9. Run a distinct read-only verifier against the exact private digest and
   evidence, then clean run-owned RBAC/resources with ownership preconditions.

Only after private acceptance may a separate authorized publication transaction
copy digest-identical neutral bytes, repeat all scans, and prove a clean
anonymous pull from an empty Docker configuration. Add a public catalog row
only after that proof; the current candidate remains quarantined.

Image-policy sweeps, simulator rollouts, rendering, and the full algorithm
matrix remain deferred.

## Local Phase A checks

Use the repository Python 3.12 interpreter:

```bash
npa/.venv/bin/python --version
npa/.venv/bin/python -m npa.cli.main workbench workflow validate-spec workflows/testing/byof-robomimic.yaml --json
npa/.venv/bin/python -m npa.cli.main workbench workflow plan-spec workflows/testing/byof-robomimic.yaml --run-id robomimic-plan-local --json
npa/.venv/bin/python -m pytest -q npa/tests/docker/test_robomimic_image_contract.py npa/tests/docker/test_robomimic_runtime_bootstrap.py npa/tests/docker/test_robomimic_image_payload_scan.py npa/tests/workflows/test_byof_robomimic.py
```

These local checks validate source contracts and refusal behavior. They do not
build an image, fetch a payload, use CUDA, submit a workflow, or claim public or
live acceptance.
