# robomimic BYOF candidate

This candidate is the smallest end-to-end robomimic proof that exercises real
learning rather than import or dataset inspection. On one B200, it runs the
upstream BC training entrypoint for four Adam steps over the official Lift PH
low-dimensional demonstrations, evaluates two forward steps on a disjoint
held-out split, saves and reloads the resulting checkpoint, and infers an action
from a held-out trajectory. It does not start robosuite or any renderer.

The executable workflow is
[`workflows/testing/byof-robomimic.yaml`](../../workflows/testing/byof-robomimic.yaml).
Its separate readiness record reports what has actually been checked; schema
validation and planning do not imply that a B200 result exists.
Normal `workflow submit` execution is intentionally refused before preflight:
the checked-in spec is planned there, while the dedicated robomimic live gate
consumes its immutable configuration and owns the build, RBAC, and cleanup
lifecycle. This prevents a direct toolRef submission from bypassing the
manager-context refusal.

## Immutable inputs, outputs, and licensing

| Boundary | Immutable identity | Delivery and permission boundary |
| --- | --- | --- |
| Source | `ARISE-Initiative/robomimic@d309eaecc18acf4152a830a895a6984b8ac71b05` | MIT source is cloned into the private BYOF image. The exact commit's `LICENSE` was checked before build preparation and has SHA-256 `7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556`. |
| Baked runtime | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2` | Operator-private derivative only. PyTorch is BSD-3-Clause. CUDA/cuDNN bytes are expected in this base and remain governed by their upstream NVIDIA terms; a private registry is only a delivery destination and supplies no permission to build, use, or redistribute them. This candidate is not selected for public NPA publication. |
| Weights | None supplied | No pretrained weights are downloaded or baked. The trained BC checkpoint is a run output, not an input weight. |
| Data and assets | `robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203`, `v1.5/lift/ph/low_dim_v15.hdf5` | The dataset card declares MIT. Bytes are fetched only by the authorized running workload and must match SHA-256 `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` and 21,084,088 bytes before HDF5 is opened. Dataset bytes and simulator assets are never baked into the image. |
| Runtime cache | Run-local only | Builds use `--no-cache-dir`. The downloaded HDF5 is kept under a separate ephemeral input directory and is never traversed by the output uploader. Training logs and the checkpoint live under the run output directory. No credential, shared cache, downloaded dataset, or generated checkpoint may survive in image layers. |
| Outputs | Run-scoped checkpoint, config, logs, BYOF summary, provenance, and `robomimic-smoke.json` | These are generated from the authorized run and uploaded only to its manager-issued private S3 prefix. Their production or storage does not derive permission from a runtime fetch, credential, registry, or input license; the operator remains responsible for applicable use and output restrictions. |

The low-dimensional dependency closure is resolved for CPython 3.11 on Linux
x86-64, pinned to 48 exact package versions, and bound to one reviewed wheel
hash per package. The build installs that complete closure with
`--only-binary=:all: --no-deps --require-hashes`, then installs the immutable
robomimic source with `--no-deps`. PyTorch 2.7.1 and TorchVision 0.22.1 come from
the digest-pinned base and are asserted during the build. Two
upstream-declared packages are deliberately excluded from this capability
closure: `egl_probe` is only an EGL device probe for simulator rendering, and
`imageio-ffmpeg` is only needed for video and adds a separately licensed static
FFmpeg wheel payload. This headless low-dimensional gate uses neither. The final
artifact records the lock SHA-256 and 48-artifact count, and the smoke binds the
BYOF build metadata to the exact declared build-command SHA-256. A release
decision still requires scanning the built image bytes and reviewing the
installed inventory.

### Runtime delivery decision

A neutral Python bootstrap plus an immutable runtime fetch was reassessed.
PyTorch publishes a 2.7.1 CUDA 12.8 wheel channel, so that delivery is
technically possible. It would still download CUDA/cuDNN-governed bytes and
would not establish authority to use them. NVIDIA's current CUDA and cuDNN
agreements require an authorized user and acceptance before download, install,
or use; neither documents a machine-readable entitlement check that NPA can
enforce. Moving those bytes from build time to Pod startup therefore changes
delivery, not permission, and adds a large first-start dependency without
removing the pending operator decision.

The candidate consequently retains the smaller digest-pinned,
operator-private build path. That path may execute only after the operator has
made the applicable runtime-use decision and the manager has published this
child's exact private registry and Kubernetes context. A credential or private
registry is not evidence of that decision. No workflow flag or locally invented
consent variable substitutes for it.

The public Lift dataset needs no model entitlement or Hugging Face token. The
CUDA/cuDNN-derived base remains subject to the current
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) and
[NVIDIA cuDNN Software License Agreement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html),
including their use and redistribution conditions. Neither agreement exposes a
vendor entitlement probe or documented environment-variable acceptance
mechanism, so NPA does not invent a local consent proxy. An authorized operator
must review and accept those terms under its own authority before the private
build or run; registry credentials and a local boolean cannot prove that legal
decision. This candidate is build-your-own and operator-private, and NPA does
not publish its CUDA/cuDNN-derived image.

The source and dataset are public and anonymously readable, so there is no
source, dataset, or model-entitlement credential refusal to fabricate. The
production BYOF runner refuses this registered robomimic request before base
resolution or Docker build unless every manager-issued selector is present:
project, private registry, factual private-visibility record, kubeconfig file,
Kubernetes context, non-default namespace, output bucket, STRICT capacity, and
the two dedicated live-suite selectors. The registry is also checked against
NPA's centralized public-registry policy. The visibility record describes a
registry property; it is not CUDA/cuDNN consent or permission. In the current
child state, authorization is false and no robomimic context is published, so
no build, runtime download, RBAC mutation, or GPU submission is permitted.

The runner also binds the fresh build image to the exact run-scoped repository
and tag under that private registry and binds `--output-root` to
`s3://<manager-bucket>/oss-solutions/robomimic`. Known public-registry hosts are
rejected even when written with a standard port. The ordinary workflow-submit
path identifies this checked-in candidate by its immutable workflow name, so
overriding mutable config values cannot bypass the plan-only boundary.
The local `run-spec --execute` surface enforces the same boundary. The
robomimic runner forbids the generic `prebuilt` profile, `--skip-push`,
`--skip-run`, and `--no-cleanup`. `--skip-build` is accepted only for the exact
private `npa-byof@sha256:...` digest published by the manager after its separate
build and byte scan; the CLI image must equal that selector. This operational
identity is not CUDA/cuDNN permission. Both fresh-build and accepted-digest
modes require the exact source revision, CUDA base digest, build command, smoke
command, capability, artifact, and semantic contents of the run-local attested
profile before any runtime pull.
Run-scoped observer objects carry a per-invocation ownership token; cleanup
verifies that token and uses the observed object UID and resource-version as
server-side deletion preconditions. It checks every object for absence even
when another cleanup operation fails. Before the workload starts, a
SelfSubjectRulesReview must show no namespaced resource authority beyond Pod
`get` (apart from Kubernetes' standard self-review resources), and the selected
kubeconfig context must resolve to the manager-issued namespace. The workload
then observes its own Pod and requires both that namespace and the exact
run-owned ServiceAccount before training.

After authorization, qualification must scan the exact pushed image digest,
including every layer and image history entry. It must prove the absence of the
Lift HDF5 payload, pretrained weights, generated checkpoints, run outputs,
runtime caches, and credentials, and it must inventory installed licenses.
CUDA/cuDNN runtime bytes are expected in this private image and are assessed at
their separate baked-runtime boundary. No image has been built in the current
unauthorized context, so this document does not claim that byte-absence gate
has passed.

## Hard gate

The workload fails before writing its final artifact unless all of these facts
are true:

- the executing image is an immutable post-push `@sha256:` reference, the
  built source metadata contains the exact commit, and the build metadata and
  48-artifact dependency lock match their exact declared hashes;
- the Kubernetes API reports the same immutable digest in the executing Pod's
  `status.containerStatuses[].imageID`; the configured image reference alone is
  not accepted as observation;
- the operator-provided target carries the explicit STRICT reserved-capacity
  attestation, and PyTorch and `nvidia-smi` independently expose exactly one
  B200 with compute capability 10.0 (`sm_100`);
- the downloaded dataset matches the immutable file hash and contains the
  official 200 trajectories plus a nonzero sample inventory;
- upstream `robomimic/scripts/split_train_val.py` produces nonempty,
  exhaustive, disjoint train and validation masks;
- upstream `robomimic/scripts/train.py` finishes one BC epoch, emits finite
  train and validation losses, and the saved Adam state proves exactly four
  optimizer steps;
- `policy_from_checkpoint` reloads that checkpoint and produces one finite
  seven-dimensional held-out action within the policy's `[-1, 1]` action range.

The exact run-derived output is
`$NPA_SMOKE_OUTPUT_DIR/robomimic-smoke.json`. The B200 profile uploads this file,
the BYOF summary, logs, config, checkpoint, and provenance metadata to the fresh
run prefix. A zero-step configuration, import-only check, overlapping split,
mutable image, wrong GPU, bad hash, missing artifact, or failed upload is a hard
failure.

Focused local workflow and contract evidence was produced with the repository
`npa/.venv` interpreter reporting CPython 3.12.14. Draft-PR CI also selects
Python 3.12. Evidence from unrelated solution runs is not relabeled as this
environment.

## Operator sequence

Use only a manager-issued runtime context. In particular, do not infer a cloud
project, cluster, registry, bucket, network, or capacity binding from local
defaults.

```bash
npa workbench health preflight --checks nebius
npa workbench workflow validate-spec workflows/testing/byof-robomimic.yaml --json
npa workbench workflow plan-spec workflows/testing/byof-robomimic.yaml \
  --run-id <fresh-run-id> --json
```

Before the dedicated gate is selected, replace `config.bucket` only in a
run-local copy, select the manager-assigned private registry and Kubernetes
context, and verify that the capacity policy is STRICT with exactly `B200:1`.
The operator must confirm its authority to use the governed CUDA/cuDNN runtime
before pulling or building; that decision stays outside workflow arguments and
environment-variable self-attestation. Publish the registry's factual
visibility as `private` in the owner-only runtime context and export the two
live-suite selectors plus `NPA_E2E_MK8S_RESERVED_CAPACITY=1` only in the reviewed
run environment.

The dedicated gate derives a collision-resistant ServiceAccount, Role, and
RoleBinding name from the run ID, creates all three in the assigned namespace,
and annotates them with a one-way run-ID digest. Before invoking the BYOF runner it
uses `kubectl auth can-i` to prove that the identity can only `get` Pods: Pod
list/watch, Pod logs, and Secret reads must all be denied. Kubernetes cannot
restrict `get` to the dynamic Pod name before SkyPilot creates it, so use the
dedicated validation namespace. A `finally` cleanup deletes all three objects
after SkyPilot cleanup or failure and verifies their absence; failure to create,
verify, delete, or prove absence fails the live gate.

Build and push through the BYOF runner so the mutable build tag is resolved to a
digest before scheduling. The dedicated live E2E requires
`NPA_BYOF_LIVE_GPU=1`, `NPA_BYOF_ROBOMIMIC_LIVE_B200=1`, the STRICT-capacity
attestation, and explicit manager-issued project, private registry plus factual
`NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY=private`, kubeconfig, context,
non-default namespace, and output-bucket selectors. The harness checks
`NPA_E2E_MK8S_RESERVED_CAPACITY=1`, then writes the corresponding boolean only
into its run-local profile copy; the checked-in profile deliberately carries no
attestation value or usable observer identity. The run is valid only in that
assigned context and after the operator has made the NVIDIA-terms decision
described above. The runner cancels and cleans up its SkyPilot workload after
the artifact is uploaded, then the harness removes the temporary RBAC. Preserve
shared infrastructure.

## Scope and deferred work

This artifact proves only a short offline BC operational path on the official
Lift PH low-dimensional dataset. It is not convergence, task-success, sim-to-real,
or physical-robot evidence. Image-policy sweeps, simulator rollouts, and the full
robomimic algorithm matrix are intentionally deferred.
