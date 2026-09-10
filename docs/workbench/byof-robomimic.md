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

## Immutable inputs and licensing

| Input | Immutable identity | Packaging boundary |
| --- | --- | --- |
| Source | `ARISE-Initiative/robomimic@d309eaecc18acf4152a830a895a6984b8ac71b05` | MIT source is cloned into the private BYOF image. The exact commit's `LICENSE` was checked before build preparation. |
| Dataset | `robomimic/robomimic_datasets@74fa018461f479cd9fd15b924a16103012096203`, `v1.5/lift/ph/low_dim_v15.hdf5` | The dataset card declares MIT. Bytes are fetched only by the running operator workload and must match SHA-256 `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` and 21,084,088 bytes before HDF5 is opened. Dataset bytes are never baked into the image. |
| Base runtime | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2` | Operator-private derivative only. PyTorch is BSD-3-Clause; CUDA/cuDNN components remain governed by their upstream NVIDIA terms. This candidate is not selected for public NPA publication. |

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

The public Lift dataset needs no model entitlement or Hugging Face token. The
CUDA/cuDNN-derived base does require the operator to review and explicitly
accept the applicable NVIDIA Deep Learning Container terms for this private
build and one-run validation scope. Registry credentials or a prior image pull
do not constitute that acceptance. Do not pull, build, or submit until the
manager records the operator's acceptance.

## Hard gate

The workload fails before writing its final artifact unless all of these facts
are true:

- the executing image is an immutable post-push `@sha256:` reference, the
  built source metadata contains the exact commit, and the build metadata and
  48-artifact dependency lock match their exact declared hashes;
- the Kubernetes API reports the same immutable digest in the executing Pod's
  `status.containerStatuses[].imageID`; the configured image reference alone is
  not accepted as observation;
- PyTorch and `nvidia-smi` independently expose exactly one B200 with compute
  capability 10.0 (`sm_100`);
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

Before submission, replace `config.bucket` only in a run-local copy, select the
manager-assigned private registry and Kubernetes context, and verify that the
capacity policy is STRICT with exactly `B200:1`. Record explicit operator
acceptance of the NVIDIA terms before pulling the base. Create the temporary
`npa-robomimic-observer` ServiceAccount, Role, and RoleBinding in the assigned
namespace immediately before launch. The Role grants only `get` on `pods`; this
lets the workload read its own Pod status using its mounted service-account
token. Prove that permission with `kubectl auth can-i`, and remove all three
objects after SkyPilot cleanup.

The exact temporary RBAC shape is below. Kubernetes RBAC cannot restrict a
dynamic Pod name before SkyPilot creates it, so this Role can get any Pod object
in the assigned namespace; it cannot list or watch Pods, read logs, or access
Secrets. Use a dedicated validation namespace and delete this binding
immediately after the single run.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: npa-robomimic-observer
  namespace: <assigned-namespace>
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: npa-robomimic-observer
  namespace: <assigned-namespace>
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: npa-robomimic-observer
  namespace: <assigned-namespace>
subjects:
  - kind: ServiceAccount
    name: npa-robomimic-observer
    namespace: <assigned-namespace>
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: npa-robomimic-observer
```

Build and push through the BYOF runner so the mutable build tag is resolved to a
digest before scheduling. The dedicated live E2E requires both
`NPA_BYOF_LIVE_GPU=1` and `BYOF_ROBOMIMIC_LIVE=1`; it is valid only in that
assigned context. The runner cancels and cleans up its SkyPilot workload after
the artifact is uploaded. Preserve shared infrastructure.

## Scope and deferred work

This artifact proves only a short offline BC operational path on the official
Lift PH low-dimensional dataset. It is not convergence, task-success, sim-to-real,
or physical-robot evidence. Image-policy sweeps, simulator rollouts, and the full
robomimic algorithm matrix are intentionally deferred.
