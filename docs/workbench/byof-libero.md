# LIBERO BYOF qualification

This candidate proves a narrow but real LIBERO behavior-cloning path on Nebius:
one official `libero_spatial` task, a trajectory-disjoint train/held-out split,
eight upstream `Sequential.observe` optimizer steps with `BCRNNPolicy`, an exact
checkpoint save/reload, and held-out loss plus finite action predictions. It
runs headlessly on exactly one STRICT-bound B200 and does not render or execute
a closed-loop policy.

The workflow is [`workflows/testing/byof-libero.yaml`](../../workflows/testing/byof-libero.yaml).
It remains a private BYOF candidate; there is no public `npa-libero` image.

## Immutable inputs and licensing

| Input | Immutable identity | Terms and packaging boundary |
|---|---|---|
| LIBERO source | `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01` | MIT. The private image retains the source, BDDL, and initial-state files but prunes `libero/libero/assets` and removes the Git object database in the source clone layer because this qualification does not render. |
| CUDA base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85` | NVIDIA Deep Learning Container License; pulling and using the image is NVIDIA's documented acceptance mechanism. Used only as the base of the operator-controlled private image. |
| SkyPilot bootstrap packages | Ubuntu packages resolved during the operator-private build and bound by the resulting image digest | Distribution packages are baked runtime, not runtime-fetched task data. The build verifies every required package capability before recording its bootstrap attestation; the resulting image remains private and receives a fresh byte/license scan. |
| Demonstration | `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`, `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5` | LIBERO's pinned README declares datasets CC BY 4.0. Runtime fetch only; never baked into the image. Expected size `508779600`, SHA-256 `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`. |
| Task-language model | `google-bert/bert-base-cased@cd5ef92a9fb2f889e972770a36d4ed042daf221e` | Apache-2.0. Runtime fetch only; the five files required by the pinned LIBERO `AutoTokenizer`/`AutoModel` path have independent size and SHA-256 pins. The smoke verifies pinned `libero/lifelong/utils.py` and uses BERT `pooler_output`, matching upstream task conditioning instead of a synthetic vector. |
| Task BDDL | `libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl` | Part of the pinned MIT checkout. SHA-256 `9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`. |
| Initial states | matching `.init` file in the pinned task folder | Part of the pinned MIT checkout. SHA-256 `c3a6a01fdc53ae1914fe24c8935088d723baee8f6ee3cd5f8d68e86aea3e2f1c`. |

The task definition names Google Scanned Objects (two bowls, a ramekin, and a
plate) and a HOPE cookies distractor. Their upstream publishers state CC BY
4.0 and CC BY-NC-SA 4.0 terms respectively. Those meshes and textures are not
needed for the stored-observation BC qualification and are excluded, together
with Git objects that could retain their bytes, in the same source clone layer;
rendered evaluation remains deferred. The BDDL and
initial-state files that bind the exact task remain in the pinned MIT checkout
and are hash-verified at runtime.

The CUDA base is governed by the
[NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf),
and NVIDIA's [official CUDA container page](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/cuda/-)
states that pulling and using the container accepts those terms. No separate
entitlement is required for this public base; the derived image remains in the
task-owned private registry.

The Hugging Face mirror card currently says `apache-2.0`, which conflicts with
the upstream LIBERO publisher's explicit CC BY 4.0 dataset declaration. The
workflow treats the upstream declaration as authoritative, preserves LIBERO
attribution in the report, pins the mirror commit and file hash, and documents
the mismatch rather than silently broadening redistribution rights.

The runtime cache is
`/workspace/libero-cache/$NPA_BYOF_RUN_ID/<dataset-and-model-revisions>/`; it
holds the hash-verified demonstration and BERT task-language files and is outside
`$NPA_SMOKE_OUTPUT_DIR` and is never uploaded. Only the report, checkpoint,
source/build receipts, and logs under `$NPA_SMOKE_OUTPUT_DIR` enter the unique
run-scoped S3 prefix. Image publication is private-registry-only.

## Before spending GPU time

Do not build, provision, or submit this qualification until the run manager has
assigned the project, cluster, namespace, private registry, bucket, and STRICT
B200 reservation. Then run the service preflight and the cheap workflow checks:

```bash
npa/.venv/bin/npa workbench health preflight --checks nebius,s3 --json
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/byof-libero.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/byof-libero.yaml \
  --run-id libero-qualification --json
```

The image must be built and pushed to the assigned private registry, resolved
to `repository@sha256:<64 hex>`, and supplied unchanged to the solution-smoke
runner. The checked-in B200 resource profile is
`byof-solution-smoke-libero-b200-gpu`; the runner's default cleanup must remain
enabled. Never substitute an on-demand GPU or a public GHCR destination for the
manager-owned runtime binding.

Build the image from the current BYOF generator. Its SkyPilot 0.12.2 bootstrap
contract installs and verifies the complete synchronous package capability set,
including `curl`, `wget`, and the Ubuntu `fuse3` provider for `fuse`. The
image-resident guard bypasses SkyPilot's redundant package-index and `fuse`
installation only after that complete set verifies. A missing package fails the
contract, and a bootstrap command that exceeds its deadline receives TERM and a
kill-after escalation for its process group plus an explicit failure marker.
An older private digest that predates this guard is not eligible for another
attempt; it must be rebuilt and byte/scanner qualified before use.

The workflow config is the source of truth for `repo_url`, `repo_ref`,
`base_image`, `source_prune_path`, `build_command`, and `smoke_command`. A live
operator passes those values to `npa/scripts/run_byof_repo.py` with:

```text
--workload solution-smoke
--source-prune-path libero/libero/assets
--solution-name libero
--capability-name libero_spatial_bc_rnn_train_reload_heldout
--smoke-artifact-name libero-smoke.json
--yaml npa/src/npa/workflows/byof/profiles/byof-solution-smoke-libero-b200-gpu.yaml
--wait-timeout -1
--cleanup
```

For LIBERO, `run_byof_repo.py` always adds `--no-direct-launch`: the real route
must submit through the managed scheduler, poll the nonempty scheduler job ID
returned by submission in the same isolated state, and treat verified absence
as a terminal failure instead of polling forever. Signals, polling failures, and
normal cleanup cancel and drain that exact scheduler ID before any run-cluster
teardown; cancellation or drain ambiguity preserves the clusters. The generic
direct-launch default is not an accepted LIBERO path, and selecting the LIBERO
profile or payload account without `--solution-name libero` refuses before any
launch preparation. The owner-only SkyPilot
config must contain exactly one nonempty `kubernetes.allowed_nodes.names` entry;
an absent, differently shaped, or multi-node allowlist refuses before submit.
Concrete node identity stays in owner-only runtime state and only its SHA-256 is
forwarded into the workload receipt. Supply that mode-0600 regular file through
host-only `NPA_LIBERO_SKYPILOT_CONFIG`; the workflow never renders its path or
contents.

Use a unique run id and S3 prefix. Do not reuse a tag: after push, run only the
resolved immutable digest. The workload itself queries its Kubernetes Pod and
requires the matching `status.containerStatuses[].imageID`; merely echoing the
requested image is not accepted as runtime evidence.

Immediately before a future submit, separately prove workload-network access to
the pinned LIBERO GitHub source and the exact Hugging Face demonstration URL.
Bootstrap-package readiness does not establish access to either official
runtime-fetch endpoint. Refuse submission if either endpoint or its immutable
identity cannot be verified.

That Pod query requires `get` access to its own Pod. The payload therefore uses
the deterministic `npa-byof-libero-payload` service account declared in the
LIBERO profile. Before launch, the run manager must create that account in the
isolated namespace and bind it to a Role with exactly `apiGroups: [""]`,
`resources: ["pods"]`, and `verbs: ["get"]`. No repository command creates
these objects. The owner-only manifest and post-create receipt must bind the
ServiceAccount, Role, and RoleBinding UIDs to the assigned namespace and cleanup
contract.

Set host-only `NPA_LIBERO_PAYLOAD_KUBECONFIG` to the reviewed mode-0600,
namespace-bound payload proof kubeconfig. Its selected context must be explicit
and distinct from the NPA/SkyPilot execution context, while their flattened
server and CA identity hashes must select the same cluster. The runner uses that
copy only to read the exact ServiceAccount, `pods/get` Role, and RoleBinding; it
refuses missing objects or rule/subject/ref drift. Every observed UID,
namespace, cluster, RBAC-specification, and allowed-node hash must equal its
`NPA_LIBERO_EXPECTED_*` value from the owner-only creation/reservation receipts;
the runner will not adopt matching-but-unreceipted objects. It forwards only
those hashes into the task. The workload then
reads its own Pod and bound service-account JWT, requiring the actual Pod
service account, service-account UID, Pod UID, namespace, and placed node to
match the preflight hashes. A rendered declaration alone is not evidence.

Keep this payload identity separate from the manager SkyPilot global config:
SkyPilot 0.12.2's controller continues to use its engine-required
`skypilot-service-account`. The preflight refuses a missing, `default`, or
SkyPilot engine account for the LIBERO payload, and also refuses a global config
that does not explicitly select the engine account for the controller. A
private kubeconfig, pull secret, or registry credential is not permission to
broaden the payload Role. Until the manager has created and receipted the exact
three-object payload RBAC contract, keep the live B200 claim deferred and do
not submit the workflow.

The generated build records the selected immutable base reference and digest in
`npa_build_metadata.json` and in the OCI base-name/base-digest config labels.
The smoke accepts those build-observed fields only when the reference is pinned
and matches the reviewed base. Before any future live use, the complete OCI
archive/config/layer and provenance scan must independently confirm the same
values; workload constants or Docker BuildKit history output are not base-image
evidence. Historical r15 bytes are bound to an older head and are not eligible
as repaired-head or live-qualified evidence.

## Acceptance artifact

Success requires `$NPA_SMOKE_OUTPUT_DIR/libero-smoke.json` with all of the
following evidence:

- solution-specific capabilities, the observed source checkout and build-command
  SHA-256, build-observed immutable base provenance, a same-layer
  source-prune/Git-object-removal receipt, exact data/task identities, and the
  observed dataset SHA-256;
- the pinned upstream LIBERO BERT-conditioning source, exact Apache-2.0
  tokenizer/model file identities, a finite 768-dimensional `pooler_output`,
  and proof that this runtime cache was not uploaded;
- the real inventory of 50 demonstrations and 5,068 action samples;
- SHA-256 proofs for disjoint train and held-out trajectory-id sets;
- eight observed optimizer steps, a changed trainable parameter, and finite
  training losses through upstream LIBERO policy code;
- a checkpoint SHA-256, strict reload, held-out negative log-likelihood, and
  finite reloaded action shape/dtype/value-range proof plus a SHA-256 over the
  actual prediction tensor bytes;
- exactly one observed B200, compute capability `10.0` / `sm_100`, compiled
  CUDA and PyTorch architecture checks, the pod-observed immutable image
  digest, exact allowed-node hash, and actual payload Pod/service-account/RBAC
  binding hashes separated from the controller identity;
- `status: passed` and `exit_status: 0`.

The Python smoke writes a failed artifact before re-raising when any gate
fails. Imports, BDDL parsing, dataset inspection, or zero optimizer steps do not
qualify. Treat missing output as a failed run and inspect the uploaded stderr,
Pod status/events, and immutable image pullability before deciding whether to
resume or rebuild.

Rendered closed-loop success sweeps, all 130 tasks, comparisons among lifelong
algorithms, and physical-robot operation are deliberately deferred.
