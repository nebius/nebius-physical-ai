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
| Demonstration | `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`, `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5` | LIBERO's pinned README declares datasets CC BY 4.0. Runtime fetch only; never baked into the image. Expected size `508779600`, SHA-256 `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`. |
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
`/workspace/libero-cache/$NPA_BYOF_RUN_ID/<dataset-revision>/`; it is outside
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

Use a unique run id and S3 prefix. Do not reuse a tag: after push, run only the
resolved immutable digest. The workload itself queries its Kubernetes Pod and
requires the matching `status.containerStatuses[].imageID`; merely echoing the
requested image is not accepted as runtime evidence.

## Acceptance artifact

Success requires `$NPA_SMOKE_OUTPUT_DIR/libero-smoke.json` with all of the
following evidence:

- solution-specific capabilities, the observed source checkout and build-command
  SHA-256, a same-layer source-prune/Git-object-removal receipt, exact data/task
  identities, and the observed dataset SHA-256;
- the real inventory of 50 demonstrations and 5,068 action samples;
- SHA-256 proofs for disjoint train and held-out trajectory-id sets;
- eight observed optimizer steps, a changed trainable parameter, and finite
  training losses through upstream LIBERO policy code;
- a checkpoint SHA-256, strict reload, held-out negative log-likelihood, and
  finite reloaded action shape/dtype/value-range proof plus a SHA-256 over the
  actual prediction tensor bytes;
- exactly one observed B200, compute capability `10.0` / `sm_100`, compiled
  CUDA and PyTorch architecture checks, and the pod-observed immutable image
  digest;
- `status: passed` and `exit_status: 0`.

The Python smoke writes a failed artifact before re-raising when any gate
fails. Imports, BDDL parsing, dataset inspection, or zero optimizer steps do not
qualify. Treat missing output as a failed run and inspect the uploaded stderr,
Pod status/events, and immutable image pullability before deciding whether to
resume or rebuild.

Rendered closed-loop success sweeps, all 130 tasks, comparisons among lifelong
algorithms, and physical-robot operation are deliberately deferred.
