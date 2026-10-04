# MolmoAct2 LIBERO on Jetson AGX Thor

[`molmoact2-jetson-thor-edge.yaml`](../../workflows/testing/molmoact2-jetson-thor-edge.yaml)
is an edge-deployment evidence workflow for the Agents2Agents MolmoAct2 LIBERO
TensorRT bundle. It is **not live-qualified**: the assigned NPA VDI is an
x86_64 host without Jetson or NVIDIA GPU runtime, so it cannot load, benchmark,
or validate a Thor engine. An x86 Kubernetes result would not establish Thor
compatibility or latency and is deliberately rejected as such.

The cloud-side adapter, contracts, local artifact pipeline, and five-stage
workflow are implemented. Target-only evidence remains blocked until an
operator supplies the exact device described below. This page separates that
blocker from the work that has been verified locally.

## Exact upstream identity and credit

| Component | Identity and credit | Use in this derivative |
| --- | --- | --- |
| Edge runtime | [Agents2AgentsAI/vla-edge](https://github.com/Agents2AgentsAI/vla-edge) at `747fd96aaf1a386d3873cf6c637e7a51141b66fd`, copyright Agents2AgentsAI contributors | Its `vla_edge.backends.tensorrt.artifacts.check_compatible`, `verify_checksums`, `vla-edge-serve`, and `ActClient` remain the native execution surfaces. |
| Engine bundle | [agents2agents/MolmoAct2-LIBERO-Jetson-Thor](https://huggingface.co/agents2agents/MolmoAct2-LIBERO-Jetson-Thor) at `7d2de215036b802c247e3c3c4db7316c99d6adfe`, published by Agents2Agents | Runtime-fetched on the operator's Thor. Its `MANIFEST.json`, checksums, host metadata, and notice are retained in the target qualification evidence. |
| Base checkpoint | [allenai/MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO) at `0d24a92bd1faf321ef497c3bbd5681af97c65aa2`, Allen Institute for AI / MolmoAct2 authors | The bundle's `host/libero/host.json` identifies this lineage. Preserve the upstream [paper citation](https://arxiv.org/abs/2605.02881) and [BibTeX](https://arxiv.org/bibtex/2605.02881); NPA does not relabel the research as original work. |

NPA's modification is only the run-scoped observation/qualification/trace
adapter at `npa.workflows.molmoact2_jetson_thor`, its workflow, tests, and this
documentation. It does not modify `vla-edge`, model parameters, TensorRT plans,
or LIBERO data.

The upstream [vla-edge license](https://github.com/Agents2AgentsAI/vla-edge/blob/main/LICENSE)
is Apache-2.0. Its [NOTICE](https://github.com/Agents2AgentsAI/vla-edge/blob/main/NOTICE)
and [third-party notices](https://github.com/Agents2AgentsAI/vla-edge/blob/main/THIRD_PARTY_NOTICES.md)
identify adapted GELLO portions as MIT and additional dependency notices. The
bundle's checked-in `LICENSE` is Apache-2.0 and its `NOTICE` attributes the
Agents2Agents conversion and the AllenAI base checkpoint; those notices travel
with an operator-fetched bundle. The source repository, model bundle, and base
checkpoint are separate upstream artifacts and retain their respective notices.

## Distribution and access boundary

| Material | Decision |
| --- | --- |
| NPA cloud evidence image | Private, unvalidated CPU-only derivative built from this adapter. It contains no bundle, checkpoint, plan, calibration capture, populated cache, JetPack, TensorRT, or `vla-edge`; it is not a target image and has no public tag or catalog release. Its exact digest and matching source SHA are supplied only at submit time. |
| `vla-edge` source | Apache-2.0, with its bundled notices preserved by the upstream installation. |
| TensorRT plans, compact host, plugins, and model bytes | Operator-owned runtime fetch from the pinned model revision on the Thor; never copied into an NPA image, layer, public bucket, or this repository. The source model card states Apache-2.0 for the bundle, but actual bytes must still be fetched and checksum-verified for the target run. |
| JetPack, TensorRT, and Jetson PyTorch runtime | Operator-installed NVIDIA product runtime on the target device; not packaged or redistributed by NPA. NVIDIA's [JetPack/TensorRT SDK terms](https://docs.nvidia.com/jetson/jetpack/eula/) govern that installation and the TensorRT SDK is licensed for applications on NVIDIA-GPU systems. The target installer is the only applicable acceptance mechanism; this onboarding adds no NPA EULA, environment acceptance flag, telemetry, or duplicate attestation. |
| LIBERO observations and action traces | Operator-provided run inputs/outputs. Do not publish captures, instructions, or traces without a separate data-distribution decision. |
| Derived metrics and RRD | Run-scoped evidence. The workflow records hashes and numerical values; it makes no upstream performance or physical-robot-success claim. |

Metadata for the exact model revision was readable during onboarding. The large
LFS payload was intentionally not downloaded on the non-Thor host, so payload
access and byte checks are **not** marked ready; a target run must obtain the
same revision through the operator's existing Hugging Face access path and let
the native checker read all declared bytes. Target execution also requires the
operator to have completed the existing NVIDIA installer acceptance if it is
present; its status is unknown on the unavailable device. Neither condition
blocks local cloud-contract work or justifies a new NPA consent mechanism.

## Target contract

The upstream bundle requires all of the following on the device that runs
`target-qualify` and `target-action-trace`:

- NVIDIA Jetson AGX Thor Developer Kit; `aarch64`; NVIDIA Thor with 20 SMs;
  `sm_110a`.
- JetPack R39 revision 2.1 and TensorRT `10.16.2.10`.
- The pinned `vla-edge` source/runtime and the pinned complete bundle.
- A matching Torch reference loaded from the pinned base-checkpoint revision,
  served separately with the same `molmoact2-libero` policy contract.

The adapter invokes upstream `check_compatible` and `verify_checksums` before
it accepts a target report. It also binds both `/act` servers to policy
`molmoact2-libero`, model family `molmoact2`, embodiment `libero`, camera names
`image`/`wrist_image`, and an eight-value state. Action width and horizon come
from the live server contract rather than being guessed.

## Connected workflow

The workflow has five substantive, serially connected leaf stages. All paths
are run-scoped S3 URIs derived from `bucket`, `prefix`, and `run.id`; no bucket,
credential, endpoint, device, or cluster identifier is committed.

| Stage | Actual operation | Consumes → emits | Current evidence |
| --- | --- | --- | --- |
| `prepare-observations` | Reads RGB arrays, state, instructions, episode IDs, seeds, and calibration/evaluation labels; validates dimensions, pixel dtype, finite state, splits, and emits canonical NPZ/manifest. | operator observations → prepared observations | locally exercised |
| `verify-target-engine` | Validates a real target report that records upstream checksum/compatibility success and binds its manifest/source/device identities to the prepared digest. | prepared observations + target qualification → verified qualification | locally exercised with a contract fixture; target execution blocked |
| `verify-target-action-trace` | Reads target-produced TensorRT and Torch action arrays plus service timing; rejects unbound, malformed, nonfinite, or wrong-backend evidence. | verified qualification + target trace → verified trace | locally exercised with a contract fixture; target execution blocked |
| `evaluate-target-action-trace` | Computes MAE, RMSE, maximum action error, and target/reference mean, p50, and p95 endpoint timings from the verified trace. | verified trace → metrics JSON | locally exercised |
| `visualize-target-evidence` | Emits a digest-bound Rerun `.rrd` with per-observation action error and both timing series. | verified trace + metrics → RRD + manifest | locally exercised and decoded |

`vla-edge` provides a policy service, not a native LIBERO closed-loop simulator
or task-success evaluator. Consequently, stage three is genuine target-device
VLA inference over evaluation observations, but it is labeled
`open_loop_observation_trace`; it cannot establish a closed-loop LIBERO success
rate or physical-robot success. Add an upstream-supported closed-loop harness
and its target evidence before making either claim.

## Target-to-cloud execution sequence

First create a real observation NPZ with `image` and `wrist_image` as `uint8`
`[N,H,W,3]`, a finite `state` `[N,8]`, and `instruction`, `episode_id`, `seed`,
and `split` arrays. Use one run ID and the same run-scoped input URI throughout.
On a Thor, install the pinned `vla-edge` revision using its documented Jetson
setup and fetch the model bundle at the exact revision into an operator-owned
directory. Do not use this x86 checkout as a substitute.

Stage the canonical preparation and then produce the two target artifacts:

```bash
npa/.venv/bin/python -m npa.workflows.molmoact2_jetson_thor prepare \
  --observations-uri "s3://<bucket>/<prefix>/input/observations.npz" \
  --prepared-uri "s3://<bucket>/<prefix>/prepare/observations.npz" \
  --manifest-uri "s3://<bucket>/<prefix>/prepare/manifest.json" \
  --run-id "<run-id>"

vla-edge-serve --policy molmoact2-libero --backend tensorrt \
  --engine-dir "<complete-pinned-bundle>" --port 8202
vla-edge-serve --policy molmoact2-libero --backend torch \
  --checkpoint "<pinned-local-MolmoAct2-LIBERO-snapshot>" --port 8203

npa/.venv/bin/python -m npa.workflows.molmoact2_jetson_thor target-qualify \
  --engine-dir "<complete-pinned-bundle>" \
  --prepared-uri "s3://<bucket>/<prefix>/prepare/observations.npz" \
  --qualification-uri "s3://<bucket>/<prefix>/target/qualification.json" \
  --run-id "<run-id>"

npa/.venv/bin/python -m npa.workflows.molmoact2_jetson_thor target-action-trace \
  --prepared-uri "s3://<bucket>/<prefix>/prepare/observations.npz" \
  --qualification-uri "s3://<bucket>/<prefix>/target/qualification.json" \
  --target-endpoint "http://127.0.0.1:8202" \
  --reference-endpoint "http://127.0.0.1:8203" \
  --trace-uri "s3://<bucket>/<prefix>/target/action-trace.npz" \
  --manifest-uri "s3://<bucket>/<prefix>/target/action-trace.json" \
  --run-id "<run-id>"
```

Then run the workflow using the same configured bucket/prefix/run ID and an
operator-private immutable cloud-evidence-image digest plus the exact source
commit baked into that digest. It repeats canonical preparation, verifies the
target artifacts, evaluates the actual trace, and publishes the RRD. Before a
live submit, run the normal health, schema, plan/render, and image preflight
commands against the selected NPA cluster. The submit matrix registers this
workflow as plan-only and rotation-skipped until a Thor dispatcher and real target evidence exist; that
is a hardware limitation, not a claim that a plan or x86 worker validated the
engine.

## Readiness and current result

The adjacent
[`readiness record`](../../workflows/testing/molmoact2-jetson-thor-edge.readiness.json)
binds the final YAML hash and separates validated local planning from blocked
target runtime, input, model-byte, and execution prerequisites. The private
cloud evidence image has a separate byte/pull qualification gate; the recorded
pre-merge digest does not certify a later changed image context and must be
rebuilt and requalified privately before it is used again. No cloud image can
contain or qualify the operator-fetched Thor plans. There is no public
container-catalog row or target-image/live-run claim for this derivative.

To resume, provide one exact Thor device with the stated software, stage the
operator-owned inputs and pinned bytes, execute the native target commands,
then submit the five-stage workflow and independently inspect its JSON and RRD
from object storage. Record a target result only after its manifest/checksum,
trace, metric, and RRD artifacts all agree.
