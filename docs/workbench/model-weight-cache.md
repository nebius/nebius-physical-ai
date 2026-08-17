# Caching runtime-downloaded model weights

The workbench images bake **no model weights**. Every NVIDIA Cosmos checkpoint and
guardrail, GR00T, the Cosmos-Curate towers, the Qwen VLMs, Wan 2.2 and LTX are
license-gated or too large to redistribute, so
`npa/docker/workbench/packaging-contract.yaml` requires them to be fetched at run
time with the operator's own token. That is what makes the images publishable (see
[container-packaging.md](container-packaging.md)).

The contract says *when* the bytes arrive. It never said *where they land* — so each
runtime used whatever directory happened to be writable inside its image
(`/tmp/hf_home`, a pod-local `emptyDir`, a path with no bind mount at all). The
download died with the container, and **the next run of the same image downloaded it
again**: tens of gigabytes of egress and minutes of GPU time per stage, on an
already-billing GPU, every run.

`npa.workbench.model_cache` is the one answer to that question. Point it at durable
storage once and every runtime NPA drives redirects its whole cache family into that
tree, so the second run of an image is a cache hit.

## Turn it on

Durability is a property of storage that only the operator can supply, so the cache
is **opt-in**. With nothing configured, `resolve_model_cache_root()` returns `""`
and every caller keeps the ephemeral default it has always used — nothing changes
implicitly.

### Kubernetes (SkyPilot tasks, sim2real sibling GPU Jobs)

```bash
kubectl get storageclass                       # confirm a ReadWriteMany class exists
kubectl apply -f npa/docker/workbench/common/model-weight-cache.yaml
kubectl wait --for=condition=complete job/npa-init-model-cache --timeout=5m
export NPA_MODEL_CACHE_PVC=npa-model-cache
```

The claim is `ReadWriteMany` on purpose: stages of one workflow land on different
nodes and parallel waves run at the same time, so a `ReadWriteOnce` volume would
serialise them or fail to attach. On Nebius mk8s that is `csi-mounted-fs-path-sc`,
the same shared-filesystem class the [Isaac runtime
cache](container-packaging.md#runtime-fetched-isaac-sim-why-the-isaac-images-are-publishable)
uses. The one-shot init Job exists because a freshly provisioned volume is owned by
root while NPA never runs a workload container as UID 0.

**Check the storage class before applying.** Not every mk8s cluster has the
shared-filesystem driver: one provisioned with only `compute.csi.nebius.com` has no
ReadWriteMany class, and the claim then sits `Pending` with
`storageclass.storage.k8s.io "csi-mounted-fs-path-sc" not found` while the pods that
want it stay `ContainerCreating` — a failure that reads like a scheduling problem
rather than a storage one. On such a cluster either add the shared filesystem, or
switch the manifest to `[ReadWriteOnce]` on the block class and accept that every
consumer must land on the volume's node (fine for a single-GPU-node cluster, not for
a parallel workflow).

Every task NPA renders then gets the cache variables in its `envs` and the claim
mounted at `/opt/npa-model-cache` in its `kubernetes.pod_config`. A spec that
already mounts something at that path keeps its own mount.

### VM / Docker (`npa deploy`, long-lived workbench containers)

```bash
export NPA_MODEL_CACHE_HOST_PATH=/mnt/data/npa-model-cache
```

The deploy creates the directory on the host, bind-mounts it into the container at
`/opt/npa-model-cache`, and exports the cache variables with `docker run -e` so they
win over any defaults baked into the image's env-file.

### Explicit root

`NPA_MODEL_CACHE_DIR` sets the in-container root directly and wins over both of the
above. Use it when the durable filesystem is already mounted by something else (a
shared NFS mount, a node-local data disk mounted by a DaemonSet). It sets the
environment only — supplying the volume is then yours.

**Object storage is not an option.** The Hugging Face hub cache is a
`blobs/`+`snapshots/` tree held together by symlinks, which S3-backed FUSE mounts do
not implement. A bucket mount would corrupt exactly the cache it was meant to
preserve. Durable weight storage has to be a real filesystem.

## What gets redirected

One flat family, applied everywhere, rather than a per-tool selection. Each variable
is read by exactly the tool that defined it, so setting all of them is inert for the
tools that ignore it — whereas selecting a subset per stage means one mis-mapped
stage silently sends a multi-gigabyte download back to a container-local directory,
which is the failure this exists to remove.

| Variables | Path under the root | Read by |
| --- | --- | --- |
| `HF_HOME`, `COSMOS_HF_CACHE` | `huggingface` | huggingface_hub; baked by the Cosmos images |
| `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE` | `huggingface/hub` | hub + transformers downloads |
| `HF_DATASETS_CACHE` | `huggingface/datasets` | `datasets` |
| `HF_XET_CACHE` | `huggingface/xet` | HF's chunked transfer backend |
| `TORCH_HOME` | `torch` | `torch.hub`, torchvision weights |
| `NLTK_DATA` | `nltk` | Cosmos-Transfer2.5 prompt tooling |
| `NPA_COSMOS3_CACHE`, `COSMOS_DOWNLOAD_CACHE_DIR` | `cosmos3`, `cosmos3/downloads` | Cosmos 3 framework checkout + checkpoints |
| `NPA_COSMOS_REASON_CACHE`, `NPA_COSMOS_REASON2_CACHE`, `NPA_COSMOS_REASON3_CACHE` | `huggingface/cosmos-reason*` | Cosmos Reason families |
| `NPA_COSMOS_CURATE_WEIGHTS_DIR` | `cosmos-curate/models` | rebinds upstream Cosmos-Curate's hardcoded `/config/models` |
| `HF_LEROBOT_HOME`, `LEROBOT_HF_HOME` | `lerobot` | LeRobot datasets and policies |
| `WAN22_CACHE_DIR`, `NPA_LTX_MODEL_CACHE` | `wan2.2`, `ltx-2.5` | the BYOF video models |

`MODEL_CACHE_LAYOUT` in `npa/src/npa/workbench/model_cache.py` is the source of
truth; `MODEL_CACHE_ENV_NAMES` is the allow-list that env-filtering call sites (the
sibling-Job builder) must pass through intact.

`HF_HOME` alone would cover the hub cache in theory, but vendor entrypoints read the
narrower names directly and a single unset name is enough to leak a download.

## Verifying a run is actually caching

Each stage logs the root it resolved before it downloads anything:

```
npa model cache: /opt/npa-model-cache (weights persist across runs)
```

Absence of that line means the stage is on ephemeral storage. The line matters
because a re-download that should have been a cache hit is otherwise invisible:
"downloading 40 GB again" looks exactly like "downloading 40 GB the first time" in a
stage log.

The decisive check is to make a download impossible and see the stage still work: set
`HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) on a rerun. A stage that succeeds
with those set read every byte from the cache. To confirm from the cluster side:

```bash
kubectl run npa-cache-du --rm -it --restart=Never --image=busybox:1.37 \
  --overrides='{"spec":{"containers":[{"name":"npa-cache-du","image":"busybox:1.37",
  "command":["du","-sh","/opt/npa-model-cache"],
  "volumeMounts":[{"name":"c","mountPath":"/opt/npa-model-cache"}]}],
  "volumes":[{"name":"c","persistentVolumeClaim":{"claimName":"npa-model-cache"}}]}}'
```

## Sizing and lifecycle

40 GiB (the manifest default) holds the Cosmos Transfer + guardrail set with room for
one more model family. Budget ~100 GiB to run Cosmos3, the Reason family, and the
curator towers against the same claim.

**Size for throughput, not only capacity.** On Nebius network block storage the
volume's throughput scales with its provisioned size, and the cache is read on the
critical path of every warm run. Measured on a 60 GiB block claim: 29 MB/s on a
direct (page-cache-bypassing) read, which is what a 3 GB checkpoint taking ~100 s to
load off the volume looks like. The download it replaces ran at a comparable rate, so
on an undersized volume the win is that the fetch disappears entirely — not that the
bytes arrive faster. Provision generously, or use the shared-filesystem class, and the
load gets faster too.

The cache is append-only from NPA's side: **nothing evicts**. Weights are immutable
per revision, so a stale entry is not a correctness problem, only a space one. Delete
the PVC (or the host directory) to reclaim space; the next run re-downloads what it
needs.

## What this does not cover

Model *weights* only. The BYOF video images also fetch a CUDA PyTorch runtime into
their own trees (`NPA_WAN_RUNTIME_CACHE`, `NPA_LTX_RUNTIME_CACHE`), which are still
per-run and are not redirected here — a wheel closure is a different artifact with a
different verification story, which is why Isaac's has its own volume rather than
sharing one.

The refusal proofs those two images run before fetching anything (`ltx-runtime
assert-refusal`, and the LTX golden eval) assert that a refusal **writes nothing**
rather than that the cache is *empty*, precisely so the second run against a warm
weight cache still proves the gate closed.

## Relationship to the Isaac runtime cache

Same motivation, different fill pattern, separate volumes.

| | Isaac runtime cache | Model weight cache |
| --- | --- | --- |
| Manifest | `common/warm-isaac-cache.yaml` | `common/model-weight-cache.yaml` |
| Contents | pinned `isaacsim`/`isaaclab` wheels + Isaac Lab source | downloaded model weights |
| Fills | one CPU warm Job, up front | lazily, by whichever stage needs a checkpoint first |
| Mounted | read-only (`NPA_ISAAC_CACHE_READONLY=1`) | read-write; it accumulates |
| Opt-in via | `NPA_ISAAC_CACHE_DIR` | `NPA_MODEL_CACHE_PVC` / `_HOST_PATH` / `_DIR` |

## Related docs

- [container-packaging.md](container-packaging.md) — why weights are fetched at run
  time in the first place, and the redistribution boundary that requires it
- [huggingface-token.md](huggingface-token.md) — the credential the gated downloads use
- [ngc-api-key.md](ngc-api-key.md) — the NGC credential for NuRec's NRE runtime
- [kubernetes.md](kubernetes.md) — cluster setup this cache plugs into
