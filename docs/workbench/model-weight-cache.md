# Caching runtime-downloaded model weights and reviewed SDKs

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

`npa.workbench.model_cache` is the one answer to that question for weights and the
Content Agents OVRTX SDK. Point it at durable
storage once and every runtime NPA drives redirects its whole cache family into that
tree, so the second run of an image is a cache hit.

## Turn it on

On by default wherever the answer is not "invent storage nobody asked for":

- **VM deploys cache without being told.** `npa deploy` creates
  `/var/lib/npa/model-cache` on the host and binds it. A directory needs no
  provisioning and costs nothing to create, so there is no reason to make an
  operator ask; the alternative default is discarding every gated download on the
  next `docker rm -f`, which this deploy runs every time.
- **Kubernetes turns itself on once the claim exists.** Submit looks for a Bound
  claim named `npa-model-cache` in the target namespace and uses it, so applying the
  manifest below is the entire opt-in — there is nothing to export afterwards, and
  no shell that forgets. A Pending claim is deliberately not adopted: it has no
  volume behind it, so mounting it would leave pods in `ContainerCreating`.
- **Serverless Jobs mount the filesystem you name.** They have no cluster and no
  host to borrow storage from, so there is nothing to detect and nothing sensible to
  default to; name a Nebius filesystem and every job attaches it:

  ```bash
  export NPA_MODEL_CACHE_FILESYSTEM=<filesystem>   # not an s3:// bucket
  ```

Nothing here provisions storage. NPA will not create a claim, guess a class, or bill
an operator for a volume they did not ask for — the Kubernetes side stays off until
the claim exists, which is the one step only they can take.

`NPA_MODEL_CACHE_DISABLED=1` switches all of it off, everywhere, including the
defaults.

### Kubernetes (SkyPilot tasks, sim2real sibling GPU Jobs)

```bash
kubectl get csinode -o custom-columns=NODE:.metadata.name,DRIVERS:.spec.drivers[*].name
kubectl apply -f npa/docker/workbench/common/model-weight-cache.yaml
kubectl wait --for=condition=complete job/npa-init-model-cache --timeout=5m
```

That is the whole setup: submit finds the claim by name and reports
`model weight cache: using claim 'npa-model-cache'`. Export
`NPA_MODEL_CACHE_PVC=<name>` only to point at a claim you named something else, and
`NPA_MODEL_CACHE_NAMESPACE=<ns>` when your SkyPilot pods do not land in `default`.

The claim is `ReadWriteMany` on purpose: stages of one workflow land on different
nodes and parallel waves run at the same time, so a `ReadWriteOnce` volume would
serialise them or fail to attach. On Nebius mk8s that is `csi-mounted-fs-path-sc`,
the same shared-filesystem class the [Isaac runtime
cache](container-packaging.md#runtime-fetched-isaac-sim-why-the-isaac-images-are-publishable)
uses. The one-shot init Job exists because a freshly provisioned volume is owned by
root while NPA never runs a workload container as UID 0.

**Check the driver, not the storage class.** The class existing proves nothing — it
can be present, and even be the cluster default, while no claim on it can ever bind.
What discriminates is whether a node registers the driver:

```bash
kubectl get csinode -o custom-columns=NODE:.metadata.name,DRIVERS:.spec.drivers[*].name
```

The init Job that opens the volume's permissions runs as root — `chmod` on a fresh
volume root is the one privileged act here — so a `restricted` PodSecurity namespace
rejects it. Either apply it somewhere at `baseline`, or skip it and rely on
`fsGroup: 1000`, which most CSI drivers apply to the volume root on first mount;
`kubectl exec <pod> -- ls -ld /opt/npa-model-cache` tells you before a workload does.

`mounted-fs-path.csi.nebius.ai` has to be in that list. It is missing in two
different situations, and only the first is obvious:

- The class does not exist, because the cluster was provisioned with only
  `compute.csi.nebius.com`. The claim sits `Pending` on
  `storageclass.storage.k8s.io "csi-mounted-fs-path-sc" not found`.
- The class exists and the node plugin is installed, but the plugin **deliberately
  refuses to start** because the node group has no Nebius shared filesystem
  attached: `Failed to initialize driver: mounted on ext4 fs, data loss may occur,
  aborting`. Backing a supposedly-shared volume with node-local disk would lose
  data, so aborting is correct. The claim then sits `Pending` on the far vaguer
  `Waiting for a volume to be created either by the external provisioner ... or
  manually by the system administrator`, and every pod that wants it stays
  `ContainerCreating`, which reads like a scheduling problem. Confirm with:

```bash
kubectl -n kube-system logs -l app=csi-mounted-fs-path-plugin -c mounted-fs-path-provisioner --tail=5
```

Either attach a shared filesystem to the node group, or take the ReadWriteOnce
fallback: `accessModes: [ReadWriteOnce]` with `storageClassName:
compute-csi-default-sc`, and pin the init Job to the node your consumers will use,
since a ReadWriteOnce volume binds to whichever node touches it first. That fits a
single-GPU-node cluster, not a parallel workflow.

### Concurrent consumers

`huggingface_hub` serialises concurrent downloads with `filelock`, taking an
advisory lock per blob under `<cache>/.locks`, so sharing one cache between stages
that start together depends on locking working on the volume. Measured with four
pods contending on one claim: locks were strictly mutually exclusive (each holder
waited for the previous one, no overlap), four simultaneous `snapshot_download`
calls for the same repository produced exactly one copy — 12 blobs, one snapshot,
no leftover `.incomplete` files — and the result re-resolved offline afterwards with
the expected content hash.

That was on a block volume, so it covers the library's coordination but not
cross-node lock propagation on the shared-filesystem class. If you run parallel
waves across nodes against one claim, expect correctness (downloads are
content-addressed and published by atomic rename) but treat "no duplicated
download" as unverified.

Every task NPA renders then gets the cache variables in its `envs` and the claim
mounted at `/opt/npa-model-cache` in its `kubernetes.pod_config`. A spec that
already mounts something at that path keeps its own mount.

### VM / Docker (`npa deploy`, long-lived workbench containers)

Already on, using `/var/lib/npa/model-cache`. Point it at a different disk when the
root filesystem is not where tens of gigabytes of weights should go:

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

### Which variable reaches which runtime

The three variables name storage in three different worlds, so each runtime acts
only on the one it can actually mount. This matters because exporting the cache
environment without the storage behind it is *worse* than exporting nothing:
`/opt` is root-owned in every workbench image and they all run unprivileged, so the
first `mkdir` fails and the stage dies where it used to work.

| Runtime | Storage it can mount | Default with none set |
| --- | --- | --- |
| SkyPilot stage on Kubernetes | `_PVC`, `_HOST_PATH` (node-local; see below) | the claim, if it exists |
| SkyPilot stage on any other cloud | none | off |
| sim2real sibling GPU Job | `_PVC`, `_HOST_PATH` | the claim, if it exists |
| `npa deploy` container on a VM | `_HOST_PATH` | `/var/lib/npa/model-cache` |
| Workbench Serverless Job | `_FILESYSTEM` | off |
| OpenPI policy server (Deployment) | `_PVC`, `_HOST_PATH` | pod-local `emptyDir` |
| LeIsaac session (Deployment) | `_PVC`, `_HOST_PATH` | pod-local `emptyDir` |
| LeRobot server on a VM | `_HOST_PATH` | `/var/lib/npa/model-cache` |
| In-container code reading its own env | none | off |

`NPA_MODEL_CACHE_DIR` is honored by every row: it is the operator asserting the path
is already mounted, which is the only claim a runtime cannot check for itself.

So an operator can export `NPA_MODEL_CACHE_PVC` for their Kubernetes workflows
without changing anything about their VM deploys or Serverless Jobs, which have no
way to reach a claim.

`NPA_MODEL_CACHE_FILESYSTEM` must name a filesystem, not a bucket. `nebius ai job
create --volume` also accepts an `s3://` source, and NPA refuses it here: an object
mount cannot represent the symlinked `blobs/`+`snapshots/` tree, so it would corrupt
the cache on the second run rather than fail on the first.

`NPA_MODEL_CACHE_HOST_PATH` on Kubernetes produces a `hostPath` volume on the
*cluster nodes*, which is node-local rather than shared: a stage that lands on
another node sees an empty cache, and a namespace with `restricted` PodSecurity
admission rejects the pod outright. Prefer a claim, and reach for the host path only
on a single-node cluster you control.

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
| `NPA_COSMOS_REASON_CACHE`, `NPA_COSMOS_REASON2_CACHE`, `NPA_COSMOS_REASON3_CACHE` | `huggingface/cosmos-reason*` | Self-hosted Cosmos Reason families |
| `NPA_COSMOS_CURATE_WEIGHTS_DIR` | `cosmos-curate/models` | rebinds upstream Cosmos-Curate's hardcoded `/config/models` |
| `HF_LEROBOT_HOME`, `LEROBOT_HF_HOME` | `lerobot` | LeRobot datasets and policies |
| `WAN22_CACHE_DIR`, `NPA_LTX_MODEL_CACHE` | `wan2.2`, `ltx-2.5` | the BYOF video models |
| `NPA_CONTENT_AGENTS_RUNTIME_CACHE` | `runtimes/content-agents` | exact OVRTX SDK delivered directly by NVIDIA to the operator |

`MODEL_CACHE_LAYOUT` in `npa/src/npa/workbench/model_cache.py` is the source of
truth; `MODEL_CACHE_ENV_NAMES` is the allow-list that env-filtering call sites (the
sibling-Job builder) must pass through intact.

`HF_HOME` alone would cover the hub cache in theory, but vendor entrypoints read the
narrower names directly and a single unset name is enough to leak a download.

## Verifying a run is actually caching

Each stage logs the root it resolved before it downloads anything:

```
npa model cache: /opt/npa-model-cache (mounted here, cached artifacts persist across runs)
```

Absence of that line means the stage is on ephemeral storage. A stage that resolved
its root from `NPA_MODEL_CACHE_DIR` says `not mounted by npa; persists only if this
path already does` instead, because NPA attached nothing and cannot vouch for what
is behind the path. The line matters
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

## Does this need rebuilt images?

Mostly no, and that is by design: the mechanism is environment variables that
`huggingface_hub`, `transformers`, and `torch` already read, injected by a renderer
that runs outside the image. The live validation above ran on a `npa-cosmos3-reason`
image published months before this existed, and the weights landed in the durable
cache. The in-container code paths were already environment-first, so an older image
resolves the same paths.

Two exceptions, both about code that is *baked*:

- **`npa-ltx2`** carries `ltx-runtime` in its layers, and its refusal proof changed.
  The cache itself works on an older build — it honours `NPA_LTX_MODEL_CACHE` like
  everything else, and nothing about the licence objects to reusing weights the
  operator already fetched under their own entitlement. What breaks is the image's
  own self-test: it asserts its caches are *empty* after the gated fetches refuse,
  which is true on a cold run and false on every run after the first. Rebuilding
  from current source, where the proof uses private directories, fixes it; until
  then a caller can get the same effect by pointing `NPA_LTX_MODEL_CACHE` and
  `NPA_LTX_RUNTIME_CACHE` at a temporary directory for the `assert-refusal` step
  alone.
- **The sim2real controller image** creates the sibling GPU Jobs from inside the
  container, so the code that mounts the claim onto them is the copy in its layers.
  Rebuild it to get the cache on sibling Jobs. The task images it launches
  (transfer, envgen, reason, Isaac) need nothing: they receive the env and the mount.

Everything rendered operator-side — SkyPilot task envs and pod volumes, Serverless
Job envs, VM deploys — takes effect as soon as the operator's `npa` is current,
whatever image the stage pulls.

## What this does not cover

Model weights plus the Content Agents OVRTX SDK only. OVRTX shares this cache
because three separate render Jobs need the same verified SDK and the public image
contains none of its bytes. Its version/architecture/lock-bound identity, writer
lock, unique temporary installation, and atomic ready marker keep it separate from
mutable model aliases. Without a mounted cache it falls back to XDG cache, which is
pod/node-ephemeral under SkyPilot and can be downloaded again by the next Job.
A durable ReadWriteOnce claim works for the sequential Content Agents render
stages; use ReadWriteMany only when jobs on different nodes must read the cache
concurrently.

The BYOF video images also fetch a CUDA PyTorch runtime into
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

| | Isaac runtime cache | Shared model/runtime cache |
| --- | --- | --- |
| Manifest | `common/warm-isaac-cache.yaml` | `common/model-weight-cache.yaml` |
| Contents | pinned `isaacsim`/`isaaclab` wheels + Isaac Lab source | downloaded model weights + exact Content Agents OVRTX SDK |
| Fills | one CPU warm Job, up front | lazily, by whichever stage needs a checkpoint first |
| Mounted | read-only (`NPA_ISAAC_CACHE_READONLY=1`) | read-write; it accumulates |
| Opt-in via | `NPA_ISAAC_CACHE_DIR` | `NPA_MODEL_CACHE_PVC` / `_HOST_PATH` / `_DIR` |

## Related docs

- [container-packaging.md](container-packaging.md) — why weights are fetched at run
  time in the first place, and the redistribution boundary that requires it
- [huggingface-token.md](huggingface-token.md) — the credential the gated downloads use
- [ngc-api-key.md](ngc-api-key.md) — the NGC credential for NuRec's NRE runtime
- [kubernetes.md](kubernetes.md) — cluster setup this cache plugs into
