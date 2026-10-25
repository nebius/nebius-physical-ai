# Shared Workbench source repairs

`repair_accelerate_checkpoints.py` repairs the actual checkpoint loader in the
reviewed Accelerate 1.12.0, 1.13.0, 1.14.0 and 1.15.0 distributions. Run it with
the interpreter whose installed distribution needs the repair. Unknown versions
and source hashes fail before modification. The repair preserves the version
and upstream Apache-2.0 license, updates wheel RECORD and checked-hash bytecode,
and writes `NPA-CHECKPOINT-SECURITY.json` inside the distribution metadata.

The repair rejects index entries that escape the checkpoint directory, absolute
shard paths, malformed mappings and nonregular shards. The index is opened with
`O_NONBLOCK` and checked through its descriptor before JSON parsing, so a named
pipe cannot block that read. Ordinary relative nested paths, flat indexes, and
Hugging Face snapshot symlinks to regular cached blobs remain supported. The
operator must own the cache: this does not sandbox an adversary who can mutate
the local filesystem or replace its symlinks during loading.

The [advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79) lists versions
through 1.14.0 and no patched release. Direct tests also reproduced traversal
and FIFO opening in the official 1.15.0 wheel. This is an NPA source repair,
not an upstream fix or a scanner waiver. Source versions remain visible to
vulnerability scanners. The five consuming recipes are Wan 2.2, Cosmos,
Cosmos3 Reason, Cosmos3 Serving and RoboCasa. Serving repairs both fresh and
reused runtime environments before executing the service. Baked consumers apply
the repair in the same build instruction as installation.

Run the real loader verification inside a disposable environment containing
the repaired Accelerate, Torch and Safetensors:

```sh
/path/to/runtime/bin/python repair_accelerate_checkpoints.py
/path/to/runtime/bin/python verify_accelerate_checkpoints.py
```

The verification loads real PyTorch and Safetensors tensors through both public
checkpoint APIs, checks exact model outputs, preserves cache symlinks, and
rejects traversal, malformed shard values, FIFO shards, FIFO symlinks and FIFO
indexes. Controlled FIFO writers make an unsafe loader observable without
leaving it waiting for a writer. This CPU verification does not qualify an
image's GPU workload or its remaining security findings.

The separate [pip source repair](secure_pip/README.md) documents the derived
installer and the constrained Setuptools manifest backport.
