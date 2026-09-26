---
name: encord
description: Use when registering S3 media with Encord SaaS, pulling curated media, or verifying an exact roundtrip with durable lineage artifacts.
---

<!-- register: operating procedure | reader: agent operating Encord transport | consumed: task-time reference -->
# Operate the Encord transport

Encord remains remote SaaS. NPA is a stateless CPU client that runs locally or
in the default workflow pod. Do not deploy a service or container for this
integration.

## Preflight

Reference credentials by environment name or NPA credential configuration.
Never embed values in a command, spec, example, receipt, or manifest.

```bash
npa workbench health preflight --checks encord
```

For workflow forwarding, use `ENCORD_SSH_KEY_B64` because a local
`ENCORD_SSH_KEY_FILE` path cannot reach a pod.

## Push

```bash
npa workbench encord push \
  --input-path s3://<bucket>/raw-media/ \
  --integration <encord-cloud-integration> \
  --folder <encord-folder> \
  --output-path s3://<bucket>/encord/push/push_receipt.json
```

Register mode retains S3 as the dataset of record. `--transfer upload` is an
explicit alternative that creates an Encord-managed copy with separate
retention and possible duplication. Never retry a registration failure as an
upload.

Require exact source URI, complete object key or URL, namespaced metadata,
stable item UUID, or a sidecar assertion. Never use a filename or basename as
identity. Stop when exact signals conflict.

## Pull

The default `--label-export none` avoids label initialization. Use
`--label-export initialize` only when the operator explicitly selects and
confirms that remote Encord label-state mutation. Retain the resulting manifest
as evidence.

## Labeling demo

Partner workflows and their runbook live in `workflows/partners/encord/`.
`encord-labeling-demo.yaml` runs push → import-labels → project pull with label
initialization → media verification → render-labels. Import creates a new
ontology and project from a SHA-256-bound video bounding-box plan. It refuses
existing project titles and output receipts. Rendering compares the actual
exported row/object identities, frames, classes, and coordinates, then produces
annotated MP4s from those exported labels. These are unreviewed programmatic
prelabels, not human-approved annotations. Keep that distinction in demos.
Select the new project and S3 targets before execution. Keep exact remote
identities and signed label-export media URLs in private evidence.

## Verify transport

Claim a roundtrip only when `verify-roundtrip` consumes both final artifacts
and passes exact identity, destination existence, size, and compatible checksum.

Checkpoints reduce evidence loss. They cannot guarantee that the newest remote
mutation is represented if artifact persistence fails immediately afterward.
The tool stops further mutation and exits nonzero.

Strict client-only Encord access can limit server-side media features. Treat
that as a capability choice, not full feature parity.

Before running any Encord transport command that writes S3 or changes Encord
state, confirm the target Encord integration or source and S3 prefixes with the
operator. Do not infer authorization for remote mutations from a read-only or
planning request.
