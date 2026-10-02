# Artifact Retention Policy

## S3 Artifact Lifecycle

Sim2Real workflow artifacts stored in S3 follow this retention policy:

| Artifact Type | Retention | Rationale |
|--------------|-----------|-----------|
| Training checkpoints | 90 days | Reproducibility window |
| Evaluation videos | 30 days | Review window |
| Intermediate rollouts | 7 days | Debugging only |
| Final models | 1 year | Production candidates |

## GC

`npa workbench gc-artifacts` enforces the policy. It scans run prefixes under
an S3 bucket, classifies each run from its `npa-workflow/manifest.json`, and
deletes only runs that are provably terminal
(`SUCCEEDED`/`FAILED`/`CANCELLED`/`FAILED_STARTUP`) **and** older than the
retention window (default 90 days, `--retention-days` to override).

The command is dry-run by default: it prints the plan without touching S3.

```bash
npa workbench gc-artifacts --bucket "<bucket>" --prefix "<prefix>"
```

Pass `--apply` (alias `--yes`) to execute the plan:

```bash
npa workbench gc-artifacts --bucket "<bucket>" --prefix "<prefix>" --apply
```

Preview with a shorter window and machine-readable output, journaling the
plan to a file:

```bash
npa workbench gc-artifacts --bucket "<bucket>" --retention-days 30 --json \
  --journal /tmp/gc-journal.json
```

The bucket and prefix can also come from the environment (`NPA_S3_BUCKET`,
`NPA_S3_PREFIX`).

## Safety rules

Deletion fails closed:

- Prefixes without a readable `npa-workflow/manifest.json` are never touched.
- Runs whose status is live (`RUNNING`, `PLANNED`, `SUBMITTED`) or unknown
  are never deleted.
- Runs whose age cannot be determined are kept.
- In apply mode each run's manifest is re-read immediately before deletion,
  so a run that restarted after planning is skipped, not deleted.

## Pinning

A run carrying a `.npa-retain` marker file under its prefix is exempt from
deletion regardless of age or status. Use it for checkpoints and curated
datasets worth keeping:

```bash
touch .npa-retain
aws s3 cp .npa-retain "s3://<bucket>/<run-prefix>/.npa-retain"
```

## Not yet implemented

Scheduled weekly GC runs and S3 lifecycle-rule enforcement are future work
(see issue #525). Until then, run `npa workbench gc-artifacts --apply`
manually or from the operator's own scheduler.
