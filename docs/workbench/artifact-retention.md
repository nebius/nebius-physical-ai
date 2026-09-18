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

A scheduled job runs weekly to delete artifacts older than their retention period.
The GC uses S3 lifecycle rules where possible, and a cleanup script for
prefix-based deletion.

## Implementation

S3 lifecycle rules are configured in the infrastructure Terraform. The `npa`
CLI includes `npa workbench gc-artifacts` for manual cleanup.
