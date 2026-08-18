---
name: teardown-and-cost
description: Use to stop spend safely and audit orphaned resources — the mandatory cancel-before-destroy ordering, which npa commands touch cloud versus local state, and how to find leaked clusters, agent VMs, controllers, and buckets.
---

# Teardown and cost hygiene

There is no `npa cost` and no `npa quota` command. Spend control here is
procedural: run things in an order that cannot orphan resources, and audit for
residue afterwards. The single most expensive mistake is destroying
infrastructure while managed jobs are still running, which leaves jobs billing
against resources nothing is tracking any more.

## The ordering is not optional

Cancel workloads, then remove the things that host them, outermost last:

```bash
npa workbench workflow cancel <run-id> --project <alias> --json   # every run
npa agent destroy --project <alias> --name <name> --yes
npa skypilot cleanup-controller --project <alias> --context <context> --yes
npa cluster down --project <alias> --force
npa storage bucket delete --project <alias> --yes --wait
npa storage service-account delete --project <alias> --dry-run
npa storage service-account delete --project <alias> --yes
npa configure --forget-project <alias>
npa cleanup --full --yes --project <alias>
```

Cancel is repeat-safe — a run that never launched is a successful no-op — so
there is no reason to skip it. `cleanup-controller` refuses while managed jobs
are in progress and retries that specific refusal after the queue drains; treat
the refusal as correct, not as something to force.

Deleting the project itself is separately opt-in and stays retained by default:

```bash
npa destroy --project <alias> --all --json          # plan only
npa destroy --project <alias> --all --yes           # execute, project retained
npa destroy --project <alias> --all --delete-project --yes --json
```

`npa destroy --all` without `--delete-project --yes` never deletes the project.
Full details of the ownership gating live in `skills/tools/nebius-infra/SKILL.md`;
this skill is the operational ordering and the audit.

## What is cloud spend and what is only local clutter

Confusing these two wastes time and money in opposite directions.

| Command | Touches cloud spend | Notes |
|---|---|---|
| `npa cleanup` | No | Local caches and state only, even with `--yes` |
| `npa cleanup --full --yes` | No | Also removes saved HF/Token Factory/NGC creds; read-only IAM verification |
| `npa workbench workflow cancel` | Yes — stops it | Repeat-safe |
| `npa skypilot cleanup-controller` | Yes | Shared; drains first |
| `npa cluster down` | Yes | Terraform-managed cluster and its nodes |
| `npa agent destroy` | Yes | Agent VM and its resources |
| `npa storage bucket delete` | Yes | Contents and versions included |
| `npa destroy --all --delete-project --yes` | Yes | Ownership-gated; all three flags required |

`npa cleanup` never deletes cloud resources. It *reports* what teardown left
behind and prints the ordered runbook, which makes it the right first command
when you inherit a machine and do not know what is still running:

```bash
npa cleanup --json                 # report only
npa cleanup --list-receipts        # durable audit trail, no secrets
```

## Auditing for orphans

Residue accumulates when a teardown was interrupted or run out of order. Check
each class explicitly — nothing sweeps them for you:

```bash
npa workbench workflow list --project <alias> --json    # non-terminal runs
npa skypilot status --project <alias> --context <ctx>   # shared controller
npa cluster list --project <alias>                      # clusters, incl. stale
npa agent list                                          # agent VMs in config
npa storage bucket list --project <alias>               # marks the configured one
npa fleet status --spec <spec.yaml>                     # cross-project fleets
```

Cross-project fleets are the easiest thing to lose track of, because a fleet spec
can create clusters in projects that never appear in your default alias. Audit
fleets from the spec that created them.

`npa cleanup` queries the SkyPilot managed-job queue as part of its report. Do
not reach for `--skip-jobs` to make it quiet: skipping the queue with
`--attest-no-active-jobs` is an explicit attestation that you verified terminal
state some other way, not a shortcut.

## Receipts: how to finish a teardown you can no longer address by alias

Teardown receipts live under `~/.npa/teardown-receipts/`, contain no secrets, and
survive removal of the project config. If you already ran
`configure --forget-project` and later discover residue, the receipt is the
recovery identity:

```bash
npa cleanup --list-receipts
npa destroy --receipt <id> --all --delete-project --yes --json
```

Prune only aged terminal receipts, and only explicitly:

```bash
npa cleanup --prune-receipts --receipt-retention-days 90 --yes
```

## Choosing a cheap tier in the first place

Most verification does not need a GPU, and the tiers are ordered by cost:

1. **Dry run / plan.** `npa workbench workflow plan-spec`, `submit --plan-only`,
   `golden-eval run <name>` (dry-run is the default), `provision-if-absent
   --dry-run`. No cloud resources.
2. **Zero-GPU hosted inference.** Token Factory (`skills/tools/token-factory`)
   answers real captioning/generation/reasoning questions with no cluster at all.
3. **Serverless single job.** `golden-eval run <name> --serverless` runs one
   container on one GPU and returns PASS/FAIL, without standing up a cluster.
4. **Cluster.** Only when the work genuinely needs a persistent multi-stage graph.

Preemptible GPU nodes change the capacity pool but **not** hard instance, disk,
or IP quotas, and a reclaim stops the node mid-run. They lower price, not
requirements.

## Gotchas

- **A PENDING job is still billing the cluster it is stuck on.** `ImagePullBackOff`
  and `Unschedulable` are retried forever and never self-terminate. Cancel them;
  see `skills/atomic/debug-failed-run/SKILL.md`.
- **`npa cluster down` going quiet for minutes is expected.** It previews the
  PodDisruptionBudgets that hold up the node drain before it destroys anything.
- **The controller is shared.** Never tear it down because *your* run finished;
  it may be hosting someone else's managed jobs. That is why `cleanup-controller`
  requires the exact project and saved context rather than an ambient
  kube-context.
- **Full cleanup exiting 2 is a real result, not a flake.** It means storage IAM
  is present or unverified: cleanup was partial and something still exists.
- **Never run a live GPU tier on a schedule.** Scheduled GPU runs are the classic
  way to discover a five-figure bill; keep GPU rotations manually dispatched.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
