# Antioch → Nebius XR1 robot learning

This pipeline fine-tunes **Xiaomi-Robotics-1 5B** on simulated dual-Franka
pick-and-place demonstrations and evaluates the resulting robot action policy
in Antioch.

| Step | Where it runs | Durable output in Nebius S3 |
| --- | --- | --- |
| Stage pinned XR1 source, base weights, processor, and episode split | Operator / Nebius | Assets, runtime manifest, and checksums |
| Collect synchronized three-camera images, robot state, and issued actions | Antioch Isaac Sim | Demonstrations and sealed train/validation/test split |
| Fine-tune XR1 and select a checkpoint using validation loss | Nebius, eight RTX PRO 6000 GPUs | Model, optimizer checkpoints, validation metrics, and provenance |
| Fetch the selected policy and compare it with the base model on held-out tasks | Antioch Isaac Sim | Robot outcomes, native camera recordings, and paired comparison |

The [workflow YAML](xr1-antioch-finetune.yaml) runs training and checkpoint
publication. Antioch collection, transfer, and closed-loop evaluation are
operator steps documented in the [complete runbook](../../../docs/workbench/cookbooks/xr1-antioch.md).
The YAML alone does not launch the entire simulation loop.

## Run it

Follow the [Antioch Workbench skill](../../../skills/workflows/antioch-workbench/SKILL.md)
for project setup, SDK version, credentials, collection, training, and evaluation.
Store an optional personal access token as `tokens.ANTIOCH_TOKEN` in
`~/.npa/credentials.yaml`; the environment overrides it and native Antioch browser
login remains a fallback. The Workbench operator supplies the stored token to
Antioch client processes. Tokens and signed S3 URLs do not belong in workflow
YAML or Git.

From the repository root, validate and inspect the training plan:

```bash
npa workbench workflow validate-spec workflows/partners/antioch/xr1-antioch-finetune.yaml
npa workbench workflow plan-spec workflows/partners/antioch/xr1-antioch-finetune.yaml --run-id preview --waves
```

Before submission, replace the example bucket through configuration overrides
and prepare the dataset, assets, and runtime manifests described in the runbook.
The YAML declares their S3 contracts and publishes the selected model under
`training_uri/candidate/` with validation and checksum manifests. Antioch uses
short-lived signed GET/PUT URLs; static S3 keys stay with the operator and
Nebius training worker. Transfers are verified by complete readback.

## Recorded result

The [measured evidence](../../../docs/workbench/evidence/xr1-antioch-rtxpro.json)
records the training YAML checksum and the run results.
The recorded run completed 10,000 optimizer steps on eight Nebius RTX PRO 6000
GPUs and selected step 9,000. Paired held-out task success improved from **0/32
to 8/32**; all 192 policy camera videos passed decoding checks. Antioch used one
RTX PRO 6000 for simulation. The low absolute success rate does not establish
real-robot readiness, and simulation paused while waiting for policy inference.
