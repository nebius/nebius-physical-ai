# Policy training stages

The Slurm adapter implements operator-owned training scripts and durable job
reconciliation. The public recipe uses `turnkey.py` stateless stages through
standard Workbench toolRefs; run the single
`workflows/testing/policy-public-training.yaml` with the public policy cookbook.

`turnkey_data` prepares, curates and splits public episodes; `turnkey_training`
trains, evaluates, gates and exports; `turnkey_runtime` prepares the native
environment and publishes verified recovery checkpoints. `turnkey_serving`,
`turnkey_server` and `turnkey_client` run independent GPU inference and native
simulation. `turnkey_report` reconciles actions and embeds actual media into
standalone HTML. `turnkey_store` carries sealed lineage between stateless workers.

Runtime-supplied environment variables: `NPA_WORKFLOW_SHA256` identifies the
resolved workflow; `NPA_VLA_RECOVERY_URI` selects private recovery storage;
`NPA_VLA_RECIPE_SHA256` and `NPA_VLA_SELECTION_SHA256` bind recovery to its inputs.
`NPA_VLA_EVIDENCE` retains native metrics and checkpoints. SkyPilot supplies the
two-worker rank/GPU/address variables. AWS credentials and the selected project's
S3 endpoint stay private and are passed by standard workflow submission.
