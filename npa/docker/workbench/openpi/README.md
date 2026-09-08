# OpenPI image

The image packages pinned OpenPI source and the NPA four-mode/full-DROID adapters.
Build it through the repository's trusted public-development workflow after its
packaging, security and licensing gates pass. See [REDISTRIBUTION.md](REDISTRIBUTION.md)
for the CUDA/cuDNN boundary and [the policy workflow documentation](../../../../docs/workbench/openpi-pi05-polaris.md)
for digest-pinned GPU submission and artifact contracts.

Gemma-derived Polaris weights and DROID data are fetched only at runtime.
Public Polaris GCS access needs no Hugging Face token. Runtime use defaults on
under the named Gemma terms; `NPA_OPENPI_ACCEPT_GEMMA_TERMS=NO` opts out before
model import or checkpoint access. An empty value also opts out. Affirmative
`Y`, `YES`, `1`, `TRUE` values and negative `N`, `NO`, `0`, `FALSE` values are
case-insensitive; invalid values fail. Unset the variable to resume the default.
No acceptance value, credential, checkpoint, dataset, or populated cache is baked.

The final non-root build step verifies command argument forwarding, writable
home and temporary directories, passwordless sudo, and the SSH daemon's complete
start/status/stop lifecycle. Password and root SSH login are disabled; generated
host keys are removed before that layer commits. This verifies the SkyPilot
bootstrap contract on the selected base.
CUDA device probes contain actual SM100 and SM120 kernels; their compilation
and ELF checks do not replace a real policy workload. Public development
validation does not promote the separate eight-node full-DROID release.

## Measured public development image

The public image `npa-openpi:dev-5dbe0fc1e87ae4da54dd7605db24383a79835d39`
resolves to
`sha256:df6910c8e8c73661b02eb55f6e62046a610a1c01a3f6c3aa570b40725b6ebb2b`.
[Trusted build 34197309452](https://github.com/nebius/nebius-physical-ai/actions/runs/34197309452)
passed its complete-layer, security, bootstrap and attestation gates; independent
anonymous registry verification bound the same source and digest.

On September 8, 2026, that digest completed the six-state four-mode workflow on
RTX PRO 6000 Blackwell (`sm_120`): explicit opt-out refusal, synthetic data
preparation, finite `15x8` direct inference, two requests between separate server
and client pods, eight real upstream LoRA/AdamW updates, checkpoint save/reload
at step 8, and evaluation on four samples excluded from the eight-sample
training split. All 34 artifacts passed storage readback, including 25
checkpoint files totaling 9,266,047,031 bytes. Exact image IDs, UID 1000 and
installed adapter hashes were observed; GPU stages used no NPA source overlay.
The CPU controller source was separately staged and fingerprinted.

Native `sm_120` probes passed in direct inference, training and evaluation.
Serving established real JAX GPU inference but did not run a server-native
`sm_120` probe. This is development functionality on synthetic inputs, with no
convergence, physical-robot, B200/B300 or full-DROID qualification claim.
The canonical `pi05-full-droid-rlds-cu128-unbuilt` pin remains quarantined.
See the [measured results](../../../../docs/workbench/openpi-pi05-polaris.md#public-development-validation-on-rtx-pro-6000)
and the hash-bound record in
[`blackwell-dc-images.json`](../blackwell-dc-images.json).
