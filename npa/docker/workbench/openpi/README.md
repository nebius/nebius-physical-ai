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
