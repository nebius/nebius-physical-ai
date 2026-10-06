# LeRobot default dependency closure, 2026-10-06

Rebuilding the releases quarantined by PR #807 exposed a separate, older runtime
regression. PR #173 forced Torch 2.12.1, torchvision 0.27.1 and Diffusers >=0.38
into the generic LeRobot 0.5.1 recipe, outside upstream's declared bounds. It
retained TorchCodec 0.10, which uses the Torch 2.10 ABI, and removed required
W&B. The separately validated B300 recipe and historical optional 0.6 evidence
have different dependency closures.

The [trusted source-correction build](https://github.com/nebius/nebius-physical-ai/actions/runs/37484670973)
published source `4f77cb3ae430aad96c34705d7219069d7b7c4494` at
`sha256:d8cd706592a37b0cff91297ef0aafd543b274b0b795699d015d0d50c04584b17`.
Anonymous manifest/config requests and signed SLSA/SPDX attestations were
verified. An independent CPU execution used those exact bytes without installing
packages or changing the vendor environment:

| Check | Actual result |
| --- | --- |
| Installed dependency closure | Failed `pip check`: incompatible Torch, torchvision and Diffusers bounds; required W&B absent. |
| Native default camera decoding | Failed loading TorchCodec against Torch 2.12.1, including the installed FFmpeg 4 path's undefined `c10::MessageLogger` symbol. |
| ACT optimizer and saved policy | Passed: finite gradients, 62 changed parameter tensors, checkpoint state equivalence, saved processor reload and equal inference. |

The overall candidate failed native CPU qualification. Its publication and
narrow ACT result cannot establish default image acceptance, PushT training or
GPU compatibility.

Simply returning to upstream's bounds is insufficient. The ordinary source
security comparison rejected compatible Diffusers 0.35.2 for
[CVE-2026-44513](https://github.com/advisories/GHSA-98h9-4798-4q5v) and
[CVE-2026-45804](https://github.com/advisories/GHSA-7wx4-6vff-v64p), both fixed in
0.38.0, and compatible Torch 2.10.0 for
[CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), fixed in 2.13.0.
The Torch advisory is LOW; the source comparison rejects newly introduced
findings at every severity, while ordinary image publication's vulnerability
gate rejects fixed CRITICAL findings. Those are separate policies.

The repair builds the explicitly versioned NPA integration
`0.5.1+npa.secure1` from upstream commit
`1396b9fab7aecddd10006c33c47a487ffdcb54b4`. It verifies the complete upstream
wheel and source archive hashes, compares every packaged model/source file to
that commit, and preserves the Apache-2.0 license. Native model source remains
unchanged. Four reviewed dependency declarations change to Torch 2.13,
torchvision 0.28, TorchCodec 0.16 and Diffusers 0.38 ranges; required W&B remains.
The rebuilt wheel has its own version, refreshed RECORD and source-integration
receipt. Unknown artifacts, changed source or unexpected dependency declarations
fail before writing a wheel. This is an NPA-maintained integration, not a claim
that unmodified upstream 0.5.1 declares these versions supported.

The supported TorchCodec/Torch pairs are documented by
[TorchCodec upstream](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions),
and the original package bounds remain visible in
[LeRobot 0.5.1](https://github.com/huggingface/lerobot/blob/v0.5.1/pyproject.toml).
The optional 0.6 resolver, VM installer pins and additive B300 recipe retain
their separate scope; this integration does not qualify them.

The image must run `pip check` and a network-disabled CPU build gate before
publication. That gate writes and decodes real camera videos through the native
default backend, executes ACT and Diffusion forward/backward and optimizer
updates, verifies changed weights, and reloads checkpoints and ACT's saved
processors. No external
model, dataset, decoder patch or GPU is involved.

The maintained integration still requires fresh ordinary vulnerability, secret,
license, image and SDK gates. No security exemptions are introduced.
Public defaults remain quarantined; exact-source
rebuild and subsequent GPU qualification are pending.
