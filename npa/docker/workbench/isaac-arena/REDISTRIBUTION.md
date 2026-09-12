# Isaac Lab-Arena redistribution boundary

`npa-isaac-arena` is public-redistributable because its added application
payload is the Apache-2.0 Isaac Lab-Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd`. The image inherits the accepted
payload-clean `npa-isaac-lab` digest and does not contain Isaac Sim, Isaac Lab,
Omniverse Kit, model weights, datasets, operator inputs, generated evaluations,
credentials, or populated runtime caches.

At first execution, NVIDIA delivers the pinned Isaac Sim/Lab wheels directly
to the operator's writable cache after the shared `ACCEPT_EULA` preflight. An
explicit negative value refuses before download. Replay data and RSL-RL
checkpoints are always operator-supplied runtime inputs and are never copied
into an image layer.

The upstream project labels 0.3.0 alpha and says its APIs are unstable and
incomplete. NPA supports the pinned policy-evaluation contract only; it does
not represent the upstream project as generally production-ready.
