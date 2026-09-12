# Isaac Lab-Arena redistribution boundary

`npa-isaac-arena` is public-redistributable because its added application
payload is the Apache-2.0 Isaac Lab-Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd` plus its hash-locked open-source
Lightwheel SDK 1.0.3, Pinocchio, Pink, ONNX Runtime, and solver dependency
closure. The Lightwheel wheel carries the full Apache-2.0 grant in its package
metadata and its modules carry Apache-2.0 headers. The image
inherits the accepted payload-clean `npa-isaac-lab` digest and does not contain
Isaac Sim, Isaac Lab, Omniverse Kit, model weights, datasets, operator inputs,
generated evaluations, Lightwheel registry assets, credentials, or populated
runtime caches.

At first execution, NVIDIA delivers the pinned Isaac Sim/Lab wheels directly
to the operator's writable cache after the shared `ACCEPT_EULA` preflight. An
explicit negative value refuses before download. Replay data and RSL-RL
checkpoints are always operator-supplied runtime inputs and are never copied
into an image layer.

Some upstream environments resolve USD content from the Lightwheel registry at
runtime. Those registry objects are separate from the Apache-2.0 SDK. NPA does
not redistribute them, supply a Lightwheel credential, or grant rights to them;
the operator remains responsible for upstream authorization and terms. The
provider controls selector resolution and bytes at execution time, so this
public image makes no immutability or general availability claim for those
assets.

The upstream project labels 0.3.0 alpha and says its APIs are unstable and
incomplete. NPA supports the pinned policy-evaluation contract only; it does
not represent the upstream project as generally production-ready.
