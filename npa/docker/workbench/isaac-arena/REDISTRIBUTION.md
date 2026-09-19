# Isaac Lab-Arena redistribution boundary

`npa-isaac-arena` is public-redistributable because its added application
payload is the Apache-2.0 Isaac Lab-Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd` plus its hash-locked open-source
Lightwheel SDK 1.0.3, Pinocchio, Pink, ONNX Runtime, and solver dependency
closure. The exact Lightwheel wheel's package description includes an
Apache-2.0 license notice and URL; several client modules repeat that notice.
The wheel has no standalone license file or structured license metadata field.
Its reviewed wheel and `METADATA` hashes are recorded in
`license-evidence.json`, which the image build verifies against the installed
distribution bytes. PyPI publishes no source distribution or stronger
standalone license artifact for 1.0.3, so a different wheel, metadata hash, or
license-file shape fails closed for renewed review.
Recipients receive the full Apache-2.0 text at `/opt/isaac-arena/LICENSE.md`,
alongside the retained Lightwheel copyright and license notices. The image
inherits the accepted payload-clean `npa-isaac-lab` digest and does not contain
Isaac Sim, Isaac Lab, or Omniverse Kit runtime payloads, Arena evaluation policy
checkpoints or replay datasets, operator inputs, generated evaluations,
Lightwheel registry objects, credentials, or populated runtime caches.

The inherited open-source dependency distributions retain packaged examples
and test fixtures, including Newton 1.2.1 example assets and its sample policy,
and ONNX 1.21.0 conformance models. Their distribution metadata declares
Apache-2.0, and their full license texts and bundled notices remain installed.
These dependency fixtures are not Arena evaluation inputs or task-success
evidence and do not grant rights to provider-controlled assets. See
`THIRD_PARTY_NOTICES.md` for their installed license locations.

NPA applies source-visible integration changes to the pinned Arena runner,
viewport recorder, revolute metric, and embodiment setup. The runner restores
the replay's recorded initial state and stores simulator metric traces with
the run. Viewport capture retains action-bound frames before automatic reset,
and the revolute trace includes its pre-action value. Only unused embodiment
camera observations are disabled; the real upstream renderer, task, policy,
and success calculation remain in use. The build requires exact upstream
contexts and removes the patch helper afterward. Modified files retain their
upstream notices and identify the NPA integration changes.

At first execution, NVIDIA delivers the pinned Isaac Sim/Lab wheels directly
to the operator's writable cache after the shared `ACCEPT_EULA` preflight. An
explicit negative value refuses before download. Replay data and RSL-RL
checkpoints are always operator-supplied runtime inputs and are never copied
into an image layer.
The Isaac Lab wheel itself declares BSD-3-Clause; Isaac Sim and its proprietary
runtime dependencies retain their separate NVIDIA terms. Runtime-fetch delivery
does not imply that all fetched components have the same license.

Viewport recording also validates usable NVIDIA headless EGL, Vulkan, and
`libnvoptix.so.1`, plus readable regular nonempty OptiX weights at
`/usr/share/nvidia/nvoptix.bin`. If a target omits these dependencies, NPA
downloads only the `libnvidia-gl-<branch>-server`
package whose version exactly matches the loaded kernel driver from Ubuntu's
signed archive. It validates package identity, architecture, and ICD metadata,
extracts the package into run-private scratch, and derives a canonical private
EGL ICD for only that simulator process. The image creates an empty weights
directory owned by the non-root runtime user. Missing weights are copied only
into that verified private container root overlay, with no symlink or submount
destination, no overwrite, and hash-verified readback. Existing different bytes
are rejected. Copied weights remain for the worker container lifetime only.
NPA never installs these bytes on the node, bakes them into an image, publishes
them as run artifacts, or redistributes them. Library, file, and settings
readiness do not establish successful denoising or task qualification.
The package remains governed by NVIDIA's driver terms accepted by the operator.

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
