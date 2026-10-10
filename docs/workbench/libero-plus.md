# LIBERO-Plus robustness benchmark

This workflow targets the upstream [LIBERO-Plus repository](https://github.com/sylvestf/LIBERO-plus) at `4976dc30028e805ff8094b55501d532c48fec182` and its [asset dataset](https://huggingface.co/datasets/Sylvest/LIBERO-plus) at `dd2bd61b7d9a6fef1abc52d606e983b41886a149`. It preserves credit to Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. The upstream citation is:

```bibtex
@article{fei25libero-plus,
  title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
  author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
  journal={arXiv preprint arXiv:2510.13626},
  year={2025},
}
```

The upstream GitHub tree has no `LICENSE`, `NOTICE`, or `COPYING` file at this revision. NPA therefore does not package, publish, or execute its source or derived assets. This is a concrete source-use and redistribution uncertainty, not an NPA consent gate: the smallest resolution is an explicit upstream license or written authorization covering the intended benchmark execution. The Hugging Face asset card declares MIT; its `assets.zip` is public, 6,395,849,578 bytes, and pinned by SHA-256 `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`. Source and asset terms remain distinct; neither access nor the card’s MIT declaration authorizes use or redistribution of the separate source tree.

The runtime enforces that conclusion fail-closed: while the reviewed pinned-source record remains unresolved, it raises before cloning source or downloading assets. This uses no operator acceptance variable or local legal attestation. A future upstream license declaration or authorization must be reviewed and recorded in code before an execution image can run this workflow.

The separate licensed-assets image is deliberately `unvalidated` and private: it
adds the tested SkyPilot worker bootstrap (non-root user, ephemeral SSH host
keys, `rsync`, and argument forwarding) to the original-MIT-LIBERO runtime, but
does not make the private derivative publicly distributable. Each immutable
private digest still needs complete byte scanning, exact-digest pull/bootstrap
verification, and a real RTX PRO 6000 artifact run before this limited camera
capability can be considered qualified.

`workflows/testing/libero-plus-robustness.yaml` has five substantive connected stages: task preparation, baseline rollout, candidate rollout on exactly the same protocol and seed, numerical per-dimension deltas, and task-level reporting. The seven native categories are Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, and Sensor Noise. Each rollout executes the upstream `OffScreenRenderEnv` against the classified upstream perturbation task rather than generating a proxy task.

The checked-in `smoke` configuration selects one real task per perturbation category. Its `smoke-zero` adapter is deliberately a simulator/IO operational smoke and writes `smoke_only: true`; it is neither an official baseline nor a performance result. A `benchmark` run selects all 10,030 upstream classification records and rejects `smoke-zero`. It needs two actual non-smoke `module:function` policy adapters in an immutable operator image, one for baseline and one for candidate. Such a full run remains distinct from training convergence and physical-robot success.

All result artifacts are run-scoped S3 objects. The implementation records native per-episode camera MP4s and independently decodable RRD metrics when an authorized runtime executes it. A full accepted run must have both artifacts before it can be described as live-ready. No extra NPA EULA flag, consent environment variable, telemetry consent, or duplicate per-image attestation is introduced.

The checked-in workflow uses an intentionally invalid example registry digest for offline validation. A future operator run must replace it with a private, immutable, independently inspected bootstrap image; no unverified image is published by this onboarding. Local contract tests validate the five-stage graph and independently decode a generated RRD, but they are not upstream simulator, benchmark, convergence, image, or physical-robot evidence.

## Licensed-assets camera compatibility profile

The separately MIT-labelled `Sylvest/LIBERO-plus` asset archive supports one
strictly smaller capability without resolving the source boundary above:
[`workflows/testing/libero-plus-licensed-assets-camera-compatibility.yaml`](../../workflows/testing/libero-plus-licensed-assets-camera-compatibility.yaml).
It uses the author-published asset revision
`dd2bd61b7d9a6fef1abc52d606e983b41886a149`, downloads and verifies the full
`assets.zip` SHA-256
`96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`, and
emits only allowlisted `assets/scenes/libero_tabletop_base_style.xml` (SHA-256
`5e69f8568bedf4a71641fcb62285182d0f6dbe498ea18adad86a13706558033f`). The
6.4 GB archive remains in an immutable revision-scoped operator cache; it is
never baked into the image or copied to result storage.

The five connected stages are (1) acquire/hash/attribute that asset, (2) compile
and `mj_forward` it through original
[`Lifelong-Robot-Learning/LIBERO`](https://github.com/Lifelong-Robot-Learning/LIBERO)
revision `8f1084e3132a39270c3a13ebe37270a43ece2a01` under MIT, (3) render
`agentview` and `agentview_60` from the same native scene state with EGL, (4)
independently decode both PNGs and calculate camera-pose plus RGB-difference
metrics, and (5) emit a gallery manifest, report, and decodable RRD. The native
executor is explicitly attributed; its `LICENSE` SHA-256 is
`e2885fd30a08381b799c4a33385522b23d637b4051b8f9a7f9f2519944b68ff6`.

Private preliminary evidence with that exact original-MIT executor compiled the
selected scene and passed `mj_forward` on MuJoCo 3.3.2 (37 cameras, 13 geoms,
two lights); actual 256×256 EGL views had mean absolute RGB difference
`52.54959615071615`. That evidence establishes only native scene/camera
compatibility. It does **not** establish a 10,030-task LIBERO-Plus result, BDDL
or randomizer parity, policy action, robustness delta, benchmark convergence, or
physical-robot success. The profile never downloads, copies, imports, or
executes the unlicensed `sylvestf/LIBERO-plus` source, BDDL/task definitions, or
benchmark randomizer.

`npa/docker/workbench/libero-plus-assets/Dockerfile.private` is an
operator-private derivative of the already inspected OpenWAM/original-LIBERO
runtime. Its build helper requires an operator-private base, verifies the native
MIT license identity and an independently observed local base-image ID, produces provenance/SBOM plus an OCI scan, refuses public
registry targets, and leaves the asset archive runtime-fetch-only. It adds no
EULA, `ACCEPT_*` flag, credential prerequisite, telemetry setting, or public
publication path. A successful future run is still reported as the narrow
camera-compatibility capability, never as full LIBERO-Plus acceptance.

## Source-admission image

`npa/docker/workbench/libero-plus/Dockerfile.admission` is an intentionally
limited private-qualification image recipe. It contains only the NPA adapter,
the digest-pinned Python/Debian bootstrap, and attribution notices; it contains
no LIBERO-Plus source, asset archive, model, simulator, renderer, cache, output,
or credential. Its only permitted live check is that `prepare` refuses before a
source clone or asset fetch while the reviewed source record is unresolved.

`build-private.sh` produces an OCI archive, scans it, loads a local private tag,
and writes a private receipt. It refuses official/public registry targets and
does not push. This source-admission check is useful image evidence, but it is
not a successful workflow stage or a benchmark result. A separate authorized
source grant (or an independently proven, licensed equivalent that is not
relabeled as LIBERO-Plus) is still required before a source-capable image or
the five-stage native path can be qualified.
