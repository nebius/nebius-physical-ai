# LeRobot FastWAM policy qualification

[`fastwam-policy-qualification.yaml`](../../../workflows/testing/fastwam-policy-qualification.yaml)
is a five-stage native LeRobot workflow for upstream FastWAM, not the separate
Cosmos3 FastWAM-K2 experiment.

FastWAM trains a video world/action model. At inference the upstream policy
calls `FastWAMPolicy.select_action`, which directly predicts an action chunk;
the workflow does not generate a future video at test time. The pipeline is:

1. validate the supplied LeRobot v3 robot dataset through FastWAM's native
   vision/state/action/text contract and seal an episode-disjoint training/
   held-out receipt;
2. run upstream `lerobot-train --policy.type=fastwam` on only training episodes;
3. run upstream `lerobot-eval` against the exact output checkpoint;
4. read the native evaluation and measure CUDA `select_action` latency on
   held-out observation frames; and
5. decode the real rollout MP4s and write/read-back a Rerun `.rrd` report.

The supplied image must be an operator-built immutable LeRobot 0.6.0 image from
the checked-in Dockerfile, whose 0.6 extra set includes `fastwam`. The image has
no model weights, datasets, credentials, or output artifacts. The stage fetches
the FastWAM base checkpoint, Wan 2.2 components, and UMT5 at the exact revisions
in the workflow into the configured Hugging Face cache, which provides the Hub's
revision-aware locking and immutable snapshot layout. Re-run an access probe if
any source revision changes.

All `REQUIRED_*` values are deliberate submit-time inputs: the dataset identity,
license, simulator adapter and task, and image digest must name real
operator-selected artifacts. The rollout task's episode length, image geometry,
batch size, policy dtype, and action-chunk count are passed directly to
`lerobot-eval`; they must match the selected simulator and model observation
contract. `libero` may be supplied as
an evaluation environment only when that contract is actually compatible. Such
a run is simulator evidence, not a physical-robot claim. Physical deployment
requires a separately reviewed robot adapter, safety controls, and an actual
physical success study.

The FastWAM base, Wan 2.2, Wan Diffusers and UMT5 model cards currently declare
Apache-2.0, but that does not create a license for an operator-provided robot
dataset or a new public checkpoint distribution. The authoritative
software/model attribution, licenses, modifications and runtime-fetch boundary are in
[`NOTICE-FASTWAM`](../../../npa/docker/workbench/lerobot/notices/NOTICE-FASTWAM).
Dataset, cache, output, and service terms must be recorded separately from that
notice in the run provenance; no NPA EULA or acceptance variable is introduced.
