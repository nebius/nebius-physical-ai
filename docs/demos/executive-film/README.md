# Intelligence that moves

A two-minute executive film about Nebius Physical AI Workbench: Cosmos synthetic
data, evaluation and curation, neural reconstruction, robotics, and Alpamayo.
The editable [storyboard](storyboard.json) contains narration and shot timings.
The renderer produces a 1920 × 1080, 30 fps H.264 MP4, an AAC narration and
original music mix, player-selectable SRT captions, a poster, and a hash manifest.

## Rebuild the film

Use Python 3.12 in this checkout's `npa/.venv`, with `npa[dev]` installed, and
FFmpeg with `libx264` plus `ffprobe` on `PATH`. The Manrope variable font and its
OFL license are bundled; [font provenance](fonts/source.json) pins the source.

Keep source media and all output outside Git. Create a private `assets.json`
mapping these editorial roles to files you are entitled to use:

| Role | Required media |
| --- | --- |
| `demonstration` | Real robot demonstration from a documented dataset workflow |
| `source` | Source video used by the chosen Cosmos augmentation |
| `cosmos_a` | Actual generated augmentation of that same source |
| `cosmos_showcase` | Actual Cosmos text-to-video output |
| `robot` | RoboCasa simulation rollout |
| `reconstruction` | NuRec rendered novel-view video |
| `depth` | Reconstructed distance video from the same scene |
| `driving` | Six-camera panel from the selected saved Alpamayo result |
| `driving_front` | Front camera projection from that exact result |
| `trajectory` | Trajectory figure from that exact result, including its legend |

Every entry has `path`, `kind` (`video` or `image`), and the complete `sha256`.
Paths may be absolute or relative to `assets.json`. Images can reference the
same original file with `crop: [x, y, width, height]` in source pixels. Crops
must fit inside the decoded image. Compute each digest with
`shasum -a 256 /path/to/source.mp4`, then place the actual digest in the manifest:

```json
{
  "source": {
    "path": "source-assets/source.mp4",
    "kind": "video",
    "sha256": "<complete SHA-256 of that file>"
  }
}
```

Populate **all ten roles** before running. Missing files, digest mismatches,
unsupported media, invalid crops, and an incorrect storyboard duration fail
before encoding. The renderer never synthesizes missing model outputs.

Generate narration once, or supply recorded narration and sentence captions as
`<scene-id>.mp3` and `<scene-id>.srt` for every scene. The optional generator uses
Microsoft Edge's online speech service through
[edge-tts](https://github.com/rany2/edge-tts); only storyboard narration is sent.
It defaults to `en-US-AndrewMultilingualNeural`; `--voice` selects another voice.
Archive the resulting audio to reproduce a delivery without depending on that
service's future voice behavior.

```bash
uv pip install --python npa/.venv/bin/python \
  -r docs/demos/executive-film/requirements.txt
npa/.venv/bin/python docs/demos/executive-film/narrate.py \
  --output-dir /path/to/private-film/narration

npa/.venv/bin/python docs/demos/executive-film/render.py \
  --assets /path/to/private-film/assets.json \
  --voice-dir /path/to/private-film/narration \
  --output-dir /path/to/private-film/render --check

npa/.venv/bin/python docs/demos/executive-film/render.py \
  --assets /path/to/private-film/assets.json \
  --voice-dir /path/to/private-film/narration \
  --output-dir /path/to/private-film/render
```

`--workers` controls concurrent scene encodes (default `2`). Narration has a
350 ms lead-in per scene and is modestly accelerated only when needed; a clip
requiring more than 15% acceleration fails with a request to shorten its text.
The original instrumental bed is deterministic and contains no sampled music.
The final check requires exactly 3,600 video frames and 120 seconds, then
decodes the complete file with FFmpeg's error-exit mode. Encoder/library
versions can change the resulting bytes; archive the output manifest and
environment alongside each delivery.

## Editorial provenance

This is a capabilities montage assembled from **separate recorded runs**. It
does not claim one continuous end-to-end execution, autonomous driving safety,
deployment readiness, training improvement, or a measured business return.
Source-conditioned augmentation is separate from the Cosmos Super generated
kitchen scene. RoboCasa footage is a simulation rollout. Alpamayo is a saved
inference visualization; the presentation zoom is not continuous driving video.
Short clips may loop to fill an editorial shot. Preserve these distinctions
when replacing media or changing narration.

The quoted driving rationale must match `trajectory.json` for all three
Alpamayo panels. The supplied storyboard uses: “Nudge left to avoid the cones
on the right side.” Update the quote, narration, and panels together when
selecting another result. Keep predicted-motion and ground-truth legends intact.

Keep exact S3 locations, raw reports, infrastructure identifiers, and access
receipts in private evidence. The public renderer manifest contains only media
roles, hashes, crop coordinates, and output properties. Media availability is
not evidence of distribution rights. In particular, PhysicalAI-AV camera
imagery remains governed by its separate dataset terms: retain that cut for
authorized internal review and resolve rights before external distribution.
Neither run footage nor the finished internal cut belongs in this public PR.

## Fresh capture through NPA

Use an isolated checkout, virtualenv, and private `NPA_CONFIG_DIR`. Discover
the tenant's actual capacity block groups and write their identifiers only into
a private Fleet spec. Use NPA's [Fleet procedure](../../../skills/tools/fleet/SKILL.md)
to create a dedicated project with a B200 inference cluster and an RTX rendering
cluster. Bind reservations explicitly with `capacity_block_group`; use
`gpu_workload_profile: rtx-rendering` for the RTX cluster. Run the existing
[Alpamayo workflow](../../../workflows/testing/alpamayo2-super-inference.yaml)
and [Cosmos workflow](../../../workflows/testing/cosmos3-generate.yaml) after
credential, model-access, image-pull, storage, and GPU readiness checks.

```bash
npa/.venv/bin/python -m npa workbench health preflight --checks nebius,hf,ngc --json
npa/.venv/bin/python -m npa workbench health access \
  --capability cosmos3,alpamayo2-super --json
npa/.venv/bin/python -m npa fleet plan --spec /path/to/private-fleet.yaml
npa/.venv/bin/python -m npa fleet deploy --spec /path/to/private-fleet.yaml --yes
```

GPU reservations do not supply boot-disk quota. A quota refusal precedes
provisioning; increase the affected tenant allowance, release only resources
you own, or revise the declared requirements, then rerun with preflight enabled.
Do not present the plan or a failed preflight as a completed GPU capture.
The film can be edited from existing qualified artifacts while fresh capture
is blocked. Cancel owned jobs before tearing down their owned clusters.

The model and platform descriptions are grounded in the repository tool docs,
the [Cosmos framework](https://github.com/NVIDIA/cosmos-framework), the
[Alpamayo model card](https://huggingface.co/nvidia/Alpamayo2-Super), and
[Nebius Physical AI](https://nebius.com/solutions/physical-ai-and-robotics).
