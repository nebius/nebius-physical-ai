# Intelligence that moves

A two-minute executive film about Nebius Physical AI Workbench: Cosmos synthetic
data, Wan video generation, FiftyOne curation, RoboCasa simulation, LeRobot
policy experiments, neural reconstruction, Alpamayo, and Rerun inspection.
SkyPilot runs the GPU workloads behind the NPA workflow interface.
The editable [storyboard](storyboard.json) contains narration and shot timings.
The renderer produces a 1920 × 1080, 30 fps H.264 MP4, an AAC narration and
original music mix, player-selectable SRT captions, a poster, and a hash manifest.

## Fast editing

Use a private project directory for the editable storyboard and media. Keep the
delivered master separately so previews cannot overwrite it. A `film-project.json`
file gives the editor four paths; relative paths resolve against that file:

```json
{
  "storyboard": "storyboard.json",
  "assets": "assets.json",
  "voice_dir": "narration",
  "output_dir": "renders",
  "voice": "en-US-AndrewMultilingualNeural"
}
```

The four paths are required. `voice` is optional and defaults to the voice shown.
Copy the bundled storyboard into this private project, supply the asset manifest
described below, and archive or generate its narration. Then use one entry point:

```bash
npa/.venv/bin/python docs/demos/executive-film/edit.py --project /path/to/film-project.json scenes
npa/.venv/bin/python docs/demos/executive-film/edit.py --project /path/to/film-project.json preview
npa/.venv/bin/python docs/demos/executive-film/edit.py --project /path/to/film-project.json preview --scene 07-lerobot --open
npa/.venv/bin/python docs/demos/executive-film/edit.py --project /path/to/film-project.json final
```

Edit headlines, labels, narration, timing, and scene order in `storyboard.json`.
Change clips and their hashes/crops in `assets.json`; shared typography and colors
live in `graphics.py`. A role can be used by multiple scenes; create a new role
and reference it in just one scene when only that occurrence should change.
The complete storyboard must remain 120 seconds, with
positive whole-second durations and unique lowercase scene IDs. Scene previews
can be shorter and retain the original chapter number, music position, and
aligned captions. `scenes` lists IDs, titles, and start times. A selected scene
only requires its own visual assets, so review footage can be edited while
another GPU output is still pending; the full narration remains required for
the continuous score and mix. Full-film renders require every referenced asset.
Use the `screen` layout with one headline for a wide application capture; its
media area is 1776 × 624 pixels beneath the title and subtitle.

`preview` composes at 960 × 540 and uses a fast H.264 preset. `final` preserves
1920 × 1080 at 30 fps with the delivery encoding settings. Both reuse unchanged
scenes, normalized narration clips, and the audio mix. A title change rebuilds its
scene; changing only narration leaves visual scenes cached. Scene reuse checks
content hashes, selected assets/crops, chapter position, font, renderer source,
and the FFmpeg/Python/Pillow/NumPy versions. A changed global graphics module
invalidates all affected visual caches. Cached output bytes are verified before
reuse. Failed builds do not publish a partial replacement.

Outputs go into `renders/preview/` or `renders/final/`; a selected scene uses its
own subdirectory, such as `renders/preview/07-lerobot/`. Each contains the video,
captions, poster, provenance, and `render-timing.json` with actual encode/reuse
counts and elapsed time. `render-storyboard.json` preserves the exact storyboard
snapshot used by that output; its bytes match the manifest's storyboard hash.
Each output also includes `watch.html`, an offline browser player with chapter
buttons and selectable English captions generated from that snapshot and its
assembled SRT. Open it directly from disk beside the MP4 and poster; it needs
no server or network access. Scene previews get one chapter starting at zero.
Changing renderer code during a render aborts publication and asks for a rerun.
`--plan` lists visual cache hits without encoding;
`--workers` controls concurrent scene encodes (default `2`); `--open` opens a
completed render in the default macOS/Linux player. The private `renders/.cache`
retains prior variants for reuse when an edit is reverted. Renders sharing a
project wait on a local lock rather than publishing over each other.

After changing spoken wording, run `edit.py ... narrate`: it regenerates only
changed or missing speech clips. Visual edits need no speech-service access.
`narrate --force` regenerates everything. For human-recorded or otherwise supplied
MP3/SRT pairs, `narrate --recorded` registers their identity without network access.
Unchanged registered recordings remain reusable even when a generated voice is
configured for other scenes.
Rendering rejects stale narration receipts so a text edit cannot silently ship
the old spoken script. The speech generator records each completed scene before
moving on, so interrupted generation resumes from its completed recordings.

Measured on the expanded two-minute film on one macOS workstation, including command
startup and validation, with the source clips and narration already local:

| Edit | Wall time |
| --- | ---: |
| Unchanged final | 1.5 seconds |
| Changed ten-second scene, 540p preview | 6.4 seconds |
| Complete 1080p film after one scene's subtitle change | 13.8 seconds |
| Reverted final, reusing the preceding version | 1.2 seconds |

These measurements exclude fetching media and running models. Source codecs,
hardware, global graphics changes, and newly generated narration affect timing.
The same exercise verified edit/revert cache reuse, generated only one changed
speech clip while reusing eleven, and kept the preceding preview intact when
renderer source changed during an encode.

## Rebuild the film

Use macOS or Linux with Python 3.12 in this checkout's `npa/.venv`, with `npa[dev]` installed, and
FFmpeg with `libx264` plus `ffprobe` on `PATH`. The Manrope variable font and its
OFL license are bundled; [font provenance](fonts/source.json) pins the source.

Keep source media and all output outside Git. Create a private `assets.json`
mapping these editorial roles to files you are entitled to use:

| Role | Required media |
| --- | --- |
| `source` | Source video used by the chosen Cosmos augmentation |
| `cosmos_a` | Actual generated augmentation of that same source |
| `cosmos_showcase` | Actual Cosmos text-to-video output |
| `wan` | Wan 2.2 warehouse video generated by the documented multi-GPU NPA workflow |
| `wan_mobility` | A second Wan 2.2 video generated by the single-GPU NPA workflow |
| `fiftyone` | Actual FiftyOne application recording of saved review data |
| `robot` | RoboCasa native simulation rollout |
| `robot_training` | Training-dataset video for the selected LeRobot run |
| `robot_eval` | Held-out ACT evaluation video from that LeRobot run |
| `reconstruction` | NuRec rendered novel-view video |
| `depth` | Reconstructed distance video from the same scene |
| `driving` | Six-camera panel from the selected saved Alpamayo result |
| `driving_front` | Front camera projection from that exact result |
| `rerun` | Actual Rerun viewer recording of the verified Wan run artifact |

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

The bundled storyboard requires **all fourteen roles**. An edited storyboard can use
additional or different roles, provided its referenced entries exist. Missing files, digest mismatches,
unsupported media, invalid crops, and an incorrect storyboard duration fail
before encoding. The renderer never synthesizes missing model outputs.

Generate narration once, or supply and register recorded narration and sentence
captions as `<scene-id>.mp3` and `<scene-id>.srt` for every scene. The optional generator uses
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

The direct renderer defaults to the bundled storyboard and the `final` profile;
use `--storyboard` for an edited copy, `--profile preview` for a smaller render,
and `--scene` for one shot. The complete film appears under `render/final/`.
`--check` verifies source assets without encoding, and `--plan` reports which
selected visual scenes need work. Narration has a
350 ms lead-in per scene and is modestly accelerated only when needed; a clip
requiring more than 15% acceleration fails with a request to shorten its text.
The original instrumental bed is deterministic and contains no sampled music.
The complete-film check requires exactly 3,600 video frames and 120 seconds, then
decodes the complete file with FFmpeg's error-exit mode. Encoder/library
versions can change the resulting bytes; archive the output manifest and
environment alongside each delivery.

## Editorial provenance

This is a capabilities montage assembled from **separate recorded runs**. It
does not claim one continuous end-to-end execution, autonomous driving safety,
deployment readiness, training improvement, or a measured business return.
Source-conditioned augmentation is separate from the Cosmos Super generated
kitchen scene. Wan is prompt-to-video generation. RoboCasa footage is a native
simulation rollout, not an expert demonstration. The LeRobot sequence shows
training data and a held-out ACT evaluation; it does not assert task success.
FiftyOne shows saved candidate decisions, including rejected candidates. Rerun
shows the actual generated video and its recorded settings. Alpamayo is a saved
inference visualization; the presentation zoom is not continuous driving video.
Short clips may loop to fill an editorial shot. Preserve these distinctions
when replacing media or changing narration.

The quoted driving rationale must match `trajectory.json` for every
Alpamayo panel. The supplied storyboard uses: “Nudge left to avoid the cones
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
