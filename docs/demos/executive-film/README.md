# Workbench film studio

An editable, cached film renderer with an example two-minute executive showcase
about Nebius Physical AI Workbench: Cosmos synthetic
data, Wan video generation, FiftyOne curation, RoboCasa simulation, LeRobot
policy experiments, neural reconstruction, Alpamayo, and Rerun inspection.
SkyPilot runs the GPU workloads behind the NPA workflow interface.
The editable [storyboard](storyboard.json) contains narration and shot timings.
The renderer produces a 1920 × 1080, 30 fps H.264 MP4, an AAC narration and
original music mix, player-selectable SRT captions, a poster, and a hash manifest.
Duration, audience and narrative are project choices. The included
[60-second technical walkthrough](storyboard-technical.json) uses the same
renderer and a subset of the same assets with a different script and scene order.

## What drives the narrative

A creative brief describes the request; an editor or coding agent turns it into
the storyboard. The renderer follows that storyboard deterministically. It does
not call a language model or interpret a new prompt during encoding. The
`brief` command prepares an authoring packet containing the request, available
asset identities and provenance, layout contracts, and evidence requirements:

```bash
npa/.venv/bin/python docs/demos/executive-film/edit.py \
  --project /path/to/film-project.json brief \
  --prompt "Explain how to inspect and reproduce a Workbench experiment" \
  --audience "ML engineers" --goal "Reproduce one workflow" \
  --tone "Practical, precise, candid about limitations" --duration 60 \
  --cta "github.com/nebius/nebius-physical-ai" > /path/to/authoring-packet.json
```

Give that packet to the coding agent or editor to write the project storyboard,
including the packet's complete `brief` object. Then run `narrate`, `preview`,
and `final`. The command prints JSON and does not change the existing project or
contact a model, speech service or cloud API. Only narration generation contacts
the configured speech service. Inspect and reproduce the chosen assets before
making claims about their origin or results.

| Brief | Narrative choices |
| --- | --- |
| Executive showcase | Lead with the opportunity, show a few capabilities, connect them to a decision, invite a pilot. |
| Technical walkthrough | Establish inputs, explain steps, inspect outputs and limitations, show how to reproduce the work. |
| Short announcement | Pick one idea and one supporting result, then close with one action. |

These are examples, not hardcoded modes. `--prompt` is free-form; `--audience`,
`--goal`, `--tone`, `--duration` and `--cta` make its constraints explicit. With
no new prompt, `brief` reuses the saved storyboard brief and applies any supplied
overrides. A new prompt starts fresh audience, goal and tone guidance; omitted
duration uses the current timeline, and omitted CTA uses the repository URL.
The author chooses scene count, order, footage, headlines, narration and timing;
changing a brief alone does not rewrite those choices.

The optional `storyboard.brief` has nonempty `prompt`, `audience`, `goal`, `tone`,
and `call_to_action` strings plus positive integer `duration_seconds`. Its duration
must match the full timeline. The output manifest preserves the brief and its
content hash; review is still needed to establish that the narrative satisfies
it. Legacy storyboards without a brief continue to render. For a scene preview,
the snapshot keeps the original full-film brief as `source_brief`.

Asset records may include a `provenance` object with the originating tool,
artifact role (input/output/reference), evidence references, description and
limitations. These fields travel into the authoring packet and output manifest.
Without them, the packet marks the asset `unattributed`; a filename or role name
does not establish which tool generated it. Keep private evidence in the private
project and consult the artifact catalog when selecting additional footage.

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
Use any positive whole-second timeline, with positive whole-second scene
durations and unique lowercase scene IDs. If top-level `duration` is present,
it must equal the sum of scene durations. A 30-second cut, 60-second walkthrough,
or three-minute demo needs an appropriately paced script, not just a changed
number or a sped-up version of the old film. Scene previews
can be shorter and retain the original chapter number, music position, and
aligned captions. `scenes` lists IDs, titles, and start times. A selected scene
only requires its own visual assets, so review footage can be edited while
another GPU output is still pending; the full narration remains required for
the continuous score and mix. Full-film renders require every referenced asset.
Use the `screen` layout with one headline for a wide application capture; its
media area is 1776 × 624 pixels beneath the title and subtitle.
Storyboard `footer` sets the shared footer (default: the film title); a scene's
`footer` overrides it. Scene `cta` supplies closing text. Optional
`quote_heading`, `quote`, `source_notes` (up to two), `evidence_note`,
`review_steps` (up to three), and `pipeline_labels` (up to three) control the
corresponding layout's callouts. They default to empty. The supplied executive
storyboard explicitly carries its slogans, attribution notes and repository CTA;
these claims are not injected into other narratives by the graphics code.

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
The complete-film check requires the storyboard's total seconds and exactly
30 times that total in video frames (3,600 for the 120-second example), then
decodes the complete file with FFmpeg's error-exit mode. Encoder/library
versions can change the resulting bytes; archive the output manifest and
environment alongside each delivery.

The logo uses the unmodified color artwork from the [official Nebius media
kit](https://nebius.com/media-kit), scaled proportionally. Its original PNG and
SVG, source archive URL, hashes, and trademark notice are in [brand/](brand/).
The PNG participates in scene cache identity and the render's source-change
check. Output manifests record the logo and its source receipt hashes. The
closing call to action points to `github.com/nebius/nebius-physical-ai`.

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
