# Workbench film studio

Use `npa studio` to select, edit and render independent film projects. Studio
loads local storyboards, media and narration; its configuration has no tenant,
project ID, region, bucket or credential fields. GPU runs and hosted inference
use the operator's separate NPA configuration. Editing a title or rendering a
film makes no infrastructure or model calls.

```bash
# From a checkout containing this change, with FFmpeg installed:
npa/.venv/bin/python -m pip install -e npa -r docs/demos/executive-film/requirements.txt
npa/.venv/bin/python -m npa studio init \
  --directory ./my-studio --renderer docs/demos/executive-film
```

Initialization copies the shared renderer, fonts and official brand assets into
the new directory. Add project aliases to `studio.json`, using the project
configuration described below. No operator configuration is copied. The
generated `studio` launcher uses `NPA_STUDIO_PYTHON` when explicitly set, then
`.venv/bin/python` inside the studio if available, then `npa` on `PATH`. It
contains no machine-specific interpreter path. Recreate virtual environments
after moving to another machine; the renderer and relative project paths move
with the studio.

```json
{
  "renderer": "renderer",
  "projects": {
    "exec": "projects/exec/film-project.json",
    "inference": "projects/inference/film-project.json"
  }
}
```

```bash
cd my-studio
./studio list
./studio inference draft --scene 01-before-reality --open
./studio inference watch --scene 01-before-reality
./studio inference narrate
./studio inference final --open
```

`draft` and `watch` render only the selected scene without narration. Cached
media and speech let an editor revise a title without repeating inference or
regenerating speech. Use `brief` to change the audience, request or duration,
then author the corresponding storyboard. Both executive and inference films
share this workflow; two minutes is an example, not a renderer limit.

## Film presentation

Use `layout: "film"` for a full-frame shot with up to two restrained headline
lines. This layout omits slide framing, scene counters and the progress bar.
Its label appears in a compact badge at the top right, so viewers can identify
the model or tool while the footage fills the screen. Keep full tool attribution
and run evidence in `assets.json`.

Film scenes accept `title_position: "bottom-left"` (default) or `"center"`, and
`brand: true` to show the bundled official Nebius logo (default false). An empty
`title` array and empty `subtitle` leave the footage clear. Supply one asset and
one label entry; use an empty label to omit the badge. The badge remains visible
independently of headline timing.

Set `transition: "cut"` for a direct cut with no fade through black. Omit it or
use `"fade"` to retain the 0.3-second fade at each end of the shot. Short scenes
with direct cuts work well for an energetic montage; scene duration and title
timing remain independent choices.

`title_delay_seconds` defaults to 0.5. `title_duration_seconds` defaults to the
rest of the scene. Both accept finite seconds; titles must fit within the scene
and display for at least 0.5 seconds, including their fades. For example, a
10-second scene with a 1-second delay and a 4-second title duration gives the
footage five clear seconds after the title fades. These settings are independent
of narration, film duration and the production tool that generated the footage.

Set optional `music_path` in `film-project.json` to a local instrumental score,
resolved relative to the project file. The score must cover the complete film;
Studio normalizes it underneath narration and includes its content hash in the
audio cache. Replacing the music rebuilds the mix while retaining cached visual
scenes. Omit this field to use the original synthesized default bed. Supply music
you are authorized to use, and record its origin with the project evidence.

## Infrastructure closing diagram

Use `layout: "architecture"` with empty `assets` and `labels` arrays to explain
where a workflow runs without reusing footage. The diagram supports one to three
compute cards feeding a storage card. All wording comes from the scene:

```json
{
  "controller": "Workbench connects the workflow",
  "compute_nodes": [
    {"title": "Token Factory", "purpose": "Hosted inference", "models": ["Vision evaluation"]},
    {"title": "GPU compute", "purpose": "Model inference", "models": ["World models"]},
    {"title": "RTX", "purpose": "Simulation + rendering", "models": ["MuJoCo", "Isaac Lab"]}
  ],
  "storage_node": {"title": "Object Storage", "detail": "Video files · Datasets · Results"},
  "architecture_note": "Reference deployment; consult artifact provenance for actual run hardware."
}
```

Keep card titles short and supply up to three model labels per card. The normal
scene title, subtitle, narration and duration remain editable. A reference
architecture describes possible placement; its labels do not rewrite the GPU or
model identity in an artifact's provenance. Draft previews need no credentials,
media generation or hosted inference calls.

## Find footage in object storage

`npa studio search` discovers assets before a film project exists. It uses the
operator's external NPA configuration and credentials, follows every listing
page, and keeps unknown formats visible as downloads. It does not provision,
upload, download media, or run a background watcher.

```bash
npa studio search --all-projects --query cosmos --kind video
npa studio search --project <your-project-alias> --discover-tenant --kind video
npa studio search --project <your-project-alias> --bucket <your-bucket> \
  --prefix runs/ --since 2026-01-01T00:00:00Z --read-metadata
```

By default, search enumerates buckets visible to the selected S3 credentials;
`--all-projects` repeats this with each configured project's own credentials.
`--discover-tenant` instead uses the selected project's tenant and the configured
Nebius CLI profile to inventory projects and buckets. Known project aliases use
their own storage credentials; other discovered buckets are probed with the
selected credential context. Resource discovery never grants object access.
Regional endpoints come from the discovered project region. This mode selects
one project context and cannot be combined with `--all-projects`.

Some credentials can list objects but cannot enumerate buckets. Search still
checks the configured bucket and reports partial discovery. An explicit
`--bucket` searches that scope without requiring bucket enumeration. JSON output
is the default; `--output-format text` prints a compact inventory. Exit code 1
means discovery or listing was incomplete, with successful results retained and
per-source error codes. A completed listing does not prove every object can be
downloaded. `--query` matches object keys, case-insensitively.

`--read-metadata` issues HEAD requests only for matching objects. It reports
declared tool, model, run ID, GPU and SHA-256 fields from S3 metadata, along with
their evidence basis. Missing attribution stays unknown; a filename or ETag is
never treated as proof of a generating tool or content hash. Unreadable metadata
is reported separately from listing coverage. Verify run manifests and actual
media before making claims in a film. Keep search JSON private: it includes
source endpoints, project identities, bucket names and object keys, but no
credentials or signed URLs. None of these values belongs in `studio.json`.

## Isaac Arena inference film

[Before the real world](storyboard-inference-arena.json) is a separate 120-second
executive story with 15 scenes. It uses the authentic Isaac Lab–Arena replay
qualified in [PR #463](https://github.com/nebius/nebius-physical-ai/pull/463),
followed by the independent MuJoCo planning experiment below. The new
`immersive` layout gives simulation the full canvas with up to two headline
lines over a lower gradient. Existing layouts remain available to both films.

The Arena source MP4 is bound to its replay input and has SHA-256
`a0adc96dd93d6f3fab01e096944127ca5fb38ace192c49ae2dac9550b70b4c7a`.
It is an RTX PRO 6000 replay evaluation, not a newly trained policy. The joint
moves, but the episode records task failure. Editorial crops, denoising,
slow playback and final-frame holds are retained in derivation records; VLM
evaluation uses the original footage. Arena remains upstream alpha.

Three real `npa workbench vlm-eval run --backend api` evaluations used
`MiniMaxAI/MiniMax-M3` through Nebius Token Factory, eight keyframes each, and
a fixed 0.80 threshold: Arena scored 0.10, blocked MuJoCo scored 0.00, and
revised MuJoCo scored 1.00. Requests, raw responses, returned model identity,
usage and exact input hashes are retained with the private film artifacts.
These individual judgments agree with the recorded task outcomes; they do not
establish general model accuracy or deployment safety. The two simulations
are explicitly separate experiments, and MuJoCo is a physics engine rather
than a learned world model.

## Other example films

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
The [120-second inference film](storyboard-inference.json) tells a third story:
how hosted model inference, simulation, and measured feedback form a development
loop. It uses fresh MuJoCo footage and five new Nebius Token Factory calls across
Qwen, MiniCPM, and MiniMax, with official Nebius branding.

## Inference film evidence

The inference example follows one newly authored Cartesian pushing workcell.
Qwen returns a direct route; a scripted controller executes its cube-center
waypoints in MuJoCo. The plan succeeds in a clear scene and fails when an
obstacle is added. MiniCPM's visual assessment incorrectly reports success.
The next MiniMax request includes original images, that assessment, measured
failure coordinates, and explicit obstacle geometry. Its revised route reaches
the target when executed by the same controller. Separate MiniMax visual checks
assess the blocked and revised outcomes, and recorded simulator state verifies
both results.

The five calls retain complete requests, raw responses, exact requested and
returned model identities, input-image hashes, and usage receipts. They use
`Qwen/Qwen3-235B-A22B-Instruct-2507`, `openbmb/MiniCPM-V-4_5`, and
`MiniMaxAI/MiniMax-M3`. Preserve the MiniCPM response as observed: its
`task_complete` field is the string `"true"`, which is also invalid for a boolean
gate. Do not silently convert it into an accepted judgment. The separate blocked
MiniMax check happened after the replan; it did not trigger that request.

MuJoCo performs CPU physics with local OpenGL rendering in this example. The
hosted model calls run through Nebius Token Factory. The controller is scripted,
and the simulation is a physics engine; this example does not demonstrate a
learned low-level policy or a neural world model. Exact oriented-corner replay
checks confirm full target containment for the two successful runs. These are
three recorded task outcomes, not a model benchmark or general safety claim.
All simulation footage is newly generated for this story, with no footage from
the executive showcase and no NVIDIA models or branding.

To adapt this story, supply these roles in the private asset manifest and retain
their original inputs, outputs, derivation scripts, and hashes:

| Roles | Evidence required |
| --- | --- |
| `nominal_overview`, `nominal_initial` | Fresh clear-scene simulation video and initial frame, plus the scene, controller, returned plan, trajectory, and result. |
| `nominal_run`, `nominal_top`, `blocked_top`, `revised_run` | Videos derived from those exact runs, with playback speed and final-frame holds recorded. |
| `blocked_final`, `revised_final` | Original final frames from the corresponding recorded runs. |
| `qwen_plan`, `minimax_route` | Editorial diagrams computed from actual returned waypoints and task geometry. |
| `measured_state` | An editorial card computed from the blocked run's recorded result. |
| `visual_assessments` | Exact summaries and boolean outcomes from the two separate MiniMax visual checks. |
| `model_roles` | A model-role card supported by the five original request/response receipts. |
| `closing` | A composition using the original official Nebius logo from [brand/](brand/). |

The [Nebius media kit](https://nebius.com/media-kit) supplies the unmodified logo.
The private editable package retains the experiment source, original simulation
frames, five inference records, independent containment checks, narration,
editorial compositions, renderer, and final output. Visual judgments are checked
against measured state. Replaying saved plans is separate from requesting new
model outputs, which may differ even with the same inputs.

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

## One studio, multiple films

The executive and inference films use the same renderer. Each project owns its
creative brief, scene order, duration, titles, narration, media and provenance.
Add any number of projects to a private `studio.json`; names are user choices:

```json
{
  "projects": {
    "exec": "projects/exec/film-project.json",
    "inference": "projects/inference/film-project.json"
  }
}
```

Project paths resolve relative to the registry. Each project retains the four
paths described below; use distinct working directories for independent edits.
Keep delivered masters outside those working output directories. The shared
launcher accepts all existing authoring and rendering commands:

```bash
npa/.venv/bin/python docs/demos/executive-film/studio.py --registry /path/to/studio.json list
npa/.venv/bin/python docs/demos/executive-film/studio.py --registry /path/to/studio.json exec scenes
npa/.venv/bin/python docs/demos/executive-film/studio.py --registry /path/to/studio.json inference brief \
  --prompt "Explain the simulation loop to engineers" --duration 75
```

The registry defaults to `studio.json` in the current directory. `list` is a
reserved name; other film names use lowercase letters, numbers, underscores and
hyphens. `brief`, `scenes`, `narrate`, `preview` and `final` forward their options
to the existing editor. `--project` comes from the selected registry entry.
There are no fixed executive/inference modes or a fixed two-minute runtime.

## Faster development previews

Use `draft` to inspect a single scene's composition and motion before speech or
other scenes' media are ready. It renders a muted 960 × 540 clip through the
same verified visual scene cache used by `preview`:

```bash
npa/.venv/bin/python docs/demos/executive-film/studio.py --registry /path/to/studio.json inference draft \
  --scene 08-minimax-revise --open
npa/.venv/bin/python docs/demos/executive-film/studio.py --registry /path/to/studio.json inference watch \
  --scene 08-minimax-revise
```

Both commands require a scene ID. `draft --open` opens the completed clip in the
default video player. Its output is
`<output_dir>/draft/<scene>/workbench-draft.mp4`, with `draft.json` recording the
scene, source identities, provenance, cache reuse and elapsed time. It contains
no narration, music or captions. The selected visual assets still require valid
hashes and decodable media; missing files from other scenes do not block it.
Project or media changes during a build fail before replacing the last good
draft. It never publishes into `preview/` or `final/`.

`watch` checks the selected scene's media, project JSON and shared renderer
files every 250 ms, then rebuilds after edits settle for 500 ms. Changes during
a build are picked up next. An invalid JSON save or stale media hash leaves the
last good clip intact and waits for the next edit. Watching uses file metadata
to detect edits; each draft still verifies content hashes. Stop with Ctrl+C.
The watcher prints the completed path; it does not reload an external player.

Measured on the executive and inference working projects on one macOS workstation,
including process startup and verification, a changed ten-second scene took
5.0–5.1 seconds. Unchanged or reverted drafts took 0.31–0.36 seconds. These runs
used local source media and made no speech-service or model calls; timings vary
with codecs, scene complexity and hardware.

For the complete review with speech, run `narrate` after changing spoken text,
then `preview` or `final`. Those commands keep their narration and full-output
validation. Drafting and watching require no speech service or model calls.

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
Video entries optionally set `playback: "hold"` to preserve their final frame
when a scene outlasts the source clip. The default, `"loop"`, repeats the clip.
Playback choice participates in scene caching and the rendered asset manifest;
changing it rebuilds the affected scenes. Neither mode changes the source file.
For a selected excerpt, add `"trim": [4.5, 12.0]` to the video entry. These are
in/out points in seconds on the original source timeline. The renderer verifies
that both points fit the source, seeks to the excerpt, and holds its final frame
if the scene is longer. Trimmed clips use hold playback; an explicit `"loop"`
is rejected. Keep the original file and its original SHA-256. Several asset
roles can select different excerpts from that same file, so adjusting a cut
requires only a manifest edit. Trim points are retained in render provenance
and invalidate only scenes that use the changed role.

Set `"unique_sources": true` at the storyboard root when each primary image or
video must appear only once. Full renders reject repeated SHA-256 identities,
including aliases with different filenames, crops or trim points. The option
defaults to false for projects that intentionally compare or revisit sources.
Single-scene previews check only the selected scene. This checks source identity;
the editor still reviews different files for visually similar content.

Set `"fit": "cover"` to fill the scene's media area with a centered crop while
preserving the source aspect ratio. The default, `"contain"`, keeps the whole
source visible with padding. An explicit pixel `crop` is applied before either
fit mode. The fit choice is recorded with the asset and invalidates its scenes.
Use the one-asset `application` scene layout for recorded software demos. Its
compact header leaves a taller media panel for cameras, controls and telemetry;
keep the subtitle to one short line. The ordinary `screen` layout reserves more
space for explanatory headings and diagrams.
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
Short clips loop by default or hold their last frame when configured. Preserve these distinctions
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
