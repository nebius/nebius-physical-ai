# Workbench Studio

Workbench Studio is a local CLI and file-based editing framework. Customers and
coding agents use the same interface: a creative brief, a storyboard, attributed
media and cached renders. There is no fixed audience, model list or film length,
and no hosted Studio UI or cloud deployment is required for editing.

## Start with your own media

Install NPA from a checkout containing Studio, with Python 3.10 or newer, and
install FFmpeg (including ffprobe and libx264) on your machine:

```bash
pip install -e 'npa[studio]'
npa studio init --directory ./my-studio
cd my-studio
npa studio create demo --input-path /path/to/your-clip.mp4 --duration 30 \
  --prompt "Explain our simulation workflow" --audience "Engineering leaders"
npa studio demo draft --scene opening --open
```

The renderer, licensed font and official Nebius brand assets are included in the
NPA wheel and source distribution. A repository checkout is not needed after
installation. `init --renderer /path/to/renderer` is an optional override for a
trusted custom renderer. Initialization creates an empty registry; `create`
copies only the selected local media, hashes it, and registers a new project.
It never copies NPA configuration, credentials, another film or a GPU run archive.
Existing project names are rejected without overwriting them.

`create` starts with one scene and placeholder narration. Edit the storyboard
to author your film; the prompt is guidance, not an automatic script generator.
Optional `--tool` records your declared attribution. Without it, provenance stays
unknown. Review the original run evidence before making tool or outcome claims.
A short clip holds its final frame; adjust scene duration or add more footage.

## Developer loop

```bash
npa studio list
npa studio demo scenes
npa studio demo brief --prompt "Explain the evaluation results" \
  --audience "Research leads" --duration 45 > projects/demo/authoring-packet.json
# Author storyboard.json from the packet and add media to assets.json.
npa studio demo watch --scene opening
npa studio demo draft --scene opening --open
npa studio demo narrate
npa studio demo preview --scene opening --open
npa studio demo final --open
```

`brief` emits an authoring packet; it does not rewrite the storyboard or call a
model. `draft` and `watch` make silent local scene previews. Scene hashes reuse
unchanged visual work. `narrate` reuses speech whose wording and voice match;
`final` reuses verified scene/audio caches and assembles the completed film.
The final MP4, optional captions, offline player and provenance manifest are in
`projects/demo/renders/final/`. File names are `film.mp4`, `film.srt` and
`watch.html`. `--plan` on preview/final reports visual cache hits before encoding.

The optional `narrate` generator sends narration text to Microsoft's Edge speech
service through edge-tts. For an entirely offline flow, supply one MP3 and SRT per
scene in the narration directory and run `npa studio demo narrate --recorded`.
Rendering retained media and recorded speech makes no cloud or model calls.
Studio does not upload the film or send messages to collaboration services.

## Project interface

```text
my-studio/
  studio.json                 # Renderer location and project aliases
  renderer/                   # Installed renderer copied for reproducibility
  projects/demo/
    film-project.json         # Relative editing paths, voice and optional music
    storyboard.json           # Brief, narration, scene order, layouts and timing
    assets.json               # Local media paths, SHA-256, transforms, provenance
    media/                    # Your selected source files
    narration/                # Generated or supplied MP3/SRT and identity records
    renders/                  # Drafts, previews, cache and final delivery
```

`studio.json` accepts only `renderer` and `projects`. For example:

```json
{"renderer": "renderer", "projects": {"demo": "projects/demo/film-project.json"}}
```

`film-project.json` accepts only these local editing fields:

```json
{
  "storyboard": "storyboard.json",
  "assets": "assets.json",
  "voice_dir": "narration",
  "output_dir": "renders"
}
```

Optional `voice` selects the narration voice. Optional `music_path` selects a
local score covering the whole film. Relative paths resolve from their owning
JSON file, so a project moves with its renderer and media. Recreate the virtual
environment on the receiving machine. The generated `studio` launcher uses
`NPA_STUDIO_PYTHON`, then a Studio-local `.venv/bin/python`, then `npa` on PATH.
No machine-specific interpreter path is embedded.

Generated projects and the registry are ignored by Git by default. This prevents
accidental commits of scripts, prompts, media, narration, caches or private
provenance. Share reviewed content deliberately. Credentials, project/tenant
identifiers, regional endpoints and storage buckets belong in external NPA
configuration, never in the registry, project template or public examples.
Studio has no default bucket or capacity reservation and does not auto-archive.

GPU footage generation remains a separate NPA workflow operation using the
customer's selected project and storage settings. See the
[native media workflow guide](../../workbench/video-generation-byof.md).

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

For selected-bucket and cross-project examples, coverage handling, provenance,
Python retrieval, and the Workbench agent browser, use the
[artifact discovery guide](../../workbench/cookbooks/find-artifacts.md).

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

## Media manifest

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

## Renderer development

The installed source is `npa/src/npa/studio_renderer/`; command dispatch, local
project creation and artifact search live in `npa/src/npa/studio*.py`. Update the
shared implementation and its unit tests, then initialize a new local studio to
verify the packaged renderer. Existing Studio directories keep their copied
renderer and are not silently upgraded. No production films or run recordings
are bundled with NPA; the sample storyboards use generic placeholder text.

```bash
npa/.venv/bin/python -m pytest npa/tests/unit/test_studio_entry.py \
  npa/tests/unit/test_studio_artifacts.py npa/tests/unit/test_film_studio.py \
  npa/tests/unit/test_film_brief.py npa/tests/unit/test_executive_film.py \
  npa/tests/unit/test_executive_film_player.py -q
```

The official Nebius logo and the font retain source and licensing records under
`npa/src/npa/studio_renderer/brand/` and `fonts/`. Brand display is optional in the
full-frame film layout. Verify source-media distribution rights separately.
