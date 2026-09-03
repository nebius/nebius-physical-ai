---
name: artifact-viz-share
description: Use to turn run outputs into something a human can look at and to hand it to someone else — `npa adapter convert` (sim demos to LeRobotDataset), `npa convert lerobot-to-rrd|lerobot-to-mp4`, and `npa rerun host|share|list-shares|revoke` for time-boxed presigned links.
---

# Artifact conversion, visualization, and sharing

Three small command groups cover the last mile between a finished run and a
human looking at it. They are standalone and local-first: none of them needs a
cluster, and all of them accept `s3://` on both sides.

Pick by what you have and what you need:

| Have | Want | Command |
|---|---|---|
| Genesis/sim episode numpy arrays | A trainable dataset | `npa adapter convert` |
| LeRobotDataset | Interactive timeline | `npa convert lerobot-to-rrd` |
| LeRobotDataset | A video to paste in a review | `npa convert lerobot-to-mp4` |
| `.rrd` recording | A link someone else can open | `npa rerun host` / `share` |

## Sim output → LeRobotDataset

```bash
npa adapter convert \
  --input-path ./data/demos/ \
  --output-path ./data/lerobot_dataset/ \
  --fps 20 --robot franka_panda \
  --task "Pick and place cube to target"
```

Converts Genesis/sim demo numpy arrays to **LeRobotDataset v3**. This is the seam
between simulation and policy training: `--fps` is the video encoding rate, and
`--task` becomes the dataset's task description, so set it to what the episodes
actually show rather than leaving the default. `-i`/`-o` are accepted aliases.

## LeRobotDataset → Rerun recording

```bash
npa convert lerobot-to-rrd \
  --input-path s3://<bucket>/datasets/<name>/ \
  --output-path s3://<bucket>/reports/<name>.rrd \
  --duration 30 \
  --predictions-path s3://<bucket>/eval/groot-predictions.json
```

`.rrd` is the interactive format — scrub the timeline, inspect per-frame state.
`--predictions-path` overlays a GR00T prediction artifact on the ground-truth
trajectory, which is how you see *where* a policy diverges rather than only that
it scored badly. `--duration` caps the recording; the default is the adapter cap.

## LeRobotDataset → MP4

```bash
npa convert lerobot-to-mp4 \
  --input-path ./data/lerobot_dataset/ \
  --output-path ./rollout.mp4 \
  --renderer matplotlib --layout single \
  --resolution 1280x720 --fps 30 --duration 10 \
  --title "<what this shows>"
```

Use MP4 when the audience will not install a viewer. `--renderer` is
`matplotlib` (default) or `rerun`. `--layout` is `single`, `side-by-side`, or
`overlay` — with `--predictions-path`, `side-by-side` and `overlay` are what make
the comparison legible; `single` throws that away.

Default duration is the source length capped at 10 seconds, so a long episode is
silently truncated unless you set `--duration`.

`npa viz lerobot` is deprecated and prints a deprecation notice; use
`npa convert lerobot-to-mp4`.

## Sharing a recording

```bash
# One-time bucket-admin setup (read-only plan, then explicit apply).
npa storage bucket cors --project <alias>
npa storage bucket cors --project <alias> --apply

npa rerun host <path.rrd> --ttl-hours 1
npa rerun share <path.rrd> --label <name> --workspace <ws> --ttl-hours 168
npa rerun list-shares --output json
npa rerun revoke <sha256-or-label>
```

`host` is the quick look: upload or reference an `.rrd` and print an
`app.rerun.io` URL, default TTL **1 hour**. `share` is the durable version:
S3-backed under `rerun-shares/<workspace>/`, labelled, default TTL **168 hours**,
which is also the maximum. Both accept a local path or an `s3://` URI.

Project scoping is explicit and worth getting right: `--source-project` is the
alias whose principal **reads** an `s3://` input, `--target-project` is the alias
whose principal **writes** the upload, and `--target-bucket` overrides the
destination (default: configured project storage). `--allow-host-creds` falls back
to host credentials for the S3 operation — an explicit opt-in, not a default.

The hosted viewer fetches the presigned object cross-origin and requests byte
ranges. A bucket administrator must therefore configure CORS once. The setup
command uses the active Nebius control-plane profile, preserves unrelated CORS
rules, and adds only the `https://app.rerun.io` origin with `GET` and `Range`.
The scoped object key created by `npa configure` remains object-only and cannot
perform this operation. Running the setup command without `--apply` is a
read-only plan.

`host` and `share` exercise a real `OPTIONS` preflight against the presigned URL
before returning it. A failed preflight exits nonzero with the bucket-admin
setup command and does not print the signed URL. After applying the policy,
retry the share. If bucket policy changes are not available, download the `.rrd`
and open it locally with `rerun <recording.rrd>`.

Revoke by label or sha256 when the work is no longer for sharing. `list-shares`
is the only way to find what you left behind; presigned links do not appear in
any other inventory.

## Related viewers

`.rrd` is not the only option. For MCAP, ROS bags, and robotics logs there are
two viewer tools: `skills/tools/foxglove/SKILL.md` (embedded in the agent UI,
`npa workbench foxglove convert-run`) and `skills/tools/lichtblick/SKILL.md`
(standalone web viewer served from S3). Use those when the artifact is a log
rather than a dataset.

## Gotchas

- **A presigned link is a credential.** Anyone with the URL can read the
  recording until it expires. Cap `--ttl-hours` to what the review needs, prefer
  `host`'s 1-hour default for a quick look, and `revoke` when done.
- **168 hours is the hard maximum.** There is no permanent share.
- **Default MP4 duration is 10 seconds.** Long episodes are truncated without a
  warning that says so.
- **`--predictions-path` with `--layout single` hides the comparison** you
  converted the file to see.
- **Check what the recording actually contains before sharing it.** A viewer that
  opens a stock demo rather than your run looks identical at a glance; confirm
  the artifact came from your run id via `npa workbench workflow artifacts`.
- **These commands do not clean up after themselves.** Uploaded shares live in the
  bucket until revoked and count toward storage; include them in the audit in
  `skills/atomic/teardown-and-cost/SKILL.md`.
- **Object access is not bucket administration.** Do not grant `PutBucketCORS`
  to the workload key or use `--allow-host-creds` as a substitute for the
  control-plane setup command.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
