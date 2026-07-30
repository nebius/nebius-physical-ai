# `npa workbench foxglove`

Tooling for the embedded Foxglove viewer: the pinned
[`@foxglove/embed` TypeScript SDK](https://docs.foxglove.dev/docs/embed/typescript-sdk)
assets, and MCAP recordings the viewer can play.

## Command Tree

```text
Usage: npa workbench foxglove [OPTIONS] COMMAND [ARGS]...

Foxglove embedded viewer: MCAP conversion, inspection, and SDK assets.

Options
--help  Show this message and exit.
Commands
convert-run  Convert a run's artifacts into an MCAP recording the Foxglove viewer can open.
inspect  Report the channels, schemas, and message counts inside an MCAP recording.
install-sdk  Install the pinned, sha512-verified Foxglove embed SDK assets.
config  Show the resolved Foxglove embed settings for this environment.
```

## Subcommands

| Command | Description |
| --- | --- |
| `convert-run` | Convert a run's artifacts into an MCAP recording the Foxglove viewer can open. |
| `inspect` | Report the channels, schemas, and message counts inside an MCAP recording. |
| `install-sdk` | Install the pinned, sha512-verified Foxglove embed SDK assets. |
| `config` | Show the resolved Foxglove embed settings for this environment. |

## `convert-run`

| Option | Description |
| --- | --- |
| `--input-path` | Directory of run artifacts (frames, metrics JSON, logs). |
| `--output-path` | Destination `.mcap` file. |
| `--run-id` | Run id recorded in MCAP metadata. |
| `--fps` | Synthetic frame rate used for timestamps (default `10`). |
| `--max-frames` | Cap the number of image frames (`0` = all). |
| `--output` | `text` (default) or `json`. |

Artifacts are packed into Foxglove well-known schemas:

| Topic | Schema | Source artifacts |
| --- | --- | --- |
| `/camera/<name>` | `foxglove.CompressedImage` | `.png`/`.jpg`/`.webp` passed through; `.ppm`/`.pgm`/`.bmp`/`.tif` transcoded to PNG |
| `/metrics/<name>` | `npa.RunMetrics.<name>` | `.json` documents (numeric fields plot directly) |
| `/log` | `foxglove.Log` | `.log`/`.txt` files, one message per non-empty line |

NPA run artifacts carry no per-frame capture time, so timestamps are generated from
`--fps`. This is recorded in the MCAP `metadata` record as
`timestamps=synthetic-fps` and reported in the command output — do not present
those times as sensor time.

Requires the optional extra: `pip install "npa[foxglove]"`.

## Examples

```bash
# Pack a downloaded run into a recording and check what is inside it
npa workbench foxglove convert-run \
  --input-path ~/runs/s2r-20260730 --output-path /tmp/run.mcap --run-id s2r-20260730 --fps 10
npa workbench foxglove inspect --input-path /tmp/run.mcap

# Install the browser SDK assets (offline-prep, mirrors supported)
npa workbench foxglove install-sdk --dest /opt/npa-agent/foxglove/sdk
npa workbench foxglove install-sdk --dest ./sdk --registry https://npm.internal.example

# Show the resolved embed configuration
npa workbench foxglove config --assets-dir /opt/npa-agent/foxglove/sdk --output json
```

## Related

- Skill: `skills/tools/foxglove/SKILL.md`
- Container: `npa/docker/workbench/foxglove-embed/README.md`
- Agent endpoints: `GET /api/foxglove/config|status`,
  `POST /api/foxglove/load-artifact|convert-run|live`
