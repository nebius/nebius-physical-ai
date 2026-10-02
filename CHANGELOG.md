# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased


### LeRobot feedback control mode requires a JSON boolean

- `POST /feedback/train-step` accepts `control: true` or `control: false`;
  omitting `control` still defaults to false. Strings (including `"true"`),
  numbers, and null now return HTTP 400 before any policy update or output
  directory creation. External clients must send a JSON boolean. Update-result
  decoding enforces the same contract instead of coercing truthy values.

### Studio videos accept S3 output paths

- `preview` and `final` accept `--output-path` for an exact S3 MP4 destination,
  with optional `--storage-project` selecting external NPA credentials. Local
  caches and renders are retained; a full readback verifies the uploaded bytes.
  Generic Python video tools share `npa.video_output.write_video_output`.

### Studio film reviews expose illustrative cues

- Film-review reports now list accepted `illustrative_cues` in assessment order,
  alongside problem and pending cues. The field is audit metadata only:
  illustrative judgments remain reviewed and non-failing under `--strict`.
