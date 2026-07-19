# Releasing `npa`

`npa` is distributed as a source checkout plus tagged releases with built
artifacts. Every release is a git tag `vX.Y.Z` that matches the `version`
field in `npa/pyproject.toml`; the `Release` workflow
(`.github/workflows/release.yml`) builds the sdist and wheel, smoke-installs
the wheel, and attaches both to a GitHub Release.

## Cutting a release

1. **Bump the version** in `npa/pyproject.toml` (`[project] version`).
   Pre-1.0, breaking CLI/SDK changes bump the minor version and fixes bump the
   patch version.
2. **Update `CHANGELOG.md`**: move the relevant `## Unreleased` entries under a
   new `## vX.Y.Z - YYYY-MM-DD` heading. Keep `## Unreleased` at the top for
   the next cycle.
3. **Merge to `main`** through the normal PR flow (CI must be green).
4. **Tag and push**:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. The `Release` workflow refuses tags whose version does not match
   `npa/pyproject.toml`, so a mismatched tag fails fast instead of shipping a
   mislabeled artifact.

## Installing a release

Consumers who do not want a source checkout can pin a released artifact:

```bash
pip install "npa @ https://github.com/nebius/nebius-physical-ai/releases/download/vX.Y.Z/npa-X.Y.Z-py3-none-any.whl"
```

The base wheel is lightweight; add extras as needed
(`npa[full]`, `npa[data]`, `npa[lancedb]`, `npa[viz]`, `npa[server]`, ...).

## Enabling PyPI publishing (optional, not yet enabled)

When the project is ready to publish to PyPI:

1. Create the `npa` project on PyPI and configure a
   [trusted publisher](https://docs.pypi.org/trusted-publishers/) pointing at
   this repository and the `release.yml` workflow.
2. Add a `pypi` environment in the repository settings.
3. Append a publish job to `release.yml`:

   ```yaml
   publish:
     needs: build
     runs-on: ubuntu-latest
     environment: pypi
     permissions:
       id-token: write
     steps:
       - uses: actions/download-artifact@v4
       - uses: pypa/gh-action-pypi-publish@release/v1
   ```

Until then, GitHub Releases are the canonical distribution channel.
