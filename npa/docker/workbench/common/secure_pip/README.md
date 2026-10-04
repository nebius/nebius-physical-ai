# Source-identified pip bootstrap repair

This builder produces `pip==26.2.1+npa.1`, not an upstream pip release. It uses
the exact upstream 26.2.1 source and vendoring machinery, replacing vendored
urllib3 2.7.0 with 2.8.0, msgpack 1.1.2 with 1.2.1, and the pkg_resources
source donor setuptools 70.3.0 with 80.9.0. Other upstream vendor pins remain
unchanged. Installing a newer top-level urllib3 does not repair pip's independent
vendored copy.

`inputs.json` pins the upstream commit/archive and every downloaded vendor
artifact. `build-tools.lock` hash-locks the temporary build environment. The
msgpack pure-Python wheel is built offline from its pinned source using those
tools, with explicit `--no-build-isolation`; its generated hash is retained in
the derivative's provenance. Upstream namespace patches are preserved, with
`pkg_resources.patch` adapting the same internal jaraco/warning behavior to the
new donor and avoiding imports from an unrelated installed setuptools tree.
The upstream BOM and vendor list describe actual replacement sources, not
rewritten version strings over old code. The donor's exact distribution license
is additionally retained as `LICENSE.setuptools`, avoiding the upstream license
collector's flattened-name collision. Existing notices are not removed.

Build with Python 3.11+ (the pinned vendoring tool's requirement), in disposable
task-owned storage:

```sh
python build.py --work-dir /tmp/npa-pip-build --output-dir /tmp/npa-pip-wheel
```

Both directories must be absent. The output wheel supports pip's original
Python 3.10+ runtime boundary; each image consumer must test its actual Python
version. This tool does not install anything into the invoking environment.
The initial unmodified pip and all build tools are temporary build inputs, not
approved final-image payloads. Use an isolated build stage, or install the
derived wheel and remove the exact temporary build tree in the **same RUN**
before exporting a layer. Do not replace pip with a shim or remove the service
bootstrap installer that SkyPilot needs.

Keep `build-receipt.json`, `pip/NPA_VENDOR_REPAIR.json`, the wheel hash and
actual installer/import/negative-control results. This is a reproducible source
repair, not an image security waiver. Every consuming image still requires its
own complete ancestor-layer, vulnerability, license, bootstrap and workload
gates. No image publication or upstream endorsement is implied.
