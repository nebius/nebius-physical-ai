# Scanner native memory proof

The final tested scanner commit is
`441c11ddaca8e5990624313c175dd49e8185b2c9`; the baseline is
`a09e1e00df1c7762aa388c6eaccc15c55eb2a4a6`. The source-bound summary records
807 Python tests, 74 Go race test/subtest results, standard and one-CPU native
integration, byte-identical complete ledgers for three synthetic OCI archives,
and the constrained helper experiments. CPU-only scanner code needs no GPU.

The scanner's committed `npa/scripts/image_byte_scan/go_helper/README.md` and
`build.py` describe the pinned native build. Its committed `main_test.go`,
`real_helper_checks.py`, and `npa/tests/docker/test_image_byte*.py` reproduce the
standard source and native checks. The archive fixtures are generated synthetic
input, not customer images.

The custom memory drivers are included here for review and reproduction.
`driver-bindings.json` hashes both the executed private originals and these
parameterized copies. The only changes are replacing the private evidence-root
path with the first command-line argument and adding `import sys`. The measured
results came from the originals, not a fresh run of these copies.

Use a fresh private evidence root with this layout:

- `base/`: the baseline source checkout.
- `final-source/`: the candidate source checkout.
- `analysis/base-tools/`: the baseline official build output.
- `analysis/final-candidate-tools/`: the candidate official build output.

Keep the official build receipts and their referenced toolchain/cache paths.
The native mutation and probe build drivers consume those receipts without
installing dependencies into shared environments. They require Python 3.12.
The cgroup drivers require Linux cgroup v2, `/usr/bin/time`, a memory controller
already enabled at the mount root, and sudo. They create and remove only fresh,
randomly named task-owned cgroups; the helper runs as uid/gid 1000, matching the
recorded test environment. Do not run on a host whose uid 1000 is another owner.

```sh
npa/.venv/bin/python build-memory-probe.py /path/to/evidence-root
sudo /usr/bin/python3 runtime-limit-probe.py /path/to/evidence-root
sudo /usr/bin/python3 cgroup-runs.py /path/to/evidence-root
sudo /usr/bin/python3 cgroup-pressure.py /path/to/evidence-root
npa/.venv/bin/python native-mutations.py /path/to/evidence-root
```

Output directories must not already exist. Synthetic credentials in the corpus
are detector fixtures, assembled inside the harness; they do not authenticate
against any service. No generated private key or operator credential is included.

At 384 MiB with a live 64 MiB sibling, the old helper was OOM-killed; the candidate
completed all 12 records and four findings. At 512 and 1536 MiB both completed.
All complete responses match byte for byte. These are individual functional
measurements, not a performance benchmark or an OOM guarantee. Go's memory limit
is soft, and the kernel's `memory.peak` was unavailable: group peaks were sampled,
while `/usr/bin/time` recorded process maximum RSS.
