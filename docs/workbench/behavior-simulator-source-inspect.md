# Inspect an Isaac application source

Use the internal `simulator-source-inspect` command in a CPU-only, read-only
preflight to derive the exact `apps` object for a later simulator startup spec:

```console
python -m npa.workflows.behavior_challenge simulator-source-inspect \
  --isaac-root /retained/isaac-sim \
  --owner-root /owned/run \
  --view-root /owned/run/isaac-view
```

The command hashes `VERSION` and every direct regular file in `apps`, records
their modes, and classifies the known extension paths `exts`, `extscache`,
`extsDeprecated`, `extsPhysics`, and `extsUser` by `lstat`. It rejects links,
special application members, and extension paths that exist as anything other
than real directories. It then runs the same production `_apps_claim` validator
used immediately before writable simulator startup.

The result is observation evidence. It starts no simulator, loads no policy,
and creates no writable view. A later startup must still bind this result to a
fresh route, exact retained mount, writable owner, evaluator context, and its
other runtime admissions.
