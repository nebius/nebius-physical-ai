# Share Rerun recordings safely

`npa rerun host` and `npa rerun share` create time-boxed `app.rerun.io` links
for `.rrd` recordings in object storage. Browser loading requires a one-time
bucket CORS rule because the viewer fetches the recording from a different
origin and uses the `Range` request header.

## Configure the bucket once

First inspect the additive plan, then apply it with a Nebius profile that can
administer the bucket:

```bash
npa storage bucket cors --project <alias>
npa storage bucket cors --project <alias> --apply
```

Use `--name <bucket>` when sharing from a bucket other than the project's
configured storage. The command preserves unrelated CORS rules. The NPA-owned
rule grants only:

- origin: `https://app.rerun.io`
- method: `GET`
- request header: `Range`
- exposed response headers: `Accept-Ranges`, `Content-Length`, `Content-Range`

This is a bucket control-plane operation. The scoped S3 key created by
`npa configure` can read and write objects but deliberately cannot update bucket
CORS. Do not broaden that workload key to fix browser sharing.

The Python SDK exposes the same plan/apply operation:

```python
from npa import rerun

plan = rerun.configure_browser_cors(target_project="<alias>")
if plan.changed:
    rerun.configure_browser_cors(target_project="<alias>", apply=True)
```

## Create and verify a share

```bash
npa rerun host recording.rrd --target-project <alias> --ttl-hours 1
npa rerun share recording.rrd \
  --target-project <alias> \
  --workspace <workspace> \
  --label <label> \
  --ttl-hours 24
```

Both commands send the same unauthenticated CORS preflight the browser sends
against the final presigned URL. They return the viewer URL only after that
preflight allows the Rerun origin, `GET`, and `Range`. On failure, the error
prints the bucket-admin setup command without printing the signed URL.

The URL is itself a temporary credential. Give it the shortest useful lifetime
and revoke durable shares when the review ends:

```bash
npa rerun list-shares --target-project <alias> --output json
npa rerun revoke <sha256-or-label> --target-project <alias>
```

## Native local fallback

When a bucket administrator cannot update CORS, download the recording and open
it in the native viewer:

```bash
rerun <recording.rrd>
```

This fallback does not require a browser origin, a presigned URL, or bucket
CORS.
