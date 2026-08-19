# Antioch OpenPI Franka camera bridge

This project is the Antioch-hosted wrapper for the shared implementation in
`npa.workbench.antioch.openpi_isaac`. It must run in a private Antioch Isaac Lab
engine after the project image installs the exact NPA revision being validated.
The OpenPI endpoint is supplied as `OPENPI_POLICY_HOST` and remains private.
Continuous soft-real-time camera/policy/control operation is the default;
one-observation behavior exists only as explicit `OPENPI_CONTROL_MODE=finite-smoke`.

The project manifest pins `NPA_SOURCE_REF` to the exact reviewed bridge commit as
a Docker build argument. Update that immutable 40-character revision whenever
the shared bridge changes, and validate it with `antioch suite collect --json`.
The engine and SDK are pinned to `0.3.47`; its derived image contains the vendor
Isaac runtime and must remain in the operator's private Antioch registry.
When an engine advertises an as-yet-unpublished NVIDIA asset prefix, the bridge
checks and uses NVIDIA's immutable 5.1 Franka USD compatibility root at runtime;
it fails closed unless the exact Franka asset is reachable there.

The project does not contain credentials. Authenticate with Antioch outside the
project, mount or inject the vendor-supported runtime session, and submit suite
`openpi_franka_streaming`. The operator must also provide the run-scoped NVIDIA and
Gemma acceptances through their respective secret channels. Never add either
acceptance, an Antioch session, or policy/checkpoint bytes to this directory.

The hosted scenario keeps the policy private with Antioch's authenticated port
tunnel. The simulator exposes the relay backend only through Antioch's declared
authenticated service port `18123`; its OpenPI frontend remains loopback-only.
On the operator host, `policy_tunnel_connector.py` pairs that tunnel with a
local `kubectl port-forward` to the digest-pinned policy Service. No Kubernetes
or registry credential is copied to Antioch, and the policy remains a ClusterIP:

```bash
kubectl --kubeconfig <task-kubeconfig> --context <task-context> \
  port-forward service/<run-policy-service> 18000:8000
npa/.venv/bin/python policy_tunnel_connector.py
antioch suite run openpi_franka_streaming --stream
```

Run `antioch services up --json` once before the connector so the declared port
is converged, then run the connector and suite concurrently. The hosted bridge
uses the authenticated RTX viewport for its exterior and wrist-mounted camera
poses, so the suite keeps Antioch's authenticated stream enabled. Console video
streaming is not observation or control readiness: require advancing camera
sequences/timestamps, repeated policy round trips, and repeated safe target
application over the configured sustained interval. Observation acquisition,
policy request, and physics control rates are separately configured and the
reported achieved rates may be lower. The Python/WebSocket/authenticated-relay
path is soft real time, never hard real time. The connector
retries the tunnel with bounded exponential backoff, caps empty sessions, and
supports the bridge's reconnect attempts; stop both after the suite reaches a
terminal state. Keep the kubeconfig and all exact live resource values outside
this project and its run artifacts.

Without an Antioch account session, validate the same bridge code in the
operator-owned Kubernetes stack rendered by:

```bash
npa workbench antioch openpi-stack --help
```

The rendered Kubernetes bridge waits for policy health and proves that the host
injected an NVIDIA Vulkan renderer before launching Isaac. A compute-only CUDA
driver mount is rejected instead of falling back or hanging during scene
creation. The bridge then fails closed on any protocol or control error. Remove
all four run objects with the same arguments plus `--delete` after collecting
the report.
