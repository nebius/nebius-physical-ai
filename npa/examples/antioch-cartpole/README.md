# Antioch cartpole offline-policy smoke

This synthetic/public example runs one Isaac Lab cartpole on Antioch-managed
infrastructure and publishes a real state/action episode. Its two RGB streams are
deterministic diagnostic renderings of the simulated state; they are not claimed
to be simulator camera sensors. The state and feedback-control action are taken
from the live simulation. The action has two physical channels for opposing cart
actuators; their difference is the joint effort applied to the simulated cart.

Package this directory as `project.tar.gz` without caches or credentials. Its
`npa.antioch.project.v1` manifest must use source name
`npa-antioch-cartpole-policy-data-v2`, source revision `2`, source license
`Apache-2.0`, and source SHA-256
`00481edd23e2ae6555e8bf3cc4f2118b90ff8a44c0fc57105501e0bc72891aaf`.
Upload the archive and manifest to an immutable S3 prefix, then run suite
`npa_cartpole_offline_smoke` through `npa workbench antioch run`.

Pass `--robot-type cartpole --task "Balance a cartpole"` explicitly (the
reference workflow exposes the same values as `antioch_robot_type` and
`antioch_task`). The adapter deliberately has no dataset-label defaults.
