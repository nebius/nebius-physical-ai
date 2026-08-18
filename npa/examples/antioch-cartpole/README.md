# Antioch cartpole offline-policy smoke

This synthetic/public example runs one Isaac Lab cartpole on Antioch-managed
infrastructure and publishes a real state/action episode. Its two RGB streams are
deterministic diagnostic renderings of the simulated state; they are not claimed
to be simulator camera sensors. The state and feedback-control action are taken
from the live simulation.

Package this directory as `project.tar.gz` without caches or credentials. Its
`npa.antioch.project.v1` manifest must use source name
`npa-antioch-cartpole-policy-data-v1`, source revision `1`, source license
`Apache-2.0`, and source SHA-256
`5e77337d11362e05439b8e110e93126dd9c74ff1221057e7409b089a0de09fee`.
Upload the archive and manifest to an immutable S3 prefix, then run suite
`npa_cartpole_offline_smoke` through `npa workbench antioch run`.
