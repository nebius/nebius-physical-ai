# Checkpoint and object download boundaries

Policy checkpoints must contain tensors and ordinary state-dict metadata.
SONIC export, MuJoCo evaluation, Genesis teacher loading, and Sim2Real policy
export use PyTorch's restricted loader. Pickled module objects are rejected.
Convert legacy checkpoints to state dictionaries in a separate trusted
environment before submitting them to these paths.

SONIC reconstructs its built-in reference policy from checkpoint metadata.
Custom architectures must be supplied explicitly through trusted `--config`
or the SDK's existing `policy=` argument. Checkpoint metadata cannot select
arbitrary Python imports or callables. Treat explicit architecture configuration
as executable operator configuration.

S3 tree downloads reject parent traversal, absolute relative paths, and
destinations that escape through existing symlinks. This applies to shared
storage downloads and the checkpoint, dataset, triage, and source staging
paths. Checkpoint cache identities include the full bucket and object prefix;
downloads are published only after the complete tree has been staged.

These checks constrain remote artifact content. They do not replace service
authentication, narrowly scoped storage credentials, or protection of local
staging directories from concurrent untrusted filesystem writers.
