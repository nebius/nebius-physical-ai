# Customer private/ directory

Copy `*.example` files here (without `.example` suffix). Never commit real values.

| File | Purpose |
| --- | --- |
| `config.yaml` | Bucket, registry, k8s_context |
| `credentials.yaml` | S3 keys, HF/NGC tokens |
| `operator.env` | TRIGGER_DATASET_URI |
| `clusters/<context>/kubeconfig` | mk8s kubeconfig |
