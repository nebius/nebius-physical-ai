# FIXME

## Pytest path assumptions in deploy template tests
**Symptom:**
Running `npa/.venv/bin/pytest npa/tests/test_deploy.py` from the repository root fails with `FileNotFoundError` for paths such as `src/npa/deploy/terraform/cloud_init.yaml.tpl`.

**Workaround:**
Run those tests from the `npa/` package directory, for example: `cd npa && .venv/bin/pytest tests/test_deploy.py`.

**Proper fix:**
Resolve Terraform template fixture paths relative to the package root or the test file instead of the process current working directory.

