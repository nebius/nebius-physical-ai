# MolmoAct2 Jetson-Thor cloud evidence worker

Status: **unvalidated; operator-private validation only**.

This is a narrow CPU worker for prepared observation validation, binding of
native target reports, numerical action/latency evaluation, and factual Rerun
recording. It is not an edge inference image and cannot load or validate the
Jetson TensorRT plans.

No public tag, public image catalog entry, or release claim is authorized. The
image must remain in an operator-controlled registry until its exact built-byte
closure, SBOM, secret and vulnerability scans, pull evidence, and intended
runtime evidence are independently accepted. An operator-owned native Thor
continues to fetch and execute all upstream model and TensorRT payloads.
