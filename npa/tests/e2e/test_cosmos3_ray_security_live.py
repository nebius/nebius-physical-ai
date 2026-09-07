"""Real Ray control-plane authentication, scoped S3, and CUDA attention checks.

Run with the patched framework image's interpreter and explicit live inputs.
This validates security changes and native numerical execution, not Cosmos model
readiness, generation quality, guarded inference, or serving throughput.
"""

import hashlib
import json
import os
from pathlib import Path
import stat
import unittest


def _attention_on_ray_worker(output_dir):
    import importlib.metadata

    import numpy as np
    import torch
    from cosmos_framework.model.attention.natten import natten_attention, natten_supported

    assert torch.cuda.is_available() and natten_supported()
    torch.manual_seed(17)
    operands = [torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16,
                            requires_grad=True) for _ in range(3)]
    query, key, value = operands
    actual = natten_attention(query, key, value)
    upstream_gradient = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, operands, upstream_gradient)
    reference_operands = [tensor.detach().float().requires_grad_() for tensor in operands]
    rq, rk, rv = reference_operands
    scores = torch.einsum("bqhd,bkhd->bhqk", rq, rk) / (rq.shape[-1] ** 0.5)
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), rv)
    reference_gradients = torch.autograd.grad(expected, reference_operands,
                                            upstream_gradient.float())
    torch.cuda.synchronize()
    # BF16 fused kernels accumulate differently from the FP32 eager reference.
    # Retain the actual error measurements and every operand for independent review.
    torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.02)
    for actual_gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
        assert bool(torch.isfinite(actual_gradient).all())
        assert float(actual_gradient.norm()) > 0
        torch.testing.assert_close(actual_gradient.float(), reference_gradient,
                                   rtol=0.03, atol=0.02)
    output = Path(output_dir) / "attention.npz"
    arrays = {"output": actual.detach().float().cpu().numpy(),
              "reference_output": expected.detach().cpu().numpy()}
    for name, operand, gradient, reference in zip(
        ("query", "key", "value"), operands, gradients, reference_gradients, strict=True,
    ):
        arrays[name] = operand.detach().float().cpu().numpy()
        arrays[name + "_gradient"] = gradient.detach().float().cpu().numpy()
        arrays[name + "_reference_gradient"] = reference.detach().cpu().numpy()
    np.savez_compressed(output, **arrays)
    return {
        "torch_version": torch.__version__,
        "natten_version": importlib.metadata.version("natten"),
        "device": torch.cuda.get_device_name(),
        "shape": list(query.shape),
        "max_output_absolute_error": float((actual.float() - expected).abs().max()),
        "max_gradient_absolute_errors": [
            float((a.float() - b).abs().max()) for a, b in zip(gradients, reference_gradients, strict=True)
        ],
        "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "worker_pid": os.getpid(),
    }


@unittest.skipUnless(
    os.environ.get("NPA_INTEGRATION_E2E") == "1"
    and all(os.environ.get(name) for name in (
        "NPA_RAY_SECURITY_E2E_INPUT_S3", "NPA_RAY_SECURITY_E2E_INPUT_SHA256",
        "NPA_RAY_SECURITY_E2E_ALLOWED_S3_ROOT", "NPA_RAY_SECURITY_E2E_OUTPUT",
    )),
    "requires an explicitly configured task S3 artifact and CUDA framework image",
)
class CosmosRaySecurityLiveTest(unittest.TestCase):
    def test_real_control_plane_scoped_input_and_attention(self):
        from npa.workbench.cosmos.ray_inputs import stage_sample_inputs
        from npa.workbench.cosmos.ray_server import _require_ray_authentication
        from npa.workbench.storage_scope import StorageAuthorizationError, StorageScope

        output = Path(os.environ["NPA_RAY_SECURITY_E2E_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        allowed = os.environ["NPA_RAY_SECURITY_E2E_ALLOWED_S3_ROOT"]
        source = os.environ["NPA_RAY_SECURITY_E2E_INPUT_S3"]
        scope = StorageScope.from_config(s3_roots=[allowed])
        cases = [
            {"vision_path": "http://169.254.169.254/latest/meta-data/"},
            {"vision_path": "/etc/passwd"},
            {"vision_path": "file:///etc/passwd"},
            {"vision_path": allowed.rstrip("/") + "-outside/frame.png"},
            {"vision_path": source, "defaults_file": "/etc/passwd"},
            {"vision_path": source, "edge": {"control_path": "/etc/passwd"}},
        ]
        for index, case in enumerate(cases):
            destination = output / ("denied-" + str(index))
            with self.assertRaises(StorageAuthorizationError):
                stage_sample_inputs({"name": "adversarial", **case}, destination, scope=scope)
            self.assertFalse(destination.exists())
        staged = stage_sample_inputs({"name": "real-frame", "vision_path": source},
                                     output / "accepted-input", scope=scope)
        staged_path = Path(staged["vision_path"])
        self.assertTrue(staged_path.is_file())
        self.assertEqual(hashlib.sha256(staged_path.read_bytes()).hexdigest(),
                         os.environ["NPA_RAY_SECURITY_E2E_INPUT_SHA256"])

        token_file = _require_ray_authentication()
        self.assertEqual(stat.S_IMODE(token_file.stat().st_mode), 0o600)
        self.assertNotIn("RAY_AUTH_TOKEN", os.environ)
        self.assertEqual(os.environ["RAY_AUTH_MODE"], "token")
        # Import only after the production helper fixes management authentication.
        import grpc
        import ray
        from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc

        try:
            context = ray.init(address="local", include_dashboard=False,
                               _node_ip_address="127.0.0.1", num_cpus=2, num_gpus=1)
            # Raw gRPC deliberately omits Ray's client interceptor so the negative
            # cases reach the real GCS server without inheriting its token.
            with grpc.insecure_channel(context.address_info["gcs_address"]) as channel:
                stub = gcs_service_pb2_grpc.NodeInfoGcsServiceStub(channel)
                for metadata in ([], [("authorization", "Bearer invalid-test-token")]):
                    with self.assertRaises(grpc.RpcError) as denied:
                        stub.GetClusterId(gcs_service_pb2.GetClusterIdRequest(),
                                          metadata=metadata, timeout=10)
                    self.assertEqual(denied.exception.code(), grpc.StatusCode.UNAUTHENTICATED)
                authorized = stub.GetClusterId(
                    gcs_service_pb2.GetClusterIdRequest(), timeout=10,
                    metadata=[("authorization", "Bearer " + token_file.read_text().strip())],
                )
                self.assertEqual(authorized.status.code, 0)
                self.assertTrue(authorized.cluster_id)
            worker = ray.remote(num_gpus=1)(_attention_on_ray_worker)
            numeric = ray.get(worker.remote(str(output)))
            self.assertNotEqual(numeric.pop("worker_pid"), os.getpid())
            report = {
                "status": "completed", "ray_version": ray.__version__,
                "management_missing_token": "UNAUTHENTICATED",
                "management_wrong_token": "UNAUTHENTICATED",
                "management_correct_token": "accepted",
                "adversarial_input_cases_rejected": len(cases),
                "real_s3_artifact_sha256": os.environ["NPA_RAY_SECURITY_E2E_INPUT_SHA256"],
                "native_ray_worker": True, "numerical": numeric,
                "scope": "production management authentication and S3 authorization; "
                         "native Cosmos-framework NATTEN forward/backward in a real Ray GPU worker",
                "limitations": ["No Cosmos model, guardrail, generation-quality or throughput qualification"],
            }
            (output / "security_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            ray.shutdown()
            token_file.unlink(missing_ok=True)
        self.assertFalse(token_file.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
