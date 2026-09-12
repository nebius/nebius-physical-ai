# Ray Train runtime and live validation

[Run the example](README.md)

This reference pins a specific two-host B200 runtime. These details support
reproduction and review; follow the README for the ordered operator commands.

## Application environment

Record the exact service-task ID from the queue. Preparation installs application
Ray **2.58.0** with Train V2, Arrow **23.0.1**, NumPy **2.4.6** and Rerun
**0.31.4** and Pillow **12.3.0**, retaining Torch **2.13.0+cu130** from the
immutable upstream PyTorch 2.13.0 CUDA 13.0 runtime image.
The image digest and the separately recorded dependency freeze identify different
parts of the runtime; this is not a fully hash-locked transitive installation.
There is no model download or external training dataset.
The Jobs driver and every new or restarted training worker also check the exact
Torch version before training, so environment drift after preparation fails closed.
Every rank records its imported Torch, CUDA and Ray versions alongside its metrics.

The pinned image runs preparation as root. Preparation requires an owned `/opt`
without shared write access and creates a fresh mode-0700 `/opt/npa-ray-train`.
Its `env`, `exports`, and Ray temporary directories stay inside that private
application directory; `preparation.json` records the verified versions and
dependency freeze. An existing directory fails preparation without changing its
contents. Preserve a failed attempt's evidence and use fresh hosting pods for a
new attempt. The service checks the directory's owner and permissions before
starting, and sets `RAY_TMPDIR` to this directory for its application processes.

## Committed live suite

The committed live suite runs baseline training, checkpoint recovery and active
cancellation against this prepared runtime. In an owner-only file outside the
checkout, set `address` to the loopback Jobs URL, `storage_uri` to the verified
fresh S3 prefix, and `evidence_dir` to an owner-only local directory. Keep the
same verified AWS variables in the test process. From the repository root:

```bash
export NPA_RAY_TRAIN_LIVE_CONFIG="$HOME/.config/ray-train/live.json"
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_ray_train_synthetic_live.py -v
```

Install the pinned Ray client and matching Torch in this checkout's own venv
first; they are optional reference dependencies. The suite independently loads
the downloaded model and SGD momentum, recomputes held-out loss, verifies two
distinct B200 hosts, decodes every RRD step and checks that recovered parameters
match uninterrupted training. It records intent before each native submission
and attempts cleanup for every owned ID even after a transport or assertion
failure. The hosting task, platform and provider teardown remain operator-owned.
Cancellation also verifies that the captured placement groups are removed and
the detached Train cleanup actor exits; terminal Jobs status alone is insufficient.

## Compatibility and licenses

The [Ray 2.58 Train storage contract](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/train/user-guides/persistent-storage.rst),
[V2 report API](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/train_fn_utils.py)
and [Torch 2.13.0 serialization](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/serialization.py)
govern this implementation. The committed live test is
`npa/tests/e2e/test_ray_train_synthetic_live.py`; it requires an explicitly
selected private runtime configuration and performs real GPU work.

This repository ships application source only. Ray and Arrow are Apache-2.0,
PyTorch uses its [BSD-style license](https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE),
and Rerun is Apache-2.0/MIT. The upstream image supplies CUDA/cuDNN under their
respective [NVIDIA terms](https://docs.nvidia.com/cuda/eula/index.html).
Runtime use requires the operator's acceptance; no new image is built or
redistributed here, and the NPA public image catalog is unchanged. Inputs are
generated tensors and outputs are this run's trained parameters and measurements.
