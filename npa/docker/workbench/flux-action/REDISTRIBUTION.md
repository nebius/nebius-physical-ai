# FLUX Action candidate packaging

This is an unbuilt, unpublished candidate. Publication is quarantined in
`npa.deploy.images.UNVALIDATED_PUBLICATION_TOOLS` pending built-layer scans,
dependency notice review, and real GPU fine-tuning/export evidence.

- Source: Black Forest Labs `flux-action`, Apache-2.0, pinned to
  `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`. Its LICENSE remains in `/opt/flux-action`.
- Runtime: digest-pinned Python Debian base, upstream locked Torch 2.10/CUDA 12.8
  environment, NATTEN 0.21.6 wheel. NVIDIA wheel terms and notices require byte
  inspection before redistribution; the source license does not establish them.
- Weights: `black-forest-labs/flux-3-action-base` at
  `eb267865d35e49e4066bde4936237f8d9f15a68c`, FLUX Kommunity License v1.0.
  Fetched directly from Hugging Face at runtime using the operator's access.
  Model use, derivatives, and commercial eligibility remain subject to those
  terms. No local acceptance checkbox substitutes for provider authorization.
- Shared text encoder: the base repository includes Qwen3-VL-4B-Instruct under
  Apache-2.0. All encoder files are runtime downloads.
- Dataset: supplied by the operator, with its own rights and training split.
- Cache: node-local HF cache by default; durable storage exists only when the
  operator configures NPA model-cache plumbing. Never bake a populated cache.
- Outputs: checkpoints and exports retain the base identity and model license
  in `result.json`; no closed-loop task-performance result is implied.

Authoritative terms and source:
https://github.com/black-forest-labs/flux-action/blob/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f/LICENSE
https://huggingface.co/black-forest-labs/flux-3-action-base/blob/eb267865d35e49e4066bde4936237f8d9f15a68c/LICENSE.md
