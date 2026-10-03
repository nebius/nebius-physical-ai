# Alpamayo 2 Super redistribution record

Review date: 2026-08-17. Engineering classification, not legal advice.

The image is `public`: it contains NVIDIA's Apache-2.0 Alpamayo 2 inference
source at immutable commit `beb2977d9a7e9d66837d4a3ad5144ff59de37519`, the
redistributable CUDA 12.8 base/runtime, and NPA's Apache-2.0 integration. It
contains no checkpoint, dataset sample, Hugging Face cache, or credential.
The one source modification is prominently marked and makes the upstream
PhysicalAI-AV interface consume NPA's immutable dataset revision environment
instead of resolving a moving `main` branch.

`nvidia/Alpamayo2-Super@00554695e729a6ff0b6281fd2c81b18d06e33dbe` is
OpenMDW-1.1. That agreement permits dealing in Model Materials without
restriction, requires retaining its agreement and applicable notices upon
distribution, imposes no output restrictions, and is accepted by exercising
its rights. NPA nevertheless runtime-fetches weights to keep provenance exact
and avoid publishing a ~70 GiB checkpoint in an image.

The sample source is separate and stricter:
`nvidia/PhysicalAI-Autonomous-Vehicles@b719eea7f0a63619ef51ec7f54178af0937ef050`
is gated by the NVIDIA Autonomous Vehicle Dataset License Agreement. It is
non-transferable and cannot be distributed or hosted for others. The operator
must accept it interactively on Hugging Face; NPA probes that real entitlement
before provisioning and downloads only into the operator's ephemeral cache.

Release qualification requires scanning the built digest and every layer for
weights, dataset assets, populated caches, and credentials, plus real inference
on each claimed GPU architecture. An image build or import smoke is not model
validation.
