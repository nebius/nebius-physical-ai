# Conditional replanning primitives

This package contains the portable TRAIN-only conditional replanning algorithm:
the preregistered split and proposal noise, exact 1,233-feature serialization,
fixed linear and exact-GELU MLP fit, deterministic NumPy artifact, and
transactional k16 queue controller.

Callers supply verified proposal arrays, a versioned contract, the Torch module
used for fitting, and the inspected native policy wrapper. Runtime, checkpoint,
dataset, provider, and infrastructure identities remain workflow inputs.

The fixed fit and exported controller have been validated against the TRAIN-only
artifact and in two fresh native serving processes. This module does not
establish DEV performance, task score, or candidate eligibility by itself.
