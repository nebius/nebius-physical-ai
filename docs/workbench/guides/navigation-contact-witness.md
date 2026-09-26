# Spherical foot contact evidence

The reference navigation adapter uses the actual static scene geometry to
distinguish floor support from obstacles. Its terrain query applies only to
verified spherical feet. Base and other body contacts retain their original
classification, and unknown or ambiguous foot geometry retains the obstacle
measurement.

At native startup, every foot must have one enabled `UsdGeom.Sphere` collider
in the composed stage, including instance proxies, and the native body view
must report one shape. The guard checks exact body ownership, active and loaded
prims, positive finite radius, and uniform, positive, unsheared transforms.
The artifact records these live identities; a separately downloaded robot USD
is insufficient to establish the loaded collision geometry.

PhysX reports pair-oriented normals and changes the signed contact force when
the sensor is the second actor. For a sphere against a triangle mesh, its GPU
contact generator reports a point on the sphere. The terrain-side witness is
`point - separation * sign(force) * normal`. These semantics are visible in the
[tensor CUDA getter](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/omni/ovruntime/source/omni.physx.tensors/plugins/gpu/CudaKernels.cu)
and [sphere/mesh contact generator](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/gpunarrowphase/src/CUDA/convexMeshOutput.cu).
This source reference does not establish binary equivalence with a runtime.
Fresh native qualification must verify the actual identities and resulting
geometry evidence before claiming support for that runtime.

The witness requires finite values, nonzero signed force and a unit normal
within the recorded float32 arithmetic tolerance. The resolved foot contact
offset bounds the witness-to-source distance and the local geometry search.
It is a contact onset distance, not a maximum negative penetration bound.
The same upward-facing, connected-surface, wall, boundary, topology and root
height checks apply around the witness. No collision shape, offset, physics
setting, reset distribution or acceptance threshold is modified. The separate
physical-failure measurements remain in force.

Contact evidence schema `npa.navigation.probe-contact-samples.v3` retains the
original point, normal, signed force and separation, and adds
`terrain_witness_world_m` and `support_witness_valid`. The index records each
foot's live collider identity, transform, radius and resolved offsets. Invalid
witnesses retain the original obstacle flag. Samples still select the strongest
original candidate per robot and control interval; they do not enumerate every
contact. Historical v1/v2 artifacts retain their original meanings and must
not be decoded as v3.
