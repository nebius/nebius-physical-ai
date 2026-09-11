# Third-party notices for the neutral robomimic candidate

The intended image contains `ARISE-Initiative/robomimic` at commit
`d309eaecc18acf4152a830a895a6984b8ac71b05`, licensed under MIT. Its exact
`LICENSE` SHA-256 is
`7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556`.

The base is the digest-pinned official Python 3.11 slim Bookworm image. The
`baked-requirements.lock` file identifies every Python distribution intended to
be added. A future authorized build must generate and review a complete package
and license inventory for the base and all locked distributions; this source
file does not itself establish redistribution eligibility for resulting bytes.

PyTorch, torchvision, Triton, and the NVIDIA CUDA/cuDNN/NCCL distributions named
by `runtime-requirements.lock` are not included in the candidate. That lock is a
compatibility declaration for an independently prepared external runtime. Its
artifact hashes and installed-file inventory must be supplied and verified at
the operator boundary after the applicable rights decision. PyTorch source
licensing is not evidence that wheel or bundled binary dependencies may be
redistributed or used as a hosted service.

The official Lift proficient-human low-dimensional dataset is also excluded. A
later authorized live gate fetches only revision
`74fa018461f479cd9fd15b924a16103012096203`, path
`v1.5/lift/ph/low_dim_v15.hdf5`, and accepts only SHA-256
`2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540`
with size 21,084,088 bytes. Dataset access and use remain the operator's separate
responsibility.
