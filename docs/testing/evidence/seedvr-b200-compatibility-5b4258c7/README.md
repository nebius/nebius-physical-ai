# SeedVR: why the current image needs a B200 rebuild

Native binary inspection of the immutable `5b4258c7…` image found **SM90 cubins only, with no PTX, in all 12 FlashAttention/Apex extensions**. The inspection ran without a GPU or network. [Exact image, tool, binary and output hashes](review.json) make the finding reviewable.

Torch itself supports SM100, but the model uses the separately compiled FlashAttention varlen kernel. We infer that this image cannot execute that path unchanged on B200: NVIDIA documents that a kernel needs compatible cubin or PTX. [NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)

The fresh H200 preflight also found zero available project quota, so no new H200 resources were created. The old failed operation was separately reconciled after four exact resources were confirmed absent.

The next candidate will retain the same dependency/model versions and native backends while adding SM100 kernels alongside SM90. It must pass fresh image qualification, actual GPU kernel controls and the unchanged paired restoration protocol. **No new GPU quality acceptance or H100/H200 performance equivalence is claimed.** Earlier quality failures remain failures.

[Verification hashes](SHA256SUMS)
