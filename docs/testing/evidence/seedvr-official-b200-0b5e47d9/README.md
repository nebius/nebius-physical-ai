# SeedVR2 7B official controls on B200

These are the two predeclared official controls used to bind the forthcoming paired VAE diagnostic to the current producer. They are the actual unedited MP4 artifacts: [main](main-official.mp4) (100 frames, 2 seconds) and [light](light-official.mp4) (25 frames, 0.5 seconds), both 640×480 at 50 fps. Independent review decoded every frame and verified output, latent, source, image and mounted-helper hashes. This accepts their identity, not their quality.

The main control improves median LPIPS from 0.11443 for bicubic to 0.09795. Its temporal error is 0.0024721, exceeding the unchanged 0.0017999 maximum. **The temporal gate fails; this is not an overall quality pass.** Full numeric aggregates and original gate results are in [measurements.json](measurements.json). Earlier quality failures remain valid. Neither control was selected for favorable metrics, and the held-out episode remains sealed.

The source is `0b5e47d97414a2fe46d6184efe9b4423b2b52198`; the private runtime manifest is identified by digest in the measurements. B200 support is an architecture change, not a quality mechanism. This pack contains no paired VAE diagnostic result. Mandatory private runtime gates passed; complete-byte image scanning and publication/readiness qualification remain pending.

The main input was a 320×240 degradation of the retained 640×480 source; the light control uses the first 25 frames at source resolution. The model output and encoding are transformations of the source footage. See [attribution.md](attribution.md). No robotics safety, downstream policy improvement, or real-sensor generalization is established by these two clips.
