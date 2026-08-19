# Isaac Lab image third-party notices

| Component | Distribution | Notice |
| --- | --- | --- |
| FFmpeg | Ubuntu snapshot packages (`ffmpeg` and shared libraries) | GPL/LGPL components under the package copyright files in `/usr/share/doc`; the image does not contain the static executable bundled by the `imageio-ffmpeg` wheel. |
| imageio-ffmpeg 0.6.0 | PyPI wrapper, BSD-2-Clause | The wrapper remains for MoviePy compatibility, is forced to `/usr/bin/ffmpeg`, and its separately licensed wheel-bundled executable is removed in the same build layer that installs it. |
