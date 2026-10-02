# Both Slurm workers must have a working video decoder

The first sixteen-B200 training attempt passed distributed communication
checks but failed while opening the actual training videos. TorchCodec on
the new worker could not load FFmpeg's `libavutil` shared libraries. Sharing
the Python environment over NFS did not share those host libraries.

The failed allocation lasted 276 seconds. Its logs contained no completed
warmup-update marker, timed optimizer update or saved checkpoint. It is
excluded from performance and quality results. `decoder-failure-16.json`
records the failure, allocation status, log hashes and diagnostic counts.
Raw logs remain private.

Both native bootstrap scripts now install `ffmpeg`. Before model loading,
each worker imports the real TorchCodec decoder and decodes the first frame
from both front and wrist camera videos in the prepared dataset. The check
requires a three-channel, 256×256 frame. Missing libraries, missing videos,
decode errors or incompatible dimensions stop the launch.

The two unmodified `video-runtime-*.json` receipts show the actual corrected
hosts decoding identical frames with TorchCodec 0.10.0+cu130 and FFmpeg
6.1.1-3ubuntu5. Each receipt includes both frame hashes, shape, dtype and the
verified recipe source hash. Their worker labels identify the storage host
and added worker, respectively; they are not Slurm node-rank assignments.

To reproduce the check on each prepared worker, set `WAM_RECIPE` to the staged
recipe directory and `WAM_SHARED_ROOT` to the prepared data and runtime root:

```bash
"$WAM_SHARED_ROOT/framework/.venv/bin/python" - "$WAM_RECIPE" "$WAM_SHARED_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("wam_recipe", Path(sys.argv[1]) / "recipe.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
recipe._verify_video_runtime(Path(sys.argv[2]))
print("Both camera decoders passed")
PY
```

Normal `recipe.py run-node` execution performs the same check automatically.
The model source, data revisions and training hyperparameters were unchanged
by this host-dependency correction. The retry starts from the base checkpoint.
