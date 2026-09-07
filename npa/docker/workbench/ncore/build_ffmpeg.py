"""Build the unmodified, pinned official FFmpeg release without GPL/nonfree code."""

import os
import shutil

from native_build_support import ANNEX, FFMPEG, ROOT, record, run, unpack


def build_ffmpeg() -> None:
    source = unpack("ffmpeg", "8.1.1")
    # These are official FFmpeg switches, on the untouched official release.
    # No PyAV FFmpeg patch, external GPL codecs, or binary vendor libraries.
    run(
        "./configure",
        f"--prefix={FFMPEG}",
        "--enable-shared",
        "--disable-static",
        "--disable-gpl",
        "--disable-nonfree",
        "--disable-version3",
        "--disable-autodetect",
        "--enable-zlib",
        "--disable-doc",
        cwd=source,
    )
    run("make", f"-j{len(os.sched_getaffinity(0))}", cwd=source)
    run("make", "install", cwd=source)
    (ANNEX / "ffmpeg-build").mkdir()
    for name in ("config.log", "config.mak", "config.sh"):
        shutil.copyfile(source / "ffbuild" / name, ANNEX / "ffmpeg-build" / name)
    for name in ("config.h", "config_components.h"):
        shutil.copyfile(source / name, ANNEX / "ffmpeg-build" / name)
    # Preserve the actual configuration/log/recipe and all original notices.
    shutil.copyfile(source / "COPYING.LGPLv2.1", FFMPEG / "COPYING.LGPLv2.1")
    shutil.copyfile(source / "LICENSE.md", FFMPEG / "LICENSE.md")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    build_ffmpeg()
    record("ffmpeg")
