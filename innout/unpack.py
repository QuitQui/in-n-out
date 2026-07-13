"""Pull-side output finalization.

The decrypted blob is opaque bytes. Directories pushed through innout are
tar.gz archives, which used to land as an unnamed `result` file that needed
two manual unpack rounds (result -> xxx.tar -> xxx/). This module inspects
the blob and hands the user `<topdir>.tar` directly. Everything is pure
Python (gzip/tarfile/shutil), so behaviour is identical on Windows and Linux.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
from pathlib import Path

_GZIP_MAGIC = b"\x1f\x8b"
_COPY_CHUNK = 1024 * 1024


def _is_gzip(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == _GZIP_MAGIC


def _safe_top_name(tar_path: Path) -> str | None:
    """Top-level name of the archive's first member, or None if unusable."""
    try:
        with tarfile.open(tar_path, "r:") as tar:
            member = tar.next()
    except tarfile.TarError:
        return None
    if member is None:
        return None
    # tar member names always use "/" separators, even for archives
    # created on Windows
    top = member.name.split("/")[0].strip()
    if not top or top in {".", ".."} or "\\" in top or "\x00" in top:
        return None
    return top


def _unclaimed(output_dir: Path, stem: str, suffix: str) -> Path:
    candidate = output_dir / f"{stem}{suffix}"
    n = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def finalize_output(blob: Path, output_dir: Path) -> Path:
    """Move the decrypted blob into output_dir under its friendliest name.

    gzip data is decompressed once so tar.gz archives become `<topdir>.tar`
    (named after the archive's first member). Unrecognized data keeps the
    legacy name `result`. Existing files are never overwritten — a `-N`
    suffix is added instead. Returns the final path.
    """
    blob = Path(blob)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work = blob
    if _is_gzip(work):
        decompressed = work.with_name(work.name + ".ungz")
        try:
            with gzip.open(work, "rb") as src, open(decompressed, "wb") as dst:
                shutil.copyfileobj(src, dst, _COPY_CHUNK)
        except (OSError, EOFError):
            # 2-byte magic false positive: not really gzip, keep original
            decompressed.unlink(missing_ok=True)
        else:
            work.unlink()
            work = decompressed

    if tarfile.is_tarfile(work):
        top = _safe_top_name(work)
        stem, suffix = (top, ".tar") if top else ("result", ".tar")
    else:
        stem, suffix = "result", ""

    final = _unclaimed(output_dir, stem, suffix)
    shutil.move(work, final)
    return final
