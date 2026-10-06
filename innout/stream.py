"""Single-pass streaming pipeline: directory -> tar -> encrypt -> parts.

The original push path wrote every stage to disk -- snapshot, then tar,
then the encrypted blob, then the parts -- so a 105 GB source needed
420 GB of free space. This module threads the stages together so only
one part exists at a time: peak cost is the source plus one part.

Concatenating every part in order reproduces exactly the file the old
``encrypt_stream`` + ``split_file`` pair produced, so the format is
unchanged and any version of in-n-out can decrypt the result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

from innout import crypto
from innout.sources import _is_excluded

BLOCK_SIZE = 4 * 1024 * 1024  # 4 MB
MAX_PARTS = 1000  # part index is zero-padded to 3 digits

_SUFFIXES = {
    "b": 1,
    "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
    "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4,
}


def parse_size(text: str) -> int:
    """Parse a human size like '1.8GiB', '512MB', '2048' (bytes) -> int.

    Decimal suffixes are powers of 1000, IEC suffixes powers of 1024. A
    bare number is bytes.
    """
    s = str(text).strip().lower().replace("_", "").replace(" ", "")
    if not s:
        raise ValueError("empty size")
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if s.endswith(suffix):
            number, unit = s[: -len(suffix)], _SUFFIXES[suffix]
            break
    else:
        number, unit = s, 1
    if not number:
        raise ValueError(f"size {text!r} has a unit but no number")
    try:
        value = float(number)
    except ValueError as exc:
        raise ValueError(f"cannot parse size {text!r}") from exc
    if value <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return int(value * unit)


def part_name(prefix: str, index: int, suffix: str = "") -> str:
    """Name of part `index`, e.g. "Video-MME-v2.part007" or, with a
    camouflage suffix, "Video-MME-v2.part007.pdf".

    The suffix is cosmetic only -- it changes how the file looks in a
    listing, not what it is. The bytes are AES-256-GCM either way, and
    the pull side recovers the suffix from the names it finds.
    """
    if index >= MAX_PARTS:
        raise ValueError(
            f"part index {index} exceeds the {MAX_PARTS}-part limit; "
            "use a larger --part-size"
        )
    return f"{prefix}.part{index:03d}{suffix}"


# Splits "Video-MME-v2.part007.pdf" into prefix, index, suffix. Greedy on
# the prefix so a name that happens to contain ".partNNN" earlier still
# anchors on the last one.
PART_RE = re.compile(r"^(?P<prefix>.+)\.part(?P<index>\d{3})(?P<suffix>.*)$")


@contextmanager
def tar_stream(
    source_dir: Path,
    excludes: list[str] | None = None,
) -> Iterator[BinaryIO]:
    """Yield a readable stream of an uncompressed tar of source_dir.

    Uncompressed on purpose: these payloads are already-compressed media
    (H.265 inside zip), where gzip costs hours of CPU for ~0% gain.

    tarfile writes, we need to read, so the archive is built by a thread
    writing into a pipe. Stream mode ("w|") never seeks, which is what
    makes a pipe a legal target.

    The byte stream is reproducible across runs for an unchanged source
    tree -- tarfile walks directories in sorted order and takes mtimes
    from disk -- which is what lets an interrupted push resume by
    regenerating and skipping the parts already uploaded.
    """
    source_dir = Path(source_dir)
    patterns = list(excludes or [])

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        rel = Path(info.name)
        if rel.parts and rel.parts[0] == source_dir.name:
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path()
        if rel.parts and _is_excluded(rel, patterns):
            return None
        return info

    read_fd, write_fd = os.pipe()
    error: list[BaseException] = []

    def _build() -> None:
        try:
            with open(write_fd, "wb", closefd=True) as writer:
                with tarfile.open(fileobj=writer, mode="w|") as tar:
                    tar.add(source_dir, arcname=source_dir.name, filter=_filter)
        except BaseException as exc:  # includes BrokenPipeError on early close
            error.append(exc)

    thread = threading.Thread(target=_build, name="innout-tar", daemon=True)
    thread.start()
    reader = open(read_fd, "rb", closefd=True)
    try:
        yield reader
    finally:
        # Close first: a consumer that bailed out leaves the writer blocked
        # on a full pipe, and closing the read end unblocks it with EPIPE.
        reader.close()
        thread.join()
    if error:
        raise error[0]


@dataclass
class PartInfo:
    """A completed part: what to verify it with, and whether it was kept."""

    index: int
    name: str
    size: int
    sha256: str
    path: Path | None = None  # None when skipped (resume) rather than written


@dataclass
class StreamResult:
    parts: list[PartInfo] = field(default_factory=list)
    plaintext_bytes: int = 0
    salt: bytes = b""
    nonce: bytes = b""

    @property
    def total_bytes(self) -> int:
        return sum(p.size for p in self.parts)


def encrypt_to_parts(
    src: BinaryIO,
    passphrase: str,
    out_dir: Path,
    prefix: str,
    part_size: int,
    *,
    salt: bytes | None = None,
    nonce: bytes | None = None,
    on_part: Callable[[PartInfo], None] | None = None,
    suffix: str = "",
    skip_before: int = 0,
    skip_indices: set[int] | None = None,
    delete_after: bool = True,
    block_size: int = BLOCK_SIZE,
) -> StreamResult:
    """Encrypt everything readable from src into numbered part files.

    Each completed part is handed to on_part (which uploads it) and then
    deleted unless delete_after is False, so disk holds one part at a
    time.

    skip_before and skip_indices support resume: those parts are hashed
    but never written to disk. The cipher stream has to run through them
    anyway -- AES-GCM is one continuous stream over the whole archive, so
    byte N depends on every byte before it -- but there is no need to
    spend the disk or the upload. Pass the original salt and nonce so the
    regenerated stream matches what was already uploaded.

    skip_indices takes an arbitrary set rather than a prefix because an
    interrupted push leaves holes, not a clean cut: if part 7 failed while
    8-20 landed, only 7 needs re-sending.

    Returns a StreamResult whose parts carry the sha256 of every part,
    including skipped ones, so a resumed run still produces a complete
    manifest.
    """
    if part_size <= 0:
        raise ValueError(f"part_size must be positive, got {part_size}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = set(skip_indices or ())
    enc = crypto.StreamEncryptor(passphrase, salt=salt, nonce=nonce)
    result = StreamResult(salt=enc.salt, nonce=enc.nonce)

    def _wanted(i: int) -> bool:
        return i >= skip_before and i not in skipped

    index = 0
    handle = None
    digest = hashlib.sha256()
    written = 0

    def _open() -> None:
        nonlocal handle, digest, written
        digest = hashlib.sha256()
        written = 0
        handle = (
            open(out_dir / part_name(prefix, index, suffix), "wb")
            if _wanted(index) else None
        )

    def _close() -> None:
        nonlocal index, handle
        if handle is not None:
            handle.close()
            handle = None
        name = part_name(prefix, index, suffix)
        path = out_dir / name if _wanted(index) else None
        info = PartInfo(index=index, name=name, size=written, sha256=digest.hexdigest(), path=path)
        result.parts.append(info)
        if on_part is not None:
            on_part(info)
        if delete_after and path is not None:
            path.unlink(missing_ok=True)
            info.path = None
        index += 1

    def _emit(data: bytes) -> None:
        nonlocal written
        view = memoryview(data)
        while view:
            room = part_size - written
            take, view = view[:room], view[room:]
            digest.update(take)
            if handle is not None:
                handle.write(take)
            written += len(take)
            if written >= part_size:
                _close()
                _open()

    _open()
    _emit(enc.header)
    while True:
        block = src.read(block_size)
        if not block:
            break
        result.plaintext_bytes += len(block)
        _emit(enc.update(block))
    _emit(enc.finalize())
    if written or not result.parts:
        _close()
    elif handle is not None:  # an exactly-full final part left an empty file open
        handle.close()
        (out_dir / part_name(prefix, index, suffix)).unlink(missing_ok=True)

    return result


def write_manifest(parts: list[PartInfo], path: Path) -> Path:
    """Write a sha256sum-compatible manifest so the pull side can verify
    every part before trusting a 105 GB decrypt.

    GCM only authenticates at the very end of the stream, so without this
    a single corrupted part is discovered after writing the whole output.
    """
    path = Path(path)
    lines = [f"{p.sha256}  {p.name}\n" for p in sorted(parts, key=lambda p: p.index)]
    path.write_text("".join(lines))
    return path


def read_manifest(path: Path) -> dict[str, str]:
    """Parse a sha256sum-style manifest into {name: sha256}."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if not name:
            digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if not name or len(digest) != 64:
            raise ValueError(f"cannot parse manifest line: {line!r}")
        out[name] = digest.lower()
    return out


def sha256_file(path: Path, block_size: int = BLOCK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    """Load resume state, or an empty dict when there is none."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: Path, state: dict) -> None:
    """Persist resume state atomically -- a half-written state file would
    strand the upload with no way to resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


# --- pull side --------------------------------------------------------------


def find_parts(
    from_dir: Path,
    prefix: str | None = None,
    suffix: str | None = None,
) -> list[Path]:
    """The part files in from_dir, in join order.

    Both the prefix and any camouflage suffix are recovered from the
    names, so a folder of ``Video-MME-v2.part000.pdf`` restores with no
    extra flags -- the operator should not have to remember what the push
    was disguised as. Pass prefix or suffix to disambiguate a directory
    holding more than one push.

    A directory holding two different pushes is refused rather than
    joined: concatenating parts from two cipher streams decrypts to
    nothing, and saying so now beats finding out after 105 GB.
    """
    from_dir = Path(from_dir)
    found: dict[tuple[str, str], dict[int, Path]] = {}
    for path in from_dir.iterdir():
        if not path.is_file():
            continue
        match = PART_RE.match(path.name)
        if not match:
            continue
        key = (match["prefix"], match["suffix"])
        if prefix is not None and key[0] != prefix:
            continue
        if suffix is not None and key[1] != suffix:
            continue
        found.setdefault(key, {})[int(match["index"])] = path

    if not found:
        wanted = "*.part???" + (suffix if suffix else "")
        raise ValueError(f"no part files ({wanted}) found in {from_dir}")

    if len(found) > 1:
        shown = ", ".join(
            f"{p}.partNNN{s}" for p, s in sorted(found)
        )
        raise ValueError(
            f"{from_dir} holds {len(found)} different pushes ({shown}); "
            "joining them would decrypt to garbage. Pass --part-prefix or "
            "--part-suffix to pick one"
        )

    (_, _), by_index = next(iter(found.items()))
    missing = sorted(set(range(max(by_index) + 1)) - set(by_index))
    if missing:
        raise ValueError(
            f"part(s) {missing} are missing from {from_dir}. The archive is "
            "one continuous cipher stream, so every part has to be present"
        )
    return [by_index[i] for i in sorted(by_index)]


def verify_parts(parts: list[Path], manifest: dict[str, str]) -> None:
    """Check every part against the manifest before decrypting anything.

    Worth the extra read: GCM only authenticates at the end of the stream,
    so without this a single corrupt part is only discovered after writing
    the entire output.
    """
    unknown = [p.name for p in parts if p.name not in manifest]
    if unknown:
        raise ValueError(f"manifest has no entry for: {', '.join(unknown)}")
    bad = [p.name for p in parts if sha256_file(p) != manifest[p.name]]
    if bad:
        raise ValueError(
            f"sha256 mismatch on {', '.join(bad)} -- re-download these parts"
        )
    absent = sorted(set(manifest) - {p.name for p in parts})
    if absent:
        raise ValueError(f"manifest lists part(s) not present: {', '.join(absent)}")


def iter_plaintext(
    parts: list[Path],
    passphrase: str,
    *,
    delete_consumed: bool = False,
    on_part: Callable[[Path], None] | None = None,
    block_size: int = BLOCK_SIZE,
) -> Iterator[bytes]:
    """Decrypt parts in order, yielding plaintext blocks.

    With delete_consumed each part is removed once it has been fed
    through, which halves what a restore needs: otherwise the parts and
    the restored tree sit side by side -- 105 GB + 105 GB for
    Video-MME-v2.
    """
    dec = crypto.StreamDecryptor(passphrase)
    for path in parts:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    break
                out = dec.update(block)
                if out:
                    yield out
        if on_part is not None:
            on_part(path)
        if delete_consumed:
            path.unlink(missing_ok=True)
    yield dec.finalize()


@contextmanager
def plaintext_stream(
    parts: list[Path],
    passphrase: str,
    *,
    delete_consumed: bool = False,
    on_part: Callable[[Path], None] | None = None,
) -> Iterator[BinaryIO]:
    """Yield a readable stream of the decrypted plaintext.

    Mirror image of tar_stream: a thread pushes decrypted blocks into a
    pipe so tarfile can pull them, letting a restore untar on the fly
    instead of materialising a 105 GB tar first.
    """
    read_fd, write_fd = os.pipe()
    error: list[BaseException] = []

    def _feed() -> None:
        try:
            with open(write_fd, "wb", closefd=True) as writer:
                for block in iter_plaintext(
                    parts, passphrase,
                    delete_consumed=delete_consumed, on_part=on_part,
                ):
                    writer.write(block)
        except BaseException as exc:  # includes BrokenPipeError on early close
            error.append(exc)

    thread = threading.Thread(target=_feed, name="innout-decrypt", daemon=True)
    thread.start()
    reader = open(read_fd, "rb", closefd=True)
    try:
        yield reader
    finally:
        # Close first: a consumer that bailed out leaves the feeder blocked
        # on a full pipe, and closing the read end unblocks it with EPIPE.
        reader.close()
        thread.join()
    if error:
        raise error[0]
