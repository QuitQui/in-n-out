"""Source acquisition for in-n-out.

Acquires a source (URL, local path, GitHub repo, or HuggingFace repo) and
returns a single file path (archive) ready for encryption.
"""

from __future__ import annotations

import subprocess
import tarfile
from fnmatch import fnmatch
from pathlib import Path

import requests
from tqdm import tqdm


def _is_excluded(rel_path: Path, patterns: list[str]) -> bool:
    """True if rel_path matches any exclude pattern.

    A pattern matches if it glob-matches the full relative path, the file
    name, or any single path component (so ``data`` skips a whole
    directory tree).
    """
    for pat in patterns:
        pat = pat.rstrip("/")
        if fnmatch(rel_path.as_posix(), pat) or fnmatch(rel_path.name, pat):
            return True
        if any(fnmatch(part, pat) for part in rel_path.parts):
            return True
    return False


def _tar_gz_dir(
    source_dir: Path, work_dir: Path, excludes: list[str] | None = None
) -> Path:
    """Create a tar.gz archive of source_dir inside work_dir and return its path.

    excludes: glob patterns; matching files/directories are left out of the
    archive (directories are skipped whole, without recursing).
    """
    archive_name = source_dir.name + ".tar.gz"
    archive_path = work_dir / archive_name
    patterns = list(excludes or [])

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        rel = Path(info.name).relative_to(source_dir.name)
        if rel.parts and _is_excluded(rel, patterns):
            return None
        return info

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name, filter=_filter)
    return archive_path


def acquire(
    source_type: str,
    source: str,
    work_dir: Path,
    excludes: list[str] | None = None,
    repo_type: str = "model",
) -> Path:
    """Download / copy / clone the source into work_dir and return the path
    to a single file (tar.gz for directories/repos, original file for URLs/local files).

    source_type: one of 'url' | 'local' | 'github' | 'hf'
    source:
      - 'url'    → https://... URL to download
      - 'local'  → path string; if it's a directory, tar.gz it first
      - 'github' → "owner/repo" or "owner/repo@branch"
      - 'hf'     → "org/repo-name" HuggingFace repo ID
    work_dir: scratch directory to place intermediate files
    repo_type: HuggingFace repo kind ('model' | 'dataset' | 'space'); only
      used when source_type is 'hf'

    Returns: Path to a single file inside work_dir
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if source_type == "url":
        return _acquire_url(source, work_dir)
    elif source_type == "local":
        return _acquire_local(source, work_dir, excludes)
    elif source_type == "github":
        return _acquire_github(source, work_dir, excludes)
    elif source_type == "hf":
        return _acquire_hf(source, work_dir, excludes, repo_type)
    else:
        raise ValueError(
            f"Unknown source_type {source_type!r}. "
            "Must be one of: 'url', 'local', 'github', 'hf'."
        )


def _acquire_url(url: str, work_dir: Path) -> Path:
    """Download a file from a URL with a tqdm progress bar."""
    filename = url.split("/")[-1].split("?")[0] or "downloaded_file"
    dest = work_dir / filename

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0)) or None
    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    return dest


def _acquire_local(
    source: str, work_dir: Path, excludes: list[str] | None = None
) -> Path:
    """Return the local path as-is for files, or tar.gz a directory."""
    path = Path(source)
    if path.is_dir():
        return _tar_gz_dir(path, work_dir, excludes)
    return path


def _acquire_github(
    source: str, work_dir: Path, excludes: list[str] | None = None
) -> Path:
    """Clone a GitHub repo (optionally at a branch) and return a tar.gz archive."""
    # Parse optional @branch suffix
    if "@" in source:
        repo_spec, branch = source.rsplit("@", 1)
    else:
        repo_spec, branch = source, None

    owner, repo = repo_spec.split("/", 1)
    clone_url = f"https://github.com/{owner}/{repo}.git"
    dest_dir = work_dir / repo

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(dest_dir)]

    subprocess.run(cmd, check=True)

    return _tar_gz_dir(dest_dir, work_dir, excludes)


def _acquire_hf(
    source: str,
    work_dir: Path,
    excludes: list[str] | None = None,
    repo_type: str = "model",
) -> Path:
    """Download a HuggingFace repo snapshot and return a tar.gz archive.

    repo_type must match the repo's kind on the Hub — models and datasets
    live in separate namespaces, so a dataset ID resolved as a model 404s.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import]

    repo_name = source.split("/")[-1]
    local_dir = work_dir / repo_name

    snapshot_download(repo_id=source, repo_type=repo_type, local_dir=str(local_dir))

    return _tar_gz_dir(local_dir, work_dir, excludes)


def acquire_dir(
    source_type: str,
    source: str,
    work_dir: Path,
    repo_type: str = "model",
) -> Path:
    """Acquire a source as a *directory*, without archiving it.

    acquire() returns a single file, which means directory sources get
    tarred to disk first -- a second full-size copy. The streaming push
    pipeline tars on the fly instead, so it needs the directory itself.

    Re-running this is cheap: snapshot_download skips files already
    present and an existing clone is left alone, so an interrupted
    multi-hour push resumes without re-fetching.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if source_type == "hf":
        from huggingface_hub import snapshot_download  # type: ignore[import]

        local_dir = work_dir / source.split("/")[-1]
        snapshot_download(
            repo_id=source, repo_type=repo_type, local_dir=str(local_dir)
        )
        return local_dir

    if source_type == "github":
        repo_spec, branch = (
            source.rsplit("@", 1) if "@" in source else (source, None)
        )
        _, repo = repo_spec.split("/", 1)
        dest_dir = work_dir / repo
        if not dest_dir.exists():
            cmd = ["git", "clone", "--depth", "1"]
            if branch:
                cmd += ["--branch", branch]
            cmd += [f"https://github.com/{repo_spec}.git", str(dest_dir)]
            subprocess.run(cmd, check=True)
        return dest_dir

    if source_type == "local":
        path = Path(source).expanduser()
        if not path.is_dir():
            raise ValueError(
                f"streaming push needs a directory, but {source!r} is not one"
            )
        return path

    raise ValueError(
        f"source_type {source_type!r} has no directory form; "
        "streaming push supports 'hf', 'github', and 'local' directories"
    )
