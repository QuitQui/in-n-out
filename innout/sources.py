"""Source acquisition for in-n-out.

Acquires a source (URL, local path, GitHub repo, or HuggingFace repo) and
returns a single file path (archive) ready for encryption.
"""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def _tar_gz_dir(source_dir: Path, work_dir: Path) -> Path:
    """Create a tar.gz archive of source_dir inside work_dir and return its path."""
    archive_name = source_dir.name + ".tar.gz"
    archive_path = work_dir / archive_name
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    return archive_path


def acquire(source_type: str, source: str, work_dir: Path) -> Path:
    """Download / copy / clone the source into work_dir and return the path
    to a single file (tar.gz for directories/repos, original file for URLs/local files).

    source_type: one of 'url' | 'local' | 'github' | 'hf'
    source:
      - 'url'    → https://... URL to download
      - 'local'  → path string; if it's a directory, tar.gz it first
      - 'github' → "owner/repo" or "owner/repo@branch"
      - 'hf'     → "org/model-name" HuggingFace repo ID
    work_dir: scratch directory to place intermediate files

    Returns: Path to a single file inside work_dir
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if source_type == "url":
        return _acquire_url(source, work_dir)
    elif source_type == "local":
        return _acquire_local(source, work_dir)
    elif source_type == "github":
        return _acquire_github(source, work_dir)
    elif source_type == "hf":
        return _acquire_hf(source, work_dir)
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


def _acquire_local(source: str, work_dir: Path) -> Path:
    """Return the local path as-is for files, or tar.gz a directory."""
    path = Path(source)
    if path.is_dir():
        return _tar_gz_dir(path, work_dir)
    return path


def _acquire_github(source: str, work_dir: Path) -> Path:
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

    return _tar_gz_dir(dest_dir, work_dir)


def _acquire_hf(source: str, work_dir: Path) -> Path:
    """Download a HuggingFace repo snapshot and return a tar.gz archive."""
    from huggingface_hub import snapshot_download  # type: ignore[import]

    repo_name = source.split("/")[-1]
    local_dir = work_dir / repo_name

    snapshot_download(repo_id=source, local_dir=str(local_dir))

    return _tar_gz_dir(local_dir, work_dir)
