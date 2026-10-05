"""Batch push of a large HuggingFace repo to Drive, one file per leaf folder.

A 100 GB dataset cannot go through `innout push --hf` in one shot: the
pipeline would need the snapshot, its tar.gz, the ciphertext and the chunks
on disk simultaneously — roughly 4x the source size. This module walks the
repo file by file instead, so peak disk stays at about 2x the *largest
single file*, and records what it did so an interrupted run resumes.

Each file lands in its own leaf folder, because `pull` joins every file it
finds in a folder — two sessions sharing one folder decrypt to garbage.

Small files (anything not matched by --split) are bundled into a single
tar.gz session so they do not each get a folder of their own.

The passphrase is read only from $INNOUT_PASSPHRASE: passing it as an
argument would expose it in shell history and in `ps` output. It is never
written to the ledger — losing it means the uploaded data is unrecoverable.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from innout import crypto, drive, splitter, unpack

_LEDGER_DIR = Path.home() / ".innout_batches"
_HASH_CHUNK = 4 * 1024 * 1024
#: Leaf folder that collects every file --split does not match.
BUNDLE_NAME = "annotations"
_BUNDLE_PREFIX = "<bundle:"


def _utc_now() -> str:
    """ISO-8601 UTC, second precision — stable and sortable in the ledger."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_HASH_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def leaf_name(repo_path: str) -> str:
    """Turn a repo-relative path into a Drive folder name.

    "videos/001.zip" -> "videos-001". Slashes become dashes so the whole
    path stays visible in a flat folder name, and the extension is dropped
    because the folder holds chunks, not the file itself.
    """
    tail = repo_path.rsplit("/", 1)[-1]
    stem = repo_path.rsplit(".", 1)[0] if "." in tail else repo_path
    name = stem.strip("/").replace("/", "-")
    if not name:
        raise ValueError(f"Cannot derive a folder name from {repo_path!r}")
    return name


def assign_leaves(repo_paths: list[str], drive_parent: str) -> dict[str, str]:
    """Map each repo path to its full nested Drive path, rejecting collisions.

    "videos/001.zip" and "videos-001.zip" both slugify to "videos-001";
    silently merging them into one folder would corrupt both uploads.
    """
    leaves: dict[str, str] = {}
    seen: dict[str, str] = {}
    for repo_path in repo_paths:
        name = leaf_name(repo_path)
        if name in seen:
            raise ValueError(
                f"Drive folder name collision: {repo_path!r} and "
                f"{seen[name]!r} both map to {name!r}. Rename or exclude one."
            )
        seen[name] = repo_path
        leaves[repo_path] = f"{drive_parent}/{name}"
    return leaves


def select_files(
    all_paths: list[str], split_globs: list[str]
) -> tuple[list[str], list[str]]:
    """Partition repo paths into (one-folder-each, bundled-together).

    Matching runs against the full repo-relative path, so "videos/*"
    matches "videos/001.zip" but not a top-level "videos.txt".
    """
    split: list[str] = []
    bundled: list[str] = []
    for path in sorted(all_paths):
        if any(fnmatch.fnmatch(path, pattern) for pattern in split_globs):
            split.append(path)
        else:
            bundled.append(path)
    return split, bundled


def bundle_key(bundle_name: str) -> str:
    """Ledger key for the bundled session.

    Angle brackets cannot appear in a HuggingFace filename, so this can
    never collide with a real repo path.
    """
    return f"{_BUNDLE_PREFIX}{bundle_name}>"


def is_bundle_key(key: str) -> bool:
    return key.startswith(_BUNDLE_PREFIX)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def ledger_path(repo_id: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return _LEDGER_DIR / f"{repo_id.replace('/', '__')}.json"


def load_ledger(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def save_ledger(path: Path, ledger: dict) -> None:
    """Write the ledger atomically — a crash mid-write must not lose it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def pending_entries(ledger: dict, targets: dict[str, str]) -> list[str]:
    """Targets not yet recorded as pushed, in deterministic order."""
    done = ledger.get("entries", {})
    return [p for p in sorted(targets) if p not in done]


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def _get_passphrase() -> str:
    passphrase = os.environ.get("INNOUT_PASSPHRASE")
    if not passphrase:
        raise SystemExit(
            "error: INNOUT_PASSPHRASE is not set.\n"
            "  Set it without leaving a shell-history trace:\n"
            "    read -rs INNOUT_PASSPHRASE && export INNOUT_PASSPHRASE\n"
            "  Losing this passphrase makes the uploaded data unrecoverable."
        )
    return passphrase


def push_one_file(
    local_path: Path,
    drive_leaf: str,
    passphrase: str,
    work_dir: Path,
    chunk_size_mb: int,
    credentials: str | None,
) -> dict:
    """Encrypt, split and upload one local file, freeing disk as it goes.

    Each intermediate is deleted as soon as the next one exists, so peak
    usage stays near 2x the file rather than 4x.
    """
    size = local_path.stat().st_size
    sha = sha256_file(local_path)
    session_id = str(uuid.uuid4())

    encrypted = work_dir / f"{session_id}.enc"
    crypto.encrypt_stream(local_path, encrypted, passphrase)

    chunks = splitter.split_file(
        encrypted, session_id, work_dir, chunk_size_mb * 1024 * 1024
    )
    encrypted.unlink(missing_ok=True)

    try:
        folder_url = drive.upload_to_drive(chunks, drive_leaf, credentials)
    finally:
        for chunk in chunks:
            chunk.unlink(missing_ok=True)

    return {
        "leaf": drive_leaf,
        "session_id": session_id,
        "sha256": sha,
        "size": size,
        "parts": len(chunks),
        "folder_url": folder_url,
        "original_name": local_path.name,
        "pushed_at": _utc_now(),
    }


def _download_one(repo_id: str, repo_type: str, repo_path: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore[import]

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=repo_type,
            local_dir=str(dest),
        )
    )


def _list_repo_files(repo_id: str, repo_type: str) -> list[str]:
    from huggingface_hub import HfApi  # type: ignore[import]

    return list(HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type))


def _resolve_targets(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    all_paths = _list_repo_files(args.repo, args.repo_type)
    split_paths, bundled_paths = select_files(all_paths, args.split)
    targets = assign_leaves(split_paths, args.drive_parent)
    if bundled_paths:
        targets[bundle_key(args.bundle_name)] = (
            f"{args.drive_parent}/{args.bundle_name}"
        )
    return targets, bundled_paths


def cmd_push(args: argparse.Namespace) -> None:
    passphrase = _get_passphrase()
    lpath = ledger_path(args.repo, args.ledger)
    ledger = load_ledger(lpath)
    targets, bundled_paths = _resolve_targets(args)

    ledger.setdefault("repo_id", args.repo)
    ledger.setdefault("repo_type", args.repo_type)
    ledger.setdefault("drive_parent", args.drive_parent)
    ledger.setdefault("chunk_size_mb", args.chunk_size)
    ledger.setdefault("created_at", _utc_now())
    ledger.setdefault("entries", {})

    todo = pending_entries(ledger, targets)
    if not todo:
        print(f"Nothing to do — all {len(targets)} targets already pushed.")
        print(f"Ledger: {lpath}")
        return

    print(f"{len(ledger['entries'])} already pushed, {len(todo)} to go.")
    print(f"Ledger: {lpath}\n")

    for index, repo_path in enumerate(todo, start=1):
        leaf = targets[repo_path]
        work_dir = Path(tempfile.mkdtemp(dir=args.work_dir or None))
        try:
            print(f"[{index}/{len(todo)}] {repo_path} -> {leaf}")
            if is_bundle_key(repo_path):
                staging = work_dir / args.bundle_name
                staging.mkdir()
                for small in bundled_paths:
                    _download_one(args.repo, args.repo_type, small, staging)
                local = Path(
                    shutil.make_archive(
                        str(work_dir / args.bundle_name), "gztar",
                        root_dir=work_dir, base_dir=args.bundle_name,
                    )
                )
                shutil.rmtree(staging, ignore_errors=True)
                entry = push_one_file(
                    local, leaf, passphrase, work_dir, args.chunk_size,
                    args.credentials,
                )
                entry["bundled_files"] = bundled_paths
            else:
                local = _download_one(args.repo, args.repo_type, repo_path, work_dir)
                entry = push_one_file(
                    local, leaf, passphrase, work_dir, args.chunk_size,
                    args.credentials,
                )

            # Record before the next download so an interruption resumes here.
            ledger["entries"][repo_path] = entry
            save_ledger(lpath, ledger)
            print(f"    ok  {entry['parts']} part(s)  "
                  f"sha256={entry['sha256'][:12]}...\n")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Done. {len(ledger['entries'])}/{len(targets)} targets pushed.")
    print(f"Ledger: {lpath} — keep it, it maps Drive folders back to filenames.")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi  # type: ignore[import]

    info = HfApi().repo_info(
        repo_id=args.repo, repo_type=args.repo_type, files_metadata=True
    )
    sizes = {
        sibling.rfilename: (
            getattr(sibling.lfs, "size", None) if getattr(sibling, "lfs", None)
            else sibling.size
        ) or 0
        for sibling in info.siblings
    }
    split_paths, bundled_paths = select_files(list(sizes), args.split)
    targets = assign_leaves(split_paths, args.drive_parent)
    done = set(load_ledger(ledger_path(args.repo, args.ledger)).get("entries", {}))

    print(f"{args.repo} ({args.repo_type}) -> Drive folder {args.drive_parent!r}\n")
    total = 0
    for repo_path in sorted(targets):
        size = sizes.get(repo_path, 0)
        total += size
        mark = "done" if repo_path in done else " -- "
        print(f"  [{mark}] {size / 1e9:7.3f} GB  {repo_path:<22} -> {targets[repo_path]}")

    if bundled_paths:
        bundle_size = sum(sizes.get(p, 0) for p in bundled_paths)
        total += bundle_size
        mark = "done" if bundle_key(args.bundle_name) in done else " -- "
        print(f"  [{mark}] {bundle_size / 1e9:7.3f} GB  "
              f"{len(bundled_paths)} small file(s) -> "
              f"{args.drive_parent}/{args.bundle_name}")
        for small in bundled_paths:
            print(f"                        {sizes.get(small, 0) / 1e9:7.3f} GB  {small}")

    count = len(targets) + (1 if bundled_paths else 0)
    largest = max(sizes.values()) if sizes else 0
    print(f"\n  {count} leaf folder(s), {total / 1e9:.2f} GB total")
    print(f"  largest single file {largest / 1e9:.2f} GB "
          f"-> peak working disk about {2 * largest / 1e9:.1f} GB")
    print(f"  {len(done)} of {count} already pushed")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def cmd_pull(args: argparse.Namespace) -> None:
    # Resolve what there is to pull before asking for the passphrase: an
    # empty or mistyped ledger should say so, not demand a key first.
    lpath = ledger_path(args.repo, args.ledger)
    entries = load_ledger(lpath).get("entries", {})
    if not entries:
        raise SystemExit(f"error: no pushed entries in ledger {lpath}")

    wanted = (
        sorted(entries) if not args.only
        else [p for p in sorted(entries) if p in args.only]
    )
    if not wanted:
        raise SystemExit(f"error: none of {args.only} are in the ledger")

    passphrase = _get_passphrase()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for index, repo_path in enumerate(wanted, start=1):
        entry = entries[repo_path]
        print(f"[{index}/{len(wanted)}] {entry['leaf']} -> {repo_path}")
        bundled = is_bundle_key(repo_path)
        dest = output / repo_path
        if not bundled and dest.exists() and not args.force:
            print("    skip (already present; --force to redo)\n")
            continue

        work_dir = Path(tempfile.mkdtemp(dir=args.work_dir or None))
        try:
            chunks = drive.download_from_drive(entry["leaf"], work_dir, args.credentials)
            if not chunks:
                raise ValueError(f"no chunks in Drive folder {entry['leaf']!r}")
            joined = work_dir / "joined"
            splitter.join_files(chunks, joined)
            for chunk in chunks:
                chunk.unlink(missing_ok=True)

            decrypted = work_dir / "decrypted"
            crypto.decrypt_stream(joined, decrypted, passphrase)
            joined.unlink(missing_ok=True)

            got = sha256_file(decrypted)
            if got != entry["sha256"]:
                failures.append(repo_path)
                print(f"    SHA256 MISMATCH\n      expected {entry['sha256']}\n"
                      f"      got      {got}\n")
                continue

            if bundled:
                final = unpack.finalize_output(decrypted, output)
                print(f"    ok  sha256 verified -> {final.name}  "
                      f"(extract with: tar -xf {final.name})\n")
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(decrypted), dest)
            print(f"    ok  sha256 verified -> {dest}\n")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    if failures:
        raise SystemExit(
            f"error: {len(failures)} target(s) failed verification: {failures}"
        )
    print(f"Done. {len(wanted)} target(s) pulled into {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="innout-batch",
        description="Push a large HuggingFace repo to Drive one file at a time",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, helptext in (
        ("plan", cmd_plan, "List what would be pushed and where (no transfer)"),
        ("push", cmd_push, "Push pending files, resuming from the ledger"),
        ("pull", cmd_pull, "Pull back, restoring filenames and verifying sha256"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--repo", metavar="org/name", required=True,
                       help="HuggingFace repo ID")
        p.add_argument("--repo-type", metavar="<kind>", default="dataset",
                       choices=["model", "dataset", "space"],
                       help="HuggingFace repo kind (default: dataset)")
        p.add_argument("--drive-parent", metavar="<folder>", default=None,
                       help="Top-level Drive folder holding every leaf "
                            "(default: the repo name)")
        p.add_argument("--split", metavar="<glob>", action="append", default=None,
                       help="Repo-path glob whose matches each get their own "
                            "leaf folder (repeatable; default: 'videos/*')")
        p.add_argument("--bundle-name", metavar="<name>", default=BUNDLE_NAME,
                       help=f"Leaf folder for all remaining small files "
                            f"(default: {BUNDLE_NAME})")
        p.add_argument("--ledger", metavar="<path>", default=None,
                       help="Ledger JSON (default: ~/.innout_batches/<repo>.json)")
        p.add_argument("--credentials", metavar="<path>", default=None,
                       help="OAuth client-secrets JSON for Drive")
        p.add_argument("--chunk-size", metavar="<MB>", type=int, default=1800,
                       help="Chunk size in MB (default: 1800)")
        p.add_argument("--work-dir", metavar="<dir>", default=None,
                       help="Where to stage downloads and chunks "
                            "(default: system temp)")
        p.set_defaults(func=func)

    pull = sub.choices["pull"]
    pull.add_argument("--output", metavar="<dir>", default=".",
                      help="Where to write recovered files (default: .)")
    pull.add_argument("--only", metavar="<repo-path>", action="append", default=None,
                      help="Pull just these repo paths (repeatable)")
    pull.add_argument("--force", action="store_true",
                      help="Re-pull files already present in --output")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.split is None:
        args.split = ["videos/*"]
    if args.drive_parent is None:
        args.drive_parent = args.repo.split("/")[-1]
    args.func(args)


if __name__ == "__main__":
    main()
