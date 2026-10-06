"""Tests for batched HuggingFace -> Drive pushes."""

import hashlib
import json
import shutil
import ssl
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from innout import batch, crypto, splitter
from innout.batch import (
    assign_leaves,
    build_parser,
    bundle_key,
    is_bundle_key,
    file_stem,
    leaf_dir,
    leaf_for,
    ledger_path,
    load_ledger,
    pending_entries,
    push_one_file,
    save_ledger,
    select_files,
    sha256_file,
)

VIDEO_MME_FILES = [
    ".gitattributes",
    "README.md",
    "assets/logo.png",
    "eval.yaml",
    "subtitle.zip",
    "test.parquet",
    "videos/001.zip",
    "videos/002.zip",
    "videos/040.zip",
]


def _piece(**overrides) -> dict:
    """One piece of a ledger entry, with synthetic values."""
    piece = {
        "name": "aa",
        "leaf": "P/videos/001/aa",
        "session_id": "00000000-0000-0000-0000-000000000000",
        "sha256": "0" * 64,
        "size": 1,
        "parts": 1,
        "folder_url": "https://drive.google.com/drive/folders/synthetic-id",
        "original_name": "aa",
        "pushed_at": "2026-10-05T07:31:04Z",
    }
    piece.update(overrides)
    return piece


def _entry(**overrides) -> dict:
    """A ledger entry in the current (piece-bearing) shape."""
    entry = {
        "sha256": "0" * 64,
        "size": 1,
        "piece_size_bytes": 1024 * 1024 * 1024,
        "pieces": [_piece()],
        "original_name": "001.zip",
        "pushed_at": "2026-10-05T07:31:04Z",
    }
    entry.update(overrides)
    return entry


def _legacy_entry(**overrides) -> dict:
    """The pre-splitting shape: one session, no pieces list."""
    entry = {
        "leaf": "P/videos/001",
        "session_id": "11111111-0000-0000-0000-000000000000",
        "sha256": "0" * 64,
        "size": 1,
        "parts": 1,
        "folder_url": "https://drive.google.com/drive/folders/synthetic-id",
        "original_name": "001.zip",
        "pushed_at": "2026-10-05T07:31:04Z",
    }
    entry.update(overrides)
    return entry


def _stub_download(repo_id, repo_type, repo_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / Path(repo_path).name
    path.write_bytes(b"synthetic payload ")
    return path


# ---------------------------------------------------------------------------
# leaf_name / assign_leaves
# ---------------------------------------------------------------------------

def test_leaf_dir_mirrors_the_repo_directory():
    """The folder is the file's directory, so videos stay together."""
    assert leaf_dir("videos/001.zip") == "videos"
    assert leaf_dir("videos/040.zip") == "videos"
    assert leaf_dir("a/b/c.tar.gz") == "a/b"


def test_leaf_dir_of_a_top_level_file_is_empty():
    assert leaf_dir("test.parquet") == ""
    assert leaf_dir("README") == ""


def test_leaf_dir_collapses_redundant_slashes():
    assert leaf_dir("videos//001.zip") == "videos"
    assert leaf_dir("/videos/001.zip") == "videos"


def test_leaf_dir_rejects_dot_segments():
    for bad in ("../secrets/001.zip", "videos/../001.zip"):
        with pytest.raises(ValueError, match="is not allowed"):
            leaf_dir(bad)


def test_file_stem_drops_the_extension():
    assert file_stem("videos/001.zip") == "001"
    assert file_stem("test.parquet") == "test"
    assert file_stem("a/b/c.tar.gz") == "c.tar"
    assert file_stem("README") == "README"


def test_file_stem_tolerates_a_trailing_slash():
    assert file_stem("videos/") == "videos"


def test_file_stem_rejects_a_path_with_no_name():
    with pytest.raises(ValueError, match="Cannot derive a folder name"):
        file_stem("/")


def test_leaf_for_per_file_gives_each_file_its_own_folder():
    """The decode-side default: one downloaded folder decodes on its own."""
    assert leaf_for("videos/001.zip") == "videos/001"
    assert leaf_for("videos/040.zip") == "videos/040"
    assert leaf_for("big.zip") == "big"


def test_leaf_for_shared_puts_a_directory_together():
    assert leaf_for("videos/001.zip", "shared") == "videos"
    assert leaf_for("videos/040.zip", "shared") == "videos"


def test_leaf_for_rejects_an_unknown_layout():
    with pytest.raises(ValueError, match="Unknown layout"):
        leaf_for("videos/001.zip", "sideways")


def test_assign_leaves_per_file_is_the_default():
    """Each folder is a self-contained unit for a manual Drive download."""
    paths = ["videos/001.zip", "videos/002.zip", "videos/040.zip"]
    assert assign_leaves(paths, "Video-MME-v2") == {
        "videos/001.zip": "Video-MME-v2/videos/001",
        "videos/002.zip": "Video-MME-v2/videos/002",
        "videos/040.zip": "Video-MME-v2/videos/040",
    }


def test_assign_leaves_shared_collapses_to_one_folder():
    paths = ["videos/001.zip", "videos/002.zip"]
    assert assign_leaves(paths, "Video-MME-v2", "shared") == {
        "videos/001.zip": "Video-MME-v2/videos",
        "videos/002.zip": "Video-MME-v2/videos",
    }


def test_assign_leaves_top_level_file_gets_its_own_folder():
    assert assign_leaves(["big.zip"], "Video-MME-v2") == {
        "big.zip": "Video-MME-v2/big",
    }
    # shared has nothing to group by, so it lands in the parent itself
    assert assign_leaves(["big.zip"], "Video-MME-v2", "shared") == {
        "big.zip": "Video-MME-v2",
    }


def test_assign_leaves_preserves_deeper_structure():
    assert assign_leaves(["videos/hd/001.zip"], "P") == {
        "videos/hd/001.zip": "P/videos/hd/001",
    }


# ---------------------------------------------------------------------------
# select_files
# ---------------------------------------------------------------------------

def test_select_files_splits_videos_bundles_the_rest():
    split, bundled = select_files(VIDEO_MME_FILES, ["videos/*"])
    assert split == ["videos/001.zip", "videos/002.zip", "videos/040.zip"]
    assert bundled == [
        ".gitattributes", "README.md", "assets/logo.png",
        "eval.yaml", "subtitle.zip", "test.parquet",
    ]


def test_select_files_matches_full_path_not_prefix():
    """'videos/*' must not match a top-level file that merely starts with it."""
    split, bundled = select_files(["videos.txt", "videos/001.zip"], ["videos/*"])
    assert split == ["videos/001.zip"]
    assert bundled == ["videos.txt"]


def test_select_files_is_sorted_for_resumable_ordering():
    split, _ = select_files(["videos/003.zip", "videos/001.zip"], ["videos/*"])
    assert split == ["videos/001.zip", "videos/003.zip"]


def test_select_files_accepts_multiple_globs():
    split, bundled = select_files(VIDEO_MME_FILES, ["videos/*", "*.parquet"])
    assert "test.parquet" in split
    assert "test.parquet" not in bundled


def test_bundle_key_cannot_collide_with_a_repo_path():
    key = bundle_key("annotations")
    assert is_bundle_key(key)
    assert not is_bundle_key("videos/001.zip")
    # Angle brackets are not legal in HuggingFace filenames.
    assert "<" in key and ">" in key


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_ledger_path_slugifies_repo_id():
    path = ledger_path("MME-Benchmarks/Video-MME-v2")
    assert path.name == "MME-Benchmarks__Video-MME-v2.json"
    assert path.parent == Path.home() / ".innout_batches"


def test_ledger_path_explicit_wins(tmp_path):
    explicit = tmp_path / "custom.json"
    assert ledger_path("any/repo", str(explicit)) == explicit


def test_load_ledger_missing_returns_empty(tmp_path):
    assert load_ledger(tmp_path / "nope.json") == {}


def test_save_ledger_round_trips(tmp_path):
    path = tmp_path / "nested" / "led.json"
    ledger = {
        "repo_id": "org/name",
        "created_at": "2026-10-05T07:31:04Z",
        "entries": {"videos/001.zip": _entry()},
    }
    save_ledger(path, ledger)
    assert load_ledger(path) == ledger


def test_save_ledger_leaves_no_temp_file(tmp_path):
    """The atomic write must not leave a .tmp file beside the ledger."""
    path = tmp_path / "led.json"
    save_ledger(path, {"entries": {}})
    assert [p.name for p in tmp_path.iterdir()] == ["led.json"]


def test_save_ledger_overwrites_atomically(tmp_path):
    path = tmp_path / "led.json"
    save_ledger(path, {"entries": {"a": 1}})
    save_ledger(path, {"entries": {"a": 1, "b": 2}})
    assert load_ledger(path)["entries"] == {"a": 1, "b": 2}


def test_save_ledger_writes_valid_json_with_trailing_newline(tmp_path):
    path = tmp_path / "led.json"
    save_ledger(path, {"entries": {}})
    text = path.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == {"entries": {}}


def test_pending_entries_skips_already_pushed():
    targets = {
        "videos/001.zip": "p/videos-001",
        "videos/002.zip": "p/videos-002",
        "videos/003.zip": "p/videos-003",
    }
    ledger = {"entries": {"videos/002.zip": _entry()}}
    assert pending_entries(ledger, targets) == ["videos/001.zip", "videos/003.zip"]


def test_pending_entries_empty_ledger_returns_all_sorted():
    assert pending_entries({}, {"b": "p/b", "a": "p/a"}) == ["a", "b"]


def test_pending_entries_all_done_returns_empty():
    assert pending_entries({"entries": {"a": _entry()}}, {"a": "p/a"}) == []


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * 5000  # larger than the read buffer
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# push_one_file
# ---------------------------------------------------------------------------

def test_push_one_file_records_metadata_and_cleans_up(tmp_path):
    src = tmp_path / "001.zip"
    src.write_bytes(b"video bytes " * 1000)
    work = tmp_path / "work"
    work.mkdir()

    with patch("innout.batch.drive.upload_to_drive") as mock_upload:
        mock_upload.return_value = "https://drive.google.com/drive/folders/leaf-id"
        entry = push_one_file(
            src, "VideoMME-v2/videos-001", "pw", work,
            chunk_size_mb=1, credentials=None,
        )

    assert entry["leaf"] == "VideoMME-v2/videos-001"
    assert entry["size"] == src.stat().st_size
    assert entry["sha256"] == sha256_file(src)
    assert entry["original_name"] == "001.zip"
    assert entry["parts"] >= 1
    assert entry["folder_url"].endswith("leaf-id")
    assert entry["pushed_at"].endswith("Z")
    # Intermediates are deleted so a 40-file run never accumulates disk.
    assert list(work.iterdir()) == [], f"work dir not clean: {list(work.iterdir())}"


def test_push_one_file_uploads_to_the_requested_leaf(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 500)
    work = tmp_path / "work"
    work.mkdir()

    with patch("innout.batch.drive.upload_to_drive") as mock_upload:
        mock_upload.return_value = "url"
        push_one_file(src, "Parent/leaf", "pw", work, 1, None)

    assert mock_upload.call_args.args[1] == "Parent/leaf"


def test_push_one_file_cleans_up_when_upload_fails(tmp_path):
    """A failed upload must not strand gigabytes of chunks on disk."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 500)
    work = tmp_path / "work"
    work.mkdir()

    with patch("innout.batch.drive.upload_to_drive") as mock_upload:
        mock_upload.side_effect = RuntimeError("network died")
        with pytest.raises(RuntimeError, match="network died"):
            push_one_file(src, "Parent/leaf", "pw", work, 1, None)

    assert list(work.iterdir()) == [], f"chunks left behind: {list(work.iterdir())}"


def test_push_one_file_splits_large_input_into_several_parts(tmp_path):
    src = tmp_path / "big.bin"
    src.write_bytes(b"y" * (3 * 1024 * 1024))
    work = tmp_path / "work"
    work.mkdir()

    with patch("innout.batch.drive.upload_to_drive") as mock_upload:
        mock_upload.return_value = "url"
        entry = push_one_file(src, "p/leaf", "pw", work, chunk_size_mb=1, credentials=None)

    assert entry["parts"] == 4  # 3 MB payload plus the crypto header
    assert len(mock_upload.call_args.args[0]) == entry["parts"]


def test_push_one_file_payload_round_trips(tmp_path):
    """The uploaded chunks must decrypt back to the original bytes."""
    original = bytes(range(256)) * 2000
    src = tmp_path / "001.zip"
    src.write_bytes(original)
    work = tmp_path / "work"
    work.mkdir()
    kept = tmp_path / "kept"
    kept.mkdir()

    def _capture(chunks, leaf, credentials):
        for chunk in chunks:
            shutil.copy(chunk, kept / chunk.name)
        return "url"

    with patch("innout.batch.drive.upload_to_drive", side_effect=_capture):
        entry = push_one_file(src, "p/leaf", "pw", work, 1, None)

    joined = tmp_path / "joined"
    splitter.join_files(sorted(kept.glob("*.part???")), joined)
    out = tmp_path / "out"
    crypto.decrypt_stream(joined, out, "pw")
    assert out.read_bytes() == original
    assert sha256_file(out) == entry["sha256"]


# ---------------------------------------------------------------------------
# Passphrase handling
# ---------------------------------------------------------------------------

def test_get_passphrase_requires_env_var(monkeypatch):
    monkeypatch.delenv("INNOUT_PASSPHRASE", raising=False)
    with pytest.raises(SystemExit, match="INNOUT_PASSPHRASE is not set"):
        batch._get_passphrase()


def test_get_passphrase_reads_env_var(monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "from-env")
    assert batch._get_passphrase() == "from-env"


@pytest.mark.parametrize("sub", ["plan", "push", "pull"])
def test_parser_rejects_passphrase_flag(sub):
    """Accepting a passphrase argument would leak it into ps and shell history."""
    assert "--passphrase" not in build_parser().format_help()
    with pytest.raises(SystemExit):
        build_parser().parse_args([sub, "--repo", "o/r", "--passphrase", "secret"])


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------

def test_main_defaults_split_to_videos_and_parent_to_repo_name(monkeypatch):
    captured = {}
    monkeypatch.setattr(batch, "cmd_plan", lambda args: captured.update(vars(args)))
    monkeypatch.setattr(
        "sys.argv", ["innout-batch", "plan", "--repo", "MME-Benchmarks/Video-MME-v2"]
    )
    batch.main()
    assert captured["split"] == ["videos/*"]
    assert captured["drive_parent"] == "Video-MME-v2"
    assert captured["repo_type"] == "dataset"
    assert captured["chunk_size"] == 512


def test_main_respects_explicit_parent_and_split(monkeypatch):
    captured = {}
    monkeypatch.setattr(batch, "cmd_push", lambda args: captured.update(vars(args)))
    monkeypatch.setattr("sys.argv", [
        "innout-batch", "push", "--repo", "o/r",
        "--drive-parent", "VideoMME-v2", "--split", "videos/*", "--split", "*.parquet",
    ])
    batch.main()
    assert captured["drive_parent"] == "VideoMME-v2"
    assert captured["split"] == ["videos/*", "*.parquet"]


def test_parser_rejects_unknown_repo_type():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan", "--repo", "o/r", "--repo-type", "nonsense"])


def test_parser_requires_repo():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plan"])


# ---------------------------------------------------------------------------
# cmd_push resume behaviour
# ---------------------------------------------------------------------------

def _push_args(led: Path, parent: str = "P", extra: list[str] | None = None):
    args = build_parser().parse_args(
        ["push", "--repo", "o/r", "--drive-parent", parent, "--ledger", str(led)]
        + (extra or [])
    )
    args.split = ["videos/*"]
    return args


def test_cmd_push_resumes_and_skips_done_entries(tmp_path, monkeypatch, capsys):
    """Interrupted run: only the unfinished files get pushed again."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    save_ledger(led, {"repo_id": "o/r", "entries": {"videos/001.zip": _entry()}})

    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: ["videos/001.zip", "videos/002.zip"],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)

    pushed = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        pushed.append(base_leaf)
        return _entry(original_name=local_path.name,
                      pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led))

    assert pushed == ["P/videos/002"], "already-pushed file must be skipped"
    assert set(load_ledger(led)["entries"]) == {"videos/001.zip", "videos/002.zip"}
    assert "1 already pushed, 1 to go" in capsys.readouterr().out


def test_cmd_push_is_a_noop_when_everything_is_done(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {"videos/001.zip": _entry()}})
    monkeypatch.setattr(
        batch, "_list_repo_files", lambda repo_id, repo_type: ["videos/001.zip"]
    )
    monkeypatch.setattr(
        batch, "push_pieces", MagicMock(side_effect=AssertionError("must not push"))
    )

    batch.cmd_push(_push_args(led))
    assert "Nothing to do" in capsys.readouterr().out


def test_cmd_push_writes_ledger_after_each_file(tmp_path, monkeypatch):
    """File 1 is on disk before file 2 is attempted, so a kill never loses it.

    Checked by reading the ledger from inside the second push: if the write
    happened only at the end, file 1 would not be there yet.
    """
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: ["videos/001.zip", "videos/002.zip"],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)

    seen_mid_run = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        seen_mid_run.append(set(load_ledger(led).get("entries", {})))
        return _entry(original_name=local_path.name,
                      pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led))

    assert seen_mid_run[0] == set(), "nothing recorded before the first push"
    assert seen_mid_run[1] == {"videos/001.zip"}, "file 1 must be durable by now"
    assert set(load_ledger(led)["entries"]) == {"videos/001.zip", "videos/002.zip"}


def test_cmd_push_bundles_small_files_into_one_leaf(tmp_path, monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: ["videos/001.zip", "eval.yaml", "test.parquet"],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)

    leaves = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        leaves.append(base_leaf)
        return _entry(original_name=local_path.name,
                      pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led, parent="Video-MME-v2"))

    assert sorted(leaves) == [
        "Video-MME-v2/annotations", "Video-MME-v2/videos/001",
    ]
    entries = load_ledger(led)["entries"]
    assert sorted(entries["<bundle:annotations>"]["bundled_files"]) == [
        "eval.yaml", "test.parquet",
    ]


def test_cmd_push_bundle_excludes_hf_download_cache(tmp_path, monkeypatch):
    """hf_hub_download drops .cache/huggingface into local_dir.

    Regression: that bookkeeping got archived with the payload, so the
    recovered tar carried a dozen .metadata files alongside the real files.
    """
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files", lambda repo_id, repo_type: ["eval.yaml"]
    )

    def _download_leaving_cache(repo_id, repo_type, repo_path, dest):
        dest = Path(dest)
        (dest / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (dest / ".cache" / "huggingface" / "download" / "eval.yaml.metadata").write_text("junk")
        path = dest / repo_path
        path.write_bytes(b"real payload")
        return path

    monkeypatch.setattr(batch, "_download_one", _download_leaving_cache)

    archived: list[list[str]] = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        import tarfile
        with tarfile.open(local_path, "r:gz") as tf:
            archived.append(tf.getnames())
        return _entry(original_name=local_path.name,
                      pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led))

    names = archived[0]
    assert any(n.endswith("eval.yaml") for n in names), names
    assert not any(".cache" in n for n in names), (
        f"HF download cache leaked into the bundle: "
        f"{[n for n in names if '.cache' in n]}"
    )


def test_cmd_push_gives_each_video_its_own_leaf(tmp_path, monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: [f"videos/{n:03d}.zip" for n in (1, 2, 3)],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)

    seen = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        seen.append((base_leaf, uuid.uuid4().hex))
        return _entry(pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led, parent="Video-MME-v2"))

    assert {leaf for leaf, _ in seen} == {
        "Video-MME-v2/videos/001", "Video-MME-v2/videos/002",
        "Video-MME-v2/videos/003",
    }
    session_ids = [sid for _, sid in seen]
    assert len(set(session_ids)) == 3
    assert all(sid for sid in session_ids)


# ---------------------------------------------------------------------------
# cmd_push failure handling
# ---------------------------------------------------------------------------

def test_cmd_push_skips_a_failed_file_and_keeps_going(tmp_path, monkeypatch, capsys):
    """A dropped connection on one file must not end a 41-file run."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: [f"videos/{n:03d}.zip" for n in (1, 2, 3)],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)
    monkeypatch.setattr(batch.drive, "delete_session_files", lambda *a, **k: 0)

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        if local_path.name == "002.zip":
            raise OSError("connection reset by peer")
        return _entry(pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)

    with pytest.raises(SystemExit) as excinfo:
        batch.cmd_push(_push_args(led))
    assert excinfo.value.code == 1

    ledger = load_ledger(led)
    # The two good files are recorded; only the bad one is outstanding.
    assert set(ledger["entries"]) == {"videos/001.zip", "videos/003.zip"}
    assert set(ledger["failures"]) == {"videos/002.zip"}
    assert "connection reset" in ledger["failures"]["videos/002.zip"]["error"]
    assert ledger["failures"]["videos/002.zip"]["failed_at"].endswith("Z")
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "1 target(s) failed" in out


def test_push_pieces_cleans_up_a_failed_piece(tmp_path, monkeypatch):
    """A piece that dies mid-upload must not leave orphan chunks on Drive.

    The retry mints a new session id, so those chunks would never be reused.
    Finished pieces are kept on purpose, so the retry can skip them.
    """
    src = tmp_path / "003.zip"
    src.write_bytes(b"x" * 2500)
    work = tmp_path / "work"
    work.mkdir()

    cleaned = []
    monkeypatch.setattr(
        batch.drive, "delete_session_files",
        lambda leaf, session_id, creds=None: (
            cleaned.append((leaf, session_id)) or 3
        ),
    )

    calls = {"n": 0}

    def _fake_one(local_path, leaf, passphrase, work_dir, chunk_size,
                  credentials, session_id=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ssl.SSLError("[SYS] unknown error")
        return _piece(name=local_path.name, leaf=leaf, session_id=session_id)

    monkeypatch.setattr(batch, "push_one_file", _fake_one)

    kept = []
    with pytest.raises(ssl.SSLError):
        batch.push_pieces(
            src, "P/videos/003", "pw", work, 1, None,
            max_piece_bytes=1000, on_piece=kept.append,
        )

    assert len(cleaned) == 1, "exactly the failed piece is cleaned"
    leaf, session_id = cleaned[0]
    assert leaf == "P/videos/003/ab", leaf
    assert session_id, "cleanup needs the session id the push actually used"
    assert [k["name"] for k in kept] == ["aa"], "piece aa must be kept for resume"


def test_push_pieces_survives_a_cleanup_failure(tmp_path, monkeypatch, capsys):
    """Failing to delete orphans warns; the original error still propagates."""
    src = tmp_path / "003.zip"
    src.write_bytes(b"x" * 1500)
    work = tmp_path / "work"
    work.mkdir()

    monkeypatch.setattr(
        batch, "push_one_file",
        MagicMock(side_effect=OSError("upload died")),
    )
    monkeypatch.setattr(
        batch.drive, "delete_session_files",
        MagicMock(side_effect=RuntimeError("cleanup also died")),
    )

    with pytest.raises(OSError, match="upload died"):
        batch.push_pieces(src, "P/videos/003", "pw", work, 1, None,
                          max_piece_bytes=1000)
    assert "WARNING could not clean up" in capsys.readouterr().out


def test_cmd_push_retries_only_the_failed_file_on_rerun(tmp_path, monkeypatch):
    """Second run picks up exactly what failed, and clears it once it lands."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    save_ledger(led, {
        "entries": {"videos/001.zip": _entry()},
        "failures": {"videos/002.zip": {
            "leaf": "P/videos", "session_id": "old-sid",
            "error": "OSError: connection reset", "failed_at": "2026-10-05T07:31:04Z",
        }},
    })
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: ["videos/001.zip", "videos/002.zip"],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)

    pushed = []

    def _fake_push(local_path, base_leaf, passphrase, work_dir, chunk_size,
                   credentials, max_piece_bytes, already=None, on_piece=None):
        pushed.append(local_path.name)
        return _entry(pieces=[_piece(leaf=f"{base_leaf}/aa")])

    monkeypatch.setattr(batch, "push_pieces", _fake_push)
    batch.cmd_push(_push_args(led))

    assert pushed == ["002.zip"]
    ledger = load_ledger(led)
    assert set(ledger["entries"]) == {"videos/001.zip", "videos/002.zip"}
    # The stale failure record is gone now that the file is in.
    assert ledger["failures"] == {}


def test_cmd_push_does_not_swallow_keyboard_interrupt(tmp_path, monkeypatch):
    """Ctrl-C must stop the run, not be treated as one more skippable file."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    monkeypatch.setattr(
        batch, "_list_repo_files",
        lambda repo_id, repo_type: ["videos/001.zip", "videos/002.zip"],
    )
    monkeypatch.setattr(batch, "_download_one", _stub_download)
    monkeypatch.setattr(
        batch, "push_pieces", MagicMock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        batch.cmd_push(_push_args(led))


# ---------------------------------------------------------------------------
# cmd_pull verification
# ---------------------------------------------------------------------------

def _seed_drive(monkeypatch, payload: bytes, passphrase: str, store: Path):
    """Encrypt+split payload into store, and make download_from_drive serve it."""
    store.mkdir(parents=True, exist_ok=True)
    src = store / "src"
    src.write_bytes(payload)
    enc = store / "enc"
    crypto.encrypt_stream(src, enc, passphrase)
    chunks = splitter.split_file(enc, "sess", store, 1024 * 1024)
    src.unlink()
    enc.unlink()

    def _fake_download(leaf, dest_dir, credentials=None, session_id=None):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = [Path(shutil.copy(c, dest_dir / c.name)) for c in chunks]
        return sorted(copied, key=lambda p: p.name)

    monkeypatch.setattr(batch.drive, "download_from_drive", _fake_download)


def _pull_args(led: Path, out: Path, extra: list[str] | None = None):
    return build_parser().parse_args(
        ["pull", "--repo", "o/r", "--ledger", str(led), "--output", str(out)]
        + (extra or [])
    )


def test_cmd_pull_restores_original_filename_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    payload = bytes(range(256)) * 300
    _seed_drive(monkeypatch, payload, "pw", tmp_path / "store")

    led = tmp_path / "led.json"
    digest = hashlib.sha256(payload).hexdigest()
    save_ledger(led, {"entries": {"videos/001.zip": _entry(
        sha256=digest, size=len(payload),
        pieces=[_piece(leaf="Video-MME-v2/videos/001/aa", sha256=digest,
                       size=len(payload))],
    )}})

    out = tmp_path / "out"
    batch.cmd_pull(_pull_args(led, out))

    # Named by its repo path, not the opaque "result" a raw pull would give.
    restored = out / "videos" / "001.zip"
    assert restored.exists()
    assert restored.read_bytes() == payload


def test_cmd_pull_fails_loudly_on_sha256_mismatch(tmp_path, monkeypatch, capsys):
    """A corrupted transfer must never be reported as success."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    _seed_drive(monkeypatch, b"actual payload", "pw", tmp_path / "store")

    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {"videos/001.zip": _entry(
        sha256="f" * 64,
        pieces=[_piece(sha256=hashlib.sha256(b"actual payload").hexdigest())],
    )}})

    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="failed verification"):
        batch.cmd_pull(_pull_args(led, out))

    assert "SHA256 MISMATCH" in capsys.readouterr().out
    assert not (out / "videos" / "001.zip").exists()


def test_cmd_pull_skips_existing_unless_forced(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    monkeypatch.setattr(
        batch.drive, "download_from_drive",
        MagicMock(side_effect=AssertionError("must not download")),
    )
    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {"videos/001.zip": _entry()}})
    out = tmp_path / "out"
    (out / "videos").mkdir(parents=True)
    (out / "videos" / "001.zip").write_bytes(b"already here")

    batch.cmd_pull(_pull_args(led, out))
    assert "skip (already present" in capsys.readouterr().out


def test_cmd_pull_force_redownloads(tmp_path, monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    payload = b"fresh payload " * 50
    _seed_drive(monkeypatch, payload, "pw", tmp_path / "store")
    led = tmp_path / "led.json"
    digest = hashlib.sha256(payload).hexdigest()
    save_ledger(led, {"entries": {"videos/001.zip": _entry(
        sha256=digest, pieces=[_piece(sha256=digest)],
    )}})
    out = tmp_path / "out"
    (out / "videos").mkdir(parents=True)
    (out / "videos" / "001.zip").write_bytes(b"stale")

    batch.cmd_pull(_pull_args(led, out, ["--force"]))
    assert (out / "videos" / "001.zip").read_bytes() == payload


def test_cmd_pull_errors_on_empty_ledger(tmp_path):
    args = _pull_args(tmp_path / "missing.json", tmp_path / "out")
    with pytest.raises(SystemExit, match="no pushed entries"):
        batch.cmd_pull(args)


def test_cmd_pull_only_filters_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {
        "videos/001.zip": _entry(pieces=[_piece(leaf="P/videos/001/aa")]),
        "videos/002.zip": _entry(pieces=[_piece(leaf="P/videos/002/aa")]),
    }})
    args = _pull_args(led, tmp_path / "out", ["--only", "videos/009.zip"])
    with pytest.raises(SystemExit, match="none of"):
        batch.cmd_pull(args)


# ---------------------------------------------------------------------------
# Piece splitting — the AES-GCM ceiling
# ---------------------------------------------------------------------------

def test_piece_names_sort_into_piece_order():
    """Two letters so `cat dec/*/result` reassembles without sorting logic."""
    names = batch.piece_names(30)
    assert names[:3] == ["aa", "ab", "ac"]
    assert names[26] == "ba"
    assert names == sorted(names)


def test_piece_names_rejects_degenerate_counts():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="count must be"):
            batch.piece_names(bad)
    with pytest.raises(ValueError, match="too many pieces"):
        batch.piece_names(26 * 26 + 1)


def test_piece_count_rounds_up_and_never_returns_zero():
    assert batch.piece_count(0, 1000) == 1, "an empty file is still one piece"
    assert batch.piece_count(1, 1000) == 1
    assert batch.piece_count(1000, 1000) == 1
    assert batch.piece_count(1001, 1000) == 2
    assert batch.piece_count(3000, 1000) == 3


def test_split_into_pieces_round_trips(tmp_path):
    src = tmp_path / "003.zip"
    payload = bytes(range(256)) * 400  # 102 400 bytes
    src.write_bytes(payload)

    pieces = batch.split_into_pieces(src, tmp_path / "pieces", 30_000)

    assert [p.name for p in pieces] == ["aa", "ab", "ac", "ad"]
    assert all(p.stat().st_size <= 30_000 for p in pieces)
    assert b"".join(p.read_bytes() for p in pieces) == payload


def test_split_into_pieces_handles_an_exact_multiple(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"x" * 2000)
    pieces = batch.split_into_pieces(src, tmp_path / "p", 1000)
    assert [p.stat().st_size for p in pieces] == [1000, 1000]


def test_split_into_pieces_handles_an_empty_file(tmp_path):
    src = tmp_path / "empty.bin"
    src.write_bytes(b"")
    pieces = batch.split_into_pieces(src, tmp_path / "p", 1000)
    assert [p.stat().st_size for p in pieces] == [0]


def test_split_into_pieces_refuses_a_piece_size_over_the_ceiling(tmp_path):
    """Regression: videos/003.zip (3.43 GB) died inside AESGCM.encrypt with
    `OverflowError: Data or associated data too long. Max 2**31 - 1 bytes`.
    A piece size above the ceiling must fail here, with that explanation,
    rather than deep in the crypto library."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="exceeds the AES-GCM ceiling"):
        batch.split_into_pieces(src, tmp_path / "p", batch.AESGCM_MAX_BYTES + 1)


def test_default_piece_size_leaves_real_headroom():
    """The user's requirement: well under 2 GB, with margin — not just under."""
    default_bytes = batch.DEFAULT_PIECE_MB * 1024 * 1024
    headroom = batch.AESGCM_MAX_BYTES - default_bytes
    assert headroom > 0, "default piece size must fit under the ceiling"
    assert headroom >= 512 * 1024 * 1024, (
        f"only {headroom / 1e6:.0f} MB of headroom; the user asked for "
        f"comfortably under 2 GB, not just under"
    )


def test_every_video_mme_archive_splits_under_the_ceiling():
    """No piece of the real dataset may reach the limit at the default size."""
    real_sizes = [  # bytes, the 5 largest archives plus the smallest
        5_095_000_000, 4_833_000_000, 4_438_000_000, 4_333_000_000,
        4_294_000_000, 1_224_000_000,
    ]
    piece_bytes = batch.DEFAULT_PIECE_MB * 1024 * 1024
    for size in real_sizes:
        count = batch.piece_count(size, piece_bytes)
        largest = min(size, piece_bytes)
        assert largest < batch.AESGCM_MAX_BYTES, size
        assert count * piece_bytes >= size, size


# ---------------------------------------------------------------------------
# push_pieces
# ---------------------------------------------------------------------------

def test_push_pieces_puts_each_piece_in_its_own_subfolder(tmp_path, monkeypatch):
    src = tmp_path / "003.zip"
    payload = b"z" * 2500
    src.write_bytes(payload)
    work = tmp_path / "work"
    work.mkdir()

    seen = []

    def _fake_one(local_path, leaf, passphrase, work_dir, chunk_size,
                  credentials, session_id=None):
        seen.append((local_path.name, leaf, local_path.stat().st_size))
        return _piece(name=local_path.name, leaf=leaf, session_id=session_id)

    monkeypatch.setattr(batch, "push_one_file", _fake_one)
    entry = batch.push_pieces(src, "P/videos/003", "pw", work, 1, None,
                              max_piece_bytes=1000)

    assert [(n, leaf) for n, leaf, _ in seen] == [
        ("aa", "P/videos/003/aa"),
        ("ab", "P/videos/003/ab"),
        ("ac", "P/videos/003/ac"),
    ]
    assert [size for _, _, size in seen] == [1000, 1000, 500]
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["size"] == len(payload)
    assert [p["name"] for p in entry["pieces"]] == ["aa", "ab", "ac"]


def test_push_pieces_frees_the_source_and_pieces(tmp_path, monkeypatch):
    """A 5 GB archive plus its pieces plus chunks would not fit otherwise."""
    src = tmp_path / "003.zip"
    src.write_bytes(b"z" * 2500)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        batch, "push_one_file",
        lambda lp, leaf, *a, **k: _piece(name=lp.name, leaf=leaf),
    )
    batch.push_pieces(src, "P/v/003", "pw", work, 1, None, max_piece_bytes=1000)

    assert not src.exists(), "the downloaded archive must go once split"
    leftovers = [p for p in (work / "pieces").rglob("*") if p.is_file()]
    assert leftovers == [], f"pieces left on disk: {leftovers}"


def test_push_pieces_skips_pieces_already_uploaded(tmp_path, monkeypatch):
    """Resuming an interrupted archive re-uploads only what is missing."""
    src = tmp_path / "003.zip"
    src.write_bytes(b"z" * 2500)
    work = tmp_path / "work"
    work.mkdir()

    pushed = []

    def _fake_one(local_path, leaf, passphrase, work_dir, chunk_size,
                  credentials, session_id=None):
        pushed.append(local_path.name)
        return _piece(name=local_path.name, leaf=leaf, session_id=session_id)

    monkeypatch.setattr(batch, "push_one_file", _fake_one)
    already = {"aa": _piece(name="aa", leaf="P/videos/003/aa")}
    entry = batch.push_pieces(src, "P/videos/003", "pw", work, 1, None,
                              max_piece_bytes=1000, already=already)

    assert pushed == ["ab", "ac"], "piece aa must not be re-uploaded"
    assert [p["name"] for p in entry["pieces"]] == ["aa", "ab", "ac"]


def test_push_pieces_reports_progress_per_piece(tmp_path, monkeypatch):
    """on_piece lets the caller persist each piece before the next upload."""
    src = tmp_path / "f.bin"
    src.write_bytes(b"z" * 2500)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        batch, "push_one_file",
        lambda lp, leaf, *a, **k: _piece(name=lp.name, leaf=leaf),
    )
    seen = []
    batch.push_pieces(src, "P/v/003", "pw", work, 1, None,
                      max_piece_bytes=1000, on_piece=seen.append)
    assert [p["name"] for p in seen] == ["aa", "ab", "ac"]


# ---------------------------------------------------------------------------
# entry_pieces — ledger shape compatibility
# ---------------------------------------------------------------------------

def test_entry_pieces_reads_a_legacy_entry_as_one_piece():
    """Entries pushed before splitting existed must stay pullable."""
    pieces = batch.entry_pieces(_legacy_entry())
    assert len(pieces) == 1
    assert pieces[0]["leaf"] == "P/videos/001"
    assert pieces[0]["session_id"] == "11111111-0000-0000-0000-000000000000"
    assert pieces[0]["sha256"] == "0" * 64


def test_entry_pieces_sorts_by_name():
    entry = _entry(pieces=[_piece(name="ab"), _piece(name="aa")])
    assert [p["name"] for p in batch.entry_pieces(entry)] == ["aa", "ab"]


# ---------------------------------------------------------------------------
# cmd_pull across pieces
# ---------------------------------------------------------------------------

def test_cmd_pull_reassembles_pieces_in_order(tmp_path, monkeypatch):
    """Three pieces must concatenate back to the original bytes."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    parts = {"aa": b"A" * 900, "ab": b"B" * 900, "ac": b"C" * 300}
    whole = b"".join(parts[n] for n in ("aa", "ab", "ac"))

    stores = {}
    for name, data in parts.items():
        store = tmp_path / f"store-{name}"
        store.mkdir()
        src = store / "src"
        src.write_bytes(data)
        enc = store / "enc"
        crypto.encrypt_stream(src, enc, "pw")
        splitter.split_file(enc, f"sess-{name}", store, 1024 * 1024)
        src.unlink()
        enc.unlink()
        stores[f"P/videos/003/{name}"] = store

    def _fake_download(leaf, dest_dir, credentials=None, session_id=None):
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            Path(shutil.copy(c, dest_dir / c.name))
            for c in sorted(stores[leaf].glob("*.part???"))
        )

    monkeypatch.setattr(batch.drive, "download_from_drive", _fake_download)

    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {"videos/003.zip": _entry(
        sha256=hashlib.sha256(whole).hexdigest(), size=len(whole),
        pieces=[
            _piece(name=n, leaf=f"P/videos/003/{n}",
                   sha256=hashlib.sha256(parts[n]).hexdigest(),
                   size=len(parts[n]))
            for n in ("aa", "ab", "ac")
        ],
    )}})

    out = tmp_path / "out"
    batch.cmd_pull(_pull_args(led, out))
    assert (out / "videos" / "003.zip").read_bytes() == whole


def test_cmd_pull_fails_on_a_bad_piece_not_just_the_whole(tmp_path, monkeypatch, capsys):
    """A corrupted piece must be named, rather than only the final hash failing."""
    monkeypatch.setenv("INNOUT_PASSPHRASE", "pw")
    _seed_drive(monkeypatch, b"piece payload", "pw", tmp_path / "store")
    led = tmp_path / "led.json"
    save_ledger(led, {"entries": {"videos/003.zip": _entry(
        sha256="0" * 64,
        pieces=[_piece(name="aa", sha256="e" * 64)],
    )}})

    with pytest.raises(SystemExit, match="failed verification"):
        batch.cmd_pull(_pull_args(led, tmp_path / "out"))
    assert "piece aa sha256 mismatch" in capsys.readouterr().out


def test_parser_default_piece_size_is_under_the_ceiling():
    args = build_parser().parse_args(["push", "--repo", "o/r"])
    assert args.max_piece_size == batch.DEFAULT_PIECE_MB
    assert args.max_piece_size * 1024 * 1024 < batch.AESGCM_MAX_BYTES
