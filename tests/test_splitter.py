import pytest
from pathlib import Path
from innout.splitter import split_file, join_files


def _make_file(path: Path, size_bytes: int) -> Path:
    path.write_bytes(bytes(range(256)) * (size_bytes // 256) + bytes(range(size_bytes % 256)))
    return path


def test_single_chunk_if_file_fits(tmp_path):
    src = _make_file(tmp_path / "data", 1024)
    chunks = split_file(src, "sess-1", tmp_path, chunk_size=4096)
    assert len(chunks) == 1
    assert chunks[0].read_bytes() == src.read_bytes()


def test_splits_into_multiple_chunks(tmp_path):
    src = _make_file(tmp_path / "data", 1000)
    chunks = split_file(src, "sess-2", tmp_path / "out", chunk_size=300)
    assert len(chunks) == 4  # ceil(1000/300) = 4


def test_join_restores_original(tmp_path):
    original = b"abcdefghij" * 500
    src = tmp_path / "original"
    src.write_bytes(original)

    chunks = split_file(src, "sess-3", tmp_path / "chunks", chunk_size=1000)
    joined = tmp_path / "joined"
    join_files(chunks, joined)

    assert joined.read_bytes() == original


def test_empty_file_produces_no_chunks(tmp_path):
    src = tmp_path / "empty"
    src.write_bytes(b"")
    chunks = split_file(src, "sess-4", tmp_path / "out", chunk_size=1024)
    assert chunks == []


def test_chunk_filenames_are_ordered(tmp_path):
    src = _make_file(tmp_path / "data", 3000)
    chunks = split_file(src, "my-session", tmp_path / "out", chunk_size=1000)
    names = [c.name for c in chunks]
    assert names == ["my-session.part000", "my-session.part001", "my-session.part002"]
