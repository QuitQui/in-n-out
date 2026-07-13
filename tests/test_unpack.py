"""Tests for innout.unpack — pull-side output finalization.

After decryption the blob is opaque bytes. finalize_output must turn it into
the friendliest single file: gzip data gets decompressed once so the user
receives `<topdir>.tar` (one extraction on Windows and Linux alike) instead
of an unnamed `result` that needs two rounds of unpacking.
"""

import io
import tarfile

import pytest

from innout import unpack


def _make_source_dir(tmp_path, name="myproj"):
    src = tmp_path / name
    src.mkdir()
    (src / "hello.txt").write_text("hello world\n")
    sub = src / "sub"
    sub.mkdir()
    (sub / "data.bin").write_bytes(b"\x00\x01\x02")
    return src


def _tar_bytes(src_dir, mode):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return buf.getvalue()


def test_gzip_tar_becomes_named_tar(tmp_path):
    src = _make_source_dir(tmp_path)
    blob = tmp_path / "decrypted"
    blob.write_bytes(_tar_bytes(src, "w:gz"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "myproj.tar"
    assert final.exists()
    with tarfile.open(final, "r:") as tar:  # plain tar, NOT gzip anymore
        names = tar.getnames()
    assert "myproj/hello.txt" in names
    assert "myproj/sub/data.bin" in names


def test_bare_tar_gets_named(tmp_path):
    src = _make_source_dir(tmp_path)
    blob = tmp_path / "decrypted"
    blob.write_bytes(_tar_bytes(src, "w"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "myproj.tar"
    with tarfile.open(final, "r:") as tar:
        assert "myproj/hello.txt" in tar.getnames()


def test_unknown_bytes_stay_result(tmp_path):
    blob = tmp_path / "decrypted"
    blob.write_bytes(b"just some plain file contents")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "result"
    assert final.read_bytes() == b"just some plain file contents"


def test_gzip_magic_false_positive_keeps_original(tmp_path):
    blob = tmp_path / "decrypted"
    blob.write_bytes(b"\x1f\x8b" + b"not really gzip data")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "result"
    assert final.read_bytes() == b"\x1f\x8b" + b"not really gzip data"


def test_existing_output_not_clobbered(tmp_path):
    src = _make_source_dir(tmp_path)
    blob = tmp_path / "decrypted"
    blob.write_bytes(_tar_bytes(src, "w:gz"))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "myproj.tar").write_bytes(b"precious existing file")

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "myproj-1.tar"
    assert (out_dir / "myproj.tar").read_bytes() == b"precious existing file"


def test_unsafe_member_name_falls_back_to_result_tar(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="../evil.txt")
        data = b"boom"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    blob = tmp_path / "decrypted"
    blob.write_bytes(buf.getvalue())
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    final = unpack.finalize_output(blob, out_dir)

    assert final == out_dir / "result.tar"


def test_cmd_pull_from_dir_emits_named_tar(tmp_path, capsys):
    """End-to-end: push-style archive → encrypt → split → pull --from-dir."""
    from innout import cli, crypto, splitter, sources

    src = _make_source_dir(tmp_path, name="mydata")
    work = tmp_path / "work"
    work.mkdir()
    archive = sources._tar_gz_dir(src, work)
    encrypted = work / "encrypted"
    crypto.encrypt_stream(archive, encrypted, "pw")
    chunks = splitter.split_file(encrypted, "sess", work, 1024 * 1024)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    for c in chunks:
        (chunk_dir / c.name).write_bytes(c.read_bytes())
    out_dir = tmp_path / "out"

    parser = cli.build_parser()
    args = parser.parse_args(
        ["pull", "--from-dir", str(chunk_dir), "--passphrase", "pw",
         "--output", str(out_dir)]
    )
    args.func(args)

    final = out_dir / "mydata.tar"
    assert final.exists()
    with tarfile.open(final, "r:") as tar:
        assert "mydata/hello.txt" in tar.getnames()
    captured = capsys.readouterr()
    assert "mydata.tar" in captured.out
    assert 'Extract with: tar -xf "mydata.tar"' in captured.out
