"""Tests for the single-pass tar -> encrypt -> parts pipeline."""

import hashlib
import io
import os
import tarfile

import pytest

from innout import crypto, splitter, stream


# --- size parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("1kb", 1000),
        ("1KiB", 1024),
        ("1.8GiB", 1932735283),
        ("512MB", 512_000_000),
        ("2g", 2 * 1024**3),
        (" 1.5 GiB ", 1610612736),
        ("1_000", 1000),
    ],
)
def test_parse_size(text, expected):
    assert stream.parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "   ", "gib", "-5", "0", "abc", "1.2.3MB"])
def test_parse_size_rejects_garbage(bad):
    with pytest.raises(ValueError):
        stream.parse_size(bad)


def test_chosen_part_size_respects_the_hard_2gb_limit():
    """The operator's constraint is that no part may reach 2 GB, with margin.

    1.8 GiB is the configured default for the Video-MME-v2 push. Guard both
    the decimal 2 GB the constraint was stated in and the 2**31-1 ceiling
    that the old one-shot crypto imposed, so nobody raises this later
    without seeing the limits.
    """
    size = stream.parse_size("1.8GiB")
    assert size < 2_000_000_000, "exceeds the stated 2 GB limit"
    assert size < 2**31 - 1, "exceeds the legacy one-shot AES-GCM ceiling"
    assert 2_000_000_000 - size > 60_000_000, "less than 60 MB of margin"


def test_part_name_zero_pads_to_three_digits():
    assert stream.part_name("Video-MME-v2", 0) == "Video-MME-v2.part000"
    assert stream.part_name("Video-MME-v2", 54) == "Video-MME-v2.part054"
    assert stream.part_name("x", 999) == "x.part999"


def test_part_name_refuses_to_overflow_the_glob():
    """The pull side globs *.part??? -- a 4-digit index would be invisible."""
    with pytest.raises(ValueError, match="exceeds the 1000-part limit"):
        stream.part_name("x", 1000)


# --- helpers ----------------------------------------------------------------


def make_tree(root, *, junk=True):
    """A miniature Video-MME-v2: zips under videos/, assets, parquet."""
    (root / "videos").mkdir(parents=True)
    (root / "assets").mkdir()
    files = {
        "videos/001.zip": b"PK\x03\x04" + os.urandom(3000),
        "videos/002.zip": b"PK\x03\x04" + os.urandom(2500),
        "assets/logo.png": os.urandom(800),
        "subtitle.zip": b"PK\x03\x04" + os.urandom(1200),
        "test.parquet": os.urandom(600),
        "README.md": b"# Video-MME-v2\n",
    }
    if junk:
        # snapshot_download(local_dir=...) leaves this bookkeeping behind
        (root / ".cache" / "huggingface").mkdir(parents=True)
        files[".cache/huggingface/.metadata"] = b"junk"
    for name, blob in files.items():
        (root / name).write_bytes(blob)
    return files


def join_and_decrypt(part_paths, passphrase):
    dec = crypto.StreamDecryptor(passphrase)
    out = bytearray()
    for path in sorted(part_paths):
        out += dec.update(path.read_bytes())
    out += dec.finalize()
    return bytes(out)


# --- the pipeline -----------------------------------------------------------


def test_roundtrip_restores_the_whole_tree(tmp_path):
    src = tmp_path / "Video-MME-v2"
    files = make_tree(src, junk=False)

    out = tmp_path / "parts"
    with stream.tar_stream(src) as reader:
        result = stream.encrypt_to_parts(
            reader, "pw", out, "Video-MME-v2", 4096, delete_after=False
        )

    blob = join_and_decrypt(list(out.glob("*.part???")), "pw")
    extracted = tmp_path / "restored"
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        tar.extractall(extracted, filter="data")

    base = extracted / "Video-MME-v2"
    assert base.is_dir()
    for name, blob_bytes in files.items():
        assert (base / name).read_bytes() == blob_bytes, name
    assert len(result.parts) > 1, "test needs multiple parts to be meaningful"


def test_parts_concatenate_to_the_legacy_format(tmp_path):
    """A stream push and an old-style push of the same bytes must produce
    the same file, or data already in Drive and data pushed from here stop
    being interchangeable."""
    payload = os.urandom(50_000)
    salt, nonce = b"S" * 16, b"N" * 12

    # old path: encrypt whole file, then split
    plain = tmp_path / "plain"
    plain.write_bytes(payload)
    legacy_blob = tmp_path / "legacy"
    enc = crypto.StreamEncryptor("pw", salt=salt, nonce=nonce)
    legacy_blob.write_bytes(enc.header + enc.update(payload) + enc.finalize())
    legacy_parts = splitter.split_file(legacy_blob, "sid", tmp_path / "legacy_parts", 8192)

    # new path: encrypt straight into parts
    out = tmp_path / "streamed"
    stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", out, "p", 8192,
        salt=salt, nonce=nonce, delete_after=False,
    )
    streamed = sorted(out.glob("*.part???"))

    assert len(streamed) == len(legacy_parts)
    assert b"".join(p.read_bytes() for p in streamed) == legacy_blob.read_bytes()
    for new, old in zip(streamed, legacy_parts):
        assert new.read_bytes() == old.read_bytes(), f"{new.name} != {old.name}"


def test_every_part_but_the_last_is_exactly_part_size(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(30_000)), "pw", out, "p", 4096, delete_after=False
    )
    sizes = [p.size for p in result.parts]
    assert all(s == 4096 for s in sizes[:-1]), sizes
    assert 0 < sizes[-1] <= 4096
    assert sum(sizes) == crypto.HEADER_SIZE + 30_000 + crypto.TAG_SIZE


def test_no_empty_trailing_part_when_size_divides_exactly(tmp_path):
    """An exact multiple must not leave a zero-byte part behind: the pull
    side would join it happily but a 0-byte file in Drive looks like a
    failed upload."""
    part_size = 1024
    payload_len = part_size * 3 - crypto.HEADER_SIZE - crypto.TAG_SIZE
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(payload_len)), "pw", out, "p", part_size,
        delete_after=False,
    )
    assert [p.size for p in result.parts] == [part_size] * 3
    assert sorted(f.name for f in out.glob("*")) == [
        "p.part000", "p.part001", "p.part002",
    ]


def test_only_one_part_exists_on_disk_at_a_time(tmp_path):
    """The whole point of the rewrite: a 105 GB push must not need 105 GB
    of parts on disk."""
    out = tmp_path / "parts"
    seen = []

    def on_part(info):
        seen.append(len(list(out.glob("*.part???"))))

    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(40_000)), "pw", out, "p", 4096, on_part=on_part
    )
    assert len(result.parts) > 5
    assert seen == [1] * len(result.parts), seen
    assert list(out.glob("*.part???")) == [], "parts left behind after upload"


def test_sha256_matches_the_written_bytes(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(20_000)), "pw", out, "p", 4096, delete_after=False
    )
    for info in result.parts:
        assert stream.sha256_file(out / info.name) == info.sha256
        assert (out / info.name).stat().st_size == info.size


# --- resume -----------------------------------------------------------------


def test_resume_skips_uploaded_parts_but_keeps_the_cipher_stream_continuous(tmp_path):
    """AES-GCM is one stream over the whole archive, so a resumed run must
    reproduce the same salt, nonce and byte offsets -- otherwise the parts
    already in Drive and the ones uploaded after the interruption cannot
    be joined."""
    payload = os.urandom(40_000)
    salt, nonce = os.urandom(16), os.urandom(12)

    full = stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", tmp_path / "full", "p", 4096,
        salt=salt, nonce=nonce, delete_after=False,
    )

    resumed = stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", tmp_path / "resumed", "p", 4096,
        salt=salt, nonce=nonce, skip_before=3, delete_after=False,
    )

    assert [p.sha256 for p in resumed.parts] == [p.sha256 for p in full.parts]
    assert [p.size for p in resumed.parts] == [p.size for p in full.parts]
    # skipped parts are hashed but never written
    written = sorted(f.name for f in (tmp_path / "resumed").glob("*.part???"))
    assert written == [p.name for p in full.parts[3:]]
    for info in resumed.parts[:3]:
        assert info.path is None


def test_resumed_parts_still_join_with_the_originals(tmp_path):
    payload = os.urandom(30_000)
    salt, nonce = os.urandom(16), os.urandom(12)
    first = tmp_path / "first"
    stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", first, "p", 4096,
        salt=salt, nonce=nonce, delete_after=False,
    )
    # pretend parts 0-2 uploaded, then the link died; resume into a new dir
    second = tmp_path / "second"
    stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", second, "p", 4096,
        salt=salt, nonce=nonce, skip_before=3, delete_after=False,
    )

    mixed = sorted(list(first.glob("*.part00[0-2]")) + list(second.glob("*.part???")))
    assert join_and_decrypt(mixed, "pw") == payload


def test_tar_stream_is_reproducible_across_runs(tmp_path):
    """Resume regenerates the tar from the source tree; if the bytes drifted
    between runs the already-uploaded parts would no longer line up."""
    src = tmp_path / "Video-MME-v2"
    make_tree(src)

    def digest():
        h = hashlib.sha256()
        with stream.tar_stream(src) as reader:
            while True:
                block = reader.read(65536)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    assert digest() == digest()


# --- excludes and errors ----------------------------------------------------


def test_tar_stream_excludes_the_hf_download_cache(tmp_path):
    """snapshot_download leaves .cache/huggingface bookkeeping inside the
    directory; shipping it would put junk in the restored dataset."""
    src = tmp_path / "Video-MME-v2"
    make_tree(src, junk=True)

    names = []
    with stream.tar_stream(src, excludes=[".cache"]) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            names = [m.name for m in tar]

    assert not any(".cache" in n for n in names), names
    assert "Video-MME-v2/videos/001.zip" in names
    assert "Video-MME-v2/test.parquet" in names


def test_tar_stream_propagates_a_failure_from_the_builder_thread(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises((FileNotFoundError, OSError)):
        with stream.tar_stream(missing) as reader:
            reader.read()


def test_tar_stream_does_not_hang_when_the_consumer_bails_out(tmp_path):
    """A consumer that raises mid-stream leaves the builder thread blocked
    on a full pipe; closing the read end has to unblock it."""
    src = tmp_path / "Video-MME-v2"
    make_tree(src)
    with pytest.raises(RuntimeError):
        with stream.tar_stream(src) as reader:
            reader.read(16)
            raise RuntimeError("consumer gave up")


def test_encrypt_to_parts_rejects_a_nonpositive_part_size(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        stream.encrypt_to_parts(io.BytesIO(b"x"), "pw", tmp_path, "p", 0)


# --- manifest ---------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(20_000)), "pw", out, "p", 4096, delete_after=False
    )
    path = stream.write_manifest(result.parts, tmp_path / "MANIFEST.sha256")
    parsed = stream.read_manifest(path)

    assert parsed == {p.name: p.sha256 for p in result.parts}
    for name, digest in parsed.items():
        assert stream.sha256_file(out / name) == digest


def test_manifest_is_sha256sum_compatible(tmp_path):
    parts = [stream.PartInfo(index=0, name="p.part000", size=1, sha256="a" * 64)]
    path = stream.write_manifest(parts, tmp_path / "M")
    assert path.read_text() == f"{'a' * 64}  p.part000\n"


def test_read_manifest_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "M"
    path.write_text("not-a-digest  p.part000\n")
    with pytest.raises(ValueError, match="cannot parse manifest"):
        stream.read_manifest(path)


def test_read_manifest_ignores_blanks_and_comments(tmp_path):
    path = tmp_path / "M"
    path.write_text(f"# header\n\n{'b' * 64}  p.part000\n")
    assert stream.read_manifest(path) == {"p.part000": "b" * 64}


# --- resume state -----------------------------------------------------------


def test_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    assert stream.load_state(path) == {}
    stream.save_state(path, {"uploaded": [0, 1], "salt": "ab"})
    assert stream.load_state(path) == {"uploaded": [0, 1], "salt": "ab"}


def test_load_state_tolerates_a_corrupt_file(tmp_path):
    """A state file truncated by a crash must not block a fresh start."""
    path = tmp_path / "state.json"
    path.write_text("{not json")
    assert stream.load_state(path) == {}


def test_save_state_leaves_no_temp_file(tmp_path):
    path = tmp_path / "state.json"
    stream.save_state(path, {"a": 1})
    assert sorted(f.name for f in tmp_path.iterdir()) == ["state.json"]


# --- pull side: finding and verifying parts ---------------------------------


def write_parts(dirpath, names):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dirpath / name).write_bytes(b"x")
    return dirpath


def test_find_parts_sorts_numerically(tmp_path):
    d = write_parts(tmp_path / "p", ["a.part002", "a.part000", "a.part001"])
    assert [p.name for p in stream.find_parts(d)] == [
        "a.part000", "a.part001", "a.part002",
    ]


def test_find_parts_refuses_an_empty_directory(tmp_path):
    (tmp_path / "p").mkdir()
    with pytest.raises(ValueError, match="no part files"):
        stream.find_parts(tmp_path / "p")


def test_find_parts_refuses_two_pushes_in_one_directory(tmp_path):
    """Concatenating parts from two cipher streams decrypts to nothing, so
    say so before the operator spends hours on it."""
    d = write_parts(tmp_path / "p", ["a.part000", "b.part000"])
    with pytest.raises(ValueError, match="2 different pushes"):
        stream.find_parts(d)


def test_find_parts_picks_one_push_by_prefix(tmp_path):
    d = write_parts(tmp_path / "p", ["a.part000", "a.part001", "b.part000"])
    assert [p.name for p in stream.find_parts(d, "a")] == ["a.part000", "a.part001"]


def test_find_parts_reports_a_missing_part_by_number(tmp_path):
    """A gap is the one failure mode a 55-part manual download invites."""
    d = write_parts(tmp_path / "p", ["a.part000", "a.part002", "a.part003"])
    with pytest.raises(ValueError, match=r"part\(s\) \[1\] are missing"):
        stream.find_parts(d)


def test_verify_parts_accepts_matching_digests(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(9000)), "pw", out, "p", 2048, delete_after=False
    )
    manifest = {p.name: p.sha256 for p in result.parts}
    stream.verify_parts(stream.find_parts(out), manifest)


def test_verify_parts_names_the_corrupt_part(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(9000)), "pw", out, "p", 2048, delete_after=False
    )
    manifest = {p.name: p.sha256 for p in result.parts}
    victim = out / "p.part001"
    victim.write_bytes(victim.read_bytes()[:-5] + b"BROKE")

    with pytest.raises(ValueError, match="sha256 mismatch on p.part001"):
        stream.verify_parts(stream.find_parts(out), manifest)


def test_verify_parts_flags_a_part_absent_from_the_manifest(tmp_path):
    d = write_parts(tmp_path / "p", ["a.part000"])
    with pytest.raises(ValueError, match="no entry for: a.part000"):
        stream.verify_parts(stream.find_parts(d), {})


def test_verify_parts_flags_a_manifest_entry_with_no_file(tmp_path):
    out = tmp_path / "parts"
    result = stream.encrypt_to_parts(
        io.BytesIO(os.urandom(5000)), "pw", out, "p", 2048, delete_after=False
    )
    manifest = {p.name: p.sha256 for p in result.parts}
    manifest["p.part099"] = "f" * 64
    with pytest.raises(ValueError, match="not present: p.part099"):
        stream.verify_parts(stream.find_parts(out), manifest)


# --- pull side: streaming decrypt -------------------------------------------


def test_iter_plaintext_restores_the_payload(tmp_path):
    payload = os.urandom(30_000)
    out = tmp_path / "parts"
    stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", out, "p", 4096, delete_after=False
    )
    joined = b"".join(stream.iter_plaintext(stream.find_parts(out), "pw"))
    assert joined == payload


def test_iter_plaintext_rejects_a_wrong_passphrase(tmp_path):
    out = tmp_path / "parts"
    stream.encrypt_to_parts(
        io.BytesIO(os.urandom(5000)), "pw", out, "p", 2048, delete_after=False
    )
    with pytest.raises(ValueError, match="wrong passphrase or corrupted"):
        list(stream.iter_plaintext(stream.find_parts(out), "nope"))


def test_delete_consumed_frees_each_part_as_it_is_read(tmp_path):
    """Without this a restore needs the parts and the result side by side --
    105 GB + 105 GB for Video-MME-v2."""
    payload = os.urandom(20_000)
    out = tmp_path / "parts"
    stream.encrypt_to_parts(
        io.BytesIO(payload), "pw", out, "p", 4096, delete_after=False
    )
    parts = stream.find_parts(out)
    remaining = []

    def on_part(_path):
        remaining.append(len(list(out.glob("*.part???"))))

    joined = b"".join(
        stream.iter_plaintext(parts, "pw", delete_consumed=True, on_part=on_part)
    )
    assert joined == payload
    # on_part fires before the delete, so counts walk down from len(parts)
    assert remaining == list(range(len(parts), 0, -1)), remaining
    assert list(out.glob("*.part???")) == []


def test_plaintext_stream_feeds_tarfile_directly(tmp_path):
    """The restore path: parts in, extracted tree out, no 105 GB tar in
    between."""
    src = tmp_path / "Video-MME-v2"
    files = make_tree(src, junk=False)
    out = tmp_path / "parts"
    with stream.tar_stream(src) as reader:
        stream.encrypt_to_parts(
            reader, "pw", out, "Video-MME-v2", 4096, delete_after=False
        )

    dest = tmp_path / "restored"
    with stream.plaintext_stream(stream.find_parts(out), "pw") as reader:
        with tarfile.open(fileobj=reader, mode="r|*") as tar:
            tar.extractall(dest, filter="data")

    for name, blob in files.items():
        assert (dest / "Video-MME-v2" / name).read_bytes() == blob


def test_plaintext_stream_does_not_hang_when_the_consumer_bails_out(tmp_path):
    out = tmp_path / "parts"
    stream.encrypt_to_parts(
        io.BytesIO(os.urandom(200_000)), "pw", out, "p", 4096, delete_after=False
    )
    with pytest.raises(RuntimeError):
        with stream.plaintext_stream(stream.find_parts(out), "pw") as reader:
            reader.read(16)
            raise RuntimeError("consumer gave up")


def test_plaintext_stream_propagates_a_bad_passphrase(tmp_path):
    out = tmp_path / "parts"
    stream.encrypt_to_parts(
        io.BytesIO(os.urandom(5000)), "pw", out, "p", 2048, delete_after=False
    )
    with pytest.raises(ValueError, match="wrong passphrase or corrupted"):
        with stream.plaintext_stream(stream.find_parts(out), "nope") as reader:
            reader.read()
