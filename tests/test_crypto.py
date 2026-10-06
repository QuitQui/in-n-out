import os

import pytest
from pathlib import Path

from innout import crypto
from innout.crypto import encrypt_stream, decrypt_stream


def test_roundtrip(tmp_path):
    src = tmp_path / "plain.bin"
    src.write_bytes(b"Hello, in-n-out world!" * 1000)
    enc = tmp_path / "encrypted"
    dec = tmp_path / "decrypted"

    encrypt_stream(src, enc, "secret")
    decrypt_stream(enc, dec, "secret")

    assert dec.read_bytes() == src.read_bytes()


def test_wrong_passphrase(tmp_path):
    src = tmp_path / "plain.bin"
    src.write_bytes(b"sensitive data")
    enc = tmp_path / "encrypted"

    encrypt_stream(src, enc, "correct")
    with pytest.raises(ValueError, match="Decryption failed"):
        decrypt_stream(enc, tmp_path / "out", "wrong")


def test_empty_file(tmp_path):
    src = tmp_path / "empty"
    src.write_bytes(b"")
    enc = tmp_path / "encrypted"
    dec = tmp_path / "decrypted"

    encrypt_stream(src, enc, "pass")
    decrypt_stream(enc, dec, "pass")
    assert dec.read_bytes() == b""


def test_different_runs_produce_different_ciphertext(tmp_path):
    src = tmp_path / "plain"
    src.write_bytes(b"same plaintext")

    enc1 = tmp_path / "enc1"
    enc2 = tmp_path / "enc2"
    encrypt_stream(src, enc1, "pass")
    encrypt_stream(src, enc2, "pass")

    # salt + nonce are random per run
    assert enc1.read_bytes() != enc2.read_bytes()


def test_truncated_file_raises(tmp_path):
    src = tmp_path / "plain"
    src.write_bytes(b"data")
    enc = tmp_path / "encrypted"
    encrypt_stream(src, enc, "pass")

    bad = tmp_path / "bad"
    bad.write_bytes(enc.read_bytes()[:5])
    with pytest.raises(ValueError):
        decrypt_stream(bad, tmp_path / "out", "pass")


# --- streaming rewrite: format compatibility and the gone 2 GiB ceiling ---


def test_output_is_byte_identical_to_one_shot_aesgcm():
    """The streaming rewrite must not change the on-disk format.

    Data already sitting in Drive was written by the old one-shot
    AESGCM.encrypt path. If the streaming encryptor produced different
    bytes for the same salt/nonce/plaintext, every existing push would
    become undecryptable by the new client and vice versa.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = b"S" * 16
    nonce = b"N" * 12
    plaintext = bytes(range(256)) * 1234  # not a block multiple

    key = crypto._derive_key("pw", salt)
    one_shot = salt + nonce + AESGCM(key).encrypt(nonce, plaintext, None)

    enc = crypto.StreamEncryptor("pw", salt=salt, nonce=nonce)
    streamed = bytearray(enc.header)
    for i in range(0, len(plaintext), 65536):
        streamed += enc.update(plaintext[i : i + 65536])
    streamed += enc.finalize()

    assert bytes(streamed) == one_shot


def test_decrypts_data_written_by_one_shot_aesgcm(tmp_path):
    """Old pushes must still decrypt -- 50 GB of them are already uploaded."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(16)
    nonce = os.urandom(12)
    plaintext = b"legacy payload" * 5000
    key = crypto._derive_key("pw", salt)
    blob = tmp_path / "old"
    blob.write_bytes(salt + nonce + AESGCM(key).encrypt(nonce, plaintext, None))

    out = tmp_path / "out"
    crypto.decrypt_stream(blob, out, "pw")
    assert out.read_bytes() == plaintext


def test_stream_decryptor_accepts_arbitrary_block_boundaries():
    """Parts arrive in whatever sizes the splitter chose; the header and the
    trailing tag may land mid-block or be split across two blocks."""
    plaintext = os.urandom(100_000)
    enc = crypto.StreamEncryptor("pw")
    blob = enc.header + enc.update(plaintext) + enc.finalize()

    for step in (1, 7, 16, 27, 28, 29, 4096, len(blob) - 1, len(blob), len(blob) + 10):
        dec = crypto.StreamDecryptor("pw")
        out = bytearray()
        for i in range(0, len(blob), step):
            out += dec.update(blob[i : i + step])
        out += dec.finalize()
        assert bytes(out) == plaintext, f"failed at step={step}"


def test_stream_decryptor_rejects_tampered_ciphertext():
    plaintext = os.urandom(50_000)
    enc = crypto.StreamEncryptor("pw")
    blob = bytearray(enc.header + enc.update(plaintext) + enc.finalize())
    blob[500] ^= 0xFF

    dec = crypto.StreamDecryptor("pw")
    dec.update(bytes(blob))
    with pytest.raises(ValueError, match="wrong passphrase or corrupted"):
        dec.finalize()


def test_stream_decryptor_rejects_truncated_stream():
    """A short read -- a part that never finished downloading -- must fail.

    GCM cannot tell truncation from tampering: the last 16 bytes of a
    truncated stream get treated as the tag and simply do not verify. Both
    land on the same honest message.
    """
    enc = crypto.StreamEncryptor("pw")
    blob = enc.header + enc.update(b"x" * 1000) + enc.finalize()

    dec = crypto.StreamDecryptor("pw")
    dec.update(blob[:-4])
    with pytest.raises(ValueError, match="wrong passphrase or corrupted"):
        dec.finalize()


def test_stream_decryptor_rejects_stream_shorter_than_tag():
    """Fewer than 16 bytes after the header: there is no tag at all."""
    enc = crypto.StreamEncryptor("pw")
    dec = crypto.StreamDecryptor("pw")
    dec.update(enc.header + b"\x00" * 4)
    with pytest.raises(ValueError, match="missing authentication tag"):
        dec.finalize()


def test_stream_decryptor_rejects_header_only():
    dec = crypto.StreamDecryptor("pw")
    dec.update(b"x" * 10)
    with pytest.raises(ValueError, match="missing salt/nonce header"):
        dec.finalize()


def test_encrypt_stream_exceeds_the_one_shot_2gib_ceiling(tmp_path):
    """AESGCM.encrypt() refuses anything over 2**31-1 bytes, which is what
    killed videos/003.zip (3.43 GB). The streaming path has no such limit;
    prove the encryptor swallows more than the ceiling in one stream
    without materialising it.
    """
    over = 2**31 - 1 + 1024
    enc = crypto.StreamEncryptor("pw")
    block = b"\0" * (8 * 1024 * 1024)
    written = len(enc.header)
    while written < over:
        written += len(enc.update(block))
    assert written > 2**31 - 1
    assert len(enc.finalize()) == crypto.TAG_SIZE


def test_decrypt_stream_removes_output_on_failure(tmp_path):
    enc = crypto.StreamEncryptor("pw")
    blob = tmp_path / "blob"
    blob.write_bytes(enc.header + enc.update(b"data" * 1000) + enc.finalize())

    out = tmp_path / "out"
    with pytest.raises(ValueError):
        crypto.decrypt_stream(blob, out, "wrong-pw")
    assert not out.exists(), "unauthenticated plaintext left on disk"


def test_header_size_constants_match_format():
    assert crypto.HEADER_SIZE == 28
    assert crypto.TAG_SIZE == 16


def test_encryptor_rejects_wrong_sized_salt_or_nonce():
    with pytest.raises(ValueError, match="salt must be"):
        crypto.StreamEncryptor("pw", salt=b"short", nonce=b"N" * 12)
    with pytest.raises(ValueError, match="nonce must be"):
        crypto.StreamEncryptor("pw", salt=b"S" * 16, nonce=b"short")


def test_encryptor_refuses_update_after_finalize():
    enc = crypto.StreamEncryptor("pw")
    enc.finalize()
    with pytest.raises(ValueError, match="after finalize"):
        enc.update(b"x")
    with pytest.raises(ValueError, match="twice"):
        enc.finalize()
