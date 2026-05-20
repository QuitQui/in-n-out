import pytest
from pathlib import Path
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
