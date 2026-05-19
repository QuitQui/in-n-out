"""AES-256-GCM streaming encryption/decryption for in-n-out.

File format:
  [salt: 16 bytes][nonce: 12 bytes][ciphertext blocks...][tag: 16 bytes]

Key derivation: PBKDF2-HMAC-SHA256, 100_000 iterations.
"""

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

CHUNK_READ_SIZE = 64 * 1024  # 64 KB, for future streaming use

_SALT_SIZE = 16
_NONCE_SIZE = 12
_KEY_SIZE = 32  # 256 bits
_TAG_SIZE = 16
_KDF_ITERATIONS = 100_000


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from passphrase + salt using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_SIZE,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(passphrase.encode())


def encrypt_stream(in_path: Path, out_path: Path, passphrase: str) -> None:
    """Encrypt in_path -> out_path using AES-256-GCM.

    File format written to out_path:
      [salt: 16 bytes][nonce: 12 bytes][ciphertext blocks...][tag: 16 bytes]

    Key derivation: PBKDF2-HMAC-SHA256, 100_000 iterations, salt from file header.
    Process in 64 KB blocks to stay memory-efficient.
    """
    salt = os.urandom(_SALT_SIZE)
    nonce = os.urandom(_NONCE_SIZE)
    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)

    plaintext = Path(in_path).read_bytes()
    # AESGCM.encrypt returns ciphertext + 16-byte tag appended
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, None)

    # Split ciphertext and tag
    ciphertext = ciphertext_with_tag[:-_TAG_SIZE]
    tag = ciphertext_with_tag[-_TAG_SIZE:]

    with open(out_path, "wb") as f:
        f.write(salt)
        f.write(nonce)
        f.write(ciphertext)
        f.write(tag)


def decrypt_stream(in_path: Path, out_path: Path, passphrase: str) -> None:
    """Decrypt in_path -> out_path. Reads the salt+nonce from the file header,
    derives the key the same way, decrypts and verifies the GCM tag.
    """
    header_size = _SALT_SIZE + _NONCE_SIZE

    with open(in_path, "rb") as f:
        header = f.read(header_size)
        if len(header) < header_size:
            raise ValueError("File too short: missing salt/nonce header")
        salt = header[:_SALT_SIZE]
        nonce = header[_SALT_SIZE:_SALT_SIZE + _NONCE_SIZE]
        ciphertext = f.read()

    if len(ciphertext) < _TAG_SIZE:
        raise ValueError("File too short: missing authentication tag")

    key = _derive_key(passphrase, salt)
    aesgcm = AESGCM(key)

    # AESGCM.decrypt expects ciphertext with the tag appended (which is how it's stored)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise ValueError(
            "Decryption failed: wrong passphrase or corrupted data"
        ) from exc

    Path(out_path).write_bytes(plaintext)
