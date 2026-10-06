"""AES-256-GCM streaming encryption/decryption for in-n-out.

File format:
  [salt: 16 bytes][nonce: 12 bytes][ciphertext blocks...][tag: 16 bytes]

Key derivation: PBKDF2-HMAC-SHA256, 100_000 iterations.

Everything here is incremental: memory stays at one block regardless of
input size. The bytes produced are identical to what the one-shot
``AESGCM.encrypt`` path produced, so data written by any version of
in-n-out decrypts with any other -- the format never changed, only the
memory profile and the 2 GiB size ceiling that the one-shot API imposed.
"""

import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

CHUNK_READ_SIZE = 4 * 1024 * 1024  # 4 MB per block

_SALT_SIZE = 16
_NONCE_SIZE = 12
_KEY_SIZE = 32  # 256 bits
_TAG_SIZE = 16
_KDF_ITERATIONS = 100_000

HEADER_SIZE = _SALT_SIZE + _NONCE_SIZE
TAG_SIZE = _TAG_SIZE


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from passphrase + salt using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_SIZE,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(passphrase.encode())


class StreamEncryptor:
    """Incremental encryptor emitting the in-n-out file format.

    ``header`` + every ``update()`` result + ``finalize()`` concatenated is
    the complete file. Feeding salt/nonce in explicitly lets a resumed
    upload continue the same cipher stream instead of starting a new one.
    """

    def __init__(
        self,
        passphrase: str,
        salt: bytes | None = None,
        nonce: bytes | None = None,
    ) -> None:
        self.salt = salt if salt is not None else os.urandom(_SALT_SIZE)
        self.nonce = nonce if nonce is not None else os.urandom(_NONCE_SIZE)
        if len(self.salt) != _SALT_SIZE:
            raise ValueError(f"salt must be {_SALT_SIZE} bytes, got {len(self.salt)}")
        if len(self.nonce) != _NONCE_SIZE:
            raise ValueError(f"nonce must be {_NONCE_SIZE} bytes, got {len(self.nonce)}")
        key = _derive_key(passphrase, self.salt)
        self._enc = Cipher(algorithms.AES(key), modes.GCM(self.nonce)).encryptor()
        self._done = False

    @property
    def header(self) -> bytes:
        """The 28-byte salt+nonce prefix that opens the file."""
        return self.salt + self.nonce

    def update(self, data: bytes) -> bytes:
        if self._done:
            raise ValueError("update() after finalize()")
        return self._enc.update(data)

    def finalize(self) -> bytes:
        """Flush and return the trailing 16-byte GCM tag."""
        if self._done:
            raise ValueError("finalize() called twice")
        self._done = True
        rest = self._enc.finalize()
        return rest + self._enc.tag


class StreamDecryptor:
    """Incremental decryptor for the in-n-out file format.

    Callers push the raw file bytes through ``update()`` from offset 0 and
    call ``finalize()`` at the end; the 28-byte header and the trailing
    16-byte tag are peeled off internally, so the caller never has to know
    where either lives. ``finalize()`` raises ValueError if the tag does
    not verify.
    """

    def __init__(self, passphrase: str) -> None:
        self._passphrase = passphrase
        self._head = bytearray()
        self._tail = b""  # last 16 bytes seen; the tag, until more arrives
        self._dec = None
        self._done = False

    @property
    def started(self) -> bool:
        """True once the header has been consumed and the key derived."""
        return self._dec is not None

    def update(self, data: bytes) -> bytes:
        if self._done:
            raise ValueError("update() after finalize()")
        if self._dec is None:
            need = HEADER_SIZE - len(self._head)
            self._head += data[:need]
            data = data[need:]
            if len(self._head) < HEADER_SIZE:
                return b""
            salt = bytes(self._head[:_SALT_SIZE])
            nonce = bytes(self._head[_SALT_SIZE:HEADER_SIZE])
            key = _derive_key(self._passphrase, salt)
            self._dec = Cipher(algorithms.AES(key), modes.GCM(nonce)).decryptor()
            if not data:
                return b""

        # Hold back the final TAG_SIZE bytes: they are the tag unless more
        # ciphertext follows, and feeding the tag to update() corrupts the
        # plaintext.
        buf = self._tail + data
        if len(buf) <= _TAG_SIZE:
            self._tail = buf
            return b""
        self._tail = buf[-_TAG_SIZE:]
        return self._dec.update(buf[:-_TAG_SIZE])

    def finalize(self) -> bytes:
        if self._done:
            raise ValueError("finalize() called twice")
        self._done = True
        if self._dec is None:
            raise ValueError("File too short: missing salt/nonce header")
        if len(self._tail) < _TAG_SIZE:
            raise ValueError("File too short: missing authentication tag")
        try:
            return self._dec.finalize_with_tag(bytes(self._tail))
        except InvalidTag as exc:
            raise ValueError(
                "Decryption failed: wrong passphrase or corrupted data"
            ) from exc


def encrypt_stream(in_path: Path, out_path: Path, passphrase: str) -> None:
    """Encrypt in_path -> out_path using AES-256-GCM.

    File format written to out_path:
      [salt: 16 bytes][nonce: 12 bytes][ciphertext blocks...][tag: 16 bytes]

    Key derivation: PBKDF2-HMAC-SHA256, 100_000 iterations, salt from file
    header. Processed in CHUNK_READ_SIZE blocks, so a 100 GB input costs
    the same memory as a 100 KB one.
    """
    enc = StreamEncryptor(passphrase)
    with open(in_path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(enc.header)
        while True:
            block = src.read(CHUNK_READ_SIZE)
            if not block:
                break
            dst.write(enc.update(block))
        dst.write(enc.finalize())


def decrypt_stream(in_path: Path, out_path: Path, passphrase: str) -> None:
    """Decrypt in_path -> out_path, verifying the GCM tag.

    Reads the salt+nonce from the file header and derives the key the same
    way as encrypt_stream. Streams in CHUNK_READ_SIZE blocks.

    The GCM tag only verifies once the whole stream has been read, so
    out_path holds unauthenticated plaintext until this returns. On
    failure out_path is removed rather than left as a plausible-looking
    partial file.
    """
    dec = StreamDecryptor(passphrase)
    out_path = Path(out_path)
    try:
        with open(in_path, "rb") as src, open(out_path, "wb") as dst:
            while True:
                block = src.read(CHUNK_READ_SIZE)
                if not block:
                    break
                dst.write(dec.update(block))
            dst.write(dec.finalize())
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
