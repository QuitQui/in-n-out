"""Upload and download encrypted chunks to/from a self-hosted HTTP server."""

import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

_CONNECT_TIMEOUT = 30       # seconds
_READ_TIMEOUT = 300         # 5 minutes
_MAX_RETRIES = 3
_RETRY_SLEEP = 2            # seconds between retries


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = api_key or os.environ.get("INNOUT_API_KEY")
    if not key:
        raise RuntimeError(
            "API key required. Pass --api-key or set the INNOUT_API_KEY env var."
        )
    return {"Authorization": f"Bearer {key}"}


class _ProgressReader:
    """Wraps a binary file object and advances a tqdm bar as bytes are read."""

    def __init__(self, fobj, progress_bar: tqdm) -> None:
        self._fobj = fobj
        self._bar = progress_bar

    def read(self, size: int = -1) -> bytes:
        data = self._fobj.read(size)
        if data:
            self._bar.update(len(data))
        return data


def upload_chunks(
    chunks: list[Path],
    server_url: str,
    session_id: str,
    api_key: str | None = None,
) -> None:
    """Upload each chunk via multipart POST to {server_url}/upload.

    Each POST includes:
      - field "session_id": the session UUID string
      - field "part": zero-padded part index (e.g. "000", "001")
      - field "total_parts": total number of parts as string
      - file field "file": the chunk binary data with filename = chunk.name

    Shows per-chunk progress with tqdm (bytes uploaded).
    Retries each chunk up to 3 times on HTTP error before raising.
    """
    url = f"{server_url.rstrip('/')}/upload"
    headers = _auth_headers(api_key)
    total_parts = len(chunks)

    for idx, chunk_path in enumerate(chunks):
        part = f"{idx:03d}"
        chunk_size = chunk_path.stat().st_size
        attempt = 0

        while True:
            attempt += 1
            try:
                with tqdm(
                    total=chunk_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Uploading {chunk_path.name}",
                    leave=True,
                ) as bar:
                    with chunk_path.open("rb") as fobj:
                        reader = _ProgressReader(fobj, bar)
                        files = {
                            "file": (chunk_path.name, reader, "application/octet-stream"),
                        }
                        data = {
                            "session_id": session_id,
                            "part": part,
                            "total_parts": str(total_parts),
                        }
                        response = requests.post(
                            url,
                            files=files,
                            data=data,
                            headers=headers,
                            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                        )

                if not response.ok:
                    raise RuntimeError(
                        f"Server returned {response.status_code} for part {part} "
                        f"of session {session_id}: {response.text}"
                    )
                # Success — move to next chunk
                break

            except requests.RequestException as exc:
                if attempt >= _MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to upload part {part} of session {session_id} "
                        f"after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(_RETRY_SLEEP)


def download_chunks(
    server_url: str,
    session_id: str,
    output_dir: Path,
    api_key: str | None = None,
) -> list[Path]:
    """Download all chunks for session_id from the server.

    First GET {server_url}/manifest/{session_id} to get JSON like:
      {"session_id": "...", "parts": ["000", "001", ...]}

    Then for each part, GET {server_url}/download/{session_id}/{part}
    and save to output_dir/<session_id>.part<NNN>.

    Shows tqdm progress per chunk.
    Returns list of downloaded Paths sorted by part number.
    """
    base = server_url.rstrip("/")
    headers = _auth_headers(api_key)

    # Fetch manifest
    manifest_url = f"{base}/manifest/{session_id}"
    try:
        manifest_resp = requests.get(
            manifest_url,
            headers=headers,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch manifest for session {session_id}: {exc}"
        ) from exc

    if not manifest_resp.ok:
        raise RuntimeError(
            f"Server returned {manifest_resp.status_code} for manifest of "
            f"session {session_id}: {manifest_resp.text}"
        )

    manifest = manifest_resp.json()
    parts: list[str] = manifest["parts"]

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for part in parts:
        download_url = f"{base}/download/{session_id}/{part}"
        dest = output_dir / f"{session_id}.part{part}"
        attempt = 0

        while True:
            attempt += 1
            try:
                with requests.get(
                    download_url,
                    headers=headers,
                    stream=True,
                    timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                ) as resp:
                    if not resp.ok:
                        raise RuntimeError(
                            f"Server returned {resp.status_code} for part {part} "
                            f"of session {session_id}: {resp.text}"
                        )

                    total_size = int(resp.headers.get("Content-Length", 0)) or None
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading part {part}",
                        leave=True,
                    ) as bar:
                        with dest.open("wb") as fobj:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    fobj.write(chunk)
                                    bar.update(len(chunk))

                # Success
                downloaded.append(dest)
                break

            except requests.RequestException as exc:
                if attempt >= _MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to download part {part} of session {session_id} "
                        f"after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(_RETRY_SLEEP)

    return sorted(downloaded, key=lambda p: p.name)
