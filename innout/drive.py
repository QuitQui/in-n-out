"""Google Drive backend for uploading and downloading encrypted chunks."""

from __future__ import annotations

import io
import os
from pathlib import Path

from tqdm import tqdm

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_TOKEN_PATH = Path.home() / ".innout_drive_token.json"

# Default location of the OAuth client-secrets file. Resolved OUTSIDE the repo
# so credentials never sit inside the (now public) project folder: an explicit
# path wins, then $INNOUT_CREDENTIALS, then a dotfile in $HOME.
_DEFAULT_CREDENTIALS_PATH = Path.home() / ".innout_credentials.json"


def _resolve_credentials_path(credentials_file: str | None) -> str:
    """Pick the OAuth client-secrets path, keeping it out of the repo by default."""
    if credentials_file:
        return credentials_file
    return os.environ.get("INNOUT_CREDENTIALS") or str(_DEFAULT_CREDENTIALS_PATH)


def _get_service(credentials_file: str | None = None):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    credentials_file = _resolve_credentials_path(credentials_file)
    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_PATH.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _get_or_create_folder(service, folder_name: str) -> str:
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"name='{safe_name}' and "
        "mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_to_drive(
    chunks: list[Path],
    folder_name: str,
    credentials_file: str | None = None,
) -> str:
    """Upload chunks to a Google Drive folder, returns the folder URL."""
    from googleapiclient.http import MediaFileUpload

    service = _get_service(credentials_file)
    folder_id = _get_or_create_folder(service, folder_name)

    for chunk in tqdm(chunks, desc="Uploading to Drive"):
        media = MediaFileUpload(
            str(chunk), mimetype="application/octet-stream", resumable=True
        )
        meta = {"name": chunk.name, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()

    return f"https://drive.google.com/drive/folders/{folder_id}"


def download_from_drive(
    folder_name: str,
    dest_dir: Path,
    credentials_file: str | None = None,
) -> list[Path]:
    """Download all chunk files from a Google Drive folder into dest_dir.

    Returns the downloaded paths sorted by name (preserves chunk order).
    """
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service(credentials_file)

    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"name='{safe_name}' and "
        "mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])
    if not folders:
        raise ValueError(f"Drive folder not found: {folder_name!r}")
    folder_id = folders[0]["id"]

    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, fields="files(id, name)", orderBy="name"
    ).execute()
    items = results.get("files", [])

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for item in tqdm(items, desc="Downloading from Drive"):
        dest_file = dest_dir / item["name"]
        request = service.files().get_media(fileId=item["id"])
        with open(dest_file, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        downloaded.append(dest_file)

    return sorted(downloaded, key=lambda p: p.name)
