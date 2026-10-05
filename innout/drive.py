"""Google Drive backend for uploading and downloading encrypted chunks."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource

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


def _get_service(credentials_file: str | None = None) -> Resource:
    """Build an authenticated Drive v3 service.

    Caches the OAuth token at ``~/.innout_drive_token.json``. If the cached
    refresh token has expired or been revoked (common with Google Cloud
    "Testing" apps whose refresh tokens expire after 7 days), the stale
    token is deleted and a fresh browser-based OAuth flow runs automatically.

    To avoid re-auth entirely: publish the Cloud Console app (move it from
    "Testing" to "Production" in the OAuth consent screen) — published apps
    get long-lived refresh tokens.
    """
    from google.auth.exceptions import RefreshError
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
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                # Only treat confirmed permanent OAuth failures (e.g. a revoked
                # or expired refresh token) as a reason to force re-auth.
                # Transient transport/network errors should propagate instead
                # of silently discarding a still-valid cached token.
                if "invalid_grant" not in str(exc):
                    raise
                print(f"[innout] Refresh token expired/revoked — "
                      f"removing {_TOKEN_PATH} and re-authenticating.")
                _TOKEN_PATH.unlink(missing_ok=True)
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_PATH.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


_FOLDER_MIME = "application/vnd.google-apps.folder"


def _escape_query_value(value: str) -> str:
    """Escape a value for interpolation into a single-quoted Drive query.

    A literal backslash or apostrophe in a folder name would otherwise
    terminate the quoted string and corrupt the query.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _split_folder_path(folder_name: str) -> list[str]:
    """Split a slash-separated Drive folder path into its segments.

    Collapses repeated and trailing slashes, so "a//b/" == "a/b". Rejects
    "." and ".." segments, which mean nothing in Drive and would otherwise
    become folders literally named "." or "..".
    """
    segments = [s for s in folder_name.split("/") if s]
    if not segments:
        raise ValueError(f"Invalid Drive folder path: {folder_name!r}")
    for segment in segments:
        if segment in (".", ".."):
            raise ValueError(
                f"Invalid Drive folder path {folder_name!r}: "
                f"segment {segment!r} is not allowed"
            )
    return segments


def _find_child_folder(service, name: str, parent_id: str) -> str | None:
    """Return the ID of folder `name` directly inside `parent_id`, else None.

    Scoping each lookup to its parent is what makes nested paths safe: a
    folder called "videos-001" elsewhere in the Drive can never match.
    """
    query = (
        f"name='{_escape_query_value(name)}' and "
        f"'{_escape_query_value(parent_id)}' in parents and "
        f"mimeType='{_FOLDER_MIME}' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def _resolve_folder_path(service, folder_name: str) -> str | None:
    """Resolve an existing slash-separated folder path to its ID, else None.

    Read-only counterpart of _get_or_create_folder, so a pull against a
    mistyped path fails loudly instead of silently creating empty folders.
    """
    parent_id = "root"
    for segment in _split_folder_path(folder_name):
        child_id = _find_child_folder(service, segment, parent_id)
        if child_id is None:
            return None
        parent_id = child_id
    return parent_id


def _get_or_create_folder(service, folder_name: str) -> str:
    """Resolve a slash-separated folder path, creating any missing levels.

    "VideoMME-v2/videos-001" resolves to the `videos-001` folder nested
    inside `VideoMME-v2`, creating either if absent, and returns the ID of
    the leaf — the folder chunks are uploaded into. One push per leaf
    folder: a pull joins *every* file it finds there, so two sessions
    sharing a folder would decrypt to garbage.

    Note: the OAuth scope is drive.file, so only folders this app created
    are visible. A `VideoMME-v2` folder made by hand in the Drive UI is
    invisible here and a same-named sibling gets created instead.
    """
    parent_id = "root"
    for segment in _split_folder_path(folder_name):
        child_id = _find_child_folder(service, segment, parent_id)
        if child_id is not None:
            parent_id = child_id
            continue
        meta = {
            "name": segment,
            "mimeType": _FOLDER_MIME,
            "parents": [parent_id],
        }
        parent_id = service.files().create(body=meta, fields="id").execute()["id"]
    return parent_id


def upload_to_drive(
    chunks: list[Path],
    folder_name: str,
    credentials_file: str | None = None,
) -> str:
    """Upload chunks to a Google Drive folder, returns the folder URL.

    folder_name may be a nested path ("VideoMME-v2/videos-001"); missing
    levels are created. Give each push its own leaf folder.
    """
    from googleapiclient.http import MediaFileUpload

    service = _get_service(credentials_file)
    folder_id = _get_or_create_folder(service, folder_name)

    for chunk in tqdm(chunks, desc="Uploading to Drive"):
        media = MediaFileUpload(
            str(chunk), mimetype="application/octet-stream", resumable=True
        )
        meta = {"name": chunk.name, "parents": [folder_id]}
        service.files().create(
            body=meta, media_body=media, fields="id"
        ).execute(num_retries=10)

    return f"https://drive.google.com/drive/folders/{folder_id}"


def download_from_drive(
    folder_name: str,
    dest_dir: Path,
    credentials_file: str | None = None,
) -> list[Path]:
    """Download all chunk files from a Google Drive folder into dest_dir.

    folder_name may be a nested path ("VideoMME-v2/videos-001"). Every file
    in the leaf folder is downloaded and later joined, so the folder must
    hold exactly one push's chunks.

    Returns the downloaded paths sorted by name (preserves chunk order).
    """
    from googleapiclient.http import MediaIoBaseDownload

    service = _get_service(credentials_file)

    folder_id = _resolve_folder_path(service, folder_name)
    if folder_id is None:
        raise ValueError(f"Drive folder not found: {folder_name!r}")

    query = f"'{_escape_query_value(folder_id)}' in parents and trashed=false"
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
                _, done = downloader.next_chunk(num_retries=10)
        downloaded.append(dest_file)

    return sorted(downloaded, key=lambda p: p.name)
