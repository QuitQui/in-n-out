"""Tests for Google Drive upload/download backend and pull modes."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from innout import crypto, splitter
from innout.drive import (
    _get_or_create_folder,
    _resolve_credentials_path,
    download_from_drive,
    upload_to_drive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(tmp: Path, content: bytes = b"hello world " * 100) -> list[Path]:
    src = tmp / "data.bin"
    src.write_bytes(content)
    return splitter.split_file(src, "test-session", tmp, chunk_size=50)


# ---------------------------------------------------------------------------
# _resolve_credentials_path — keep credentials OUT of the repo
# ---------------------------------------------------------------------------

def test_resolve_credentials_explicit_path_wins(monkeypatch):
    monkeypatch.setenv("INNOUT_CREDENTIALS", "/from/env.json")
    assert _resolve_credentials_path("/explicit/creds.json") == "/explicit/creds.json"


def test_resolve_credentials_uses_env_var(monkeypatch):
    monkeypatch.setenv("INNOUT_CREDENTIALS", "/home/me/.secrets/creds.json")
    assert _resolve_credentials_path(None) == "/home/me/.secrets/creds.json"


def test_resolve_credentials_defaults_outside_repo(monkeypatch):
    monkeypatch.delenv("INNOUT_CREDENTIALS", raising=False)
    resolved = Path(_resolve_credentials_path(None))
    # Must live under $HOME, never as a repo-relative "credentials.json".
    assert resolved == Path.home() / ".innout_credentials.json"
    assert resolved.is_absolute()
    assert resolved.name != "credentials.json" or resolved.parent == Path.home()


# ---------------------------------------------------------------------------
# _get_or_create_folder
# ---------------------------------------------------------------------------

def test_get_or_create_folder_reuses_existing():
    service = MagicMock()
    service.files().list().execute.return_value = {"files": [{"id": "abc123", "name": "my-folder"}]}
    folder_id = _get_or_create_folder(service, "my-folder")
    assert folder_id == "abc123"
    service.files().create.assert_not_called()


def test_get_or_create_folder_creates_new():
    service = MagicMock()
    service.files().list().execute.return_value = {"files": []}
    service.files().create().execute.return_value = {"id": "newid"}
    folder_id = _get_or_create_folder(service, "new-folder")
    assert folder_id == "newid"


# ---------------------------------------------------------------------------
# upload_to_drive
# ---------------------------------------------------------------------------

@patch("innout.drive._get_service")
def test_upload_to_drive_returns_folder_url(mock_get_service):
    service = MagicMock()
    mock_get_service.return_value = service
    service.files().list().execute.return_value = {"files": [{"id": "folder123", "name": "test"}]}
    service.files().create().execute.return_value = {"id": "file1"}

    tmp = Path(tempfile.mkdtemp())
    try:
        chunks = _make_chunks(tmp)
        url = upload_to_drive(chunks, "test-folder", credentials_file="fake.json")
        assert url == "https://drive.google.com/drive/folders/folder123"
    finally:
        shutil.rmtree(tmp)


@patch("innout.drive._get_service")
def test_upload_to_drive_uploads_each_chunk(mock_get_service):
    service = MagicMock()
    mock_get_service.return_value = service
    service.files().list().execute.return_value = {"files": [{"id": "folder123"}]}
    service.files().create.return_value.execute.return_value = {"id": "file1"}

    tmp = Path(tempfile.mkdtemp())
    try:
        content = b"x" * 200
        chunks = _make_chunks(tmp, content)
        upload_to_drive(chunks, "test-folder", credentials_file="fake.json")
        assert service.files().create.call_count == len(chunks)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# pull --from-dir (cmd_pull with local chunks)
# ---------------------------------------------------------------------------

def test_pull_from_dir_round_trip():
    """Encrypt + split → copy to 'downloaded' dir → join + decrypt via --from-dir logic."""
    tmp = Path(tempfile.mkdtemp())
    try:
        original = b"secret repo content " * 500
        src = tmp / "src.bin"
        src.write_bytes(original)

        encrypted = tmp / "encrypted"
        passphrase = "test-passphrase"
        crypto.encrypt_stream(src, encrypted, passphrase)

        chunks = splitter.split_file(encrypted, "sess-abc", tmp / "chunks", chunk_size=200)

        # Simulate manual download: copy chunks to a separate dir
        downloaded = tmp / "downloaded"
        downloaded.mkdir()
        for chunk in chunks:
            shutil.copy(chunk, downloaded / chunk.name)

        found = sorted(downloaded.glob("*.part???"))
        assert len(found) == len(chunks)

        joined = tmp / "joined"
        splitter.join_files(found, joined)

        output = tmp / "output"
        crypto.decrypt_stream(joined, output, passphrase)

        assert output.read_bytes() == original
    finally:
        shutil.rmtree(tmp)


def test_pull_from_dir_empty_raises():
    """--from-dir with no chunk files should raise SystemExit."""
    import argparse
    from innout.cli import cmd_pull

    tmp = Path(tempfile.mkdtemp())
    try:
        empty_dir = tmp / "empty"
        empty_dir.mkdir()
        args = argparse.Namespace(
            from_dir=str(empty_dir),
            drive=None,
            server=None,
            session_id=None,
            passphrase="p",
            api_key=None,
            output=str(tmp / "out"),
        )
        with pytest.raises(SystemExit, match="no chunk files"):
            cmd_pull(args)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# download_from_drive
# ---------------------------------------------------------------------------

@patch("innout.drive._get_service")
def test_download_from_drive_raises_if_folder_missing(mock_get_service):
    """Raises ValueError when the named Drive folder does not exist."""
    service = MagicMock()
    mock_get_service.return_value = service
    service.files().list().execute.return_value = {"files": []}

    tmp = Path(tempfile.mkdtemp())
    try:
        with pytest.raises(ValueError, match="Drive folder not found"):
            download_from_drive("no-such-folder", tmp, credentials_file="fake.json")
    finally:
        shutil.rmtree(tmp)


@patch("innout.drive._get_service")
def test_download_from_drive_downloads_all_files(mock_get_service, tmp_path):
    """Downloads every file listed in the Drive folder."""
    from innout.drive import download_from_drive

    service = MagicMock()
    mock_get_service.return_value = service

    folder_list = MagicMock()
    folder_list.execute.return_value = {"files": [{"id": "folder1", "name": "my-folder"}]}

    file_list = MagicMock()
    file_list.execute.return_value = {
        "files": [
            {"id": "f1", "name": "sess.part000"},
            {"id": "f2", "name": "sess.part001"},
        ]
    }

    service.files().list.side_effect = [folder_list, file_list]

    fake_content = b"chunk data"

    def _fake_get_media(fileId):
        mock_request = MagicMock()
        mock_request.http = None
        mock_request.uri = "http://fake"

        class _FakeDownloader:
            def __init__(self, fh, req):
                self._fh = fh
                self._done = False

            def next_chunk(self):
                if not self._done:
                    self._fh.write(fake_content)
                    self._done = True
                    return None, True
                return None, True

        return mock_request

    call_count = [0]

    with patch("googleapiclient.http.MediaIoBaseDownload") as mock_dl_cls:
        def _make_downloader(fh, req):
            call_count[0] += 1
            fh.write(fake_content)
            dl = MagicMock()
            dl.next_chunk.return_value = (None, True)
            return dl

        mock_dl_cls.side_effect = _make_downloader
        files = download_from_drive("my-folder", tmp_path, credentials_file="fake.json")

    assert len(files) == 2
    assert call_count[0] == 2
    assert all(f.parent == tmp_path for f in files)


@patch("innout.drive._get_service")
def test_download_from_drive_returns_sorted(mock_get_service, tmp_path):
    """Returned paths are sorted by name regardless of Drive listing order."""
    service = MagicMock()
    mock_get_service.return_value = service

    folder_list = MagicMock()
    folder_list.execute.return_value = {"files": [{"id": "fld", "name": "my-folder"}]}

    file_list = MagicMock()
    file_list.execute.return_value = {
        "files": [
            {"id": "f2", "name": "sess.part001"},
            {"id": "f1", "name": "sess.part000"},
        ]
    }
    service.files().list.side_effect = [folder_list, file_list]

    with patch("googleapiclient.http.MediaIoBaseDownload") as mock_dl_cls:
        def _make_downloader(fh, req):
            fh.write(b"x")
            dl = MagicMock()
            dl.next_chunk.return_value = (None, True)
            return dl

        mock_dl_cls.side_effect = _make_downloader
        files = download_from_drive("my-folder", tmp_path, credentials_file="fake.json")

    assert [f.name for f in files] == ["sess.part000", "sess.part001"]
