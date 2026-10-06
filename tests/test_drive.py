"""Tests for Google Drive upload/download backend and pull modes."""

import http.client
import shutil
import socket
import ssl
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from innout import crypto, drive, splitter
from innout.drive import (
    _is_transient,
    _list_folder_files,
    _with_retry,
    delete_session_files,
    list_sessions,
    _escape_query_value,
    _find_child_folder,
    _get_or_create_folder,
    _resolve_credentials_path,
    _resolve_folder_path,
    _split_folder_path,
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
# Nested folder paths
# ---------------------------------------------------------------------------

def _mock_service(list_results, create_ids=()):
    """Service whose list/create calls return the given results in order."""
    service = MagicMock()
    service.files.return_value.list.return_value.execute.side_effect = list(list_results)
    service.files.return_value.create.return_value.execute.side_effect = [
        {"id": cid} for cid in create_ids
    ]
    return service


def _list_queries(service):
    return [c.kwargs["q"] for c in service.files.return_value.list.call_args_list]


def _create_bodies(service):
    return [c.kwargs["body"] for c in service.files.return_value.create.call_args_list]


def test_split_folder_path_collapses_redundant_slashes():
    assert _split_folder_path("VideoMME-v2/videos-001") == ["VideoMME-v2", "videos-001"]
    assert _split_folder_path("a//b/") == ["a", "b"]
    assert _split_folder_path("/leading") == ["leading"]
    assert _split_folder_path("solo") == ["solo"]


def test_split_folder_path_rejects_empty():
    for bad in ("", "/", "///"):
        with pytest.raises(ValueError, match="Invalid Drive folder path"):
            _split_folder_path(bad)


def test_split_folder_path_rejects_dot_segments():
    """'.' and '..' mean nothing in Drive; they'd become literal folder names."""
    for bad in ("a/../b", "./a", "a/."):
        with pytest.raises(ValueError, match="is not allowed"):
            _split_folder_path(bad)


def test_escape_query_value_escapes_quotes_and_backslashes():
    assert _escape_query_value("it's") == "it\\'s"
    assert _escape_query_value("back\\slash") == "back\\\\slash"
    # Backslash is escaped before the quote, so an escaped quote is not
    # double-escaped into a literal backslash + unescaped quote.
    assert _escape_query_value("a\\'b") == "a\\\\\\'b"


def test_find_child_folder_query_is_scoped_to_parent():
    """Regression: the lookup used to match a folder name Drive-wide.

    Any same-named folder elsewhere in the user's Drive could be picked up
    (and `files[0]` chosen arbitrarily), so chunks landed in the wrong place.
    """
    service = _mock_service([{"files": [{"id": "child"}]}])
    assert _find_child_folder(service, "videos-001", "parent-id") == "child"
    query = _list_queries(service)[0]
    assert "'parent-id' in parents" in query
    assert "name='videos-001'" in query
    assert "trashed=false" in query


def test_find_child_folder_returns_none_when_absent():
    service = _mock_service([{"files": []}])
    assert _find_child_folder(service, "nope", "root") is None


def test_get_or_create_folder_nested_creates_each_level_under_its_parent():
    """Regression: create() used to omit `parents`, so every folder landed
    at My Drive root — 42 sibling folders instead of a tidy tree."""
    service = _mock_service(
        list_results=[{"files": []}, {"files": []}],
        create_ids=["parent-id", "leaf-id"],
    )
    assert _get_or_create_folder(service, "VideoMME-v2/videos-001") == "leaf-id"

    bodies = _create_bodies(service)
    assert len(bodies) == 2
    assert bodies[0]["name"] == "VideoMME-v2"
    assert bodies[0]["parents"] == ["root"]
    assert bodies[1]["name"] == "videos-001"
    assert bodies[1]["parents"] == ["parent-id"]


def test_get_or_create_folder_nested_reuses_existing_parent():
    """Pushing videos-002 must reuse the VideoMME-v2 folder, not duplicate it."""
    service = _mock_service(
        list_results=[{"files": [{"id": "parent-id"}]}, {"files": []}],
        create_ids=["leaf-id"],
    )
    assert _get_or_create_folder(service, "VideoMME-v2/videos-002") == "leaf-id"

    bodies = _create_bodies(service)
    assert len(bodies) == 1
    assert bodies[0]["name"] == "videos-002"
    assert bodies[0]["parents"] == ["parent-id"]
    # Second lookup is scoped to the parent found by the first.
    assert "'parent-id' in parents" in _list_queries(service)[1]


def test_get_or_create_folder_nested_reuses_whole_path():
    service = _mock_service(
        list_results=[{"files": [{"id": "parent-id"}]}, {"files": [{"id": "leaf-id"}]}]
    )
    assert _get_or_create_folder(service, "VideoMME-v2/videos-001") == "leaf-id"
    service.files.return_value.create.assert_not_called()


def test_get_or_create_folder_single_segment_is_rooted():
    """A plain name keeps working and is anchored at root, not matched globally."""
    service = _mock_service(list_results=[{"files": []}], create_ids=["vlm-id"])
    assert _get_or_create_folder(service, "VLMEvalKit") == "vlm-id"
    assert _create_bodies(service)[0]["parents"] == ["root"]
    assert "'root' in parents" in _list_queries(service)[0]


def test_resolve_folder_path_never_creates():
    """pull must not create folders — a typo should fail, not make an empty dir."""
    service = _mock_service(list_results=[{"files": [{"id": "parent-id"}]}, {"files": []}])
    assert _resolve_folder_path(service, "VideoMME-v2/typo") is None
    service.files.return_value.create.assert_not_called()


def test_resolve_folder_path_returns_leaf_id():
    service = _mock_service(
        list_results=[{"files": [{"id": "parent-id"}]}, {"files": [{"id": "leaf-id"}]}]
    )
    assert _resolve_folder_path(service, "VideoMME-v2/videos-001") == "leaf-id"


def test_resolve_folder_path_stops_at_first_missing_level():
    """A missing parent short-circuits: no lookup for the leaf is attempted."""
    service = _mock_service(list_results=[{"files": []}])
    assert _resolve_folder_path(service, "missing-parent/videos-001") is None
    assert len(_list_queries(service)) == 1


@patch("innout.drive._get_service")
def test_upload_to_drive_accepts_nested_path(mock_get_service):
    """End-to-end: upload_to_drive resolves a nested path and returns the leaf URL."""
    service = _mock_service(
        list_results=[{"files": [{"id": "parent-id"}]}, {"files": []}],
        create_ids=["leaf-id"],
    )
    mock_get_service.return_value = service

    tmp = Path(tempfile.mkdtemp())
    try:
        chunks = _make_chunks(tmp)
        # create() is consumed by the folder first, then once per chunk.
        service.files.return_value.create.return_value.execute.side_effect = (
            [{"id": "leaf-id"}] + [{"id": f"file{i}"} for i in range(len(chunks))]
        )
        url = upload_to_drive(
            chunks, "VideoMME-v2/videos-001", credentials_file="fake.json"
        )
        assert url == "https://drive.google.com/drive/folders/leaf-id"
        # Chunks are parented to the leaf folder, not to root.
        chunk_bodies = _create_bodies(service)[1:]
        assert chunk_bodies, "expected at least one chunk upload"
        assert all(b["parents"] == ["leaf-id"] for b in chunk_bodies)
    finally:
        shutil.rmtree(tmp)


@patch("innout.drive._get_service")
def test_download_from_drive_resolves_nested_path(mock_get_service, tmp_path):
    """pull --drive 'A/B' walks the path, then lists only the leaf's files."""
    service = MagicMock()
    mock_get_service.return_value = service

    parent_lookup = MagicMock()
    parent_lookup.execute.return_value = {"files": [{"id": "parent-id"}]}
    leaf_lookup = MagicMock()
    leaf_lookup.execute.return_value = {"files": [{"id": "leaf-id"}]}
    file_list = MagicMock()
    file_list.execute.return_value = {"files": [{"id": "f1", "name": "sess.part000"}]}
    service.files().list.side_effect = [parent_lookup, leaf_lookup, file_list]

    with patch("googleapiclient.http.MediaIoBaseDownload") as mock_dl_cls:
        def _make_downloader(fh, req):
            fh.write(b"x")
            dl = MagicMock()
            dl.next_chunk.return_value = (None, True)
            return dl

        mock_dl_cls.side_effect = _make_downloader
        files = download_from_drive(
            "VideoMME-v2/videos-001", tmp_path, credentials_file="fake.json"
        )

    assert [f.name for f in files] == ["sess.part000"]
    # The final listing must be parented to the leaf, not the intermediate folder.
    assert "'leaf-id' in parents" in service.files().list.call_args_list[-1].kwargs["q"]


@patch("innout.drive._get_service")
def test_download_from_drive_raises_on_missing_nested_leaf(mock_get_service, tmp_path):
    service = _mock_service(
        list_results=[{"files": [{"id": "parent-id"}]}, {"files": []}]
    )
    mock_get_service.return_value = service
    with pytest.raises(ValueError, match="Drive folder not found"):
        download_from_drive("VideoMME-v2/typo", tmp_path, credentials_file="fake.json")


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


@patch("innout.drive._get_service")
def test_upload_to_drive_retries_on_transient_failure(mock_get_service):
    """A resumable upload must retry transient errors instead of failing outright.

    Regression test: upload_to_drive used to call .execute() with no
    num_retries, so a single transient SSLError/ConnectionError mid-upload
    killed the whole (potentially 100+ MB) transfer with no retry at all.
    """
    service = MagicMock()
    mock_get_service.return_value = service
    service.files().list().execute.return_value = {"files": [{"id": "folder123"}]}
    execute_mock = service.files().create.return_value.execute
    execute_mock.return_value = {"id": "file1"}

    tmp = Path(tempfile.mkdtemp())
    try:
        chunks = _make_chunks(tmp)
        upload_to_drive(chunks, "test-folder", credentials_file="fake.json")
        for call in execute_mock.call_args_list:
            assert call.kwargs.get("num_retries", 0) > 0
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
    """--from-dir with no part files should raise SystemExit.

    Wording note: these were "chunks" in the server era, but the files are
    literally named *.part???, so the message names what the operator sees
    in their folder.
    """
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
        with pytest.raises(SystemExit, match="no part files"):
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
def test_download_from_drive_retries_on_transient_failure(mock_get_service, tmp_path):
    """Regression test: downloader.next_chunk() must pass num_retries too."""
    service = MagicMock()
    mock_get_service.return_value = service

    folder_list = MagicMock()
    folder_list.execute.return_value = {"files": [{"id": "fld", "name": "my-folder"}]}
    file_list = MagicMock()
    file_list.execute.return_value = {"files": [{"id": "f1", "name": "sess.part000"}]}
    service.files().list.side_effect = [folder_list, file_list]

    with patch("googleapiclient.http.MediaIoBaseDownload") as mock_dl_cls:
        dl = MagicMock()
        dl.next_chunk.return_value = (None, True)
        mock_dl_cls.return_value = dl

        download_from_drive("my-folder", tmp_path, credentials_file="fake.json")

        for call in dl.next_chunk.call_args_list:
            assert call.kwargs.get("num_retries", 0) > 0


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


# ---------------------------------------------------------------------------
# Transient-failure classification
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status):
        self.status = status


class _FakeHttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.resp = _FakeResp(status)


def test_is_transient_covers_dropped_connections():
    """`ssl.SSLError: [SYS] unknown error` is what a mobile link produces."""
    assert _is_transient(ssl.SSLError("[SYS] unknown error"))
    assert _is_transient(ConnectionResetError("reset by peer"))
    assert _is_transient(socket.timeout("timed out"))
    assert _is_transient(TimeoutError("timed out"))
    assert _is_transient(http.client.RemoteDisconnected("closed"))


def test_is_transient_covers_rate_limits_and_5xx():
    for status in (408, 429, 500, 502, 503, 504):
        assert _is_transient(_FakeHttpError(status)), status


def test_is_transient_rejects_permanent_failures():
    """Retrying a 404 or a bad passphrase just wastes time."""
    for status in (400, 401, 403, 404):
        assert not _is_transient(_FakeHttpError(status)), status
    assert not _is_transient(ValueError("bad argument"))
    assert not _is_transient(KeyError("missing"))


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------

def test_with_retry_returns_immediately_on_success(monkeypatch):
    slept = []
    monkeypatch.setattr(drive.time, "sleep", slept.append)
    assert _with_retry(lambda: "value", "op") == "value"
    assert slept == []


def test_with_retry_recovers_from_a_transient_failure(monkeypatch):
    """Regression: a connection dropped mid-upload used to kill the whole run.

    googleapiclient's num_retries only covers the request that opens a
    resumable upload, not the PUTs that carry the bytes.
    """
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    def _flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ssl.SSLError("[SYS] unknown error")
        return "uploaded"

    assert _with_retry(_flaky, "uploading chunk") == "uploaded"
    assert attempts["n"] == 3


def test_with_retry_backs_off_with_jitter(monkeypatch):
    """Delays grow and are never identical, so parallel failures don't resync."""
    slept = []
    monkeypatch.setattr(drive.time, "sleep", slept.append)
    monkeypatch.setattr(drive.random, "uniform", lambda a, b: b)

    with pytest.raises(ssl.SSLError):
        _with_retry(lambda: (_ for _ in ()).throw(ssl.SSLError("boom")), "op")

    assert len(slept) == drive._MAX_ATTEMPTS - 1
    assert slept == sorted(slept), "delays must be non-decreasing"
    assert all(d <= drive._BACKOFF_CAP * 1.5 for d in slept)


def test_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    def _always_fails():
        attempts["n"] += 1
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        _with_retry(_always_fails, "op")
    assert attempts["n"] == drive._MAX_ATTEMPTS


def test_with_retry_does_not_retry_a_permanent_failure(monkeypatch):
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    attempts = {"n": 0}

    def _bad_request():
        attempts["n"] += 1
        raise _FakeHttpError(404)

    with pytest.raises(_FakeHttpError):
        _with_retry(_bad_request, "op")
    assert attempts["n"] == 1, "a 404 must fail on the first try"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def test_list_folder_files_follows_every_page():
    """Regression: Drive caps a listing at 100 files and says nothing.

    An unpaginated list silently returned a prefix of the chunks, which
    would join into a truncated, undecryptable blob.
    """
    page1 = {"files": [{"id": f"f{i}", "name": f"s.part{i:03d}"} for i in range(100)],
             "nextPageToken": "tok1"}
    page2 = {"files": [{"id": f"g{i}", "name": f"s.part{i + 100:03d}"} for i in range(100)],
             "nextPageToken": "tok2"}
    page3 = {"files": [{"id": "h0", "name": "s.part200"}]}

    service = _mock_service([page1, page2, page3])
    items = _list_folder_files(service, "folder-id")

    assert len(items) == 201
    tokens = [c.kwargs.get("pageToken")
              for c in service.files.return_value.list.call_args_list]
    assert tokens == [None, "tok1", "tok2"]


def test_list_folder_files_requests_a_large_page_size():
    service = _mock_service([{"files": []}])
    _list_folder_files(service, "folder-id")
    call = service.files.return_value.list.call_args
    assert call.kwargs["pageSize"] == drive._PAGE_SIZE
    assert "nextPageToken" in call.kwargs["fields"]


# ---------------------------------------------------------------------------
# Session filtering — many pushes in one folder
# ---------------------------------------------------------------------------

_SID_A = "aaaaaaaa-0000-0000-0000-000000000000"
_SID_B = "bbbbbbbb-0000-0000-0000-000000000000"


def _two_level_resolution():
    """List results for resolving "A/B": one lookup per path segment."""
    return [{"files": [{"id": "parent-id"}]}, {"files": [{"id": "leaf-id"}]}]


def _shared_folder_listing():
    return {"files": [
        {"id": "a0", "name": f"{_SID_A}.part000"},
        {"id": "a1", "name": f"{_SID_A}.part001"},
        {"id": "b0", "name": f"{_SID_B}.part000"},
    ]}


def _patch_downloader(monkeypatch):
    def _make(fh, req):
        fh.write(b"x")
        dl = MagicMock()
        dl.next_chunk.return_value = (None, True)
        return dl
    monkeypatch.setattr("googleapiclient.http.MediaIoBaseDownload", _make)


@patch("innout.drive._get_service")
def test_download_from_drive_takes_only_the_requested_session(
    mock_get_service, tmp_path, monkeypatch
):
    """40 pushes share Video-MME-v2/videos; only one session may come back."""
    service = _mock_service([*_two_level_resolution(), _shared_folder_listing()])
    mock_get_service.return_value = service
    _patch_downloader(monkeypatch)

    files = download_from_drive(
        "Video-MME-v2/videos", tmp_path, credentials_file="fake.json",
        session_id=_SID_A,
    )
    assert [f.name for f in files] == [f"{_SID_A}.part000", f"{_SID_A}.part001"]


@patch("innout.drive._get_service")
def test_download_from_drive_raises_when_the_session_is_absent(
    mock_get_service, tmp_path
):
    service = _mock_service([*_two_level_resolution(), _shared_folder_listing()])
    mock_get_service.return_value = service
    with pytest.raises(ValueError, match="No chunks for session"):
        download_from_drive(
            "Video-MME-v2/videos", tmp_path, credentials_file="fake.json",
            session_id="cccccccc-0000-0000-0000-000000000000",
        )


@patch("innout.drive._get_service")
def test_download_from_drive_refuses_an_ambiguous_folder(mock_get_service, tmp_path):
    """Joining two pushes would decrypt to garbage, so say so instead."""
    service = _mock_service([*_two_level_resolution(), _shared_folder_listing()])
    mock_get_service.return_value = service
    with pytest.raises(ValueError, match="holds 2 pushes"):
        download_from_drive(
            "Video-MME-v2/videos", tmp_path, credentials_file="fake.json"
        )


@patch("innout.drive._get_service")
def test_download_from_drive_single_session_needs_no_id(
    mock_get_service, tmp_path, monkeypatch
):
    """The old one-push-per-folder usage keeps working unchanged."""
    listing = {"files": [{"id": "a0", "name": f"{_SID_A}.part000"}]}
    service = _mock_service([{"files": [{"id": "fld"}]}, listing])
    mock_get_service.return_value = service
    _patch_downloader(monkeypatch)

    files = download_from_drive("solo", tmp_path, credentials_file="fake.json")
    assert [f.name for f in files] == [f"{_SID_A}.part000"]


# ---------------------------------------------------------------------------
# list_sessions / delete_session_files
# ---------------------------------------------------------------------------

@patch("innout.drive._get_service")
def test_list_sessions_reports_each_push_once(mock_get_service):
    service = _mock_service([*_two_level_resolution(), _shared_folder_listing()])
    mock_get_service.return_value = service
    assert list_sessions("Video-MME-v2/videos", "fake.json") == [_SID_A, _SID_B]


@patch("innout.drive._get_service")
def test_list_sessions_raises_for_a_missing_folder(mock_get_service):
    service = _mock_service([{"files": []}])
    mock_get_service.return_value = service
    with pytest.raises(ValueError, match="Drive folder not found"):
        list_sessions("nope", "fake.json")


@patch("innout.drive._get_service")
def test_delete_session_files_removes_only_that_session(mock_get_service):
    """A failed push leaves orphans; the resume uses a new id and can't reuse them."""
    service = _mock_service([*_two_level_resolution(), _shared_folder_listing()])
    mock_get_service.return_value = service

    removed = delete_session_files("Video-MME-v2/videos", _SID_A, "fake.json")

    assert removed == 2
    deleted_ids = [c.kwargs["fileId"]
                   for c in service.files.return_value.delete.call_args_list]
    assert sorted(deleted_ids) == ["a0", "a1"], "the other push must be untouched"


@patch("innout.drive._get_service")
def test_delete_session_files_is_a_noop_for_a_missing_folder(mock_get_service):
    service = _mock_service([{"files": []}])
    mock_get_service.return_value = service
    assert delete_session_files("nope", _SID_A, "fake.json") == 0
    service.files.return_value.delete.assert_not_called()


@patch("innout.drive._get_service")
def test_upload_to_drive_retries_a_dropped_chunk(mock_get_service, monkeypatch):
    """The whole point: a mid-upload drop costs one chunk, not the transfer."""
    monkeypatch.setattr(drive.time, "sleep", lambda _: None)
    service = MagicMock()
    mock_get_service.return_value = service
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "folder123"}]
    }

    attempts = {"n": 0}

    def _execute(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ssl.SSLError("[SYS] unknown error")
        return {"id": "file1"}

    service.files.return_value.create.return_value.execute.side_effect = _execute

    tmp = Path(tempfile.mkdtemp())
    try:
        chunks = _make_chunks(tmp, b"z" * 60)
        url = upload_to_drive(chunks, "P/videos", credentials_file="fake.json")
        assert url == "https://drive.google.com/drive/folders/folder123"
        assert attempts["n"] == len(chunks) + 1, "exactly one retry happened"
    finally:
        shutil.rmtree(tmp)
