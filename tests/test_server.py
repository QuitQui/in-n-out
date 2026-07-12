import io
import pytest
from innout.server import create_app

_API_KEY = "test-key"
_AUTH = {"Authorization": f"Bearer {_API_KEY}"}


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "store", api_key=_API_KEY)


@pytest.fixture
def client(app):
    app.config["TESTING"] = True
    return app.test_client()


def _upload(client, session_id: str, part: str, data: bytes, total_parts: str = "1"):
    return client.post(
        "/upload",
        data={
            "session_id": session_id,
            "part": part,
            "total_parts": total_parts,
            "file": (io.BytesIO(data), "chunk.bin"),
        },
        content_type="multipart/form-data",
        headers=_AUTH,
    )


def test_upload_and_manifest(client):
    sid = "11111111-1111-1111-1111-111111111111"
    resp = _upload(client, sid, "000", b"chunk-zero")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["part"] == "000"

    manifest = client.get(f"/manifest/{sid}", headers=_AUTH).get_json()
    assert manifest["session_id"] == sid
    assert "000" in manifest["parts"]


def test_download_roundtrip(client):
    sid = "22222222-2222-2222-2222-222222222222"
    payload = b"encrypted payload data"
    _upload(client, sid, "000", payload)

    resp = client.get(f"/download/{sid}/000", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.data == payload


def test_manifest_unknown_session(client):
    sid = "33333333-3333-3333-3333-333333333333"
    resp = client.get(f"/manifest/{sid}", headers=_AUTH)
    assert resp.status_code == 404


def test_download_unknown_part(client):
    sid = "44444444-4444-4444-4444-444444444444"
    _upload(client, sid, "000", b"data")
    resp = client.get(f"/download/{sid}/999", headers=_AUTH)
    assert resp.status_code == 404


def test_upload_missing_fields(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data", headers=_AUTH)
    assert resp.status_code == 400


def test_upload_invalid_session_id(client):
    resp = _upload(client, "not-a-uuid", "000", b"data")
    assert resp.status_code == 400


def test_upload_invalid_part(client):
    sid = "44444444-4444-4444-4444-444444444444"
    resp = _upload(client, sid, "99", b"data")
    assert resp.status_code == 400


def test_upload_part_out_of_range(client):
    sid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    resp = _upload(client, sid, "001", b"data", total_parts="1")
    assert resp.status_code == 400


def test_multiple_parts_manifest_sorted(client):
    sid = "55555555-5555-5555-5555-555555555555"
    for part in ["002", "000", "001"]:
        _upload(client, sid, part, b"data", total_parts="3")

    manifest = client.get(f"/manifest/{sid}", headers=_AUTH).get_json()
    assert manifest["parts"] == ["000", "001", "002"]


def test_no_auth_returns_401(client):
    sid = "66666666-6666-6666-6666-666666666666"
    resp = client.get(f"/manifest/{sid}")
    assert resp.status_code == 401


def test_wrong_auth_returns_403(client):
    sid = "77777777-7777-7777-7777-777777777777"
    resp = client.get(f"/manifest/{sid}", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 403


def test_invalid_session_id_returns_400(client):
    resp = client.get("/manifest/not-a-uuid", headers=_AUTH)
    assert resp.status_code == 400


def test_invalid_part_returns_400(client):
    sid = "88888888-8888-8888-8888-888888888888"
    resp = client.get(f"/download/{sid}/99", headers=_AUTH)
    assert resp.status_code == 400
