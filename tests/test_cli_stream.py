"""End-to-end tests for the streaming push/pull through the real CLI.

These stand in for the Video-MME-v2 run: a tree shaped like the dataset
goes up as encrypted parts and has to come back as the same tree.

tests/test_stream.py covers innout/stream.py's functions in isolation;
this file drives the actual argparse CLI against a Drive stand-in, which
is the only place the push and pull halves are proven to agree.
"""

import os
import tarfile
from pathlib import Path

import pytest

from innout import cli, stream


def make_dataset(root: Path) -> dict[str, bytes]:
    """A miniature Video-MME-v2, including the junk snapshot_download leaves."""
    (root / "videos").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / ".cache" / "huggingface").mkdir(parents=True)
    files = {
        "videos/001.zip": b"PK\x03\x04" + os.urandom(9000),
        "videos/002.zip": b"PK\x03\x04" + os.urandom(8000),
        "videos/003.zip": b"PK\x03\x04" + os.urandom(7000),
        "assets/demo.mp4": os.urandom(4000),
        "assets/logo.png": os.urandom(900),
        "subtitle.zip": b"PK\x03\x04" + os.urandom(1500),
        "test.parquet": os.urandom(1200),
        "eval.yaml": b"dataset: Video-MME-v2\n",
        "README.md": b"# Video-MME-v2\n",
        ".gitattributes": b"*.zip filter=lfs\n",
    }
    for name, blob in files.items():
        (root / name).write_bytes(blob)
    (root / ".cache" / "huggingface" / ".metadata").write_bytes(b"junk")
    return files


class FakeDrive:
    """A Drive folder backed by a local directory.

    Records every upload so a test can assert on resume and replacement
    without touching the network.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads: list[str] = []
        self.fail_on: set[str] = set()
        self.next_id = 0

    def install(self, monkeypatch, module):
        monkeypatch.setattr(module, "open_folder", self._open, raising=False)
        monkeypatch.setattr(module, "folder_contents", self._contents, raising=False)
        monkeypatch.setattr(module, "upload_one", self._upload, raising=False)
        monkeypatch.setattr(
            module, "folder_url", lambda fid: f"fake://{fid}", raising=False
        )
        monkeypatch.setattr(module, "delete_file", lambda s, i: None, raising=False)

    def _open(self, folder_name, credentials_file=None):
        return object(), folder_name

    def _contents(self, service, folder_id):
        return {
            p.name: {"id": p.name, "size": p.stat().st_size}
            for p in sorted(self.root.iterdir()) if p.is_file()
        }

    def _upload(self, service, folder_id, path, name=None, replace_id=None):
        name = name or Path(path).name
        if name in self.fail_on:
            self.fail_on.discard(name)
            raise OSError(f"simulated upload failure on {name}")
        (self.root / name).write_bytes(Path(path).read_bytes())
        self.uploads.append(name)
        self.next_id += 1
        return str(self.next_id)


@pytest.fixture
def fake_drive(tmp_path, monkeypatch):
    from innout import drive

    fake = FakeDrive(tmp_path / "drive")
    fake.install(monkeypatch, drive)
    return fake


def run(argv):
    args = cli.build_parser().parse_args(argv)
    args.func(args)


def push_argv(src, part_size="8KiB", work=None, extra=()):
    return [
        "push", "--local", str(src), "--drive", "VideoMMEv2",
        "--part-size", part_size, "--part-prefix", "Video-MME-v2",
        "--passphrase", "pw", "--exclude", ".cache",
        "--work-dir", str(work), *extra,
    ]


def download_all(fake_drive, dest: Path) -> Path:
    """What the operator does by hand: pull every file into one folder."""
    dest.mkdir(parents=True, exist_ok=True)
    for path in fake_drive.root.iterdir():
        (dest / path.name).write_bytes(path.read_bytes())
    return dest


def test_push_then_pull_restores_the_dataset_exactly(tmp_path, fake_drive, capsys):
    """The whole point: parts on Drive, download them, get the tree back.

    Mirrors the operator's plan -- a flat set of <prefix>.partNNN in one
    folder, downloaded by hand into one directory, restored with one
    command.
    """
    src = tmp_path / "Video-MME-v2"
    files = make_dataset(src)

    run(push_argv(src, work=tmp_path / "work"))

    names = sorted(p.name for p in fake_drive.root.iterdir())
    parts = [n for n in names if ".part" in n]
    assert set(names) == set(parts) | {"MANIFEST.sha256"}, (
        f"Drive should hold only parts plus the manifest, got {names}"
    )
    assert len(parts) > 3, f"need several parts to be meaningful: {parts}"
    assert parts[0] == "Video-MME-v2.part000"
    assert parts == sorted(parts), "part names must sort into join order"

    downloaded = download_all(fake_drive, tmp_path / "downloaded")
    run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
         "--output", str(tmp_path / "out"), "--extract"])

    base = tmp_path / "out" / "Video-MME-v2"
    for name, blob in files.items():
        assert (base / name).read_bytes() == blob, name
    assert not (base / ".cache").exists(), "HF download cache leaked into the push"
    assert "All parts match the manifest." in capsys.readouterr().out


def test_pull_writes_a_named_tar_when_not_extracting(tmp_path, fake_drive):
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    run(push_argv(src, work=tmp_path / "work"))
    downloaded = download_all(fake_drive, tmp_path / "downloaded")

    run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
         "--output", str(tmp_path / "out")])

    tar_path = tmp_path / "out" / "Video-MME-v2.tar"
    assert tar_path.exists()
    with tarfile.open(tar_path, "r:") as tar:
        names = tar.getnames()
    assert "Video-MME-v2/videos/001.zip" in names
    assert not any(".cache" in n for n in names)
    assert not (tmp_path / "out" / ".innout-incoming").exists()


def test_pull_refuses_a_corrupted_part_before_decrypting(tmp_path, fake_drive):
    """A 55-part manual download will truncate one sooner or later; the
    manifest has to catch it before 105 GB of output is written."""
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    run(push_argv(src, work=tmp_path / "work"))
    downloaded = download_all(fake_drive, tmp_path / "downloaded")

    victim = downloaded / "Video-MME-v2.part001"
    victim.write_bytes(victim.read_bytes()[:-10] + b"CORRUPTED!")

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="sha256 mismatch on Video-MME-v2.part001"):
        run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
             "--output", str(out_dir), "--extract"])
    assert not (out_dir / "Video-MME-v2").exists()


def test_pull_names_a_missing_part(tmp_path, fake_drive):
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    run(push_argv(src, work=tmp_path / "work"))

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    for path in fake_drive.root.iterdir():
        if path.name != "Video-MME-v2.part001":
            (downloaded / path.name).write_bytes(path.read_bytes())

    with pytest.raises(SystemExit, match=r"part\(s\) \[1\] are missing"):
        run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
             "--output", str(tmp_path / "out")])


def test_delete_consumed_leaves_no_parts_behind(tmp_path, fake_drive):
    src = tmp_path / "Video-MME-v2"
    files = make_dataset(src)
    run(push_argv(src, work=tmp_path / "work"))
    downloaded = download_all(fake_drive, tmp_path / "downloaded")

    run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
         "--output", str(tmp_path / "out"), "--extract", "--delete-consumed"])

    assert list(downloaded.glob("*.part???")) == []
    assert (downloaded / "MANIFEST.sha256").exists(), "manifest must survive"
    base = tmp_path / "out" / "Video-MME-v2"
    for name, blob in files.items():
        assert (base / name).read_bytes() == blob


def test_interrupted_push_resumes_without_re_uploading(tmp_path, fake_drive):
    """The real run is ~15 hours over a home link; it will be interrupted.

    A resumed push must re-send only what is missing, and the result must
    still join into the same archive -- every part was cut from one
    continuous GCM stream.
    """
    src = tmp_path / "Video-MME-v2"
    files = make_dataset(src)
    work = tmp_path / "work"

    fake_drive.fail_on = {"Video-MME-v2.part002"}
    with pytest.raises(OSError, match="simulated upload failure"):
        run(push_argv(src, work=work))

    first_round = list(fake_drive.uploads)
    assert "Video-MME-v2.part000" in first_round
    assert "Video-MME-v2.part002" not in first_round

    fake_drive.uploads.clear()
    run(push_argv(src, work=work))

    assert "Video-MME-v2.part000" not in fake_drive.uploads, "re-sent a done part"
    assert "Video-MME-v2.part002" in fake_drive.uploads

    downloaded = download_all(fake_drive, tmp_path / "downloaded")
    run(["pull", "--from-dir", str(downloaded), "--passphrase", "pw",
         "--output", str(tmp_path / "out"), "--extract"])

    base = tmp_path / "out" / "Video-MME-v2"
    for name, blob in files.items():
        assert (base / name).read_bytes() == blob, name


def test_changing_part_size_starts_a_fresh_stream(tmp_path, fake_drive, capsys):
    """Resuming with a different part size would cut the stream at new
    offsets, so the recorded parts cannot be reused."""
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    work = tmp_path / "work"

    run(push_argv(src, part_size="8KiB", work=work))
    capsys.readouterr()
    run(push_argv(src, part_size="16KiB", work=work))
    assert "starting a fresh stream" in capsys.readouterr().out


def test_streaming_push_requires_a_destination(tmp_path):
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    with pytest.raises(SystemExit, match="one of --server or --drive"):
        run(["push", "--local", str(src), "--part-size", "8KiB",
             "--passphrase", "pw"])


def test_streaming_push_refuses_a_file_source(tmp_path, fake_drive):
    """The pipeline tars a directory on the fly; a single file has no tree."""
    blob = tmp_path / "one.bin"
    blob.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="is not one"):
        run(push_argv(blob, work=tmp_path / "work"))


def test_manifest_lists_every_part_with_its_digest(tmp_path, fake_drive):
    src = tmp_path / "Video-MME-v2"
    make_dataset(src)
    run(push_argv(src, work=tmp_path / "work"))

    manifest = stream.read_manifest(fake_drive.root / "MANIFEST.sha256")
    parts = sorted(p.name for p in fake_drive.root.glob("*.part???"))
    assert sorted(manifest) == parts
    for name, digest in manifest.items():
        assert stream.sha256_file(fake_drive.root / name) == digest
