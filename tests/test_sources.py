import tarfile
from pathlib import Path

import pytest

from innout import sources
from innout.cli import build_parser


def _make_tree(root: Path) -> Path:
    """A small project tree with code files and data files mixed in."""
    src = root / "proj"
    (src / "sub").mkdir(parents=True)
    (src / "keep.py").write_text("print('hi')")
    (src / "model.pt").write_text("weights")
    (src / "sub" / "results.json").write_text("{}")
    (src / "sub" / "keep.md").write_text("# doc")
    (src / "data").mkdir()
    (src / "data" / "sample.npz").write_text("blob")
    return src


def _tar_names(archive: Path) -> set[str]:
    with tarfile.open(archive) as tar:
        return set(tar.getnames())


def test_local_dir_no_excludes_keeps_everything(tmp_path):
    src = _make_tree(tmp_path)
    out = sources.acquire("local", str(src), tmp_path / "work")
    names = _tar_names(out)
    assert "proj/keep.py" in names
    assert "proj/model.pt" in names
    assert "proj/sub/results.json" in names


def test_local_dir_excludes_glob_patterns(tmp_path):
    src = _make_tree(tmp_path)
    out = sources.acquire(
        "local", str(src), tmp_path / "work", excludes=["*.pt", "*.json"]
    )
    names = _tar_names(out)
    assert "proj/keep.py" in names
    assert "proj/sub/keep.md" in names
    assert not any(n.endswith(".pt") for n in names)
    assert not any(n.endswith(".json") for n in names)


def test_local_dir_excludes_whole_directory(tmp_path):
    src = _make_tree(tmp_path)
    out = sources.acquire(
        "local", str(src), tmp_path / "work", excludes=["data"]
    )
    names = _tar_names(out)
    assert "proj/keep.py" in names
    assert not any("data" in Path(n).parts for n in names)


def test_local_single_file_ignores_excludes(tmp_path):
    f = tmp_path / "single.bin"
    f.write_text("x")
    out = sources.acquire("local", str(f), tmp_path / "work", excludes=["*.pt"])
    assert out == f


def test_push_parser_accepts_repeated_exclude():
    parser = build_parser()
    args = parser.parse_args(
        ["push", "--local", "x", "--drive", "f",
         "--exclude", "*.pt", "--exclude", "*.json"]
    )
    assert args.exclude == ["*.pt", "*.json"]


def test_push_parser_exclude_defaults_empty():
    parser = build_parser()
    args = parser.parse_args(["push", "--local", "x", "--drive", "f"])
    assert args.exclude == []


def _fake_snapshot_download(recorder: dict):
    """Stand-in for huggingface_hub.snapshot_download that records kwargs."""

    def _fake(**kwargs):
        recorder.update(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "README.md").write_text("# snapshot")
        return str(local_dir)

    return _fake


def test_acquire_hf_forwards_dataset_repo_type(tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_snapshot_download(calls)
    )

    out = sources.acquire(
        "hf",
        "MME-Benchmarks/Video-MME-v2",
        tmp_path / "work",
        repo_type="dataset",
    )

    assert calls["repo_id"] == "MME-Benchmarks/Video-MME-v2"
    assert calls["repo_type"] == "dataset"
    assert out.name == "Video-MME-v2.tar.gz"
    assert "Video-MME-v2/README.md" in _tar_names(out)


def test_acquire_hf_defaults_to_model_repo_type(tmp_path, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", _fake_snapshot_download(calls)
    )

    sources.acquire("hf", "org/some-model", tmp_path / "work")

    assert calls["repo_type"] == "model"


def test_acquire_hf_applies_excludes(tmp_path, monkeypatch):
    def _fake(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "keep.json").write_text("{}")
        (local_dir / "huge.mp4").write_text("video")
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake)

    out = sources.acquire(
        "hf",
        "MME-Benchmarks/Video-MME-v2",
        tmp_path / "work",
        excludes=["*.mp4"],
        repo_type="dataset",
    )

    names = _tar_names(out)
    assert "Video-MME-v2/keep.json" in names
    assert not any(n.endswith(".mp4") for n in names)


def test_push_parser_accepts_dataset_repo_type():
    parser = build_parser()
    args = parser.parse_args(
        ["push", "--hf", "MME-Benchmarks/Video-MME-v2",
         "--repo-type", "dataset", "--drive", "f"]
    )
    assert args.repo_type == "dataset"


def test_push_parser_repo_type_defaults_to_model():
    parser = build_parser()
    args = parser.parse_args(["push", "--hf", "org/m", "--drive", "f"])
    assert args.repo_type == "model"


def test_push_parser_rejects_unknown_repo_type():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["push", "--hf", "org/m", "--repo-type", "datasets", "--drive", "f"]
        )
