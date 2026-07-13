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
