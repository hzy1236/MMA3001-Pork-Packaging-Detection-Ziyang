"""Synthetic software tests; these are NOT evidence of ML detection accuracy."""
from pathlib import Path
import pytest
import yaml
from mma3001_pork.dataset import read_class_names, make_config, count_annotations


def demo_dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": ["bad_seal", "fold"]}), encoding="utf-8")
    for split in ("train", "valid", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    return tmp_path


def test_reads_class_names(tmp_path):
    root = demo_dataset(tmp_path)
    assert read_class_names(root) == ["bad_seal", "fold"]


def test_config_uses_absolute_dataset_path(tmp_path):
    root = demo_dataset(tmp_path / "input")
    path = make_config(root, tmp_path / "config.yaml")
    conf = yaml.safe_load(path.read_text())
    assert conf["path"] == str(root.resolve())
    assert conf["val"] == "valid/images"


def test_counts_all_classes_including_zero(tmp_path):
    root = demo_dataset(tmp_path)
    (root / "train/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    result = count_annotations(root, 2)
    assert result["train"] == {0: 2, 1: 0}
    assert result["test"] == {0: 0, 1: 0}


def test_rejects_out_of_range_annotation(tmp_path):
    root = demo_dataset(tmp_path)
    (root / "test/labels/b.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        count_annotations(root, 2)


def test_rejects_missing_split(tmp_path):
    root = demo_dataset(tmp_path)
    (root / "test/images").rmdir()
    with pytest.raises(FileNotFoundError):
        make_config(root, tmp_path / "out.yaml")
