"""Check experimental settings without needing GPU or training."""
from pathlib import Path
import pytest
from mma3001_pork.experiment import training_arguments
from mma3001_pork.evaluation import extract_metrics


def test_only_resolution_and_name_change():
    a = training_arguments("baseline", Path("data.yaml"), Path("runs"))
    b = training_arguments("highres", Path("data.yaml"), Path("runs"))
    assert a["imgsz"] == 416 and b["imgsz"] == 640
    assert {k: v for k, v in a.items() if k not in ("imgsz", "name")} == {k: v for k, v in b.items() if k not in ("imgsz", "name")}


def test_invalid_variant():
    with pytest.raises(ValueError):
        training_arguments("wrong", Path("data.yaml"), Path("runs"))


def test_serialise_metrics():
    class Box:
        mp = .5063
        mr = .4136
        map50 = .3553
        map = .1969
    assert extract_metrics(Box())["mAP50_95"] == pytest.approx(.1969)
