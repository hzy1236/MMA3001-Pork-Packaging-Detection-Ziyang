"""Prepare YOLO dataset paths and audit label-class imbalance.

Only local, student-authorised datasets are accepted. This module does not
acquire or relicense third-party images, and does not assign new ground truth.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import yaml

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def read_class_names(dataset_root: Path) -> list[str]:
    """Return YOLO class names from a dataset's ``data.yaml``.

    Args:
        dataset_root: Directory containing the original ``data.yaml``.

    Returns:
        Ordered list of label names for class IDs 0..N-1.

    Raises:
        FileNotFoundError: The expected YAML does not exist.
        ValueError: Class names are missing or class IDs are not consecutive.
    """
    source = Path(dataset_root) / "data.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        mapping = {int(i): str(name) for i, name in names.items()}
        if sorted(mapping) != list(range(len(mapping))):
            raise ValueError("Class IDs must be contiguous from zero")
        return [mapping[i] for i in range(len(mapping))]
    if isinstance(names, list) and names:
        return [str(item) for item in names]
    raise ValueError("Missing or invalid 'names' in dataset data.yaml")


def make_config(dataset_root: Path, output: Path) -> Path:
    """Create a YOLO YAML whose image-split paths are unambiguous and absolute.

    Args:
        dataset_root: Extracted Roboflow dataset root.
        output: Path for generated YAML configuration.

    Returns:
        The created config file's resolved path.
    """
    root = Path(dataset_root).expanduser().resolve()
    names = read_class_names(root)
    for location in SPLITS.values():
        for section in ("images", "labels"):
            if not (root / location / section).is_dir():
                raise FileNotFoundError(root / location / section)
    result = {"path": str(root), "train": "train/images", "val": "valid/images", "test": "test/images", "names": names}
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return output


def count_annotations(dataset_root: Path, n_classes: int) -> dict[str, dict[int, int]]:
    """Count bounding-box instances per class, including absent classes as zero.

    Args:
        dataset_root: Extracted Roboflow dataset root.
        n_classes: Number of declared class IDs.

    Returns:
        Counts indexed by split and integer class ID.

    Raises:
        ValueError: Malformed class ID or bounding-box annotation line.
    """
    root = Path(dataset_root)
    result = {}
    for split, folder in SPLITS.items():
        counts: Counter[int] = Counter()
        for file in sorted((root / folder / "labels").glob("*.txt")):
            for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Expected 5 YOLO fields: {file}:{line_no}")
                try:
                    label_id = int(fields[0])
                    coords = [float(x) for x in fields[1:]]
                except ValueError as exc:
                    raise ValueError(f"Invalid YOLO annotation: {file}:{line_no}") from exc
                if not (0 <= label_id < n_classes) or any(not 0 <= c <= 1 for c in coords):
                    raise ValueError(f"Class or box coordinate out of range: {file}:{line_no}")
                counts[label_id] += 1
        result[split] = {i: counts[i] for i in range(n_classes)}
    return result


def main() -> None:
    """Generate absolute-path config and report class imbalance to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = make_config(args.dataset, args.output)
    names = read_class_names(args.dataset)
    counts = count_annotations(args.dataset, len(names))
    print("Training config:", config)
    print(json.dumps({split: {names[i]: n for i, n in values.items()} for split, values in counts.items()}, indent=2))


if __name__ == "__main__":
    main()
