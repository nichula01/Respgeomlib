import yaml
from pathlib import Path
from typing import List, Any


REQUIRED_KEYS = {"id", "kind", "params", "parent_id", "parent_port_index"}


def load_yaml_list(path: Path) -> List[Any]:
    if not path.exists():
        raise RuntimeError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a top-level list in {path}, got {type(data).__name__}")
    return data


def _validate_entries(entries: List[Any], source: Path) -> None:
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise RuntimeError(f"Entry {idx} in {source} is not a mapping/dict")
        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            raise RuntimeError(f"Entry {idx} in {source} is missing required keys: {sorted(missing)}")


def merge_trees(
    base_path: Path,
    subtree_path: Path,
    out_path: Path,
) -> None:
    base_entries = load_yaml_list(base_path)
    subtree_entries = load_yaml_list(subtree_path)

    _validate_entries(base_entries, base_path)
    _validate_entries(subtree_entries, subtree_path)

    merged = base_entries + subtree_entries

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    print(f"Base entries:    {len(base_entries)} from {base_path}")
    print(f"Subtree entries: {len(subtree_entries)} from {subtree_path}")
    print(f"Merged entries:  {len(merged)} written to {out_path}")


def main():
    base_path = Path("trees") / "human_central_v2_weibel.yaml"
    subtree_path = Path("trees") / "right_lung_subtree_auto.yaml"
    out_path = Path("trees") / "human_central_v2_weibel_with_right_subtree.yaml"
    merge_trees(base_path, subtree_path, out_path)


if __name__ == "__main__":
    main()
