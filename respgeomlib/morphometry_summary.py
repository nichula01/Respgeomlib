import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .tree_builder import load_specs_from_yaml, SegmentSpec


def compute_generations(specs: List[SegmentSpec]) -> Dict[str, int]:
    """
    Determine generation index for each segment.

    Priority: use spec.meta["generation"] if present; otherwise, infer by
    walking up parents (root=0, child=parent+1).
    """
    spec_by_id: Dict[str, SegmentSpec] = {s.id: s for s in specs}
    gen_cache: Dict[str, int] = {}

    def infer(seg_id: str) -> int:
        if seg_id in gen_cache:
            return gen_cache[seg_id]
        spec = spec_by_id.get(seg_id)
        if spec is None:
            raise RuntimeError(f"Missing spec for id {seg_id!r}")
        meta_gen = None
        if spec.meta and "generation" in spec.meta:
            try:
                meta_gen = int(spec.meta["generation"])
            except Exception as e:
                raise RuntimeError(f"Invalid generation in meta for {seg_id!r}: {e}")
        if meta_gen is not None:
            gen_cache[seg_id] = meta_gen
            return meta_gen
        if spec.parent_id is None:
            gen_cache[seg_id] = 0
            return 0
        parent_gen = infer(spec.parent_id)
        gen = parent_gen + 1
        gen_cache[seg_id] = gen
        return gen

    for s in specs:
        infer(s.id)
    return gen_cache


def segment_diameter_and_length(spec: SegmentSpec) -> Tuple[Optional[float], Optional[float]]:
    """
    Approximate a representative diameter and length for a segment.

    Returns (diameter, length) or (None, None) if not applicable.
    """
    kind = spec.kind.lower()
    p = spec.params

    if kind in {"pipe", "pipe_stenosis", "pipe_dilation"}:
        try:
            d_in = float(p["d_in"])
            d_out = float(p["d_out"])
            length = float(p["length"])
        except KeyError:
            return None, None
        diameter = 0.5 * (d_in + d_out)
        return diameter, length

    if kind == "y2":
        try:
            d_trunk = float(p["d_trunk"])
            length_trunk = float(p["length_trunk"])
        except KeyError:
            return None, None
        return d_trunk, length_trunk

    if kind == "y3":
        try:
            d_trunk = float(p["d_trunk"])
            length_trunk = float(p["length_trunk"])
        except KeyError:
            return None, None
        return d_trunk, length_trunk

    return None, None


def main() -> None:
    yaml_path = "trees/human_central_v2_weibel_with_right_subtree.yaml"
    specs = load_specs_from_yaml(yaml_path)
    if not specs:
        raise RuntimeError(f"No specs loaded from {yaml_path}")

    gen_map = compute_generations(specs)

    by_gen = defaultdict(list)
    for spec in specs:
        gen = gen_map.get(spec.id)
        d, L = segment_diameter_and_length(spec)
        if d is None or L is None:
            continue
        by_gen[gen].append((d, L))

    print(f"Morphometry summary for: {yaml_path}")
    for gen in sorted(by_gen.keys()):
        entries = by_gen[gen]
        if not entries:
            continue
        diameters = np.array([e[0] for e in entries], dtype=float)
        lengths = np.array([e[1] for e in entries], dtype=float)
        print(f"Generation {gen}:")
        print(f"  count = {len(entries)}")
        print(
            f"  diameter: mean={diameters.mean():.2f} cm, "
            f"min={diameters.min():.2f}, max={diameters.max():.2f}"
        )
        print(
            f"  length  : mean={lengths.mean():.2f} cm, "
            f"min={lengths.min():.2f}, max={lengths.max():.2f}"
        )
        print()


if __name__ == "__main__":
    main()
