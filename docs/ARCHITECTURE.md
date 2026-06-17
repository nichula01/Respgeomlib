# RespGeomLib architecture

This note maps the paper's method onto the code so contributors can navigate
the engine quickly. (Drop the paper PDF into this `docs/` folder for reference.)

## Pipeline overview

```
YAML spec ─▶ load_specs_from_yaml ─▶ build_tree ─▶ merge_built_segments ─▶ mesh
            (tree_builder.py)        (port-based     (concatenate         (STL/PLY/
                                      assembly)        segment meshes)      VTP)
```

1. **Specification** — a YAML list of segments, each with a primitive `kind`,
   `params`, and a `(parent_id, parent_port_index)` attachment.
   → `respgeomlib/tree_builder.py` (`SegmentSpec`, `load_specs_from_yaml`).
2. **Local coordinate transform** — each primitive is built in a canonical
   local frame and mapped to world coordinates with a rigid transform
   `x_world = R · x_local + o`. Child axes come from elevation/azimuth angles via
   a minimal-twist frame.
   → `respgeomlib/frames.py` (`direction_from_angles`, `make_child_frame`, `Frame`).
3. **Primitive / junction geometry**
   - Analytic pipes & tapers → `respgeomlib/primitives.py`,
     `respgeomlib/curved_primitives.py`.
   - Segment abstraction with ports → `respgeomlib/segments.py`
     (`build_pipe_segment`, `build_y2_segment`, `build_y3_segment`, plus the
     `pipe_stenosis` / `pipe_dilation` variant builders).
4. **Implicit smooth-min junctions (local extraction)** — junctions are the zero
   level-set of a blended tube field
   `φ(x) = -(1/κ) · log Σ exp(-κ φ_i(x))`, sampled on a tight local grid around
   the junction and extracted with marching cubes.
   → `respgeomlib/implicit_y.py` (Y2), `respgeomlib/implicit_y3.py` (Y3).
   The tube-based baseline lives in `respgeomlib/junctions.py`.
5. **Assembly & merge** — `build_tree` resolves a deterministic build order and
   places each segment; `merge_built_segments` concatenates the per-segment
   meshes into the full lumen surface.

## Morphometry-guided growth

`respgeomlib/morphometry_rules.py` implements generation-wise diameter/length
trends (Weibel/ICRP) with an optional Murray-type branching constraint, and can
emit a YAML subtree. `respgeomlib/merge_trees.py` merges a base spec with a
generated subtree, and `respgeomlib/morphometry_summary.py` reports
per-generation statistics for a built tree.

## Controlled synthetic variants

Localized lumen remodeling is `r(s) = r0(s) · (1 + α · w(s))`, with `w` a smooth
window and `α` the severity (negative → stenosis, positive → dilation). It is
applied to a fixed base segment while preserving topology and ports.
→ `gaussian_radius_profile`, `build_pipe_stenosis_segment`,
`build_pipe_dilation_segment` in `respgeomlib/segments.py`.

## Airway Studio (GUI)

`app/airway_studio.py` is a self-contained PyQt application that wraps the
engine: Weibel-model generation, disease variants, an embedded interactive
viewer, mesh validity checks, and an ANSYS Fluent export package (wall STL,
inlet/outlet caps, solver journal, ZIP bundle).
