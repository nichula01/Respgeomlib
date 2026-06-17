"""
RespGeomLib
===========

A reproducible parametric engine for generating analysis-ready human airway
lumen geometry from compact YAML specifications.

The library combines:
  * a modular port-based assembly interface for hierarchical tree construction
    (:mod:`respgeomlib.frames`, :mod:`respgeomlib.segments`,
    :mod:`respgeomlib.tree_builder`),
  * implicit smooth-min junction blending for seamless bifurcations (Y2) and
    trifurcations (Y3) (:mod:`respgeomlib.implicit_y`,
    :mod:`respgeomlib.implicit_y3`),
  * morphometry-guided tree growth using Weibel/ICRP-style trends with optional
    Murray-type constraints (:mod:`respgeomlib.morphometry_rules`).

Typical entry points::

    from respgeomlib import load_specs_from_yaml, build_tree, merge_built_segments

    specs = load_specs_from_yaml("trees/stenosis_test.yaml")
    built = build_tree(specs)
    points, faces = merge_built_segments(built)
"""

from __future__ import annotations

# --- Coordinate frames & primitives ----------------------------------------
from .frames import unit, direction_from_angles, Frame, make_child_frame
from .primitives import make_cylinder_local, cylinder_polydata
from .curved_primitives import make_curved_pipe

# --- Segment-local geometry building blocks ---------------------------------
from .segments import (
    Port,
    SegmentGeom,
    gaussian_radius_profile,
    build_pipe_segment,
    build_pipe_stenosis_segment,
    build_pipe_dilation_segment,
    build_y2_segment,
    build_y3_segment,
)

# --- Implicit smooth-min junctions ------------------------------------------
from .junctions import make_two_way_y_local, two_way_y_polydata
from .implicit_y import make_two_way_y_implicit_local, two_way_y_implicit_polydata
from .implicit_y3 import (
    make_three_way_y_implicit_local,
    three_way_y_implicit_polydata,
)

# --- Tree assembly -----------------------------------------------------------
from .tree_builder import (
    SegmentSpec,
    BuiltSegment,
    build_tree,
    build_segment_geom,
    segment_to_world,
    merge_built_segments,
    load_specs_from_yaml,
    make_example_specs,
)

# --- Morphometry-guided growth ----------------------------------------------
from .morphometry_rules import (
    ChildBranch,
    length_from_diameter,
    murray_child_diameters,
    weibel_like_children,
    generate_binary_subtree,
)

__version__ = "0.1.0"

__all__ = [
    "unit",
    "direction_from_angles",
    "Frame",
    "make_child_frame",
    "make_cylinder_local",
    "cylinder_polydata",
    "make_curved_pipe",
    "Port",
    "SegmentGeom",
    "gaussian_radius_profile",
    "build_pipe_segment",
    "build_pipe_stenosis_segment",
    "build_pipe_dilation_segment",
    "build_y2_segment",
    "build_y3_segment",
    "make_two_way_y_local",
    "two_way_y_polydata",
    "make_two_way_y_implicit_local",
    "two_way_y_implicit_polydata",
    "make_three_way_y_implicit_local",
    "three_way_y_implicit_polydata",
    "SegmentSpec",
    "BuiltSegment",
    "build_tree",
    "build_segment_geom",
    "segment_to_world",
    "merge_built_segments",
    "load_specs_from_yaml",
    "make_example_specs",
    "ChildBranch",
    "length_from_diameter",
    "murray_child_diameters",
    "weibel_like_children",
    "generate_binary_subtree",
    "__version__",
]
