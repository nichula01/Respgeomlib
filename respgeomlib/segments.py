"""
Small abstraction layer for pipe segments, 2-way Y segments, and 3-way Y segments,
all defined in segment-local coordinates with port metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .frames import unit
from .primitives import make_cylinder_local
from .implicit_y import make_two_way_y_implicit_local, _build_child_centerline
from .implicit_y3 import make_three_way_y_implicit_local


@dataclass
class Port:
    """
    A connection port on a segment.

    Attributes
    ----------
    position : (3,) float
        Port center in segment-local coordinates.
    direction : (3,) float
        Unit vector pointing into the segment from this port.
    """

    position: np.ndarray
    direction: np.ndarray


@dataclass
class SegmentGeom:
    """
    Geometry and port metadata for a single segment in its local frame.

    Attributes
    ----------
    points : (N, 3) float
        Segment surface vertices in segment-local coordinates.
    faces : (M, 3) int
        Triangle indices into points.
    ports : list of Port
        Ports available for connecting this segment to others.
    parent_port_index : int
        Index into `ports` that is considered the parent port (usually 0).
    """

    points: np.ndarray
    faces: np.ndarray
    ports: List[Port]
    parent_port_index: int = 0


def gaussian_radius_profile(
    z: np.ndarray,
    r_prox: float,
    r_dist: float,
    r_min_factor: float = 0.5,
    center: float = 0.5,
    width: float = 0.1,
    mode: str = "stenosis",
) -> np.ndarray:
    """
    Build a radius profile along a normalized axial coordinate z in [0, 1].

    Parameters
    ----------
    z : (N,) float
        Axial coordinates normalized to [0,1].
    r_prox : float
        Radius at the proximal end.
    r_dist : float
        Radius at the distal end.
    r_min_factor : float
        For stenosis: minimal radius is r_min_factor * min(r_prox, r_dist).
        For dilation: maximal radius is (1 / r_min_factor) * max(r_prox, r_dist).
    center : float
        Centre of the lesion in [0,1].
    width : float
        Controls lesion extent (Gaussian standard deviation in [0,1]).
    mode : {"stenosis", "dilation"}
        Shape type: narrowing or widening.

    Returns
    -------
    radii : (N,) float
        Radius at each z position.
    """
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array")
    if z_arr.size == 0:
        return z_arr.copy()
    if r_prox <= 0 or r_dist <= 0:
        raise ValueError("r_prox and r_dist must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if r_min_factor <= 0:
        raise ValueError("r_min_factor must be positive")

    center_clamped = float(np.clip(center, 0.0, 1.0))
    base = r_prox + (r_dist - r_prox) * z_arr
    base = np.maximum(base, 1e-9)

    base_center = r_prox + (r_dist - r_prox) * center_clamped
    base_center = max(base_center, 1e-9)

    gaussian = np.exp(-((z_arr - center_clamped) ** 2) / (2.0 * width ** 2))

    mode_l = (mode or "").lower()
    if mode_l == "stenosis":
        target = r_min_factor * min(r_prox, r_dist)
        target = max(target, 1e-9)
        A = 1.0 - target / base_center
        A = float(np.clip(A, 0.0, 0.999))
        factor = 1.0 - A * gaussian
    elif mode_l == "dilation":
        target = (1.0 / r_min_factor) * max(r_prox, r_dist)
        target = max(target, 1e-9)
        A = target / base_center - 1.0
        A = max(A, 0.0)
        factor = 1.0 + A * gaussian
    else:
        raise ValueError("mode must be 'stenosis' or 'dilation'")

    radii = base * factor
    return radii


def _dir_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return unit(np.array([x, y, z], dtype=float))


def build_pipe_segment(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int = 32,
    n_z: int = 16,
) -> SegmentGeom:
    """
    Build a straight pipe segment in segment-local coordinates.

    Local convention:
      - Parent port (inlet) at (0, 0, 0).
      - Segment axis along +z, outlet at (0, 0, length).
      - `direction` for the parent port points into the segment along +z.
      - Child port (outlet) at (0, 0, length), direction into segment is -z.
    """
    points, faces = make_cylinder_local(
        length, d_in, d_out, n_theta=n_theta, n_z=n_z, cap_ends=False
    )

    parent_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    parent_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    child_pos = np.array([0.0, 0.0, length], dtype=float)
    child_dir = np.array([0.0, 0.0, -1.0], dtype=float)

    ports = [
        Port(position=parent_pos, direction=parent_dir),
        Port(position=child_pos, direction=child_dir),
    ]

    return SegmentGeom(points=points, faces=faces, ports=ports, parent_port_index=0)


def _build_pipe_with_profile(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int,
    n_z: int,
    center: float,
    width: float,
    r_min_factor: float,
    mode: str,
) -> SegmentGeom:
    if length <= 0:
        raise ValueError("length must be positive")
    if d_in <= 0 or d_out <= 0:
        raise ValueError("diameters must be positive")
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")
    if n_z < 1:
        raise ValueError("n_z must be at least 1")

    r_in = d_in / 2.0
    r_out = d_out / 2.0

    z_vals = np.linspace(0.0, length, n_z + 1)
    z_norm = z_vals / length
    radii = gaussian_radius_profile(
        z_norm,
        r_in,
        r_out,
        r_min_factor=r_min_factor,
        center=center,
        width=width,
        mode=mode,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    pts = []
    for z_val, r_val in zip(z_vals, radii):
        x_ring = r_val * cos_t
        y_ring = r_val * sin_t
        z_ring = np.full_like(theta, z_val)
        ring = np.column_stack((x_ring, y_ring, z_ring))
        pts.append(ring)
    points = np.vstack(pts)

    faces = []
    n_rings = n_z + 1
    for j in range(n_rings - 1):
        base0 = j * n_theta
        base1 = (j + 1) * n_theta
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            v00 = base0 + i
            v01 = base0 + i_next
            v10 = base1 + i
            v11 = base1 + i_next
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    parent_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    parent_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    child_pos = np.array([0.0, 0.0, length], dtype=float)
    child_dir = np.array([0.0, 0.0, -1.0], dtype=float)

    ports = [
        Port(position=parent_pos, direction=parent_dir),
        Port(position=child_pos, direction=child_dir),
    ]

    return SegmentGeom(
        points=points.astype(float),
        faces=np.asarray(faces, dtype=int),
        ports=ports,
        parent_port_index=0,
    )


def build_pipe_stenosis_segment(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int = 32,
    n_z: int = 32,
    center: float = 0.5,
    width: float = 0.1,
    r_min_factor: float = 0.5,
) -> SegmentGeom:
    """
    Build a pipe segment with a local stenosis (narrowing) in the middle.
    Geometry is in local coordinates (axis along +z).
    """
    return _build_pipe_with_profile(
        length=length,
        d_in=d_in,
        d_out=d_out,
        n_theta=n_theta,
        n_z=n_z,
        center=center,
        width=width,
        r_min_factor=r_min_factor,
        mode="stenosis",
    )


def build_pipe_dilation_segment(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int = 32,
    n_z: int = 32,
    center: float = 0.5,
    width: float = 0.1,
    r_min_factor: float = 0.5,
) -> SegmentGeom:
    """
    Build a pipe segment with a local dilatation (widening) in the middle.
    Geometry is in local coordinates (axis along +z).
    """
    return _build_pipe_with_profile(
        length=length,
        d_in=d_in,
        d_out=d_out,
        n_theta=n_theta,
        n_z=n_z,
        center=center,
        width=width,
        r_min_factor=r_min_factor,
        mode="dilation",
    )


def build_y2_segment(
    length_trunk: float,
    length_child1: float,
    length_child2: float,
    d_trunk: float,
    d_child1: float,
    d_child2: float,
    theta1_deg: float,
    phi1_deg: float,
    theta2_deg: float,
    phi2_deg: float,
    n_s_trunk: int = 40,
    n_s_child: int = 40,
    blend_length: float = 2.0,
    grid_resolution_per_radius: float = 3.0,
) -> SegmentGeom:
    """
    Build a 2-way Y (bifurcation) segment in segment-local coordinates.

    Segment-local convention:
      - Parent port is the trunk inlet orifice.
      - Parent port is placed at the origin (0, 0, 0).
      - Local +z at the parent port points along the trunk towards the carina.
      - Child ports are located at the ends of the two child branches.
      - Each child port's direction points from the orifice back into the segment.
    """
    pts_carina, faces = make_two_way_y_implicit_local(
        length_trunk=length_trunk,
        length_child1=length_child1,
        length_child2=length_child2,
        d_trunk=d_trunk,
        d_child1=d_child1,
        d_child2=d_child2,
        theta1_deg=theta1_deg,
        phi1_deg=phi1_deg,
        theta2_deg=theta2_deg,
        phi2_deg=phi2_deg,
        n_s_trunk=n_s_trunk,
        n_s_child=n_s_child,
        blend_length=blend_length,
        grid_resolution_per_radius=grid_resolution_per_radius,
    )

    trunk_orifice_carina = np.array([0.0, 0.0, -length_trunk], dtype=float)
    offset = -trunk_orifice_carina
    pts_local = pts_carina + offset

    parent_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    parent_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    carina_pos = offset  # carina was near (0,0,0) before translation

    d1 = _dir_from_angles(theta1_deg, phi1_deg)
    d2 = _dir_from_angles(theta2_deg, phi2_deg)

    trunk_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    blend_len_child1 = min(max(blend_length, 0.0), length_child1)
    blend_len_child2 = min(max(blend_length, 0.0), length_child2)

    pts_child1_carina = _build_child_centerline(
        direction=d1,
        length=length_child1,
        n_s=n_s_child,
        blend_length=blend_len_child1,
        trunk_dir=trunk_dir,
    )
    pts_child2_carina = _build_child_centerline(
        direction=d2,
        length=length_child2,
        n_s=n_s_child,
        blend_length=blend_len_child2,
        trunk_dir=trunk_dir,
    )

    child1_orifice_carina = pts_child1_carina[-1]
    child2_orifice_carina = pts_child2_carina[-1]

    child1_pos = child1_orifice_carina + offset
    child2_pos = child2_orifice_carina + offset

    t1 = pts_child1_carina[-1] - pts_child1_carina[-2]
    t2 = pts_child2_carina[-1] - pts_child2_carina[-2]

    child1_dir = -unit(t1)
    child2_dir = -unit(t2)

    ports = [
        Port(position=parent_pos, direction=parent_dir),
        Port(position=child1_pos, direction=child1_dir),
        Port(position=child2_pos, direction=child2_dir),
    ]

    return SegmentGeom(
        points=pts_local.astype(float),
        faces=faces.astype(int),
        ports=ports,
        parent_port_index=0,
    )


def build_y3_segment(
    length_trunk: float,
    length_child1: float,
    length_child2: float,
    length_child3: float,
    d_trunk: float,
    d_child1: float,
    d_child2: float,
    d_child3: float,
    theta1_deg: float,
    phi1_deg: float,
    theta2_deg: float,
    phi2_deg: float,
    theta3_deg: float,
    phi3_deg: float,
    n_s_trunk: int = 40,
    n_s_child: int = 40,
    blend_length: float = 2.0,
    grid_resolution_per_radius: float = 3.0,
) -> SegmentGeom:
    """
    Build a 3-way Y (trifurcation) segment in segment-local coordinates.

    Segment-local convention:
      - Parent port is the trunk inlet orifice at (0, 0, 0).
      - Local +z at the parent port points along the trunk towards the carina.
      - Three child ports are located at the ends of the three child branches.
      - Each child port's direction points from the orifice back into the segment.
    """
    pts_carina, faces = make_three_way_y_implicit_local(
        length_trunk=length_trunk,
        length_child1=length_child1,
        length_child2=length_child2,
        length_child3=length_child3,
        d_trunk=d_trunk,
        d_child1=d_child1,
        d_child2=d_child2,
        d_child3=d_child3,
        theta1_deg=theta1_deg,
        phi1_deg=phi1_deg,
        theta2_deg=theta2_deg,
        phi2_deg=phi2_deg,
        theta3_deg=theta3_deg,
        phi3_deg=phi3_deg,
        n_s_trunk=n_s_trunk,
        n_s_child=n_s_child,
        blend_length=blend_length,
        grid_resolution_per_radius=grid_resolution_per_radius,
    )

    trunk_orifice_carina = np.array([0.0, 0.0, -length_trunk], dtype=float)
    offset = -trunk_orifice_carina
    pts_local = pts_carina + offset

    parent_pos = np.array([0.0, 0.0, 0.0], dtype=float)
    parent_dir = np.array([0.0, 0.0, 1.0], dtype=float)

    carina_pos = offset  # carina was near (0,0,0) before translation

    d1 = _dir_from_angles(theta1_deg, phi1_deg)
    d2 = _dir_from_angles(theta2_deg, phi2_deg)
    d3 = _dir_from_angles(theta3_deg, phi3_deg)

    trunk_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    blend_len_child1 = min(max(blend_length, 0.0), length_child1)
    blend_len_child2 = min(max(blend_length, 0.0), length_child2)
    blend_len_child3 = min(max(blend_length, 0.0), length_child3)

    pts_child1_carina = _build_child_centerline(
        direction=d1,
        length=length_child1,
        n_s=n_s_child,
        blend_length=blend_len_child1,
        trunk_dir=trunk_dir,
    )
    pts_child2_carina = _build_child_centerline(
        direction=d2,
        length=length_child2,
        n_s=n_s_child,
        blend_length=blend_len_child2,
        trunk_dir=trunk_dir,
    )
    pts_child3_carina = _build_child_centerline(
        direction=d3,
        length=length_child3,
        n_s=n_s_child,
        blend_length=blend_len_child3,
        trunk_dir=trunk_dir,
    )

    child1_orifice_carina = pts_child1_carina[-1]
    child2_orifice_carina = pts_child2_carina[-1]
    child3_orifice_carina = pts_child3_carina[-1]

    child1_pos = child1_orifice_carina + offset
    child2_pos = child2_orifice_carina + offset
    child3_pos = child3_orifice_carina + offset

    t1 = pts_child1_carina[-1] - pts_child1_carina[-2]
    t2 = pts_child2_carina[-1] - pts_child2_carina[-2]
    t3 = pts_child3_carina[-1] - pts_child3_carina[-2]

    child1_dir = -unit(t1)
    child2_dir = -unit(t2)
    child3_dir = -unit(t3)

    ports = [
        Port(position=parent_pos, direction=parent_dir),
        Port(position=child1_pos, direction=child1_dir),
        Port(position=child2_pos, direction=child2_dir),
        Port(position=child3_pos, direction=child3_dir),
    ]

    return SegmentGeom(
        points=pts_local.astype(float),
        faces=faces.astype(int),
        ports=ports,
        parent_port_index=0,
    )


if __name__ == "__main__":
    # Simple sanity checks for each builder.
    pipe = build_pipe_segment(length=5.0, d_in=2.0, d_out=2.0)
    print("Pipe:", pipe.points.shape, pipe.faces.shape, "ports:", len(pipe.ports))
    for i, port in enumerate(pipe.ports):
        print(f"  Pipe port {i}: pos={port.position}, dir={port.direction}")

    pipe_stenosis = build_pipe_stenosis_segment(length=5.0, d_in=2.0, d_out=2.0)
    print("Pipe stenosis:", pipe_stenosis.points.shape, pipe_stenosis.faces.shape, "ports:", len(pipe_stenosis.ports))

    pipe_dilation = build_pipe_dilation_segment(length=5.0, d_in=2.0, d_out=2.0)
    print("Pipe dilation:", pipe_dilation.points.shape, pipe_dilation.faces.shape, "ports:", len(pipe_dilation.ports))

    y2 = build_y2_segment(
        length_trunk=6.0,
        length_child1=4.0,
        length_child2=4.0,
        d_trunk=2.0,
        d_child1=2.0,
        d_child2=2.0,
        theta1_deg=45.0,
        phi1_deg=0.0,
        theta2_deg=45.0,
        phi2_deg=120.0,
    )
    print("Y2:", y2.points.shape, y2.faces.shape, "ports:", len(y2.ports))

    y3 = build_y3_segment(
        length_trunk=6.0,
        length_child1=4.0,
        length_child2=4.0,
        length_child3=4.0,
        d_trunk=2.0,
        d_child1=2.0,
        d_child2=2.0,
        d_child3=2.0,
        theta1_deg=45.0,
        phi1_deg=0.0,
        theta2_deg=45.0,
        phi2_deg=120.0,
        theta3_deg=45.0,
        phi3_deg=240.0,
    )
    print("Y3:", y3.points.shape, y3.faces.shape, "ports:", len(y3.ports))
