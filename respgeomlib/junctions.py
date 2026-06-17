"""
Two-way Y junctions built in local coordinates using PyVista's tube filter and lines_from_points.
No boolean operations are used; three centerlines are tubed into a single Y-shaped surface.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - optional dependency
    pv = None

from .frames import direction_from_angles


def _build_child_centerline(
    direction: np.ndarray,
    length: float,
    n_s: int,
    blend_length: float,
    trunk_dir: np.ndarray,
) -> np.ndarray:
    """
    Build a child centerline polyline starting at the origin.

    If blend_length <= 0, the centerline is straight: C(s) = s * direction.
    Otherwise, for 0 <= s <= blend_length, the tangent smoothly rotates from
    trunk_dir to direction using a smoothstep profile; beyond that it is constant.
    """
    dir_norm = np.linalg.norm(direction)
    trunk_norm = np.linalg.norm(trunk_dir)
    if dir_norm == 0 or trunk_norm == 0:
        raise ValueError("direction and trunk_dir must be nonzero vectors")
    direction = direction / dir_norm
    trunk_dir = trunk_dir / trunk_norm

    s = np.linspace(0.0, length, n_s + 1)
    if blend_length <= 0:
        return s[:, None] * direction[None, :]

    ds = length / n_s
    pts = np.zeros((n_s + 1, 3), dtype=float)
    for k in range(n_s):
        s_k = k * ds
        if s_k < blend_length:
            t = s_k / blend_length
            w = 3.0 * t**2 - 2.0 * t**3  # smoothstep with zero slope endpoints
            u = (1.0 - w) * trunk_dir + w * direction
        else:
            u = direction
        u = u / np.linalg.norm(u)
        pts[k + 1] = pts[k] + u * ds
    return pts


def make_two_way_y_local(
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
    n_s_trunk: int = 20,
    n_s_child: int = 20,
    n_theta: int = 24,
    blend_length: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a parametric two-way Y junction in local coordinates.

    Local convention:
      - Junction point is at the origin (0, 0, 0).
      - Trunk centerline goes from the origin along negative z.
      - Child centerlines start at the origin and extend in directions
        defined by (theta_deg, phi_deg) in this local frame.
      - blend_length controls smooth bending of the child tangents near the origin;
        blend_length = 0 recovers the sharp Y.

    Returns:
        points : (N, 3) float
            Vertex positions of the Y surface in local coordinates.
        faces : (M, 3) int
            Triangle indices into points.
    """
    if pv is None:
        raise RuntimeError("pyvista is required to build Y junctions")

    if length_trunk <= 0 or length_child1 <= 0 or length_child2 <= 0:
        raise ValueError("all lengths must be positive")
    if d_trunk <= 0 or d_child1 <= 0 or d_child2 <= 0:
        raise ValueError("all diameters must be positive")
    if n_s_trunk < 1 or n_s_child < 1:
        raise ValueError("n_s_trunk and n_s_child must be at least 1")
    if n_theta < 6:
        raise ValueError("n_theta must be at least 6")
    if blend_length < 0:
        raise ValueError("blend_length must be non-negative")

    if not np.allclose([d_trunk, d_child1, d_child2], d_trunk, rtol=1e-6, atol=1e-9):
        raise ValueError(
            "Current implementation expects equal diameters at the junction; "
            "support for different diameters will be added later."
        )

    blend_len_child1 = min(max(blend_length, 0.0), length_child1)
    blend_len_child2 = min(max(blend_length, 0.0), length_child2)

    r_trunk = d_trunk / 2.0

    # Straight trunk along -z, starting at origin.
    s_trunk = np.linspace(0.0, length_trunk, n_s_trunk + 1)
    pts_trunk = np.column_stack(
        [np.zeros_like(s_trunk), np.zeros_like(s_trunk), -s_trunk]
    )

    trunk_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    d1 = direction_from_angles(theta1_deg, phi1_deg)
    d2 = direction_from_angles(theta2_deg, phi2_deg)

    # Build child centerlines; near s=0 tangents follow trunk_dir then smoothly rotate.
    pts_child1 = _build_child_centerline(
        direction=d1,
        length=length_child1,
        n_s=n_s_child,
        blend_length=blend_len_child1,
        trunk_dir=trunk_dir,
    )
    pts_child2 = _build_child_centerline(
        direction=d2,
        length=length_child2,
        n_s=n_s_child,
        blend_length=blend_len_child2,
        trunk_dir=trunk_dir,
    )

    poly_trunk = pv.lines_from_points(pts_trunk, close=False)
    poly_child1 = pv.lines_from_points(pts_child1, close=False)
    poly_child2 = pv.lines_from_points(pts_child2, close=False)
    centerlines = pv.merge([poly_trunk, poly_child1, poly_child2])

    tube = centerlines.tube(
        radius=r_trunk,
        n_sides=n_theta,
        capping=False,
        progress_bar=False,
    )

    tube = tube.triangulate().clean()

    pts_out = tube.points
    faces_flat = tube.faces
    if faces_flat.size == 0:
        raise RuntimeError("Tube filter returned zero faces for Y junction")
    faces_vtk = faces_flat.reshape(-1, 4)
    faces = faces_vtk[:, 1:4]
    return pts_out.astype(float), faces.astype(int)


def two_way_y_polydata(
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
    n_s_trunk: int = 20,
    n_s_child: int = 20,
    n_theta: int = 24,
    blend_length: float = 0.0,
):
    """
    Convenience wrapper: return a pyvista.PolyData for the two-way Y junction.
    """
    if pv is None:
        raise RuntimeError("pyvista is required to build Y junctions")

    points, faces = make_two_way_y_local(
        length_trunk,
        length_child1,
        length_child2,
        d_trunk,
        d_child1,
        d_child2,
        theta1_deg,
        phi1_deg,
        theta2_deg,
        phi2_deg,
        n_s_trunk=n_s_trunk,
        n_s_child=n_s_child,
        n_theta=n_theta,
        blend_length=blend_length,
    )
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(points, faces_flat)


if __name__ == "__main__":
    try:
        for blend_len in (0.0, 2.0):
            pts, fcs = make_two_way_y_local(
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
                n_s_trunk=20,
                n_s_child=20,
                n_theta=24,
                blend_length=blend_len,
            )
            print(f"Y junction mesh (blend_length={blend_len}):")
            print(" points:", len(pts))
            print(" faces:", len(fcs))
            print(" x/y/z min:", pts.min(axis=0))
            print(" x/y/z max:", pts.max(axis=0))

        if pv is not None:
            mesh = two_way_y_polydata(
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
                blend_length=2.0,
            )
            print("PyVista Y mesh points:", mesh.n_points)
            print("PyVista Y mesh cells:", mesh.n_cells)
        else:
            print("PyVista not available; skipping visualization test.")
    except Exception as e:
        print("Error in two-way Y self-test:", e)
