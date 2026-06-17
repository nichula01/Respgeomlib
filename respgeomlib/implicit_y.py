"""
Two-way Y junction via an implicit distance field and marching cubes in local coordinates.
Local convention: junction at origin, trunk along -z, children set by elevation/azimuth angles.
Blending comes from the min of tube distance fields; no boolean operations are used.
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

    If blend_length <= 0, the centerline is straight:
      C(s) = s * direction

    If blend_length > 0, then for 0 <= s <= blend_length the tangent
    smoothly rotates from trunk_dir to direction using a smoothstep
    profile, and for s > blend_length it is constant = direction.
    """
    direction = np.asarray(direction, dtype=float)
    trunk_dir = np.asarray(trunk_dir, dtype=float)
    direction /= np.linalg.norm(direction)
    trunk_dir /= np.linalg.norm(trunk_dir)

    if blend_length <= 0:
        s = np.linspace(0.0, length, n_s + 1)
        return s[:, None] * direction[None, :]

    ds = length / n_s
    pts = np.zeros((n_s + 1, 3), dtype=float)
    for k in range(n_s):
        s_k = k * ds
        if s_k < blend_length:
            t = s_k / blend_length  # in [0,1)
            w = 3.0 * t**2 - 2.0 * t**3  # smoothstep
            u = (1.0 - w) * trunk_dir + w * direction
        else:
            u = direction.copy()
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            u = direction.copy()
        else:
            u /= norm
        pts[k + 1] = pts[k] + u * ds
    return pts


def _approx_distance_to_polyline(
    grid_points: np.ndarray,
    poly_points: np.ndarray,
) -> np.ndarray:
    """
    Approximate distance from each grid point to a polyline, by taking
    the minimum Euclidean distance to the sampled points on the line.

    This is not the exact distance to the continuous line, but it is
    sufficient if poly_points are dense along the centerline.

    Parameters
    ----------
    grid_points : (N, 3) float
        Points where the distance is evaluated.
    poly_points : (M, 3) float
        Sampled points along the polyline centerline.

    Returns
    -------
    dist : (N,) float
        Approximate distance for each grid point.
    """
    dist2 = np.full(grid_points.shape[0], np.inf, dtype=float)
    for p in poly_points:
        diff = grid_points - p
        d2 = np.einsum("ij,ij->i", diff, diff)
        dist2 = np.minimum(dist2, d2)
    return np.sqrt(dist2)


def make_two_way_y_implicit_local(
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
    blend_length: float = 0.0,
    grid_resolution_per_radius: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a smooth two-way Y junction in local coordinates using an
    implicit distance field and marching cubes.

    Local convention:
      - Junction at the origin (0, 0, 0).
      - Trunk centerline along negative z.
      - Child centerlines defined by elevation/azimuth angles in this frame.

    The implicit field is:
        F(x) = min_i (dist_to_branch_i(x) - r_i),
    and we extract the F=0 iso-surface.

    Returns
    -------
    points : (N, 3) float
        Vertex positions of the Y surface in local coordinates.
    faces : (M, 3) int
        Triangle indices into points.
    """
    if pv is None:
        raise RuntimeError("pyvista is required for implicit Y junctions")

    if length_trunk <= 0 or length_child1 <= 0 or length_child2 <= 0:
        raise ValueError("all lengths must be positive")
    if d_trunk <= 0 or d_child1 <= 0 or d_child2 <= 0:
        raise ValueError("all diameters must be positive")
    if n_s_trunk < 4 or n_s_child < 4:
        raise ValueError("n_s_trunk and n_s_child must be at least 4")
    if grid_resolution_per_radius <= 0:
        raise ValueError("grid_resolution_per_radius must be positive")

    r_trunk = d_trunk / 2.0
    r_child1 = d_child1 / 2.0
    r_child2 = d_child2 / 2.0

    # Trunk centerline (straight along -z).
    s_trunk = np.linspace(0.0, length_trunk, n_s_trunk + 1)
    pts_trunk = np.column_stack(
        [np.zeros_like(s_trunk), np.zeros_like(s_trunk), -s_trunk]
    )

    trunk_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    d1 = direction_from_angles(theta1_deg, phi1_deg)
    d2 = direction_from_angles(theta2_deg, phi2_deg)

    blend_len_child1 = min(max(blend_length, 0.0), length_child1)
    blend_len_child2 = min(max(blend_length, 0.0), length_child2)

    # Child centerlines: start tangent = trunk_dir, smoothly bend over blend_length, then straight.
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

    # Bounds and grid sizing.
    all_pts = np.vstack([pts_trunk, pts_child1, pts_child2])
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    r_max = max(r_trunk, r_child1, r_child2)
    margin = 2.0 * r_max

    dx = r_max / grid_resolution_per_radius
    dy = dx
    dz = dx

    nx = int(np.ceil((xyz_max[0] - xyz_min[0] + 2 * margin) / dx)) + 1
    ny = int(np.ceil((xyz_max[1] - xyz_min[1] + 2 * margin) / dy)) + 1
    nz = int(np.ceil((xyz_max[2] - xyz_min[2] + 2 * margin) / dz)) + 1
    nx = max(nx, 3)
    ny = max(ny, 3)
    nz = max(nz, 3)

    origin = np.array(
        [
            xyz_min[0] - margin,
            xyz_min[1] - margin,
            xyz_min[2] - margin,
        ],
        dtype=float,
    )

    # Support pyvista builds that expose ImageData but not UniformGrid.
    if hasattr(pv, "UniformGrid"):
        grid_cls = pv.UniformGrid
    else:
        grid_cls = getattr(pv, "ImageData", None)
    if grid_cls is None:
        raise RuntimeError("pyvista is missing both UniformGrid and ImageData classes")
    grid = grid_cls()
    grid.dimensions = (nx, ny, nz)
    grid.origin = origin
    grid.spacing = (dx, dy, dz)

    # Distance-field union of tubes; blending comes from min of distances.
    pts_grid = grid.points
    dist_trunk = _approx_distance_to_polyline(pts_grid, pts_trunk)
    dist_child1 = _approx_distance_to_polyline(pts_grid, pts_child1)
    dist_child2 = _approx_distance_to_polyline(pts_grid, pts_child2)

    phi_trunk = dist_trunk - r_trunk
    phi_child1 = dist_child1 - r_child1
    phi_child2 = dist_child2 - r_child2

    phi = np.minimum(phi_trunk, np.minimum(phi_child1, phi_child2))
    grid["phi"] = phi

    surface = grid.contour(
        isosurfaces=[0.0],
        scalars="phi",
    )

    surface = surface.triangulate().clean()

    pts_out = surface.points
    faces_flat = surface.faces
    if faces_flat.size == 0:
        raise RuntimeError("Implicit Y contour returned zero faces")
    faces_vtk = faces_flat.reshape(-1, 4)
    faces = faces_vtk[:, 1:4]
    return pts_out.astype(float), faces.astype(int)


def two_way_y_implicit_polydata(
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
    blend_length: float = 0.0,
    grid_resolution_per_radius: float = 3.0,
):
    """
    Convenience wrapper: return a pyvista.PolyData for the implicit Y junction.
    """
    if pv is None:
        raise RuntimeError("pyvista is required for implicit Y junctions")

    points, faces = make_two_way_y_implicit_local(
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
        blend_length=blend_length,
        grid_resolution_per_radius=grid_resolution_per_radius,
    )
    faces_flat = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=int), faces]
    ).ravel()
    return pv.PolyData(points, faces_flat)


if __name__ == "__main__":
    try:
        pts, fcs = make_two_way_y_implicit_local(
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
            n_s_trunk=40,
            n_s_child=40,
            blend_length=2.0,
            grid_resolution_per_radius=3.0,
        )
        print("Implicit Y junction mesh:")
        print(" points:", len(pts))
        print(" faces:", len(fcs))
        print(" x/y/z min:", pts.min(axis=0))
        print(" x/y/z max:", pts.max(axis=0))

        if pv is not None:
            mesh = two_way_y_implicit_polydata(
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
            print("PyVista implicit Y mesh points:", mesh.n_points)
            print("PyVista implicit Y mesh cells:", mesh.n_cells)
        else:
            print("PyVista not available; skipping PolyData test.")
    except Exception as e:
        print("Error in implicit two-way Y self-test:", e)
