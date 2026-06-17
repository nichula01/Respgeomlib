"""
Geometric primitives for respiratory geometry pipelines.

Provides tapered cylindrical meshes in local coordinates (axis aligned with
+z, inlet at z=0, outlet at z=length). Transform to world coordinates with a
Frame from frames.py if needed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - optional dependency
    pv = None


def make_cylinder_local(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int = 32,
    n_z: int = 16,
    cap_ends: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a (possibly tapered) cylinder mesh in local coordinates.

    Local frame: axis along +z; inlet center at (0,0,0); outlet center at
    (0,0,length). Radius varies linearly from d_in/2 at z=0 to d_out/2 at
    z=length. Intended to be mapped to world space later (e.g., via Frame).

    Returns:
        points: (N,3) array of vertex positions in local coordinates.
        faces: (M,3) array of triangle indices.
    """
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

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(0.0, length, n_z + 1)
    # Radii per ring interpolate linearly along z.
    r_vals = r_in + (r_out - r_in) * (z_vals / length)

    # Build ring points; index mapping: j*n_theta + i for ring j, angle i.
    pts = []
    for z_idx, (z_val, r_val) in enumerate(zip(z_vals, r_vals)):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
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
            # Split quad into two triangles (v00, v10, v11) and (v00, v11, v01)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    if cap_ends:
        # Inlet cap (z=0)
        c0 = len(points)
        ring0_indices = np.arange(0, n_theta, dtype=int)
        points = np.vstack((points, np.array([[0.0, 0.0, 0.0]])))
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            faces.append((c0, ring0_indices[i_next], ring0_indices[i]))

        # Outlet cap (z=length)
        c1 = len(points)
        ring1_indices = np.arange((n_rings - 1) * n_theta, n_rings * n_theta, dtype=int)
        points = np.vstack((points, np.array([[0.0, 0.0, length]])))
        for i in range(n_theta):
            i_next = (i + 1) % n_theta
            # Orientation chosen so normals point roughly +z.
            faces.append((c1, ring1_indices[i], ring1_indices[i_next]))

    faces_array = np.asarray(faces, dtype=int)
    return points.astype(float), faces_array


def cylinder_polydata(
    length: float,
    d_in: float,
    d_out: float,
    n_theta: int = 32,
    n_z: int = 16,
    cap_ends: bool = False,
):
    """
    Convenience wrapper: return a pyvista.PolyData of the tapered cylinder.
    """
    if pv is None:
        raise RuntimeError("pyvista is not installed; cylinder_polydata is unavailable")

    points, faces = make_cylinder_local(length, d_in, d_out, n_theta, n_z, cap_ends)
    # PyVista expects a flat array: [3, i, j, k, 3, i2, j2, k2, ...]
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(points, faces_flat)


if __name__ == "__main__":
    pts, fcs = make_cylinder_local(length=10.0, d_in=4.0, d_out=2.0, n_theta=16, n_z=8, cap_ends=True)
    print("Cylinder mesh:")
    print(" points:", len(pts))
    print(" faces:", len(fcs))
    print(" z min/max:", pts[:, 2].min(), pts[:, 2].max())

    if pv is not None:
        mesh = cylinder_polydata(length=10.0, d_in=4.0, d_out=2.0, n_theta=16, n_z=8, cap_ends=True)
        print("PyVista mesh points:", mesh.n_points)
        print("PyVista mesh cells:", mesh.n_cells)
    else:
        print("PyVista not available; skipping PolyData test.")
