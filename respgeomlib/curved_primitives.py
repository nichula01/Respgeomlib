"""
Curved airway primitive generation utilities.

Provides a curved, possibly tapered pipe mesh following an arbitrary 3D
centreline polyline in world coordinates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .frames import Frame, unit


def _stable_frame_axes(tangent: np.ndarray, prev_x: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build orthonormal x, y axes given a z-axis (tangent), reusing prev_x when possible
    to minimize twist.
    """
    z_axis = tangent
    if prev_x is not None:
        x_proj = prev_x - np.dot(prev_x, z_axis) * z_axis
        if np.linalg.norm(x_proj) > 1e-8:
            x_axis = unit(x_proj)
            y_axis = unit(np.cross(z_axis, x_axis))
            return x_axis, y_axis

    ref_candidates = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    ref = None
    for cand in ref_candidates:
        if abs(np.dot(cand, z_axis)) < 0.9:
            ref = cand
            break
    if ref is None:
        ref = np.array([1.0, 0.0, 0.0])

    x_proj = ref - np.dot(ref, z_axis) * z_axis
    x_axis = unit(x_proj)
    y_axis = unit(np.cross(z_axis, x_axis))
    return x_axis, y_axis


def make_curved_pipe(
    centerline: np.ndarray,
    d_in: float,
    d_out: float,
    n_theta: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (possibly tapered) curved pipe surface following a 3D centreline.

    Parameters
    ----------
    centerline : (M, 3) float
        Polyline of points describing the centreline in world coordinates.
        Assumed to be ordered from inlet to outlet.
    d_in : float
        Diameter at the first centreline point.
    d_out : float
        Diameter at the last centreline point.
    n_theta : int
        Number of angular samples per ring.

    Returns
    -------
    points : (N, 3) float
        Vertex positions in world coordinates.
    faces : (K, 3) int
        Triangle indices.
    """
    pts = np.asarray(centerline, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("centerline must have shape (M,3)")
    if pts.shape[0] < 2:
        raise ValueError("centerline must contain at least 2 points")
    if d_in <= 0 or d_out <= 0:
        raise ValueError("diameters must be positive")
    if n_theta < 3:
        raise ValueError("n_theta must be at least 3")

    n_pts = pts.shape[0]
    tangents = []
    for i in range(n_pts):
        if i < n_pts - 1:
            diff = pts[i + 1] - pts[i]
        else:
            diff = pts[i] - pts[i - 1]
        if np.linalg.norm(diff) < 1e-12:
            raise ValueError("Degenerate centerline segment detected")
        tangents.append(unit(diff))
    tangents = np.vstack(tangents)

    diameters = np.linspace(d_in, d_out, n_pts)
    radii = diameters / 2.0

    frames = []
    prev_x = None
    for p, t in zip(pts, tangents):
        x_axis, y_axis = _stable_frame_axes(t, prev_x=prev_x)
        R = np.column_stack((x_axis, y_axis, t))
        frames.append(Frame(origin=p, R=R))
        prev_x = x_axis

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ring_points = []
    for frame, r in zip(frames, radii):
        local_ring = np.column_stack((r * cos_t, r * sin_t, np.zeros_like(theta)))
        ring_points.append(frame.to_world(local_ring))
    points = np.vstack(ring_points)

    faces = []
    n_rings = n_pts
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

    return points.astype(float), np.asarray(faces, dtype=int)


if __name__ == "__main__":
    # Quarter-circle arc in x-z plane, radius 5 cm.
    radius = 5.0
    n_samples = 20
    angles = np.linspace(0.0, np.pi / 2.0, n_samples)
    centerline = np.column_stack(
        (radius * np.sin(angles), np.zeros_like(angles), radius * np.cos(angles))
    )

    pts, fcs = make_curved_pipe(centerline=centerline, d_in=2.0, d_out=1.0, n_theta=24)
    print("Curved pipe mesh:")
    print(" points:", len(pts))
    print(" faces:", len(fcs))
