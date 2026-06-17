"""
3D frame utilities for respiratory geometry pipelines.

Provides a lightweight Frame dataclass, helpers to build direction vectors from
elevation/azimuth angles, and a minimal-twist child-frame constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def unit(v: Iterable[float] | np.ndarray) -> np.ndarray:
    """
    Normalize a 3D vector.

    Raises:
        ValueError: If the vector norm is too small to normalize.
    """
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError("Input must be a length-3 vector")
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return arr / norm


def direction_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    """
    Convert elevation/azimuth angles (degrees) to a unit direction vector.

    Args:
        theta_deg: Elevation from +z (0 along +z, 90 in x–y plane).
        phi_deg: Azimuth in x–y from +x, counter-clockwise looking from +z.
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return unit(np.array([x, y, z], dtype=float))


@dataclass
class Frame:
    """A 3D coordinate frame defined by an origin and rotation matrix."""

    origin: np.ndarray  # shape (3,)
    R: np.ndarray  # shape (3, 3); columns are x, y, z axes in world coords

    def __post_init__(self) -> None:
        origin_arr = np.asarray(self.origin, dtype=float).reshape(-1)
        R_arr = np.asarray(self.R, dtype=float)
        if origin_arr.shape[0] != 3:
            raise ValueError("origin must be length 3")
        if R_arr.shape != (3, 3):
            raise ValueError("R must be a 3x3 matrix")
        self.origin = origin_arr
        self.R = R_arr

    @property
    def x(self) -> np.ndarray:
        return self.R[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.R[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.R[:, 2]

    def to_world(self, pts_local: np.ndarray) -> np.ndarray:
        """
        Transform points from this frame into world coordinates.

        Args:
            pts_local: Shape (N,3) or (3,) array of local coordinates.
        """
        pts = np.asarray(pts_local, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[1] != 3:
            raise ValueError("pts_local must have shape (N,3)")
        return pts @ self.R.T + self.origin


def make_child_frame(parent: Frame, theta_deg: float, phi_deg: float) -> Frame:
    """
    Create a child frame with minimal twist relative to a parent frame.

    The (theta_deg, phi_deg) angles define the child z-axis direction expressed
    in the parent frame (elevation from +z, azimuth from +x in the x–y plane).
    The resulting child frame is returned in world coordinates; its origin is
    the same as the parent's, its z-axis follows the requested direction, and
    its x-axis is the closest orthogonal direction to the parent's x-axis
    (falling back to the parent's y-axis if needed), with y completing a
    right-handed orthonormal basis.
    """
    z_child_parent = direction_from_angles(theta_deg, phi_deg)
    z_child_world = unit(parent.R @ z_child_parent)

    x_parent_world = parent.x
    x_proj = x_parent_world - np.dot(x_parent_world, z_child_world) * z_child_world
    if np.linalg.norm(x_proj) < 1e-8:
        y_parent_world = parent.y
        x_proj = y_parent_world - np.dot(y_parent_world, z_child_world) * z_child_world
    x_child_world = unit(x_proj)

    y_child_world = unit(np.cross(z_child_world, x_child_world))

    R_child_world = np.column_stack((x_child_world, y_child_world, z_child_world))
    origin_child_world = parent.origin.copy()

    return Frame(origin_child_world, R_child_world)


if __name__ == "__main__":
    parent = Frame(origin=np.zeros(3), R=np.eye(3))

    tests = [
        (0.0, 0.0, "+z expected"),
        (90.0, 0.0, "+x expected"),
        (90.0, 90.0, "+y expected"),
    ]

    for theta_deg, phi_deg, label in tests:
        child = make_child_frame(parent, theta_deg, phi_deg)
        print(f"Test {label}: theta={theta_deg}, phi={phi_deg}")
        print(" child z:", child.z)
        print(" dot(x,z):", np.dot(child.x, child.z))
        print(" dot(y,z):", np.dot(child.y, child.z))
        print(" dot(x,y):", np.dot(child.x, child.y))
        print(" orthonormal:",
              np.allclose(np.dot(child.x, child.z), 0.0, atol=1e-7)
              and np.allclose(np.dot(child.y, child.z), 0.0, atol=1e-7)
              and np.allclose(np.dot(child.x, child.y), 0.0, atol=1e-7))
        print("---")
