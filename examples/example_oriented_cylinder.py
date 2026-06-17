import numpy as np
import pyvista as pv

from respgeomlib.frames import Frame, make_child_frame
from respgeomlib.primitives import make_cylinder_local


def build_polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(points, faces_flat)


def main():
    # Parent cylinder sits at the origin aligned with +z in the world frame.
    parent = Frame(origin=np.zeros(3), R=np.eye(3))
    length_parent = 8.0
    d_parent = 3.0
    points_p_local, faces_p = make_cylinder_local(length_parent, d_parent, d_parent, cap_ends=True)
    points_p_world = parent.to_world(points_p_local)

    # Child cylinder is rotated via make_child_frame and mapped to world coords.
    child = make_child_frame(parent, theta_deg=45.0, phi_deg=30.0)
    length_child = 6.0
    d_child = 2.0
    points_c_local, faces_c = make_cylinder_local(length_child, d_child, d_child, cap_ends=True)
    points_c_world = child.to_world(points_c_local)

    mesh_parent = build_polydata(points_p_world, faces_p)
    mesh_child = build_polydata(points_c_world, faces_c)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh_parent, color="lightgray", smooth_shading=True, show_edges=False)
    plotter.add_mesh(mesh_child, color="lightblue", smooth_shading=True, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.set_background("white")
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while visualizing oriented cylinders:", e)
