import numpy as np
import pyvista as pv

from respgeomlib.frames import Frame, make_child_frame
from respgeomlib.primitives import make_cylinder_local


def build_polydata(points, faces) -> pv.PolyData:
    """
    Convert (points, faces) arrays into a PyVista PolyData.
    faces: (M,3) triangle indices.
    """
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(points, faces_flat)


def main():
    parent = Frame(origin=np.zeros(3), R=np.eye(3))

    length_parent = 8.0
    d_parent = 3.0
    pts_p_local, faces_p = make_cylinder_local(
        length_parent, d_parent, d_parent,
        n_theta=48, n_z=24, cap_ends=True
    )
    pts_p_world = parent.to_world(pts_p_local)
    mesh_parent = build_polydata(pts_p_world, faces_p)

    child = make_child_frame(parent, theta_deg=45.0, phi_deg=30.0)

    length_child = 6.0
    d_child = 2.0
    pts_c_local, faces_c = make_cylinder_local(
        length_child, d_child, d_child,
        n_theta=48, n_z=24, cap_ends=True
    )
    pts_c_world = child.to_world(pts_c_local)
    mesh_child = build_polydata(pts_c_world, faces_c)

    mesh_parent_clean = mesh_parent.triangulate().clean()
    mesh_child_clean = mesh_child.triangulate().clean()

    mesh_Y = mesh_parent_clean.boolean_union(
        mesh_child_clean,
        progress_bar=False,
    )

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(
        mesh_Y,
        color="lightblue",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while computing boolean union:", e)
