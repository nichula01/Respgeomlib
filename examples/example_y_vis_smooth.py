import numpy as np
import pyvista as pv

from respgeomlib.junctions import two_way_y_polydata


def main():
    # Y is in local coordinates: junction at origin, trunk along -z.
    length_trunk = 6.0
    length_child1 = 4.0
    length_child2 = 4.0
    d_trunk = d_child1 = d_child2 = 2.0
    theta1_deg, phi1_deg = 45.0, 0.0
    theta2_deg, phi2_deg = 45.0, 120.0

    mesh_sharp = two_way_y_polydata(
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
        blend_length=0.0,
    )

    mesh_smooth = two_way_y_polydata(
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
        blend_length=2.0,
    )

    print("Sharp Y:", mesh_sharp.n_points, "points,", mesh_sharp.n_cells, "cells")
    print("Smooth Y:", mesh_smooth.n_points, "points,", mesh_smooth.n_cells, "cells")

    # blend_length controls how far the child directions blend, smoothing the junction.
    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Sharp (blend_length=0)", font_size=12)
    plotter.add_mesh(
        mesh_sharp,
        color="lightgray",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()

    plotter.subplot(0, 1)
    plotter.add_text("Smoothed (blend_length=2)", font_size=12)
    plotter.add_mesh(
        mesh_smooth,
        color="lightblue",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while visualizing sharp vs smoothed Y:", e)
