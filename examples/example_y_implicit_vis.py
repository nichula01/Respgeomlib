import numpy as np
import pyvista as pv

from respgeomlib.junctions import two_way_y_polydata
from respgeomlib.implicit_y import two_way_y_implicit_polydata


def main():
    # Both meshes are in local coordinates: junction at origin, trunk along -z.
    length_trunk = 6.0
    length_child1 = 4.0
    length_child2 = 4.0
    d_trunk = d_child1 = d_child2 = 2.0
    theta1_deg, phi1_deg = 45.0, 0.0
    theta2_deg, phi2_deg = 45.0, 120.0
    blend_length = 2.0  # for tube-based Y centerline smoothing

    mesh_tube = two_way_y_polydata(
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
        blend_length=blend_length,
    )

    # Implicit version uses a distance-field union and marching cubes for smoother blending.
    mesh_imp = two_way_y_implicit_polydata(
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
        blend_length=blend_length,
    )

    print("Tube-based Y:", mesh_tube.n_points, "points,", mesh_tube.n_cells, "cells")
    print("Implicit Y:", mesh_imp.n_points, "points,", mesh_imp.n_cells, "cells")
    print("Tube bounds:", mesh_tube.bounds)
    print("Implicit bounds:", mesh_imp.bounds)

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background("white")

    plotter.subplot(0, 0)
    plotter.add_text("Tube-based Y", font_size=12)
    plotter.add_mesh(
        mesh_tube,
        color="lightgray",
        smooth_shading=True,
        show_edges=True,
    )
    plotter.add_axes()
    plotter.show_grid()

    plotter.subplot(0, 1)
    plotter.add_text("Implicit blended Y", font_size=12)
    plotter.add_mesh(
        mesh_imp,
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
        print("Error while visualizing tube vs implicit Y:", e)
