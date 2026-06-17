import pyvista as pv

from respgeomlib.primitives import cylinder_polydata


def main():
    try:
        mesh = cylinder_polydata(length=10.0, d_in=4.0, d_out=2.0, n_theta=32, n_z=16, cap_ends=True)
    except RuntimeError:
        print("PyVista not available, cannot show cylinder.")
        return

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, smooth_shading=True, show_edges=False)
    plotter.set_background("lightgray")
    plotter.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error while visualizing cylinder:", e)
