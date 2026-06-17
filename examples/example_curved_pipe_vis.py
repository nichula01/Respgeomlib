import numpy as np
import pyvista as pv

from respgeomlib.curved_primitives import make_curved_pipe


def build_quarter_circle_centerline(radius: float = 5.0, n_samples: int = 40) -> np.ndarray:
    angles = np.linspace(0.0, np.pi / 2.0, n_samples)
    return np.column_stack(
        (radius * np.sin(angles), np.zeros_like(angles), radius * np.cos(angles))
    )


def main() -> None:
    centerline = build_quarter_circle_centerline()
    points, faces = make_curved_pipe(centerline=centerline, d_in=1.0, d_out=0.5)

    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=int), faces]).ravel()
    mesh = pv.PolyData(points, faces_flat)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text("Curved tapered pipe", font_size=12)
    plotter.add_mesh(mesh, color="lightcoral", smooth_shading=True, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


if __name__ == "__main__":
    main()
