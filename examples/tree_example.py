import numpy as np
import pyvista as pv

from respgeomlib.frames import Frame, unit
from respgeomlib.segments import (
    SegmentGeom,
    Port,
    build_pipe_segment,
    build_y2_segment,
)


def frame_from_origin_and_z(origin: np.ndarray, z_world: np.ndarray) -> Frame:
    """
    Build a Frame whose origin is `origin` and whose local +z axis
    aligns with `z_world`, with minimal twist using a fixed global
    x-axis as reference.
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    z = unit(z_world)

    # Use global x as a reference; if it's too parallel to z, use global y.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])

    x_proj = ref - np.dot(ref, z) * z
    x = unit(x_proj)
    y = unit(np.cross(z, x))

    R = np.column_stack((x, y, z))
    return Frame(origin=origin, R=R)


def segment_to_world(seg: SegmentGeom, frame: Frame) -> tuple[np.ndarray, np.ndarray, list[Port]]:
    """
    Transform a segment's points and ports from segment-local coordinates
    into world coordinates using the given Frame.

    Returns
    -------
    points_world : (N, 3)
        Transformed vertices.
    faces : (M, 3)
        Same faces array (unchanged indices).
    ports_world : list of Port
        Ports with position and direction expressed in world coordinates.
    """
    pts_world = frame.to_world(seg.points)

    ports_world: list[Port] = []
    for port in seg.ports:
        pos_w = frame.to_world(port.position)
        # Directions transform by rotation only (no translation).
        dir_w = frame.R @ port.direction
        ports_world.append(Port(position=pos_w, direction=dir_w))

    return pts_world, seg.faces.copy(), ports_world


def main():
    # 1) Build root pipe in its own local frame.
    root_seg = build_pipe_segment(length=6.0, d_in=2.0, d_out=2.0)

    # 2) Choose a world frame for the root:
    #    origin at (0,0,0), local +z pointing "downwards" along -global z.
    #    This means airways go downwards into the chest.
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])
    root_frame = frame_from_origin_and_z(root_origin, root_z_world)

    # 3) Transform root to world and get its world ports.
    root_pts_w, root_faces, root_ports_w = segment_to_world(root_seg, root_frame)

    # 4) Attach a 2-way Y segment to the outlet port of the root pipe.
    #    Root ports: 0 = parent (inlet), 1 = child (outlet).
    root_child_port = root_ports_w[1]

    # For connecting, we want the direction along the path *away* from the root,
    # i.e. opposite to the "into segment" direction of the root outlet port.
    conn_dir = -root_child_port.direction
    conn_origin = root_child_port.position

    # 5) Build a Y2 segment in its own local frame.
    y2_seg = build_y2_segment(
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
    )

    # 6) Place the Y2 segment so that:
    #    - its parent port (at local origin, +z into segment)
    #    - is attached at conn_origin,
    #    - with its local +z aligned to conn_dir.
    y2_frame = frame_from_origin_and_z(conn_origin, conn_dir)
    y2_pts_w, y2_faces, y2_ports_w = segment_to_world(y2_seg, y2_frame)

    # 7) Collect geometry of root + Y2 into lists for later merging.
    all_points = []
    all_faces = []

    # Root segment indices
    offset = 0
    all_points.append(root_pts_w)
    all_faces.append(root_faces + offset)
    offset += root_pts_w.shape[0]

    # Y2 segment indices
    all_points.append(y2_pts_w)
    all_faces.append(y2_faces + offset)

    points_merged = np.vstack(all_points)
    faces_merged = np.vstack(all_faces)

    # 8) Visualize in PyVista.
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text("Pipe -> Y2 airway tree (local primitives)", font_size=12)

    faces_flat = np.hstack(
        [np.full((faces_merged.shape[0], 1), 3, dtype=int), faces_merged]
    ).ravel()
    mesh = pv.PolyData(points_merged, faces_flat)

    plotter.add_mesh(
        mesh,
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
        print("Error while building or visualizing tree:", e)
