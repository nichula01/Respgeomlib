import numpy as np
import pyvista as pv

from respgeomlib.frames import Frame, unit
from respgeomlib.segments import (
    SegmentGeom,
    Port,
    build_pipe_segment,
    build_y2_segment,
    build_y3_segment,
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


def segment_to_world(seg: SegmentGeom, frame: Frame):
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
        dir_w = frame.R @ port.direction
        ports_world.append(Port(position=pos_w, direction=dir_w))

    return pts_world, seg.faces.copy(), ports_world


def main():
    # ----------------------------
    # 1) Root pipe segment
    # ----------------------------
    root_seg = build_pipe_segment(length=6.0, d_in=2.0, d_out=2.0)

    # World frame: origin at (0,0,0), local +z goes along -global z
    root_origin = np.array([0.0, 0.0, 0.0])
    root_z_world = np.array([0.0, 0.0, -1.0])
    root_frame = frame_from_origin_and_z(root_origin, root_z_world)

    root_pts_w, root_faces, root_ports_w = segment_to_world(root_seg, root_frame)

    # ----------------------------
    # 2) 2-way Y attached to root outlet
    # ----------------------------
    # Root ports: 0 = parent (inlet), 1 = child (outlet)
    root_out_port = root_ports_w[1]

    # Direction along the path away from the root:
    conn_origin_y2 = root_out_port.position
    conn_dir_y2 = -root_out_port.direction

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

    y2_frame = frame_from_origin_and_z(conn_origin_y2, conn_dir_y2)
    y2_pts_w, y2_faces, y2_ports_w = segment_to_world(y2_seg, y2_frame)

    # ----------------------------
    # 3) 3-way Y attached to one child branch of Y2
    # ----------------------------
    # Y2 ports: index 0 = parent (trunk inlet), 1 and 2 = children.
    # Let's attach the 3-way Y to child port 1.
    y2_child_port = y2_ports_w[1]

    conn_origin_y3 = y2_child_port.position
    # Again, we want to go "downstream", so take the opposite of the into-segment direction.
    conn_dir_y3 = -y2_child_port.direction

    y3_seg = build_y3_segment(
        length_trunk=4.0,
        length_child1=3.0,
        length_child2=3.0,
        length_child3=3.0,
        d_trunk=2.0,
        d_child1=2.0,
        d_child2=2.0,
        d_child3=2.0,
        theta1_deg=45.0,
        phi1_deg=0.0,
        theta2_deg=45.0,
        phi2_deg=120.0,
        theta3_deg=45.0,
        phi3_deg=240.0,
    )

    y3_frame = frame_from_origin_and_z(conn_origin_y3, conn_dir_y3)
    y3_pts_w, y3_faces, y3_ports_w = segment_to_world(y3_seg, y3_frame)

    # ----------------------------
    # 4) Merge all geometry
    # ----------------------------
    all_points = []
    all_faces = []
    offset = 0

    # Root
    all_points.append(root_pts_w)
    all_faces.append(root_faces + offset)
    offset += root_pts_w.shape[0]

    # Y2
    all_points.append(y2_pts_w)
    all_faces.append(y2_faces + offset)
    offset += y2_pts_w.shape[0]

    # Y3
    all_points.append(y3_pts_w)
    all_faces.append(y3_faces + offset)

    points_merged = np.vstack(all_points)
    faces_merged = np.vstack(all_faces)

    # ----------------------------
    # 5) Visualize
    # ----------------------------
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_text("Pipe -> Y2 -> Y3 airway tree", font_size=12)

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
        print("Error while building or visualizing Pipe -> Y2 -> Y3 tree:", e)
