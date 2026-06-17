from pathlib import Path

import trimesh


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "assets" / "figures"
DST_DIR = ROOT / "assets" / "models"

MODELS = {
    "airway_closed_validation.stl": (
        "airway_baseline.glb",
        [116, 201, 194, 255],
    ),
    "airway_closed_validation_stenosis.stl": (
        "airway_stenosis.glb",
        [111, 197, 190, 255],
    ),
    "airway_closed_validation_dilation.stl": (
        "airway_dilation.glb",
        [126, 208, 202, 255],
    ),
}


def _as_mesh(loaded):
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            return None
        return trimesh.util.concatenate(geometries)
    return loaded


def clean_and_center(mesh):
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())

    mesh.remove_unreferenced_vertices()

    if mesh.bounds is not None:
        centroid = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-centroid)

    return mesh


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for stl_name, (glb_name, rgba) in MODELS.items():
        stl_path = SRC_DIR / stl_name
        if not stl_path.exists():
            print(f"[WARN] Missing {stl_path}")
            continue

        mesh = _as_mesh(trimesh.load_mesh(stl_path, process=True))
        if mesh is None:
            print(f"[WARN] No geometry found in {stl_path}")
            continue

        mesh = clean_and_center(mesh)
        mesh.visual.face_colors = rgba

        glb_path = DST_DIR / glb_name
        mesh.export(glb_path)
        print(f"[OK] Exported {glb_path}")


if __name__ == "__main__":
    main()
