import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DBSCAN_DIR
import open3d as o3d
import numpy as np
import os
import glob

# =========================
# PARAMÈTRES
# =========================

ply_folder = DBSCAN_DIR
ply_files = sorted(glob.glob(os.path.join(ply_folder, "*.ply")))

point_names = [
    "tete",
    "epaule_gauche",
    "epaule_droite",
    "coude_gauche",
    "coude_droit",
    "poignet_gauche",
    "poignet_droit",
    "torse",
    "hanche_gauche",
    "hanche_droite",
    "genou_gauche",
    "genou_droit",
    "pied_gauche",
    "pied_droit"
]

skeleton_links = [
    ("tete", "torse"),
    ("torse", "epaule_gauche"),
    ("torse", "epaule_droite"),
    ("epaule_gauche", "coude_gauche"),
    ("coude_gauche", "poignet_gauche"),
    ("epaule_droite", "coude_droit"),
    ("coude_droit", "poignet_droit"),
    ("torse", "hanche_gauche"),
    ("hanche_gauche", "genou_gauche"),
    ("genou_gauche", "pied_gauche"),
    ("torse", "hanche_droite"),
    ("hanche_droite", "genou_droit"),
    ("genou_droit", "pied_droit")
]

search_radius = 0.15  # rayon de recherche en mètres


# =========================
# SÉLECTION MANUELLE
# =========================

def pick_points(pcd):
    print("\nSélectionne les points dans cet ordre :")
    for name in point_names:
        print("-", name)

    print("""
Instructions Open3D :
1. Shift + clic gauche pour sélectionner un point
2. Shift + clic droit pour annuler le dernier point
3. Ferme la fenêtre quand tu as fini
""")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Selection points squelette")
    vis.add_geometry(pcd)
    vis.run()

    picked_ids = vis.get_picked_points()
    vis.destroy_window()

    # Supprimer les doublons en gardant l'ordre
    unique_picked_ids = []
    for idx in picked_ids:
        if idx not in unique_picked_ids:
            unique_picked_ids.append(idx)

    print("Points sélectionnés bruts :", len(picked_ids))
    print("Points après suppression doublons :", len(unique_picked_ids))
    print("Indices :", unique_picked_ids)

    if len(unique_picked_ids) > len(point_names):
        print("Trop de points sélectionnés. Je garde seulement les 14 premiers.")
        unique_picked_ids = unique_picked_ids[:len(point_names)]

    if len(unique_picked_ids) != len(point_names):
        raise ValueError(
            f"Tu as sélectionné {len(unique_picked_ids)} points uniques, mais il faut {len(point_names)} points."
        )

    points = np.asarray(pcd.points)
    selected_3d_points = {}

    for i, idx in enumerate(unique_picked_ids):
        selected_3d_points[point_names[i]] = points[idx]

    return selected_3d_points


def create_skeleton_geometry(points_dict):
    geometries = []

    # Points rouges
    sphere_radius = 0.03

    for name, point in points_dict.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)

    # Lignes bleues
    line_points = []
    lines = []

    for p1, p2 in skeleton_links:
        line_points.append(points_dict[p1])
        line_points.append(points_dict[p2])

        start_idx = len(line_points) - 2
        end_idx = len(line_points) - 1
        lines.append([start_idx, end_idx])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])

    geometries.append(line_set)

    return geometries


# =========================
# TRACKING 3D
# =========================

def track_points(previous_points_dict, current_pcd, radius=0.15):
    current_points = np.asarray(current_pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(current_pcd)

    new_points_dict = {}

    for name, old_point in previous_points_dict.items():
        k, idx, dist = pcd_tree.search_radius_vector_3d(old_point, radius)

        if k > 0:
            candidates = current_points[idx]

            distances = np.linalg.norm(candidates - old_point, axis=1)
            best_candidate = candidates[np.argmin(distances)]

            new_points_dict[name] = best_candidate
        else:
            # Si aucun voisin trouvé, on garde l'ancien point
            new_points_dict[name] = old_point
            print(f"Attention : point perdu -> {name}")

    return new_points_dict


# =========================
# PROGRAMME PRINCIPAL
# =========================

if len(ply_files) == 0:
    raise FileNotFoundError("Aucun fichier .ply trouvé.")

first_pcd = o3d.io.read_point_cloud(ply_files[0])

# 1. Sélection manuelle sur la première frame
tracked_points = pick_points(first_pcd)

# 2. Sauvegarde des points initiaux
np.save("skeleton_points_initial.npy", tracked_points)

print("\nPoints sélectionnés :")
for name, point in tracked_points.items():
    print(name, point)

# 3. Tracking sur les frames suivantes
for frame_id, ply_file in enumerate(ply_files):
    print(f"Frame {frame_id} : {ply_file}")

    pcd = o3d.io.read_point_cloud(ply_file)

    if frame_id > 0:
        tracked_points = track_points(
            previous_points_dict=tracked_points,
            current_pcd=pcd,
            radius=search_radius
        )

    skeleton_geometry = create_skeleton_geometry(tracked_points)

    o3d.visualization.draw_geometries(
        [pcd] + skeleton_geometry,
        window_name=f"Tracking squelette 3D - Frame {frame_id}"
    )