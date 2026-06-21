import os
import sys
import glob
import time
import json
import numpy as np
import open3d as o3d
import cv2

# Pour permettre l'import depuis src/preprocessing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.OrientationExporter import OrientationExporter
from skeleton_def import JOINT_NAMES, EDGES, SPHERE_RADIUS

SKELETON_COLOR = (1.0, 0.15, 0.15)
JOINT_COLOR    = (0.0, 1.0, 0.3)
from config import SLICE_DIR,RANSAC_DIR, DBSCAN_DIR, JOINTS_JSON_DIR, OUT_VIDEO

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES SQUELETTE
# ─────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────
# HELPERS SQUELETTE
# ─────────────────────────────────────────────────────────────────────────────

def load_joints_from_json(json_path: str) -> dict:
    """
    Charge un fichier JSON de joints produit par l1_skeleton.save_joints().
    Retourne { nom -> np.ndarray(3,) | None }.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    joints = {}
    for name, val in data.items():
        joints[name] = np.array(val, dtype=np.float64) if val is not None else None
    return joints


def joints_json_path(ply_path: str, joints_dir: str) -> str:
    """
    Construit le chemin du JSON de joints correspondant à un PLY.
    Ex: frame_0040_clean_slice.ply → joints_dir/frame_0040_clean_slice_joints.json
    """
    base = os.path.splitext(os.path.basename(ply_path))[0]
    return os.path.join(joints_dir, base + "_joints.json")


def build_skeleton_lineset(
    joints: dict,
    color: tuple = SKELETON_COLOR,
) -> o3d.geometry.LineSet:
    """Construit le LineSet Open3D du squelette à partir des joints."""
    pts, name_to_i = [], {}
    for name in JOINT_NAMES:
        p = joints.get(name)
        if p is None:
            continue
        name_to_i[name] = len(pts)
        pts.append(p)

    if len(pts) < 2:
        return None

    lines = [
        [name_to_i[a], name_to_i[b]]
        for a, b in EDGES
        if a in name_to_i and b in name_to_i
    ]
    if not lines:
        return None

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([color]), (len(lines), 1))
    )
    return ls


def build_joint_spheres(
    joints: dict,
    radius: float = SPHERE_RADIUS,
    color: tuple = JOINT_COLOR,
) -> list:
    """Construit une liste de sphères Open3D pour chaque articulation."""
    spheres = []
    for name in JOINT_NAMES:
        p = joints.get(name)
        if p is None:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(p.astype(float))
        s.paint_uniform_color(list(color))
        spheres.append(s)
    return spheres


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class CloudFramesVis:

    def __init__(
        self,
        ply_dir: str,
        pattern: str = "*.ply",
        width: int = 1280,
        height: int = 720,
        use_saved_orientation: bool = True,
        joints_dir: str = None,         # dossier contenant les JSON L1
        show_skeleton: bool = True,     # afficher le squelette L1
        show_joint_spheres: bool = True,
    ):
        self.PLY_DIR  = ply_dir
        self.PATTERN  = pattern
        self.W        = width
        self.H        = height
        self.use_saved_orientation = use_saved_orientation
        self.joints_dir        = joints_dir
        self.show_skeleton     = show_skeleton and (joints_dir is not None)
        self.show_joint_spheres = show_joint_spheres

        orientation_dir = ply_dir if os.path.isdir(ply_dir) else os.path.dirname(ply_dir)
        self.orientation = OrientationExporter(
            ply_dir=orientation_dir,
            pattern=pattern,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNES
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ply_files(self):
        ply_files = sorted(glob.glob(os.path.join(self.PLY_DIR, self.PATTERN)))
        if not ply_files:
            raise RuntimeError(f"Aucun PLY trouvé : {os.path.join(self.PLY_DIR, self.PATTERN)}")
        return ply_files

    def _apply_orientation(self, vis, ply_path: str):
        if self.use_saved_orientation:
            self.orientation.apply_to_visualizer(vis, ply_path)

    def _load_skeleton(self, ply_path: str):
        """Charge les joints L1 correspondant à ce PLY. Retourne None si absent."""
        if not self.show_skeleton:
            return None
        jpath = joints_json_path(ply_path, self.joints_dir)
        if not os.path.exists(jpath):
            return None
        return load_joints_from_json(jpath)

    # ─────────────────────────────────────────────────────────────────────────
    # AFFICHER UNE SEULE FRAME
    # ─────────────────────────────────────────────────────────────────────────

    def visua_frame(self):
        ply_path = self.PLY_DIR   # mode frame unique : PLY_DIR est un fichier
        pcd = o3d.io.read_point_cloud(ply_path)

        if len(pcd.points) == 0:
            raise RuntimeError(f"PLY vide ou illisible : {ply_path}")

        vis = o3d.visualization.Visualizer()
        vis.create_window("Frame viewer", width=self.W, height=self.H, visible=True)
        vis.add_geometry(pcd)

        # Squelette L1
        joints = self._load_skeleton(ply_path)
        if joints:
            ls = build_skeleton_lineset(joints)
            if ls:
                vis.add_geometry(ls)
            if self.show_joint_spheres:
                for s in build_joint_spheres(joints):
                    vis.add_geometry(s)

        vis.poll_events()
        vis.update_renderer()
        self._apply_orientation(vis, ply_path)

        vis.run()
        vis.destroy_window()

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING (séquence de frames)
    # ─────────────────────────────────────────────────────────────────────────

    def stream_frames(self, sleep_s: float = 0.05):
        ply_files = self._get_ply_files()

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            "Human Cluster Stream",
            width=self.W,
            height=self.H,
            visible=True,
        )

        # ── Première frame ───────────────────────────────────────────────────
        pcd = o3d.io.read_point_cloud(ply_files[0])
        if len(pcd.points) == 0:
            raise RuntimeError(f"Premier PLY vide : {ply_files[0]}")
        vis.add_geometry(pcd)

        # Géométrie squelette : on la garde en mémoire et on met à jour ses points
        skel_ls = o3d.geometry.LineSet()
        vis.add_geometry(skel_ls)

        vis.poll_events()
        vis.update_renderer()
        self._apply_orientation(vis, ply_files[0])

        # ─────────────────────────────────────────────────────────────────────
        # OPTIMISATION : créer les sphères UNE SEULE FOIS
        # Ensuite, à chaque frame, on modifie seulement leur position.
        # Cela évite remove_geometry() + create_sphere() + add_geometry()
        # qui ralentissaient fortement l'affichage.
        # ─────────────────────────────────────────────────────────────────────
        joint_spheres = {}
        sphere_positions = {}

        first_joints = self._load_skeleton(ply_files[0])

        if first_joints and self.show_joint_spheres:
            for name in JOINT_NAMES:
                p = first_joints.get(name)

                if p is None:
                    continue

                s = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
                s.translate(p.astype(float))
                s.paint_uniform_color(list(JOINT_COLOR))

                joint_spheres[name] = s
                sphere_positions[name] = p.copy()

                vis.add_geometry(s, reset_bounding_box=False)

        # ── Frames suivantes ─────────────────────────────────────────────────
        for ply in ply_files:
            new_pcd = o3d.io.read_point_cloud(ply)
            if len(new_pcd.points) == 0:
                print(f"Frame ignorée (vide) : {ply}")
                continue

            # Mise à jour nuage
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)

            # Mise à jour squelette
            joints = self._load_skeleton(ply)
            if joints:
                ls = build_skeleton_lineset(joints)
                if ls:
                    skel_ls.points = ls.points
                    skel_ls.lines  = ls.lines
                    skel_ls.colors = ls.colors
                    vis.update_geometry(skel_ls)

                # ─────────────────────────────────────────────────────────────
                # Mise à jour RAPIDE des sphères
                # On ne recrée pas les sphères : on les déplace seulement.
                # ─────────────────────────────────────────────────────────────
                if self.show_joint_spheres:
                    for name, sphere in joint_spheres.items():
                        p_new = joints.get(name)

                        if p_new is None:
                            continue

                        p_old = sphere_positions[name]
                        delta = p_new - p_old

                        sphere.translate(delta)
                        sphere_positions[name] = p_new.copy()

                        vis.update_geometry(sphere)

            vis.poll_events()
            vis.update_renderer()

            if sleep_s > 0:
                time.sleep(sleep_s)

        vis.destroy_window()

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT MP4
    # ─────────────────────────────────────────────────────────────────────────

    def export_mp4(self, out_video: str, fps: int = 10, sleep_s: float = 0.0):
        ply_files = self._get_ply_files()

        vis = o3d.visualization.Visualizer()
        vis.create_window("capture", width=self.W, height=self.H, visible=True)

        pcd = o3d.io.read_point_cloud(ply_files[0])
        if len(pcd.points) == 0:
            raise RuntimeError(f"Premier PLY vide : {ply_files[0]}")
        vis.add_geometry(pcd)

        skel_ls = o3d.geometry.LineSet()
        vis.add_geometry(skel_ls)

        vis.poll_events()
        vis.update_renderer()
        self._apply_orientation(vis, ply_files[0])
        vis.poll_events()
        vis.update_renderer()

        # ─────────────────────────────────────────────────────────────────────
        # OPTIMISATION : créer les sphères UNE SEULE FOIS pour la vidéo aussi.
        # Ensuite, on les déplace avec translate(delta).
        # ─────────────────────────────────────────────────────────────────────
        joint_spheres = {}
        sphere_positions = {}

        first_joints = self._load_skeleton(ply_files[0])

        if first_joints and self.show_joint_spheres:
            for name in JOINT_NAMES:
                p = first_joints.get(name)

                if p is None:
                    continue

                s = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
                s.translate(p.astype(float))
                s.paint_uniform_color(list(JOINT_COLOR))

                joint_spheres[name] = s
                sphere_positions[name] = p.copy()

                vis.add_geometry(s, reset_bounding_box=False)

        vis.poll_events()
        vis.update_renderer()

        # Capture première frame
        img = self._capture(vis)
        if img is None:
            vis.destroy_window()
            raise RuntimeError("Capture Open3D vide.")

        h, w = img.shape[:2]
        print(f"Taille vidéo : {w}×{h}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video  = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        if not video.isOpened():
            vis.destroy_window()
            raise RuntimeError("VideoWriter n'a pas pu s'ouvrir.")

        video.write(img)

        for ply in ply_files[1:]:
            new_pcd = o3d.io.read_point_cloud(ply)
            if len(new_pcd.points) == 0:
                print(f"Frame ignorée (vide) : {ply}")
                continue

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)

            joints = self._load_skeleton(ply)
            if joints:
                ls = build_skeleton_lineset(joints)
                if ls:
                    skel_ls.points = ls.points
                    skel_ls.lines  = ls.lines
                    skel_ls.colors = ls.colors
                    vis.update_geometry(skel_ls)

                # Mise à jour rapide des sphères
                if self.show_joint_spheres:
                    for name, sphere in joint_spheres.items():
                        p_new = joints.get(name)

                        if p_new is None:
                            continue

                        p_old = sphere_positions[name]
                        delta = p_new - p_old

                        sphere.translate(delta)
                        sphere_positions[name] = p_new.copy()

                        vis.update_geometry(sphere)

            vis.poll_events()
            vis.update_renderer()

            img = self._capture(vis)
            if img is not None:
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                video.write(img)

            if sleep_s > 0:
                time.sleep(sleep_s)

        video.release()
        vis.destroy_window()
        print(f"Vidéo exportée → {out_video}")


    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS INTERNES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _capture(vis) -> np.ndarray:
        """Capture l'écran Open3D et retourne un tableau BGR uint8."""
        img = np.asarray(vis.capture_screen_float_buffer(False))
        if img.size == 0:
            return None
        img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    PLY_FILE  = next(__import__("glob").iglob(__import__("os").path.join(DBSCAN_DIR, "*.ply")), None)
    PLY_DIR   = DBSCAN_DIR
    JOINTS_DIR = JOINTS_JSON_DIR
    OUT_VIDEO_PATH = OUT_VIDEO

    # ── 1) Frame unique avec squelette L1 ────────────────────────────────────
    # app = CloudFramesVis(
    #     PLY_FILE,
    #     use_saved_orientation=False,
    #     joints_dir=JOINTS_DIR,
    # )
    # app.visua_frame()

    # ── 2) Streaming séquence avec squelette L1 ──────────────────────────────
    # app = CloudFramesVis(
    #     PLY_DIR,
    #     pattern="frame_*_clean_slice.ply",
    #     use_saved_orientation=True,
    #     joints_dir=JOINTS_DIR,
    #     show_skeleton=True,
    #     show_joint_spheres=True,
    # )
    # app.stream_frames(sleep_s=0.15)

    # ── 3) Export MP4 avec squelette L1 ──────────────────────────────────────
    app = CloudFramesVis(
         PLY_DIR,
         pattern="frame_*_clean_slice.ply",
         use_saved_orientation=True,
         joints_dir=JOINTS_DIR,
    )
    app.export_mp4(OUT_VIDEO, fps=10,sleep_s = 0.2)

# import os
# import glob
# import json
#
# INPUT_DIR = r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\donnee\joints_json"
# OUTPUT_JSON = r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\donnee\all_frames_joints.json"
#
# FPS = 30
# FRAME_DURATION_MS = int(1000 / FPS)
#
# JOINT_MAPPING = {
#     "HEAD": "NOSE",
#     "L_SHOULDER": "L_SHOULDER",
#     "R_SHOULDER": "R_SHOULDER",
#     "L_ELBOW": "L_ELBOW",
#     "R_ELBOW": "R_ELBOW",
#     "L_WRIST": "L_WRIST",
#     "R_WRIST": "R_WRIST",
#     "L_HIP": "L_HIP",
#     "R_HIP": "R_HIP",
#     "L_KNEE": "L_KNEE",
#     "R_KNEE": "R_KNEE",
#     "L_ANKLE": "L_ANKLE",
#     "R_ANKLE": "R_ANKLE",
# }


# def convert_all_json(input_dir, output_json):
#     json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
#
#     all_frames = []
#
#     for frame_id, json_path in enumerate(json_files):
#         with open(json_path, "r", encoding="utf-8") as f:
#             old_joints = json.load(f)
#
#         new_joints = {}
#
#         for old_name, new_name in JOINT_MAPPING.items():
#             if old_name in old_joints and old_joints[old_name] is not None:
#                 new_joints[new_name] = old_joints[old_name]
#
#         frame_data = {
#             "frame": frame_id,
#             "timestamp_ms": frame_id * FRAME_DURATION_MS,
#             "joints": new_joints
#         }
#
#         all_frames.append(frame_data)
#
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(all_frames, f, indent=2)
#
#     print(f"Conversion terminée : {len(all_frames)} frames")
#     print(f"Fichier sauvegardé : {output_json}")
#
# if __name__ == "__main__":
#     convert_all_json(INPUT_DIR, OUTPUT_JSON)