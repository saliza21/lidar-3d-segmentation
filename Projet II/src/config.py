import os

# ─── Racines ────────────────────────────────────────────────────────────────

DATA_ROOT = os.getenv(
    "LIDAR_DATA",
    r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\donnee"
)

SRC_ROOT = os.getenv(
    "LIDAR_SRC",
    r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\src"
)

# MODEL_ROOT = os.getenv(
#     "LIDAR_MODELS",
#     r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\SMPL\models"
# )

# ─── Données brutes ──────────────────────────────────────────────────────────

PCAP_DIR       = os.path.join(DATA_ROOT, "PCAP")
PLY_FRAMES_DIR = os.path.join(PCAP_DIR, "ply_frames")
NETT_DIR       = os.path.join(PCAP_DIR, "nett")
RANSAC_DIR     = os.path.join(PCAP_DIR, "ransac")
DBSCAN_DIR     = os.path.join(PCAP_DIR, "dbscan")
SLICE_DIR     = os.path.join(PCAP_DIR, "slice_scene")
RGB_FRAMES_DIR = os.path.join(PCAP_DIR, "rgb_frames")
RGB_MP_DIR     = os.path.join(PCAP_DIR, "rgb_MediaPipe")

# ─── Sorties ─────────────────────────────────────────────────────────────────

JOINTS_JSON_DIR      = os.path.join(DATA_ROOT, "joints_json")
MORPHO_OUT_DIR       = os.path.join(DATA_ROOT, "morpho_skeleton_output")
OUT_VIDEO            = os.path.join(PCAP_DIR, "human_cluster.mp4")

# ─── Fichiers spécifiques ────────────────────────────────────────────────────

POSE_CSV       = os.path.join(RGB_MP_DIR, "pose_by_frame.csv")
TASK_PATH      = os.path.join(SRC_ROOT, "squelette", "landmarker", "pose_landmarker_lite.task")

# ─── Exemple de frame (pour les tests) ───────────────────────────────────────

PLY_PATH_EXAMPLE = os.path.join(RANSAC_DIR, "frame_0003_clean_slice.ply")
