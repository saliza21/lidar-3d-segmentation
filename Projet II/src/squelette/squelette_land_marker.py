import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import TASK_PATH
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
MODEL_PATH = TASK_PATH
CAM_INDEX = 0
WINDOW_NAME = "PoseLandmarker (Tasks)"

# =========================
# Connexions (33 points MediaPipe Pose)
# Même squelette que POSE_CONNECTIONS, mais sans mp.solutions
# =========================
POSE_CONNECTIONS = {
    "NOSE": 0,

    "L_SHOULDER": 11,
    "R_SHOULDER": 12,

    "L_ELBOW": 13,
    "R_ELBOW": 14,

    "L_WRIST": 15,
    "R_WRIST": 16,

    "L_INDEX": 19,
    "R_INDEX": 20,

    "L_HIP": 23,
    "R_HIP": 24,

    "L_KNEE": 25,
    "R_KNEE": 26,

    "L_ANKLE": 27,
    "R_ANKLE": 28,
}

EDGES = [
    ("NOSE", "L_SHOULDER"),
    ("NOSE", "R_SHOULDER"),

    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("L_WRIST", "L_INDEX"),

    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("R_WRIST", "R_INDEX"),

    ("L_SHOULDER", "L_HIP"),
    ("R_SHOULDER", "R_HIP"),

    ("L_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),

    ("R_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE"),

    ("L_HIP", "R_HIP"),
]


COLOR_LEFT  = (255, 0, 0)    # bleu
COLOR_RIGHT = (0, 0, 255)    # rouge
COLOR_CENTER = (0, 255, 0)  # vert (tête / bassin)

# ---- EMA CONFIG ----
SMOOTH_ALPHA = 0.6     # 0.6-0.85 recommandé (plus haut = plus stable, mais plus lent)
MIN_VIS = 0.6           # ignore les points peu fiables (si 'visibility' existe)

# Mémoire des points par articulation (1 personne, MVP)
prev_points = {name: None for name in POSE_CONNECTIONS.keys()}

def ema(prev, curr, alpha):
    """prev et curr: (x,y) ; retourne (x,y) lissé."""
    if prev is None:
        return curr
    px, py = prev
    cx, cy = curr
    return (alpha * px + (1 - alpha) * cx, alpha * py + (1 - alpha) * cy)

def is_left(name):
    return name.startswith("L_")

def is_right(name):
    return name.startswith("R_")

def color_for_joint(name):
    if is_left(name):
        return COLOR_LEFT
    if is_right(name):
        return COLOR_RIGHT
    return COLOR_CENTER





# =========================
# Callback async
# =========================
latest_result = None

def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# =========================
# Create landmarker
# =========================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# =========================
# OpenCV
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Webcam introuvable. Essaie CAM_INDEX=1")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1200, 800)

timestamp_ms = 0

def draw_pose_essential(frame, pose_landmarks):
    global prev_points
    h, w = frame.shape[:2]

    points = {}

    # 1) Extraire seulement les points essentiels + EMA
    for name, idx in POSE_CONNECTIONS.items():
        lm = pose_landmarks[idx]

        # Si visibility existe, on peut filtrer
        vis = getattr(lm, "visibility", 1.0)
        if vis < MIN_VIS:
            points[name] = None
            continue

        x = lm.x * w
        y = lm.y * h

        # EMA smoothing
        x_s, y_s = ema(prev_points[name], (x, y), SMOOTH_ALPHA)
        prev_points[name] = (x_s, y_s)

        points[name] = (int(x_s), int(y_s))

    # 2) Dessiner les segments essentiels
    for a, b in EDGES:
        pa, pb = points.get(a), points.get(b)
        if pa is None or pb is None:
            continue

        if is_left(a) and is_left(b):
            color = COLOR_LEFT
        elif is_right(a) and is_right(b):
            color = COLOR_RIGHT
        else:
            color = COLOR_CENTER

        cv2.line(frame, pa, pb, color, 3)

    # 3) Dessiner les joints essentiels
    for name, pt in points.items():
        if pt is None:
            continue
        cv2.circle(frame, pt, 6, color_for_joint(name), -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms += 33
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.pose_landmarks:
        draw_pose_essential ( frame, latest_result.pose_landmarks[0] )

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
