import cv2
import numpy as np
import mediapipe as mp

# =========================================================
# 0) CONFIG — adapte ici à ton cas
# =========================================================

CAM_INDEX = 0            # 0 ou 1 selon webcam
MIN_VIS = 0.60           # seuil de confiance (0.0 -> 1.0). Monte si ça "bug"
SMOOTH_ALPHA = 0.7       # lissage (0=pas de lissage, 0.7=stable)




# Articulations "principales" (MediaPipe Pose = 33 landmarks)
# On garde: tête(nez), épaules, coudes, poignets, index (main), hanches, genoux, chevilles
KEEP_LANDMARKS = {
    "NOSE": mp.solutions.pose.PoseLandmark.NOSE,
    "L_SHOULDER": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "R_SHOULDER": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "L_ELBOW": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "R_ELBOW": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "L_WRIST": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "R_WRIST": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "L_INDEX": mp.solutions.pose.PoseLandmark.LEFT_INDEX,   # proxy "main"
    "R_INDEX": mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
    "L_HIP": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "R_HIP": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "L_KNEE": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "R_KNEE": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "L_ANKLE": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "R_ANKLE": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
}

# Connexions utiles (segments) pour visualiser ton "bonhomme"
# (tu peux en enlever/ajouter selon ton besoin)
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
    ("L_HIP", "R_HIP"),  # bassin
]

# Couleurs BGR (OpenCV)
COLOR_LEFT = (255, 0, 0)    # bleu
COLOR_RIGHT = (0, 0, 255)   # rouge
COLOR_CENTER = (0, 255, 0)  # vert (tête/bassin)

# =========================================================
# 1) OUTILS — angles, lissage, etc.
# =========================================================

def angle_abc(a, b, c):
    """
    Calcule l'angle (en degrés) au point b, formé par a-b-c.
    a, b, c sont des points 2D (x,y) en pixels.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def is_left(name):   # simple règle
    return name.startswith("L_")

def is_right(name):
    return name.startswith("R_")

def choose_color(joint_name):
    if is_left(joint_name):
        return COLOR_LEFT
    if is_right(joint_name):
        return COLOR_RIGHT
    return COLOR_CENTER

def ema_smooth(prev_xy, new_xy, alpha):
    """Lissage exponentiel (anti-tremblement)."""
    if prev_xy is None:
        return new_xy
    return (alpha * np.array(prev_xy) + (1 - alpha) * np.array(new_xy)).tolist()

# =========================================================
# 2) PIPELINE — Pose estimation (MediaPipe Solutions Pose)
# =========================================================
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(CAM_INDEX)

cv2.namedWindow("Skeleton (MediaPipe)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Skeleton (MediaPipe)", 1200, 800)

if not cap.isOpened():
    raise RuntimeError("Webcam introuvable. Essaie CAM_INDEX=1.")

# Pour le lissage: on stocke la dernière position de chaque joint
prev_points = {name: None for name in KEEP_LANDMARKS.keys()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # MediaPipe attend du RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Dictionnaire joints => (x_px, y_px, visibility)
    points = {}

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ---- 2.1 Extraire uniquement les points "KEEP_LANDMARKS" ----
        for name, idx in KEEP_LANDMARKS.items():
            p = lm[idx]
            vis = float(p.visibility) if hasattr(p, "visibility") else 1.0

            # Conversion coords normalisées -> pixels
            x_px = int(p.x * w)
            y_px = int(p.y * h)

            # Filtre "confidence"
            if vis < MIN_VIS:
                points[name] = None
                continue

            # Lissage
            smoothed = ema_smooth(prev_points[name], [x_px, y_px], SMOOTH_ALPHA)
            prev_points[name] = smoothed
            points[name] = (int(smoothed[0]), int(smoothed[1]))

        # ---- 2.2 Dessiner les segments (EDGES) ----
        for a, b in EDGES:
            pa, pb = points.get(a), points.get(b)
            if pa is None or pb is None:
                continue
            # Couleur selon le côté (si segment gauche -> bleu, droite -> rouge, sinon vert)
            col = COLOR_CENTER
            if is_left(a) and is_left(b):
                col = COLOR_LEFT
            elif is_right(a) and is_right(b):
                col = COLOR_RIGHT
            cv2.line(frame, pa, pb, col, 3)

        # ---- 2.3 Dessiner les joints (points) ----
        for name, pt in points.items():
            if pt is None:
                continue
            cv2.circle(frame, pt, 6, choose_color(name), -1)

        # ---- 2.4 Exemple: calcul d'angles (coude + genou) ----
        # Coude gauche: épaule - coude - poignet
        if points["L_SHOULDER"] and points["L_ELBOW"] and points["L_WRIST"]:
            a = angle_abc(points["L_SHOULDER"], points["L_ELBOW"], points["L_WRIST"])
            if a is not None:
                cv2.putText(frame, f"L_elbow: {a:.0f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LEFT, 2)

        # Coude droit
        if points["R_SHOULDER"] and points["R_ELBOW"] and points["R_WRIST"]:
            a = angle_abc(points["R_SHOULDER"], points["R_ELBOW"], points["R_WRIST"])
            if a is not None:
                cv2.putText(frame, f"R_elbow: {a:.0f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RIGHT, 2)

        # Genou gauche: hanche - genou - cheville
        if points["L_HIP"] and points["L_KNEE"] and points["L_ANKLE"]:
            a = angle_abc(points["L_HIP"], points["L_KNEE"], points["L_ANKLE"])
            if a is not None:
                cv2.putText(frame, f"L_knee: {a:.0f} deg", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LEFT, 2)

        # Genou droit
        if points["R_HIP"] and points["R_KNEE"] and points["R_ANKLE"]:
            a = angle_abc(points["R_HIP"], points["R_KNEE"], points["R_ANKLE"])
            if a is not None:
                cv2.putText(frame, f"R_knee: {a:.0f} deg", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RIGHT, 2)

    cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Skeleton (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
