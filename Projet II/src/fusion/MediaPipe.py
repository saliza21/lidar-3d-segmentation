import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RGB_FRAMES_DIR, RGB_MP_DIR, POSE_CSV, TASK_PATH
import os
import glob
import csv
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseLandmarkerFrameAnnotator:

    def __init__(self,frames_dir: str,out_dir: str,task_path: str,keep: list,edges: list,out_csv: str,
                score_min: float = 0.20,
                radius: int = 4,
                thickness: int = 2):

        self.frames_dir = frames_dir
        self.out_dir = out_dir
        self.task_path = task_path
        self.keep = keep
        self.edges = edges
        self.out_csv = out_csv
        self.score_min = score_min
        self.radius = radius
        self.thickness = thickness

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)

        PL = mp.solutions.pose.PoseLandmark
        self.idx = {
            "NOSE": PL.NOSE.value,
            "L_SHOULDER": PL.LEFT_SHOULDER.value,
            "R_SHOULDER": PL.RIGHT_SHOULDER.value,
            "L_ELBOW": PL.LEFT_ELBOW.value,
            "R_ELBOW": PL.RIGHT_ELBOW.value,
            "L_WRIST": PL.LEFT_WRIST.value,
            "R_WRIST": PL.RIGHT_WRIST.value,
            "L_HIP": PL.LEFT_HIP.value,
            "R_HIP": PL.RIGHT_HIP.value,
            "L_KNEE": PL.LEFT_KNEE.value,
            "R_KNEE": PL.RIGHT_KNEE.value,
            "L_ANKLE": PL.LEFT_ANKLE.value,
            "R_ANKLE": PL.RIGHT_ANKLE.value,
        }

        base_options = python.BaseOptions(model_asset_path=self.task_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Prépare CSV
        self._init_csv()

    def _init_csv(self):
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "image", "detected", "lm", "x_px", "y_px", "score", "W", "H"])

    def _lm_to_px(self, lm, W, H):
        x = int(np.clip(lm.x, 0.0, 1.0) * (W - 1))
        y = int(np.clip(lm.y, 0.0, 1.0) * (H - 1))
        return x, y

    def _get_points(self, pose_landmarks, W, H):

        pts = {}
        # 1) points "normaux"
        for name in self.keep:
            if name == "MID_HIP":
                continue

            if name not in self.idx:
                continue

            i = self.idx[name]
            lm = pose_landmarks[i]
            score = float ( lm.visibility ) if hasattr ( lm, "visibility" ) else 1.0
            if score < self.score_min:
                continue

            x, y = self._lm_to_px ( lm, W, H )
            pts[name] = (x, y, score)

        # 2) MID_HIP = moyenne L_HIP et R_HIP
        if "MID_HIP" in self.keep:
            if ("L_HIP" in self.idx) and ("R_HIP" in self.idx):
                lmL = pose_landmarks[self.idx["L_HIP"]]
                lmR = pose_landmarks[self.idx["R_HIP"]]

                sL = float ( lmL.visibility ) if hasattr ( lmL, "visibility" ) else 1.0
                sR = float ( lmR.visibility ) if hasattr ( lmR, "visibility" ) else 1.0

                # on exige que les deux hanches soient assez fiables
                if (sL >= self.score_min) and (sR >= self.score_min):
                    xL, yL = self._lm_to_px ( lmL, W, H )
                    xR, yR = self._lm_to_px ( lmR, W, H )

                    xM = int ( (xL + xR) / 2 )
                    yM = int ( (yL + yR) / 2 )
                    sM = float ( min ( sL, sR ) )  # score conservateur

                    pts["MID_HIP"] = (xM, yM, sM)

        return pts

    def _draw(self, img_bgr, pts):
        # segments
        for a, b in self.edges:
            if a in pts and b in pts:
                xa, ya, _ = pts[a]
                xb, yb, _ = pts[b]
                cv2.line(img_bgr, (xa, ya), (xb, yb), (0, 255, 0), self.thickness)

        # points
        for name, (x, y, s) in pts.items():
            cv2.circle(img_bgr, (x, y), self.radius, (0, 0, 255), -1)

        return img_bgr

    def _append_csv_rows(self, frame_idx, image_name, detected, pts, W, H):

        if detected == 0:
            return

        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for lm_name in self.keep:
                if lm_name in pts:
                    x, y, score = pts[lm_name]
                    w.writerow([frame_idx, image_name, 1, lm_name, x, y, score, W, H])

    def run(self, pattern="*.jpg"):
        paths = sorted(glob.glob(os.path.join(self.frames_dir, pattern)))
        if not paths:
            raise RuntimeError(f"Aucune image trouvée dans {self.frames_dir} avec pattern={pattern}")

        for frame_idx, p in enumerate(paths):
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue

            H, W = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = self.landmarker.detect(mp_image)

            image_name = os.path.basename(p)

            # Pas de pose
            if not result.pose_landmarks:
                out_path = os.path.join(self.out_dir, os.path.splitext(image_name)[0] + "_pose.jpg")
                cv2.imwrite(out_path, img_bgr)
                self._append_csv_rows(frame_idx, image_name, detected=0, pts={}, W=W, H=H)
                continue

            pose_landmarks = result.pose_landmarks[0]
            pts = self._get_points(pose_landmarks, W, H)

            # Dessin + save image
            ann = self._draw(img_bgr, pts)
            out_path = os.path.join(self.out_dir, os.path.splitext(image_name)[0] + "_pose.jpg")
            cv2.imwrite(out_path, ann)

            # Save CSV
            self._append_csv_rows(frame_idx, image_name, detected=1, pts=pts, W=W, H=H)

        print(f"Done. Annotated frames saved in: {self.out_dir}")
        print(f"Done. Landmarks CSV saved at: {self.out_csv}")

KEEP = [
    "NOSE",
    "L_SHOULDER","R_SHOULDER",
    "L_ELBOW","R_ELBOW",
    "L_WRIST","R_WRIST",
    "MID_HIP",
    "L_KNEE","R_KNEE",
    "L_ANKLE","R_ANKLE"
]

EDGES = [
    ("NOSE", "MID_HIP"),
    ("L_SHOULDER", "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("MID_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),
    ("MID_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE"),
]

frames_dir = RGB_FRAMES_DIR
out_dir    = RGB_MP_DIR
out_csv    = POSE_CSV
task_path  = TASK_PATH



annot = PoseLandmarkerFrameAnnotator(
    frames_dir=frames_dir,
    out_dir=out_dir,
    task_path=task_path,
    keep=KEEP,
    edges=EDGES,
    out_csv=out_csv,
    score_min=0.20
)

annot.run(pattern="*.jpg")