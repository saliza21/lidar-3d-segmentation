import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import RGB_FRAMES_DIR
import cv2
import os
import math


class VideoFrameExtractor:

    def __init__(self, input_video, output_dir, target_fps=10):

        self.input_video = input_video
        self.output_dir = output_dir
        self.target_fps = target_fps

        os.makedirs(self.output_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.input_video)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo: {self.input_video}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Original FPS:", self.original_fps)
        print("Total frames :", self.total_frames)

    def extract(self):

        # Fallback si FPS invalide (0 ou NaN)
        if self.original_fps is None or self.original_fps <= 1e-3 or math.isnan(self.original_fps):
            print("[WARN] FPS non lisible (0/NaN) ")
        else:
            frame_skip = max(1, int(round(self.original_fps / self.target_fps)))
            print("Frame skip:", frame_skip)

            frame_id = 0
            saved = 0

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_id % frame_skip == 0:
                    cv2.imwrite(os.path.join(self.output_dir, f"frame_{saved:05d}.jpg"), frame)
                    saved += 1

                frame_id += 1

            self.cap.release()
            print("Done. Saved frames:", saved)


INPUT_VIDEO = r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\donnee\PCAP\salma_26s.mp4"
OUTPUT_DIR  = RGB_FRAMES_DIR
TARGET_FPS  = 10

extractor = VideoFrameExtractor(INPUT_VIDEO, OUTPUT_DIR, TARGET_FPS)
extractor.extract()
