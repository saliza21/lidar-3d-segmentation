import sys

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QPushButton, QGridLayout, QLineEdit, QLabel, QVBoxLayout
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt

import json
import numpy as np
from PyQt6.QtCore import QUrl, QTimer
from PyQt6.QtMultimedia import QMediaPlayer

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def body_center(joints):
    points = [
        joints["L_HIP"],
        joints["R_HIP"],
        joints["L_SHOULDER"],
        joints["R_SHOULDER"]
    ]
    return np.mean(np.array(points), axis=0)

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player (PyQt6)")
        self.resize(1200, 600)

        self.is_playing = False
        self.videos_loaded = False


        self.label = QLabel()
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)
        self.input.setPlaceholderText("Folder…")
        self.input.setFixedHeight(36)
        self.input.setFixedWidth(200)

        self.info_label = QLabel("Skeleton info will appear here")
        self.info_label.setWordWrap(True)
        self.info_label.setFixedWidth(260)
        self.info_label.setStyleSheet("""
                    font-size: 14px;
                """)

        self.frames_data = []
        self.current_frame_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_skeleton_info)


        self.video_widget1 = QVideoWidget()
        self.video_widget2 = QVideoWidget()
        self.video_widget3 = QVideoWidget()
        self.video_widget4 = QVideoWidget()

        # players
        self.player1 = QMediaPlayer()
        self.player1.setVideoOutput(self.video_widget1)
        self.player2 = QMediaPlayer()
        self.player2.setVideoOutput(self.video_widget2)
        self.player3 = QMediaPlayer()
        self.player3.setVideoOutput(self.video_widget3)
        self.player4 = QMediaPlayer()
        self.player4.setVideoOutput(self.video_widget4)
        self.player1.mediaStatusChanged.connect(
            lambda status: self.loop_video(self.player1, status)
        )
        self.player2.mediaStatusChanged.connect(
            lambda status: self.loop_video(self.player2, status)
        )
        self.player3.mediaStatusChanged.connect(
            lambda status: self.loop_video(self.player3, status)
        )
        self.player4.mediaStatusChanged.connect(
            lambda status: self.loop_video(self.player4, status)
        )

        # button
        self.play_button1 = QPushButton("▶ Play")
        # self.play_button1.clicked.connect(self.play_video)
        self.play_button1.clicked.connect(self.toggle_video)
        self.play_button1.setFixedHeight(60)
        self.play_button1.setStyleSheet("""
                    background-color: #03224C;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                    font-weight: bold;
                    font-size: 13px;
                """)

        # layout
        layout = QHBoxLayout()
        layoutvideo = QGridLayout()
        container_menu = QWidget()
        container_menu.setStyleSheet("background-color: white; color: black;")
        container_menu.setFixedWidth(250)
        layoutVertical = QVBoxLayout(container_menu)

        layoutvideo.addWidget(self.video_widget1, 2, 2)
        layoutvideo.addWidget(self.video_widget2, 2, 3)
        layoutvideo.addWidget(self.video_widget3, 3, 2)
        layoutvideo.addWidget(self.video_widget4, 3, 3)
        layout.setContentsMargins(0, 0, 0, 0)
        layoutVertical.setContentsMargins(20, 20, 20, 20)

        lidarlabel = QLabel('Lidar Projet')
        lidarlabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lidarlabel.setStyleSheet("""
            background-color: #03224C;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
            font-size: 15px;
        """)
        lidarlabel.setFixedHeight(60)
        layoutVertical.addWidget(lidarlabel)

        menulabel = QLabel('Menu')
        menulabel.setStyleSheet("""
                color: #03224C;
                font-weight: bold;
                font-size: 14px;
            """)
        menulabel.setFixedHeight(40)
        menulabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layoutVertical.addWidget(menulabel)

        layoutVertical.addWidget(self.info_label)
        layoutVertical.addWidget(self.input)
        layoutVertical.addWidget(self.play_button1)


        # layout.addLayout(layoutVertical)
        layout.addWidget(container_menu)
        layout.addLayout(layoutvideo)



        self.setLayout(layout)

    def loop_video(self, player, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            player.setPosition(0)
            player.play()

    def update_skeleton_info(self):
        if not self.frames_data:
            return

        frame = self.frames_data[self.current_frame_index]
        if frame["frame"] % 10 == 0:
            joints = frame["joints"]

            left_elbow = angle(
                joints["L_SHOULDER"],
                joints["L_ELBOW"],
                joints["L_WRIST"]
            )

            right_elbow = angle(
                joints["R_SHOULDER"],
                joints["R_ELBOW"],
                joints["R_WRIST"]
            )

            left_knee = angle(
                joints["L_HIP"],
                joints["L_KNEE"],
                joints["L_ANKLE"]
            )

            right_knee = angle(
                joints["R_HIP"],
                joints["R_KNEE"],
                joints["R_ANKLE"]
            )

            shoulder_width = distance(
                joints["L_SHOULDER"],
                joints["R_SHOULDER"]
            )

            hip_width = distance(
                joints["L_HIP"],
                joints["R_HIP"]
            )

            center = body_center(joints)

            #              Temps: {frame["timestamp_ms"]} ms<br><br>
            info = f"""
            <b>Frame:</b> {frame["frame"]}<br><br>

            <b>Angles:</b><br>
            <b>Coude gauche:</b> {left_elbow:.1f}°<br>
            <b>Coude droit:</b> {right_elbow:.1f}°<br>
            <b>Genou gauche:</b> {left_knee:.1f}°<br>
            <b>Genou droit:</b> {right_knee:.1f}°<br><br>

            <b>Distances:</b><br>
            <b>Largeur des épaules:</b> {shoulder_width:.3f}<br>
            <b>Largeur des hanches:</b> {hip_width:.3f}<br><br>

            <b>Centre du corps:</b><br>
            x = {center[0]:.3f}<br>
            y = {center[1]:.3f}<br>
            z = {center[2]:.3f}
            """

            self.info_label.setText(info)

        self.current_frame_index += 1

        if self.current_frame_index >= len(self.frames_data):
            # self.timer.stop()
            self.current_frame_index = 0


    def load_videos(self):
        file_source = self.input.text()

        self.player1.setSource(QUrl.fromLocalFile(f"{file_source}/rgb.mp4"))
        self.player2.setSource(QUrl.fromLocalFile(f"{file_source}/lidar.mp4"))
        self.player3.setSource(QUrl.fromLocalFile(f"{file_source}/unity.mp4"))
        self.player4.setSource(QUrl.fromLocalFile(f"{file_source}/unity-demo2.mp4"))

        with open("C:/Users/kaktu/Projet annuel/Assets/rgb5.json", "r") as f:
            self.frames_data = json.load(f)

        self.current_frame_index = 0

        self.resume_videos()

    def resume_videos(self):
        self.player1.play()
        self.player2.play()
        self.player3.play()
        self.player4.play()

        self.timer.start(40)

        self.is_playing = True
        self.play_button1.setText("II Pause")

    def pause_videos(self):
        self.player1.pause()
        self.player2.pause()
        self.player3.pause()
        self.player4.pause()

        self.timer.stop()

        self.is_playing = False
        self.play_button1.setText("▶ Play")

    def toggle_video(self):

        # First click → load videos
        if not self.videos_loaded:
            self.load_videos()
            self.videos_loaded = True

        if self.is_playing:
            self.pause_videos()
        else:
            self.resume_videos()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = VideoPlayer()
    window.setStyleSheet("""
        QWidget {
            background-color: black;
            color: white;
        }""")

    
    window.show()
    sys.exit(app.exec())
