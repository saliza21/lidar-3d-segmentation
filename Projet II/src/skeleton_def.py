
JOINT_NAMES = [
    "HEAD", "NECK",
    "L_SHOULDER", "R_SHOULDER",
    "L_ELBOW",    "R_ELBOW",
    "L_WRIST",    "R_WRIST",
    "MID_HIP",
    "L_HIP",      "R_HIP",
    "L_KNEE",     "R_KNEE",
    "L_ANKLE",    "R_ANKLE",
]

EDGES = [
    ("HEAD",       "NECK"),
    ("NECK",       "L_SHOULDER"), ("NECK",   "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),    ("L_ELBOW","L_WRIST"),
    ("R_SHOULDER", "R_ELBOW"),    ("R_ELBOW","R_WRIST"),
    ("NECK",       "MID_HIP"),
    ("MID_HIP",    "L_HIP"),      ("MID_HIP","R_HIP"),
    ("L_HIP",      "L_KNEE"),     ("L_KNEE", "L_ANKLE"),
    ("R_HIP",      "R_KNEE"),     ("R_KNEE", "R_ANKLE"),
]

SPHERE_RADIUS = 0.025
