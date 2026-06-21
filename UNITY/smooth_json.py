import json
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter, medfilt
from copy import deepcopy

def dist(a, b):
    return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float))


def remove_impossible_points(frame):
    joints = frame["joints"]

    for side in ["L", "R"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        shoulder = f"{side}_SHOULDER"

        if not all(j in joints for j in [hip, knee, ankle, shoulder]):
            continue

        hip_p = joints[hip]
        knee_p = joints[knee]
        ankle_p = joints[ankle]
        shoulder_p = joints[shoulder]

        # 1) Knee should not be above shoulder
        # MediaPipe image coords: smaller y = higher
        if knee_p[1] < shoulder_p[1]:
            joints[knee] = [np.nan, np.nan, np.nan]

        # 2) Ankle should not be above hip
        if ankle_p[1] < hip_p[1]:
            joints[ankle] = [np.nan, np.nan, np.nan]

        # 3) Bone lengths should not be absurdly large
        upper_leg = dist(hip_p, knee_p)
        lower_leg = dist(knee_p, ankle_p)

        if upper_leg > 0.6:
            joints[knee] = [np.nan, np.nan, np.nan]

        if lower_leg > 0.6:
            joints[ankle] = [np.nan, np.nan, np.nan]

    return frame

def is_bad_leg_position(joints, side="L"):
    hip = joints[f"{side}_HIP"]
    knee = joints[f"{side}_KNEE"]
    ankle = joints[f"{side}_ANKLE"]
    shoulder = joints[f"{side}_SHOULDER"]

    # In image coordinates, smaller y = higher
    # Knee should usually be below hip/shoulder, not near head
    if knee[1] < shoulder[1]:
        return True

    # Leg length should not explode
    upper_leg = np.linalg.norm(np.array(knee) - np.array(hip))
    lower_leg = np.linalg.norm(np.array(ankle) - np.array(knee))

    if upper_leg > 0.5 or lower_leg > 0.5:
        return True

    return False

def clean_signal(signal, jump_threshold=0.06, window=20, smooth_window=25, polyorder=3):
    signal = np.array(signal, dtype=float)

    jumps = np.abs(np.diff(signal, prepend=signal[0])) > jump_threshold
    bad = binary_dilation(jumps, iterations=window)

    clean = signal.copy()
    clean[bad] = np.nan

    clean = pd.Series(clean).interpolate(method="polynomial",limit_direction="both", order=5).to_numpy()

    if smooth_window >= len(clean):
        smooth_window = len(clean) - 1

    if smooth_window % 2 == 0:
        smooth_window += 1

    if smooth_window <= polyorder:
        return clean

    smooth = savgol_filter(clean, window_length=smooth_window, polyorder=polyorder)
    smooth = medfilt(smooth)
    return smooth


def smooth_json_file(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    # 1) Remove anatomically impossible points BEFORE smoothing
    for frame in data:
        remove_impossible_points(frame)

    for frame in data:
        joints = frame["joints"]

        for side in ["L", "R"]:
            required = [
                f"{side}_HIP",
                f"{side}_KNEE",
                f"{side}_ANKLE",
                f"{side}_SHOULDER"
            ]

            if all(k in joints for k in required):
                if is_bad_leg_position(joints, side):
                    joints[f"{side}_KNEE"] = [np.nan, np.nan, np.nan]
                    joints[f"{side}_ANKLE"] = [np.nan, np.nan, np.nan]

    smoothed_data = deepcopy(data)

    # 2) Then smooth all joints normally
    joint_names = set()
    for frame in data:
        joint_names.update(frame["joints"].keys())

    for joint_name in joint_names:
        for axis in range(3):
            values = []
            frame_indices = []

            for i, frame in enumerate(data):
                if joint_name in frame["joints"]:
                    values.append(frame["joints"][joint_name][axis])
                    frame_indices.append(i)

            values = pd.Series(values).interpolate(limit_direction="both").to_numpy()

            if len(values) < 5:
                continue

            smooth_values = clean_signal(values)

            for idx, frame_index in enumerate(frame_indices):
                smoothed_data[frame_index]["joints"][joint_name][axis] = float(smooth_values[idx])

    with open(output_path, "w") as f:
        json.dump(smoothed_data, f, indent=4)

    print(f"Smoothed JSON saved to: {output_path}")


smooth_json_file(
    "all_frames_joints_smoothed2.json",
    "all_frames_joints_smoothed3.json"
)