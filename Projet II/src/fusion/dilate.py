import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DBSCAN_DIR, MORPHO_OUT_DIR
import glob
import json
import cv2
import numpy as np
import open3d as o3d
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as _mst
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


class LidarMorphoSkeleton:

    JOINT_NAMES = [
        "HEAD", "NECK",
        "L_SHOULDER", "R_SHOULDER",
        "L_ELBOW", "R_ELBOW",
        "L_WRIST", "R_WRIST",
        "MID_HIP",
        "L_HIP", "R_HIP",
        "L_KNEE", "R_KNEE",
        "L_ANKLE", "R_ANKLE",
    ]

    def __init__(
        self,
        ply_dir,
        out_dir,
        pattern="frame_*_clean_slice.ply",
        img_size=512,
        dilation_iter=5,
        kernel_size=5,
        base_scale=180.0,
        save_debug=True,
    ):
        self.ply_dir = ply_dir
        self.out_dir = out_dir
        self.pattern = pattern
        self.img_size = img_size
        self.dilation_iter = dilation_iter
        self.kernel_size = kernel_size
        self.base_scale = base_scale
        self.save_debug = save_debug

        self._global_eU = None
        self._global_eV = None

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "debug"), exist_ok=True)

    def get_ply_files(self):
        files = sorted(glob.glob(os.path.join(self.ply_dir, self.pattern)))
        if not files:
            raise RuntimeError("Aucun fichier PLY trouvé.")
        return files

    def get_orientation_path(self, ply_path):
        base = os.path.splitext(os.path.basename(ply_path))[0]
        return os.path.join(self.ply_dir, base + "_orientation.json")

    def load_orientation(self, ply_path):
        path = self.get_orientation_path(ply_path)
        if not os.path.exists(path):
            raise RuntimeError(f"Orientation manquante : {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        center = np.array(data["centroid"], dtype=np.float64)
        eU_raw = np.array(data["right_dir"], dtype=np.float64)
        eV_raw = np.array(data["up_dir"], dtype=np.float64)

        eU_raw /= np.linalg.norm(eU_raw) + 1e-12
        eV_raw /= np.linalg.norm(eV_raw) + 1e-12

        if self._global_eU is None:
            self._global_eU = eU_raw.copy()
            self._global_eV = eV_raw.copy()

        eU = eU_raw if np.dot(eU_raw, self._global_eU) >= 0 else -eU_raw
        eV = eV_raw if np.dot(eV_raw, self._global_eV) >= 0 else -eV_raw

        eD = np.cross(eU, eV)
        eD /= np.linalg.norm(eD) + 1e-12

        zoom = float(data.get("zoom", 0.7))
        scale = self.base_scale * zoom

        return center, eU, eV, eD, scale

    def project_to_frontal_image(self, points, ply_path):
        center, eU, eV, eD, scale = self.load_orientation(ply_path)

        P = points - center
        a = P @ eU
        b = P @ eV

        x_img = (a * scale + self.img_size / 2).astype(np.int32)
        y_img = (self.img_size / 2 - b * scale).astype(np.int32)

        mask = (
            (x_img >= 0) & (x_img < self.img_size) &
            (y_img >= 0) & (y_img < self.img_size)
        )

        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        img[y_img[mask], x_img[mask]] = 255

        meta = {
            "center": center.tolist(),
            "eU": eU.tolist(),
            "eV": eV.tolist(),
            "eD": eD.tolist(),
            "scale": scale,
            "img_size": self.img_size,
        }

        return img, meta

    def build_silhouette(self, img):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
        dilated = cv2.dilate(closed, kernel, iterations=self.dilation_iter)
        return dilated

    def skeletonize_silhouette(self, silhouette):
        binary = silhouette > 0
        skel = skeletonize(binary)
        return skel.astype(np.uint8)

    def skeleton_to_graph(self, skel):
        ys, xs = np.where(skel > 0)
        G = nx.Graph()
        pixels = set(zip(xs, ys))

        for x, y in pixels:
            G.add_node((x, y))

        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]

        for x, y in pixels:
            for dx, dy in neighbors:
                nx_, ny_ = x + dx, y + dy
                if (nx_, ny_) in pixels:
                    dist = np.sqrt(dx * dx + dy * dy)
                    G.add_edge((x, y), (nx_, ny_), weight=dist)

        return G

    def find_endpoints_and_junctions(self, skel):
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ], dtype=np.uint8)

        count = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)

        endpoints_yx = np.argwhere(count == 12)
        junctions_yx = np.argwhere(count >= 14)

        endpoints = [(int(x), int(y)) for y, x in endpoints_yx]
        junctions = [(int(x), int(y)) for y, x in junctions_yx]

        return endpoints, junctions

    def nearest_node(self, G, target):
        tx, ty = target
        nodes = np.array(list(G.nodes), dtype=np.float64)

        if len(nodes) == 0:
            return None

        d = np.linalg.norm(nodes - np.array([tx, ty]), axis=1)
        return tuple(nodes[np.argmin(d)].astype(int))

    def point_on_path_ratio(self, path, ratio):
        if not path:
            return None

        idx = int(np.clip(ratio * (len(path) - 1), 0, len(path) - 1))
        return path[idx]

    def stable_zone_point(
        self,
        nodes,
        G,
        x_min,
        x_max,
        y_min,
        y_max,
        y_ratio,
        y_tol_ratio,
        side=None,
    ):
        y_target = y_min + y_ratio * (y_max - y_min)
        y_tol = y_tol_ratio * (y_max - y_min)

        zone = nodes[np.abs(nodes[:, 1] - y_target) < y_tol]

        if len(zone) == 0:
            return self.nearest_node(G, ((x_min + x_max) / 2, y_target))

        x_mid = (x_min + x_max) / 2

        if side == "left":
            zone_side = zone[zone[:, 0] < x_mid]
            if len(zone_side) > 0:
                zone = zone_side

        elif side == "right":
            zone_side = zone[zone[:, 0] > x_mid]
            if len(zone_side) > 0:
                zone = zone_side

        cx = int(np.mean(zone[:, 0]))
        cy = int(np.mean(zone[:, 1]))

        return self.nearest_node(G, (cx, cy))

    def estimate_joints_2d(self, skel):
        G = self.skeleton_to_graph(skel)

        if len(G.nodes) < 10:
            return {}

        nodes = np.array(list(G.nodes), dtype=np.int32)

        y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        x_mid = int((x_min + x_max) / 2)

        head = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.02,
            y_tol_ratio=0.03,
            side=None
        )

        neck = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.18,
            y_tol_ratio=0.04,
            side=None
        )

        left_shoulder = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.25,
            y_tol_ratio=0.05,
            side="left"
        )

        right_shoulder = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.25,
            y_tol_ratio=0.05,
            side="right"
        )

        mid_hip = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.58,
            y_tol_ratio=0.05,
            side=None
        )

        left_hip = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.62,
            y_tol_ratio=0.05,
            side="left"
        )

        right_hip = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.62,
            y_tol_ratio=0.05,
            side="right"
        )

        left_wrist = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.50,
            y_tol_ratio=0.20,
            side="left"
        )

        right_wrist = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.50,
            y_tol_ratio=0.20,
            side="right"
        )

        left_foot = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.96,
            y_tol_ratio=0.05,
            side="left"
        )

        right_foot = self.stable_zone_point(
            nodes, G, x_min, x_max, y_min, y_max,
            y_ratio=0.96,
            y_tol_ratio=0.05,
            side="right"
        )

        def mid_path(a, b):
            if a is None or b is None:
                return None
            try:
                path = nx.shortest_path(G, a, b, weight="weight")
                return self.point_on_path_ratio(path, 0.5)
            except Exception:
                return None

        left_elbow = mid_path(left_shoulder, left_wrist)
        right_elbow = mid_path(right_shoulder, right_wrist)

        left_knee = mid_path(left_hip, left_foot)
        right_knee = mid_path(right_hip, right_foot)

        joints = {
            "HEAD": head,
            "NECK": neck,
            "L_SHOULDER": left_shoulder,
            "R_SHOULDER": right_shoulder,
            "L_ELBOW": left_elbow,
            "R_ELBOW": right_elbow,
            "L_WRIST": left_wrist,
            "R_WRIST": right_wrist,
            "MID_HIP": mid_hip,
            "L_HIP": left_hip,
            "R_HIP": right_hip,
            "L_KNEE": left_knee,
            "R_KNEE": right_knee,
            "L_ANKLE": left_foot,
            "R_ANKLE": right_foot,
        }

        return joints

    def image_point_to_3d(self, p, meta):
        if p is None:
            return None

        x_img, y_img = p

        center = np.array(meta["center"], dtype=np.float64)
        eU = np.array(meta["eU"], dtype=np.float64)
        eV = np.array(meta["eV"], dtype=np.float64)
        scale = float(meta["scale"])

        a = (x_img - self.img_size / 2) / scale
        b = (self.img_size / 2 - y_img) / scale

        point_3d = center + a * eU + b * eV

        return [
            float(point_3d[0]),
            float(point_3d[1]),
            float(point_3d[2]),
        ]

    def joints_2d_to_3d(self, joints_2d, meta):
        joints_3d = {}

        for name in self.JOINT_NAMES:
            joints_3d[name] = self.image_point_to_3d(joints_2d.get(name), meta)

        return joints_3d

    def draw_debug(self, silhouette, skel, joints_2d):
        debug = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)

        ys, xs = np.where(skel > 0)
        debug[ys, xs] = (0, 0, 255)

        for name, p in joints_2d.items():
            if p is None:
                continue

            x, y = p
            cv2.circle(debug, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(
                debug,
                name,
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        return debug

    def save_json(self, frame_data, frame_id):
        path = os.path.join(
            self.out_dir,
            "json",
            f"frame_{frame_id:04d}_morpho_skeleton.json"
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(frame_data, f, indent=2)

    def save_debug_image(self, debug, frame_id):
        path = os.path.join(
            self.out_dir,
            "debug",
            f"frame_{frame_id:04d}_debug.png"
        )

        cv2.imwrite(path, debug)

    def process_one(self, ply_path, frame_id):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        if len(points) == 0:
            print("Frame vide :", ply_path)
            return None

        raw_img, meta = self.project_to_frontal_image(points, ply_path)
        silhouette = self.build_silhouette(raw_img)
        skel = self.skeletonize_silhouette(silhouette)

        joints_2d = self.estimate_joints_2d(skel)
        joints_3d = self.joints_2d_to_3d(joints_2d, meta)

        frame_data = {
            "frame": frame_id,
            "timestamp_ms": int(frame_id * 33),
            "joints": joints_3d,
        }

        if self.save_debug:
            debug = self.draw_debug(silhouette, skel, joints_2d)
            self.save_debug_image(debug, frame_id)

            try:
                cv2.imshow("Morphological Skeleton", debug)
                cv2.waitKey(1)
            except cv2.error:
                pass

        self.save_json(frame_data, frame_id)

        return frame_data

    def process_all(self):
        ply_files = self.get_ply_files()
        all_frames = []

        for frame_id, ply_path in enumerate(ply_files):
            print("Processing:", frame_id, os.path.basename(ply_path))

            try:
                frame_data = self.process_one(ply_path, frame_id)
            except Exception as e:
                print("Erreur :", e)
                continue

            if frame_data is not None:
                all_frames.append(frame_data)

        final_path = os.path.join(self.out_dir, "all_frames_morpho_skeleton.json")

        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(all_frames, f, indent=2)

        cv2.destroyAllWindows()

        print("Terminé.")
        print("JSON final :", final_path)


if __name__ == "__main__":

    PLY_DIR = DBSCAN_DIR
    OUT_DIR = MORPHO_OUT_DIR

    app = LidarMorphoSkeleton(
        ply_dir=PLY_DIR,
        out_dir=OUT_DIR,
        pattern="frame_*_clean_slice.ply",
        img_size=512,
        dilation_iter=5,
        kernel_size=5,
        base_scale=230.0,
        save_debug=True,
    )
    app.process_all()