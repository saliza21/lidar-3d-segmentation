import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PLY_FRAMES_DIR, NETT_DIR
import os
import re
import open3d as o3d
import numpy as np


class PlyCleaner:


    def __init__(self):

        self.ply_in_dir  = PLY_FRAMES_DIR
        self.ply_out_dir = NETT_DIR
        os.makedirs(self.ply_out_dir, exist_ok=True)

        self.DIST_MIN=1     # mètres
        self.DIST_MAX =5     # mètres
        self.VOXEL=0.0   # 4 cm
        self.NB_NEI=10
        self.STD_R=5


    def natural_key(self, s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    def clean_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:

        if pcd.is_empty():
            return pcd

        pts = np.asarray(pcd.points)


        # 2) Filtre distance
        d = np.linalg.norm(pts, axis=1)
        mask_d = (d >= self.DIST_MIN) & (d <= self.DIST_MAX)
        pts = pts[mask_d]
        if pts.shape[0] == 0:
            return o3d.geometry.PointCloud()

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts)

        # 3) Voxel downsample
        if self.VOXEL and self.VOXEL > 0:
            pcd2 = pcd2.voxel_down_sample(voxel_size=self.VOXEL)

        # 4) Suppression outliers (statistical)
        if len(pcd2.points) > self.NB_NEI:
            pcd2, _ = pcd2.remove_statistical_outlier(nb_neighbors=self.NB_NEI, std_ratio=self.STD_R)

        return pcd2

    def run(self):

        ply_files = sorted(
            [f for f in os.listdir(self.ply_in_dir) if f.lower().endswith(".ply")],
            key=self.natural_key
        )

        print(f"[INFO] {len(ply_files)} fichiers PLY trouvés.")

        for idx, fname in enumerate(ply_files, start=1):
            in_path = os.path.join(self.ply_in_dir, fname)
            out_path = os.path.join(self.ply_out_dir, fname.replace(".ply", "_clean.ply"))

            pcd = o3d.io.read_point_cloud(in_path)
            before = len(pcd.points)

            pcd_clean = self.clean_pcd(pcd)

            after = len(pcd_clean.points)

            if after == 0:
                print(f"[ERREUR] {fname}: nuage vide après nettoyage (before={before}).")
                continue

            o3d.io.write_point_cloud(out_path, pcd_clean, write_ascii=False)
            print(f"[OK] {fname} -> {os.path.basename(out_path)} | points: {before} -> {after}")

        print("Nettoyage terminé.")


if __name__ == "__main__":

    cleaner = PlyCleaner()
    cleaner.run()
