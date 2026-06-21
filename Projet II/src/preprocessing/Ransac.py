import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import NETT_DIR, RANSAC_DIR
import os
import glob
import numpy as np
import open3d as o3d


class PlyRansacPlanes:

    def __init__(self):
        self.INPUT_DIR = NETT_DIR
        self.OUT_DIR = RANSAC_DIR

        self.VOXEL= 0.0     # 3 cm
        self.DIST= 0.04     # 2 cm (sol/murs)
        self.MAX_PLANES=3
        self.MIN_INLIERS_RATIO= 0.05

        # Orientation
        self.FLOOR_NZ_MIN = 0.85
        self.WALL_NZ_MAX  = 0.30


    def remove_planes_ransac_by_normal(self, pcd: o3d.geometry.PointCloud,
        dist=None,
        max_planes=None,
        min_inliers_ratio=None,
        floor_nz_min=None,
        wall_nz_max=None
    ) -> o3d.geometry.PointCloud:

        if dist is None:
            dist = self.DIST
        if max_planes is None:
            max_planes = self.MAX_PLANES
        if min_inliers_ratio is None:
            min_inliers_ratio = self.MIN_INLIERS_RATIO
        if floor_nz_min is None:
            floor_nz_min = self.FLOOR_NZ_MIN
        if wall_nz_max is None:
            wall_nz_max = self.WALL_NZ_MAX


        remaining = pcd
        for k in range(max_planes):
            n_pts = len(remaining.points)
            if n_pts < 200:
                break

            model, inliers = remaining.segment_plane(
                distance_threshold=dist,
                ransac_n=3,
                num_iterations=4000
            )

            inlier_ratio = len(inliers) / max(1, n_pts)
            if inlier_ratio < min_inliers_ratio:
                break

            a, b, c, d = model
            n = np.array([a, b, c], dtype=float)
            # print ( n )
            n = n/(np.linalg.norm(n) + 1e-12)
            # print ( n )
            # exit ()
            nz = abs(n[2])

            rest = remaining.select_by_index(inliers, invert=True)

            # Supprimer uniquement sol ou mur (orientation)
            if nz >= floor_nz_min or nz <= wall_nz_max:
                remaining = rest

            else:
                break

        return remaining



    def run(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)

        ply_files = sorted(glob.glob(os.path.join(self.INPUT_DIR, "*.ply")))
        if not ply_files:
            raise RuntimeError(f"Aucun .ply trouvé dans: {self.INPUT_DIR}")

        print(f"Fichiers trouvés: {len(ply_files)}")
        print(f"Output: {self.OUT_DIR}")

        for i, ply_path in enumerate(ply_files, 1):
            name = os.path.basename(ply_path)
            out_path = os.path.join(self.OUT_DIR, name)

            pcd = o3d.io.read_point_cloud(ply_path)
            if pcd.is_empty():
                print(f"[{i}/{len(ply_files)}] {name} -> vide, skip")
                continue

            if self.VOXEL is not None and self.VOXEL > 0:
                pcd = pcd.voxel_down_sample(self.VOXEL)

            pcd_clean = self.remove_planes_ransac_by_normal(pcd)

            ok = o3d.io.write_point_cloud(out_path, pcd_clean)
            n_in = len(pcd.points)
            n_out = len(pcd_clean.points)

            print(f"[{i}/{len(ply_files)}] {name}: {n_in} -> {n_out} points | saved={ok}")

        print("Terminé.")


if __name__ == "__main__":
    app = PlyRansacPlanes()
    app.run()