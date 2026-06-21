import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import  RANSAC_DIR,SLICE_DIR
import os
import glob
import numpy as np
import open3d as o3d


class PlyFirstSliceCropper:

    def __init__(self):

        self.IN_DIR  = RANSAC_DIR
        self.OUT_DIR = SLICE_DIR
        os.makedirs(self.OUT_DIR, exist_ok=True)

        self.AXIS ="x"
        self.START =-4
        self.STEP =3
        self.MIN_POINTS_KEEP =10

        self.axis_idx = {"x": 0, "y": 1, "z": 2}[self.AXIS]

    def crop_first_slice(self, pcd: o3d.geometry.PointCloud, a0: float, a1: float) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return o3d.geometry.PointCloud()

        mn = pts.min(axis=0).astype(float)
        mx = pts.max(axis=0).astype(float)

        mn[self.axis_idx] = a0
        mx[self.axis_idx] = a1

        box = o3d.geometry.AxisAlignedBoundingBox(mn, mx)
        return pcd.crop(box)

    def run(self):
        ply_files = sorted(glob.glob(os.path.join(self.IN_DIR, "*.ply")))
        print(f"[INFO] {len(ply_files)} fichiers trouvés dans: {self.IN_DIR}")

        a0, a1 = self.START, self.START + self.STEP

        kept = 0
        for fp in ply_files:
            name = os.path.splitext(os.path.basename(fp))[0]

            pcd = o3d.io.read_point_cloud(fp)

            chunk = self.crop_first_slice(pcd, a0, a1)

            n = len(chunk.points)
            if n < self.MIN_POINTS_KEEP:
                print(f"[skip] {name} | points={n}")
                continue

            out_path = os.path.join(self.OUT_DIR, f"{name}_slice.ply")
            ok = o3d.io.write_point_cloud(out_path, chunk)
            print(f"[ok] {name} -> {os.path.basename(out_path)} | points={n} | Save={ok}")
            kept += 1

        print(f"[DONE] {kept}/{len(ply_files)} fichiers sauvegardés dans: {self.OUT_DIR}")


if __name__ == "__main__":
    app = PlyFirstSliceCropper()
    app.run()
