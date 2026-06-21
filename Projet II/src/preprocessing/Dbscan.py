import os
import sys
import glob
import numpy as np
import open3d as o3d

# Fix import path — fonctionne quel que soit le répertoire de lancement
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.OrientationExporter import OrientationExporter
from config import RANSAC_DIR, DBSCAN_DIR,SLICE_DIR


class PlyDbscanHumanSelector:

    def __init__(self):

        self.INPUT_DIR = SLICE_DIR
        self.OUT_DIR = DBSCAN_DIR
        os.makedirs(self.OUT_DIR, exist_ok=True)

        # DBSCAN PARAMS
        self.VOXEL = 0.0      #si > 0, on réduit le nombre de points (downsample) pour accélérer DBSCAN.
        self.EPS = 0.22       #distance max pour considérer des points “voisins” (donc pour faire un cluster).
        self.MIN_POINTS = 10  #nombre minimal de voisins pour qu’un point soit considéré “noyau” du cluster.

        # Human constraints
        self.H_MIN, self.H_MAX = 1, 2
        self.W_MAX = 1
        self.D_MAX = 1

        self.MIN_CLUSTER_PTS = 25

        # PARAMS silhouette
        self.SLICE_H = 0.08
        self.MIN_PTS_SLICE = 10
        self.MIN_VALID_SLICES = 8
        self.MAX_CENTER_WIGGLE = 0.25
        self.MIN_WIDTH_VARIATION = 0.15
        self.MAX_THINNESS = 0.10


    def _points_to_obb_frame(self, cluster: o3d.geometry.PointCloud):

        obb = cluster.get_oriented_bounding_box()
        R = np.asarray(obb.R)
        c = np.asarray(obb.center)
        pts = np.asarray(cluster.points)

        pts_local = (pts - c) @ R
        ext = np.array(obb.extent, dtype=float)
        ext_sorted = np.sort(ext)
        return pts_local, ext_sorted, obb

    def _silhouette_signature(self, pts_local: np.ndarray):

        spans = pts_local.max(axis=0) - pts_local.min(axis=0) # la taille du cluster selon chaque axe local
        up = int(np.argmax(spans)) # l’axe où l’objet est le plus “long” HAUTEUR
        # print("####################")
        # print(spans)
        # print(up)
        # exit()
        axes = [0, 1, 2]
        axes.remove(up)
        a_lat0, a_lat1 = axes[0], axes[1]

        u = pts_local[:, up] # coordonnée des points sur l’axe “vertical”
        umin, umax = float(u.min()), float(u.max())
        H = umax - umin
        if H <= 1e-6:
            return None

        nbins = int(np.ceil(H / self.SLICE_H)) # Nb tranches
        widths = []
        centers0 = []
        centers1 = []
        valid = 0

        for b in range(nbins):
            # de tranche intervalle [uO min_tranche , u1 max_tranche]
            u0 = umin + b * self.SLICE_H
            u1 = min(umin + (b + 1) * self.SLICE_H, umax)

            mask = (u >= u0) & (u < u1)

            if not np.any(mask):
                continue

            slab = pts_local[mask] # points de cette tranche

            if slab.shape[0] < self.MIN_PTS_SLICE:
                continue

            lat0 = slab[:, a_lat0]
            w = float(lat0.max() - lat0.min())

            c0 = float(np.mean(slab[:, a_lat0]))
            c1 = float(np.mean(slab[:, a_lat1]))

            widths.append(w)
            centers0.append(c0)
            centers1.append(c1)
            valid += 1

        if valid < self.MIN_VALID_SLICES:
            return None

        widths = np.array(widths, dtype=float)
        centers0 = np.array(centers0, dtype=float)
        centers1 = np.array(centers1, dtype=float)


        w_p90 = float(np.percentile(widths, 90))#valeur sous laquelle 90% des données se trouvent
        w_p10 = float(np.percentile(widths, 10))#valeur sous laquelle 10% des données se trouvent
        w_var = float(np.std(widths)) # écart-type -> Ça mesure à quel point les largeurs varient autour de la moyenne
        w_range = w_p90 - w_p10

        center_wiggle = float(np.sqrt(np.var(centers0) + np.var(centers1)))

        return {
            "H": H,
            "valid_slices": valid,
            "w_p90": w_p90,
            "w_p10": w_p10,
            "w_var": w_var,
            "w_range": w_range,
            "center_wiggle": center_wiggle,
        }


    def select_human_cluster(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud | None:

        if pcd.is_empty():
            return None

        work = pcd

        if self.VOXEL and self.VOXEL > 0:
            work = work.voxel_down_sample(self.VOXEL)

        labels = np.array(work.cluster_dbscan(eps=self.EPS, min_points=self.MIN_POINTS, print_progress=False))
        max_label = labels.max()
        if max_label < 0:
            return None

        best = None
        best_score = -1e18

        for cid in range(max_label + 1):

            idx = np.where(labels == cid)[0]
            n = len(idx)

            if n < self.MIN_CLUSTER_PTS:
                continue

            cluster = work.select_by_index(idx)

            if len(cluster.points) <= 200:
                continue

            pts_local, ext_sorted, _ = self._points_to_obb_frame(cluster)
            S, M, L = ext_sorted  # épaisseur, largeur, longueur

            if S < self.MAX_THINNESS:
                continue

            sig = self._silhouette_signature(pts_local)
            if sig is None:
                continue

            h = sig["H"]
            if not (self.H_MIN <= h <= self.H_MAX):
                continue

            # humain: variation largeur (épaules/torse/taille)
            if sig["w_range"] < self.MIN_WIDTH_VARIATION:
                continue

            if sig["center_wiggle"] < 0.02:
                continue

            # score (simple mais efficace)
            score = (
                4.0 * h
                + 0.02 * n
                + 2.0 * sig["w_range"]
                + 0.5 * sig["w_var"]
                - 1.5 * (L / max(M, 1e-6))
                - 2.0 * max(0.0, 0.18 - S)
                + 0.2 * sig["center_wiggle"]
            )

            if score > best_score:
                best_score = score
                best = cluster

        return best

    def run(self):
        ply_files = sorted(glob.glob(os.path.join(self.INPUT_DIR, "*.ply")))
        if not ply_files:
            raise RuntimeError(f"Aucun .ply trouvé dans: {self.INPUT_DIR}")

        ok, fail = 0, 0
        print("Files:", len(ply_files))
        print(f"DBSCAN eps={self.EPS} min_points={self.MIN_POINTS} voxel={self.VOXEL}")
        print(f"Human: H[{self.H_MIN},{self.H_MAX}] W<={self.W_MAX} D<={self.D_MAX}\n")

        for i, path in enumerate(ply_files, 1):
            name = os.path.basename(path)
            out_path = os.path.join(self.OUT_DIR, name)

            pcd = o3d.io.read_point_cloud(path)
            n_in = len(pcd.points)

            human = self.select_human_cluster(pcd)

            if human is None or len(human.points) == 0:
                print(f"[{i}/{len(ply_files)}] {name}: in={n_in} -> no human cluster")
                fail += 1
                continue

            o3d.io.write_point_cloud(out_path, human)
            print(f"[{i}/{len(ply_files)}] {name}: in={n_in} -> human={len(human.points)} ✅")
            ok += 1

        print("\nDone.")
        print("Success:", ok, "| Failed:", fail)


if __name__ == "__main__":
    app = PlyDbscanHumanSelector()
    app.run()

    # corriger l'orientation de nuage (vue frontal)
    PLY_DIR = DBSCAN_DIR
    exporter = OrientationExporter(
        ply_dir=PLY_DIR,
        pattern="frame_*_clean_slice.ply"
    )
    exporter.export_all()
