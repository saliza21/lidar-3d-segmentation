import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Dict, List, Optional, Tuple
import os
import glob
import json
import sys

# Pour permettre l'import depuis src/preprocessing
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing.OrientationExporter import OrientationExporter
from skeleton_def import JOINT_NAMES, EDGES, SPHERE_RADIUS
from config import DBSCAN_DIR, JOINTS_JSON_DIR, PLY_PATH_EXAMPLE as PLY_PATH

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

N_CONTRACTORS   = 120    # nombre de points contracteurs initiaux
N_ITER          = 40     # itérations de contraction L1
R_LOCAL         = 0.12   # rayon de voisinage (mètres)
SIGMA           = 0.05   # paramètre de pondération gaussienne (m)
MIN_BRANCH_LEN  = 0.08   # longueur minimale d'une branche MST (en m)
N_GRAPH_NEIGHBORS = 8    # voisins K pour construire le graphe MST


# ─────────────────────────────────────────────────────────────────────────────
# REPÈRE LOCAL (OBB)
# ─────────────────────────────────────────────────────────────────────────────

def make_body_frame(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construit le repère local du corps (centre, eU latéral, eV vertical, eD profondeur)."""
    center = P.mean(axis=0)
    eV = np.array([0.0, 0.0, 1.0])  # vertical = axe Z monde

    # PCA dans le plan horizontal pour trouver l'axe latéral
    XY = P[:, :2] - center[:2]
    C = np.cov(XY.T)
    _, vecs = np.linalg.eigh(C)
    eU_xy = vecs[:, 1]
    eU = np.array([eU_xy[0], eU_xy[1], 0.0])
    eU /= np.linalg.norm(eU) + 1e-12

    eD = np.cross(eV, eU)
    eD /= np.linalg.norm(eD) + 1e-12

    return center, eU, eV, eD


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACTION L1
# ─────────────────────────────────────────────────────────────────────────────

def l1_contract(
    P: np.ndarray,
    n_contractors: int = N_CONTRACTORS,
    n_iter: int = N_ITER,
    r_local: float = R_LOCAL,
    sigma: float = SIGMA,
) -> np.ndarray:
    """
    Contraction L1-médiale.

    Pour chaque contracteur q, à chaque itération :
      q_new = Σ w_i * p_i  /  Σ w_i
    avec w_i = exp(-||p_i - q||² / (2σ²)) / ||p_i - q||   (pondération L1 + gaussienne)

    Les contracteurs convergent vers la médiale géométrique du nuage.
    """
    # Initialisation : échantillonnage aléatoire dans P
    idx = np.random.choice(len(P), size=min(n_contractors, len(P)), replace=False)
    Q = P[idx].copy().astype(np.float64)

    tree = KDTree(P)

    for iteration in range(n_iter):
        Q_new = np.empty_like(Q)

        # Optimisation : requête batch pour tous les contracteurs en une fois
        nbrs_list = tree.query_ball_point(Q, r=r_local)

        for i, nbrs in enumerate(nbrs_list):
            q = Q[i]
            if len(nbrs) < 3:
                # Fallback : 5 plus proches voisins
                _, knn = tree.query(q, k=5)
                nbrs = list(knn)

            pts = P[nbrs]                                     # (M, 3)
            dists = np.linalg.norm(pts - q, axis=1) + 1e-8   # (M,)

            # Pondération L1-gaussienne
            w = np.exp(-dists**2 / (2 * sigma**2)) / dists
            Q_new[i] = (pts * w[:, None]).sum(axis=0) / (w.sum() + 1e-12)

        # Critère d'arrêt anticipé
        delta = np.linalg.norm(Q_new - Q, axis=1).max()
        Q = Q_new
        if delta < 1e-4:
            print(f"      Convergence à l'itération {iteration+1} (delta={delta:.6f})")
            break

    return Q


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE MST + SIMPLIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def build_mst(Q: np.ndarray, k: int = N_GRAPH_NEIGHBORS) -> List[Tuple[int, int]]:
    """Construit le MST sur les contracteurs Q."""
    tree = KDTree(Q)
    n = len(Q)
    W = lil_matrix((n, n))

    _, neighbors = tree.query(Q, k=k + 1)  # k+1 car le premier est Q lui-même

    for i in range(n):
        for j in neighbors[i][1:]:
            d = float(np.linalg.norm(Q[i] - Q[j]))
            W[i, j] = d
            W[j, i] = d

    mst = minimum_spanning_tree(W.tocsr()).tocoo()
    return list(zip(mst.row.tolist(), mst.col.tolist()))


def prune_short_branches(
    Q: np.ndarray,
    edges: List[Tuple[int, int]],
    min_len: float = MIN_BRANCH_LEN,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Supprime les branches du MST dont la longueur totale est inférieure à min_len.
    Une "branche" est un chemin depuis une feuille jusqu'à une bifurcation.
    """
    from collections import defaultdict

    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Identifier les feuilles (degré 1)
    leaves = [i for i in range(len(Q)) if len(adj[i]) == 1]

    removed = set()
    for leaf in leaves:
        if leaf in removed:
            continue
        path = [leaf]
        prev = -1
        cur = leaf
        branch_len = 0.0

        while True:
            effective_neighbors = [n for n in adj[cur] if n != prev and n not in removed]
            if len(effective_neighbors) != 1:
                break
            nxt = effective_neighbors[0]
            branch_len += float(np.linalg.norm(Q[nxt] - Q[cur]))
            if branch_len >= min_len:
                break
            path.append(nxt)
            prev, cur = cur, nxt

        if branch_len < min_len:
            removed.update(path[:-1])

    keep_idx = [i for i in range(len(Q)) if i not in removed]
    remap = {old: new for new, old in enumerate(keep_idx)}

    Q_pruned = Q[keep_idx]
    edges_pruned = [
        (remap[a], remap[b])
        for a, b in edges
        if a in remap and b in remap
    ]

    return Q_pruned, edges_pruned


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING ARTICULATOIRE
# ─────────────────────────────────────────────────────────────────────────────

def map_joints_from_skeleton(
    Q: np.ndarray,
    center: np.ndarray,
    eU: np.ndarray,
    eV: np.ndarray,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Mappe les nœuds du squelette médial aux articulations anatomiques
    en utilisant leur position dans le repère local (u = latéral, v = vertical).
    """
    X = Q - center[None, :]
    u_coords = X @ eU
    v_coords = X @ eV

    v_min, v_max = v_coords.min(), v_coords.max()
    v_range = v_max - v_min + 1e-8
    v_norm = (v_coords - v_min) / v_range

    def nearest(u_t: float, v_t_norm: float, u_w: float = 1.0, v_w: float = 1.0) -> np.ndarray:
        dist = u_w * (u_coords - u_t)**2 + v_w * (v_norm - v_t_norm)**2
        return Q[int(np.argmin(dist))]

    v_head     = 1.00
    v_neck     = 0.88
    v_shoulder = 0.78
    v_elbow    = 0.62
    v_wrist    = 0.48
    v_hip      = 0.42
    v_knee     = 0.22
    v_ankle    = 0.04

    u_left  = np.percentile(u_coords, 20)
    u_right = np.percentile(u_coords, 80)
    u_mid   = 0.0

    joints: Dict[str, Optional[np.ndarray]] = {}

    joints["HEAD"]       = nearest(u_mid,   v_head,     u_w=0.2)
    joints["NECK"]       = nearest(u_mid,   v_neck,     u_w=0.2)
    joints["MID_HIP"]   = nearest(u_mid,   v_hip,      u_w=0.2)

    joints["L_SHOULDER"] = nearest(u_left,  v_shoulder)
    joints["R_SHOULDER"] = nearest(u_right, v_shoulder)
    joints["L_ELBOW"]    = nearest(u_left,  v_elbow)
    joints["R_ELBOW"]    = nearest(u_right, v_elbow)
    joints["L_WRIST"]    = nearest(u_left,  v_wrist)
    joints["R_WRIST"]    = nearest(u_right, v_wrist)
    joints["L_HIP"]      = nearest(u_left,  v_hip)
    joints["R_HIP"]      = nearest(u_right, v_hip)
    joints["L_KNEE"]     = nearest(u_left,  v_knee)
    joints["R_KNEE"]     = nearest(u_right, v_knee)
    joints["L_ANKLE"]    = nearest(u_left,  v_ankle)
    joints["R_ANKLE"]    = nearest(u_right, v_ankle)

    return joints


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def save_joints(joints: Dict[str, Optional[np.ndarray]], ply_path: str, output_dir: str):
    """Sauvegarde les joints dans un fichier JSON."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    output_path = os.path.join(output_dir, base_name + "_joints.json")

    data = {}
    for name, p in joints.items():
        if p is None:
            data[name] = None
        else:
            data[name] = [float(p[0]), float(p[1]), float(p[2])]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVE] {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_lineset(
    joints: Dict[str, Optional[np.ndarray]],
    color: Tuple[float, float, float] = (1.0, 0.1, 0.1),
) -> Optional[o3d.geometry.LineSet]:
    pts, name_to_i = [], {}
    for name in JOINT_NAMES:
        p = joints.get(name)
        if p is None:
            continue
        name_to_i[name] = len(pts)
        pts.append(p)
    if len(pts) < 2:
        return None
    lines = [[name_to_i[a], name_to_i[b]] for a, b in EDGES if a in name_to_i and b in name_to_i]
    if not lines:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([color]), (len(lines), 1)))
    return ls


def joints_as_spheres(
    joints: Dict[str, Optional[np.ndarray]],
    radius: float = SPHERE_RADIUS,
    color: Tuple[float, float, float] = (0.0, 1.0, 0.3),
) -> List[o3d.geometry.TriangleMesh]:
    meshes = []
    for name in JOINT_NAMES:
        p = joints.get(name)
        if p is None:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(p.astype(float))
        s.paint_uniform_color(list(color))
        meshes.append(s)
    return meshes


def build_medial_lineset(
    Q: np.ndarray,
    edges: List[Tuple[int, int]],
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(Q.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(edges, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([[0.5, 0.5, 1.0]], (len(edges), 1))
    )
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run(ply_path: Optional[str] = None, visualize: bool = True) -> Dict[str, Optional[np.ndarray]]:
    if ply_path is None:
        ply_path = PLY_PATH
    print("=" * 55)
    print("  L1-Medial Skeleton — reconstruction géométrique")
    print("=" * 55)

    # 1. Chargement
    print(f"\n[1/5] Chargement : {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    P = np.asarray(pcd.points, dtype=np.float64)
    if P.shape[0] < 50:
        raise RuntimeError(f"Nuage trop sparse : {P.shape[0]} points")
    print(f"      {P.shape[0]} points chargés")

    # 2. Repère local
    print("\n[2/5] Chargement du repère local depuis JSON orientation...")
    orient = OrientationExporter(
        ply_dir=os.path.dirname(ply_path),
        pattern="frame_*_clean_slice.ply"
    )
    center, eU, eV, eD = orient.load_body_frame(ply_path)

    # 3. Contraction L1
    bbox_diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    r_local_auto = max(R_LOCAL, bbox_diag * 0.15)
    sigma_auto   = max(SIGMA,   bbox_diag * 0.05)
    print(f"\n[3/5] Contraction L1 ({N_CONTRACTORS} contracteurs, {N_ITER} itérations)...")
    print(f"      r_local={r_local_auto:.3f} m  sigma={sigma_auto:.3f} m  (bbox_diag={bbox_diag:.3f} m)")
    Q = l1_contract(P, n_contractors=N_CONTRACTORS, n_iter=N_ITER,
                    r_local=r_local_auto, sigma=sigma_auto)
    print(f"      {len(Q)} nœuds médians obtenus")

    # 4. MST + élagage
    print("\n[4/5] Construction MST + élagage des branches courtes...")
    edges = build_mst(Q, k=N_GRAPH_NEIGHBORS)
    Q, edges = prune_short_branches(Q, edges, min_len=MIN_BRANCH_LEN)
    print(f"      {len(Q)} nœuds après élagage, {len(edges)} arêtes")

    # 5. Mapping articulatoire
    print("\n[5/5] Mapping vers les articulations anatomiques...")
    joints = map_joints_from_skeleton(Q, center, eU, eV)
    detected = sum(1 for p in joints.values() if p is not None)
    print(f"      {detected}/{len(JOINT_NAMES)} articulations mappées")

    save_joints(joints, ply_path, JOINTS_JSON_DIR)

    # Visualisation (désactivée en batch pour aller plus vite)
    if visualize:
        print("\nVisualisation Open3D...")
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

        medial_ls = build_medial_lineset(Q, edges)
        joint_ls  = build_lineset(joints, color=(1.0, 0.15, 0.15))
        spheres   = joints_as_spheres(joints)

        geoms = [pcd, medial_ls]
        if joint_ls is not None:
            geoms.append(joint_ls)
        geoms.extend(spheres)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="L1-Medial Skeleton (bleu=médial, rouge=anatomique)",
            width=1200,
            height=800,
            visible=True,
        )
        for g in geoms:
            vis.add_geometry(g)

        orient.apply_to_visualizer_robust(vis, ply_path, n_tries=5)
        vis.run()
        vis.destroy_window()

    return joints


def run_all_dbscan(dbscan_dir: str, output_dir: str):
    ply_files = sorted(glob.glob(os.path.join(dbscan_dir, "*.ply")))
    print(f"{len(ply_files)} fichiers trouvés")

    for ply_path in ply_files:
        print("\n" + "=" * 80)
        print(f"Traitement : {ply_path}")
        try:
            # visualize=False pour ne pas bloquer sur chaque frame en batch
            run(ply_path, visualize=False)
        except Exception as e:
            print(f"[ERREUR] {ply_path}")
            print(e)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_dbscan(DBSCAN_DIR, JOINTS_JSON_DIR)