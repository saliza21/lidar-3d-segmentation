
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PLY_PATH_EXAMPLE as PLY_PATH, POSE_CSV
import numpy as np
import pandas as pd
import open3d as o3d

PLY_PATH = PLY_PATH
POSE_CSV = POSE_CSV
#OUT_PATH= r"E:\IAFH\M1\Projet 25-26\Lidar_squelette\donnee\PCAP\bu-salle-fev\ply\CLASS\skeleton_out"


FRAME_IDX = 50 #   index -> pose_by_frame.csv

SCORE_MIN = 0.20
R_UV = 0.10
MIN_NEI = 15

QLO, QHI  = 0.02, 0.98
SCALE_PAD = 1.1


KEEP = [
    "NOSE",
    "L_SHOULDER","R_SHOULDER",
    "L_ELBOW","R_ELBOW",
    "L_WRIST","R_WRIST",
    "MID_HIP",
    "L_KNEE","R_KNEE",
    "L_ANKLE","R_ANKLE"
]

EDGES = [
    ("NOSE", "MID_HIP"),
    ("L_SHOULDER", "R_SHOULDER"),
    ("L_SHOULDER", "L_ELBOW"),
    ("L_ELBOW", "L_WRIST"),
    ("R_SHOULDER", "R_ELBOW"),
    ("R_ELBOW", "R_WRIST"),
    ("MID_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),
    ("MID_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE")
]


def robust_center(a, qlo=0.10, qhi=0.90):
    lo = np.quantile(a, qlo)
    hi = np.quantile(a, qhi)
    m = (a >= lo) & (a <= hi)
    if np.any(m):
        return float(np.mean(a[m]))
    return float(np.mean(a))

def recenter_skeleton_uvd(points3d, center, eU, eV, eD, cloud_u, cloud_v, cloud_d,
                          qlo=0.10, qhi=0.90, keep_d=True):

    cu = robust_center(cloud_u, qlo, qhi)
    cv = robust_center(cloud_v, qlo, qhi)
    cd = robust_center(cloud_d, qlo, qhi)

    names = [k for k, p in points3d.items() if p is not None]
    if len(names) < 2:
        return points3d

    P = np.stack([points3d[k] for k in names], axis=0)
    su, sv, sd = to_uvd(P, center, eU, eV, eD)

    scu = robust_center(su, qlo, qhi)
    scv = robust_center(sv, qlo, qhi)
    scd = robust_center(sd, qlo, qhi)

    du = cu - scu
    dv = cv - scv
    dd = (cd - scd) if (not keep_d) else 0.0

    out = {}
    for k, p in points3d.items():
        if p is None:
            out[k] = None
            continue
        u1, v1, d1 = to_uvd_one(p, center, eU, eV, eD)
        out[k] = from_uvd(u1 + du, v1 + dv, d1 + dd, center, eU, eV, eD)

    return out


def make_body_frame(points_xyz: np.ndarray):

    P = points_xyz
    center = P.mean(axis=0)
    eV = np.array([0.0, 0.0, 1.0])

    XY = P[:, :2] - center[:2]
    C = np.cov(XY.T)
    vals, vecs = np.linalg.eigh(C)
    eU_xy = vecs[:, np.argmax(vals)]
    eU = np.array([eU_xy[0], eU_xy[1], 0.0])
    eU = eU / (np.linalg.norm(eU) + 1e-12)

    eD = np.cross(eV, eU)
    eD = eD / (np.linalg.norm(eD) + 1e-12)
    return center, eU, eV, eD

def to_uvd(P, center, eU, eV, eD):
    X = P - center[None, :]
    return X @ eU, X @ eV, X @ eD

def to_uvd_one(p, center, eU, eV, eD):
    X = p - center
    return float(X @ eU), float(X @ eV), float(X @ eD)

def from_uvd(u, v, d, center, eU, eV, eD):
    return center + u*eU + v*eV + d*eD

def estimate_depth_for_uv(u_t, v_t, u, v, d):
    du = u - u_t
    dv = v - v_t
    dist2 = du*du + dv*dv
    idx = np.where(dist2 <= (R_UV * R_UV))[0]
    if idx.size >= MIN_NEI:
        return float(np.median(d[idx]))
    return float(d[int(np.argmin(dist2))])

def qrange(a, qlo=0.02, qhi=0.98):
    return float(np.quantile(a, qlo)), float(np.quantile(a, qhi))

def rescale_to_cloud(points3d, center, eU, eV, eD, cloud_u, cloud_v, pad=1.2):
    u0, u1 = qrange(cloud_u, QLO, QHI)
    v0, v1 = qrange(cloud_v, QLO, QHI)
    cloud_w = max(u1-u0, 1e-3)
    cloud_h = max(v1-v0, 1e-3)
    cloud_cu = 0.5*(u0+u1)
    cloud_cv = 0.5*(v0+v1)

    names = [k for k,p in points3d.items() if p is not None]
    if len(names) < 2:
        return points3d

    P = np.stack([points3d[k] for k in names], axis=0)
    su, sv, sd = to_uvd(P, center, eU, eV, eD)
    su0, su1 = qrange(su, QLO, QHI)
    sv0, sv1 = qrange(sv, QLO, QHI)
    sk_w = max(su1-su0, 1e-3)
    sk_h = max(sv1-sv0, 1e-3)
    sk_cu = 0.5*(su0+su1)
    sk_cv = 0.5*(sv0+sv1)

    sU = (cloud_w/sk_w)*pad
    sV = (cloud_h/sk_h)*pad

    out = {}
    for k,p in points3d.items():
        if p is None:
            out[k]=None; continue
        uu, vv, dd = to_uvd_one(p, center, eU, eV, eD)
        uu2 = (uu - sk_cu)*sU + cloud_cu
        vv2 = (vv - sk_cv)*sV + cloud_cv
        out[k] = from_uvd(uu2, vv2, dd, center, eU, eV, eD)
    return out

def joints_as_spheres(points3d, radius=0.03, color=(0, 1.0, 0.0)):
    meshes = []
    for name in KEEP:
        p = points3d.get(name)
        if p is None:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(np.asarray(p, dtype=float))
        s.paint_uniform_color(color)
        meshes.append(s)
    return meshes


def build_lineset(points3d):
    pts = []
    name_to_i = {}
    for name in KEEP:
        p = points3d.get(name)
        if p is None:
            continue
        name_to_i[name] = len(pts)
        pts.append(p)
    if len(pts) < 2:
        return None

    lines = []
    for a,b in EDGES:
        if a in name_to_i and b in name_to_i:
            lines.append([name_to_i[a], name_to_i[b]])
    if not lines:
        return None

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))

    colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls



# LOAD DATA
df = pd.read_csv(POSE_CSV)


df["x_norm"] = df["x_px"] / df["W"]
df["y_norm"] = df["y_px"] / df["H"]

frame_col = "frame" if "frame" in df.columns else "frame_idx"
sub = df[(df["frame_idx"] == FRAME_IDX) & (df["detected"] == 1)]

if sub.empty:
    raise RuntimeError(f"No pose for FRAME_IDX={FRAME_IDX} in CSV")

pcd = o3d.io.read_point_cloud(PLY_PATH)
P = np.asarray(pcd.points)
if P.shape[0] < 50:
    raise RuntimeError("Too few points in PLY")

center, eU, eV, eD = make_body_frame(P)
u, v, d = to_uvd(P, center, eU, eV, eD)

u_min, u_max = float(np.min(u)), float(np.max(u))
v_min, v_max = float(np.min(v)), float(np.max(v))

points3d = {}
for name in KEEP:
    row = sub[sub["lm"] == name]
    if row.empty:
        points3d[name] = None
        continue

    r = row.iloc[0]
    score = float(r["score"]) if "score" in sub.columns else 1.0
    if score < SCORE_MIN:
        points3d[name] = None
        continue

    x = float(r["x_norm"])
    y = float(r["y_norm"])

    u_t = u_min + x * (u_max - u_min)
    v_t = v_max - y * (v_max - v_min)

    d_t = estimate_depth_for_uv(u_t, v_t, u, v, d)
    points3d[name] = from_uvd(u_t, v_t, d_t, center, eU, eV, eD)


points3d = rescale_to_cloud(points3d, center, eU, eV, eD, u, v, pad=SCALE_PAD)

points3d = recenter_skeleton_uvd(
    points3d, center, eU, eV, eD,
    cloud_u=u, cloud_v=v, cloud_d=d,
    qlo=0.10, qhi=0.90,
    keep_d=True
)

ls = build_lineset(points3d)
pcd.paint_uniform_color([0.7, 0.7, 0.7])
joint_meshes = joints_as_spheres(points3d, radius=0.02)
geoms = [pcd, ls]+joint_meshes


o3d.visualization.draw_geometries(geoms, window_name="Nuage + Squelette")