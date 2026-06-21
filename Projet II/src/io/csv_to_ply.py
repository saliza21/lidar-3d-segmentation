import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import PLY_FRAMES_DIR
import re
import os
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement


class CsvToPly:

    def make_ply_safe_columns(self, columns):
        safe = []
        used = set()
        mapping = {}
        for c in columns:
            c2 = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
            if c2 == "":
                c2 = "field"
            if c2[0].isdigit():
                c2 = "f_" + c2
            base = c2
            k = 1
            while c2 in used:
                c2 = f"{base}_{k}"
                k += 1
            used.add(c2)
            safe.append(c2)
            mapping[c] = c2
        return safe, mapping

    def csv_to_ply(self,
        csv_path: str,
        ply_path: str = "cloud_enriched.ply",
        xyz_preference=("Point_X", "Point_Y", "Point_Z"),
        xyz_fallback=("Points_m_XYZ:0", "Points_m_XYZ:1", "Points_m_XYZ:2"),
        colorize_by: str | None = "intensity",
        drop_nan: bool = True,
    ):
        df = pd.read_csv(csv_path)

        if all(c in df.columns for c in xyz_preference):
            xcol, ycol, zcol = xyz_preference
        elif all(c in df.columns for c in xyz_fallback):
            xcol, ycol, zcol = xyz_fallback
        else:
            raise ValueError(f"XYZ introuvables. Colonnes: {list(df.columns)}")

        if drop_nan:
            mask = np.isfinite(df[[xcol, ycol, zcol]].to_numpy(dtype=np.float64)).all(axis=1)
            df = df.loc[mask].reset_index(drop=True)

        df2 = df.copy()
        for col in df2.columns:
            try:
                df2[col] = pd.to_numeric(df2[col])
            except Exception:
                pass

        safe_cols, mapping = self.make_ply_safe_columns(df2.columns)

        df2.columns = safe_cols

        xcol_s, ycol_s, zcol_s = mapping[xcol], mapping[ycol], mapping[zcol]
        colorize_by_s = mapping.get(colorize_by, None) if colorize_by is not None else None

        df2["x"] = df2[xcol_s].astype(np.float32)
        df2["y"] = df2[ycol_s].astype(np.float32)
        df2["z"] = df2[zcol_s].astype(np.float32)

        def ply_dtype_for(series: pd.Series):
            if pd.api.types.is_float_dtype(series):
                return np.float32
            if pd.api.types.is_integer_dtype(series):
                return np.int32
            if pd.api.types.is_bool_dtype(series):
                return np.uint8
            return "S64"

        ordered_cols = ["x", "y", "z"] + [c for c in df2.columns if c not in ("x", "y", "z")]

        dtype = [(col, ply_dtype_for(df2[col])) for col in ordered_cols]
        vertex = np.empty(len(df2), dtype=dtype)


        for col, dt in dtype:
            if dt == "S64":
                vertex[col] = df2[col].astype(str).str.encode("utf-8")
            else:
                vertex[col] = df2[col].to_numpy(dtype=dt)

        if colorize_by_s is not None and colorize_by_s in df2.columns:
            vals = pd.to_numeric(df2[colorize_by_s], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(vals)
            if ok.any():
                lo, hi = np.nanpercentile(vals[ok], [1, 99])
                if hi <= lo:
                    lo, hi = np.nanmin(vals[ok]), np.nanmax(vals[ok])
                norm = (vals - lo) / (hi - lo + 1e-12)
                norm = np.clip(norm, 0.0, 1.0)
                gray = (norm * 255).astype(np.uint8)

                if not all(c in vertex.dtype.names for c in ("red", "green", "blue")):
                    new_dtype = vertex.dtype.descr + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
                    v2 = np.empty(vertex.shape, dtype=new_dtype)
                    for name in vertex.dtype.names:
                        v2[name] = vertex[name]
                    v2["red"] = gray
                    v2["green"] = gray
                    v2["blue"] = gray
                    vertex = v2

        el = PlyElement.describe(vertex, "vertex")
        PlyData([el], text=False).write(ply_path)

        print(f"PLY enrichi écrit: {ply_path}")
        print(f"Points: {len(df2)}")
        print(f"Open3D XYZ: x,y,z (copiés depuis {xcol}->{xcol_s}, etc.)")

        return ply_path



if __name__ == "__main__":
    converter = CsvToPly()

    csv_dir = PLY_FRAMES_DIR

    ply_out_dir = PLY_FRAMES_DIR
    os.makedirs(ply_out_dir, exist_ok=True)

    csv_files = sorted([
        f for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv")
    ])

    for i, csv_file in enumerate(csv_files, start=1):
        csv_path = os.path.join(csv_dir, csv_file)

        ply_name = f"frame_{i:04d}.ply"
        ply_path = os.path.join(ply_out_dir, ply_name)

        print(f"[INFO] Conversion {csv_file} -> {ply_name}")

        converter.csv_to_ply(
            csv_path=csv_path,
            ply_path=ply_path,
            colorize_by="intensity",
            drop_nan=True
        )

    print("Conversion terminée.")