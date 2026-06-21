import os
import glob
import json
import numpy as np
import open3d as o3d


class OrientationExporter:

    def __init__(self,ply_dir: str,pattern: str = "frame_*_clean_slice.ply",out_dir: str = None,
            lidar_origin: np.ndarray = None,  # position 3D du capteur dans la scène
    ):
        self.ply_dir = ply_dir
        self.pattern = pattern
        self.out_dir = out_dir if out_dir else ply_dir
        os.makedirs ( self.out_dir, exist_ok = True )

        # Origine du capteur (optionnelle mais recommandée)
        if lidar_origin is not None:
            self.lidar_origin = np.asarray ( lidar_origin, dtype = np.float64 )
        else:
            # Valeur par défaut : capteur à l'origine monde (0,0,0)
            # À adapter selon votre setup expérimental
            self.lidar_origin = np.array ( [0.0, 0.0, 0.0], dtype = np.float64 )
            self._origin_known = False
        self._origin_known = lidar_origin is not None
        # Cache mémoire : évite de relire le JSON d'orientation à chaque frame
        self._orientation_cache: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # FICHIERS
    # ─────────────────────────────────────────────────────────────────────────

    def get_ply_files(self):
        files = sorted ( glob.glob ( os.path.join ( self.ply_dir, self.pattern ) ) )
        if not files:
            raise RuntimeError (
                f"Aucun fichier PLY trouvé : {os.path.join ( self.ply_dir, self.pattern )}"
            )
        return files

    def get_orientation_json(self, ply_path: str) -> str:
        base = os.path.splitext ( os.path.basename ( ply_path ) )[0]
        return os.path.join ( self.out_dir, base + "_orientation.json" )

    # ─────────────────────────────────────────────────────────────────────────
    # CALCUL D'ORIENTATION
    # ─────────────────────────────────────────────────────────────────────────

    def compute_orientation(self, points: np.ndarray) -> dict:
        """
        Calcule l'orientation du corps dans le repère LiDAR.

        Stratégie front_dir (par priorité) :
          1. Vecteur centroid → lidar_origin  (stable, recommandé)
          2. Eigenvecteur PCA minimal          (fallback si origine inconnue)

        up_dir : toujours axe Z monde (le LiDAR est fixe et horizontal).
        right_dir : produit vectoriel front × up.
        """
        centroid = points.mean ( axis = 0 )

        # ── up_dir : axe Z monde ────────────────────────────────────────────
        up_dir = np.array ( [0.0, 0.0, 1.0], dtype = np.float64 )

        # ── front_dir ───────────────────────────────────────────────────────
        if self._origin_known:
            # Méthode 1 : vecteur centroid → capteur (toujours stable)
            front_dir = self.lidar_origin - centroid
            front_dir[2] = 0.0  # on projette dans le plan horizontal
            norm = np.linalg.norm ( front_dir )
            if norm < 1e-6:
                front_dir = self._pca_front ( points, centroid )
            else:
                front_dir /= norm
        else:
            # Méthode 2 : PCA dans le plan XY (fallback)
            front_dir = self._pca_front ( points, centroid )

        # ── right_dir ───────────────────────────────────────────────────────
        right_dir = np.cross ( front_dir, up_dir )
        right_norm = np.linalg.norm ( right_dir )
        if right_norm < 1e-6:
            # Dégénérescence (front // up) → fallback
            right_dir = np.array ( [1.0, 0.0, 0.0] )
        else:
            right_dir /= right_norm

        # Ré-orthogonalisation de front_dir
        front_dir = np.cross ( up_dir, right_dir )
        front_dir /= np.linalg.norm ( front_dir ) + 1e-12

        # ── zoom adaptatif selon la hauteur du nuage ─────────────────────────
        height = float ( points[:, 2].max () - points[:, 2].min () )
        zoom = float ( np.clip ( 1.2 / (height + 0.5), 0.3, 0.9 ) )

        return {
            "centroid": centroid.tolist (),
            "front_dir": front_dir.tolist (),
            "up_dir": up_dir.tolist (),
            "right_dir": right_dir.tolist (),
            "zoom": zoom,
            "method": "lidar_origin" if self._origin_known else "pca_fallback",
        }

    def _pca_front(self, points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Calcule front_dir par PCA dans le plan XY.
        Eigenvecteur de variance MINIMALE = direction de profondeur du corps.
        Signe choisi pour que front_dir pointe vers Y négatif (convention
        LiDAR : le capteur est généralement derrière le sujet).
        """
        pts_xy = points[:, :2] - centroid[:2]
        C = np.cov ( pts_xy.T )
        eigenvalues, eigenvectors = np.linalg.eigh ( C )  # valeurs croissantes

        # Eigenvecteur de variance minimale = axe de profondeur
        front_2d = eigenvectors[:, 0]
        front_dir = np.array ( [front_2d[0], front_2d[1], 0.0] )

        # Convention : pointe vers Y négatif si le capteur est dans cette direction
        # Ajuster selon votre setup (Y-, Y+, X-, X+)
        if front_dir[1] > 0:
            front_dir = -front_dir

        norm = np.linalg.norm ( front_dir )
        if norm < 1e-6:
            return np.array ( [0.0, -1.0, 0.0] )
        return front_dir / norm

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_one(self, ply_path: str) -> str:
        """Calcule et sauvegarde l'orientation d'un PLY."""
        pcd = o3d.io.read_point_cloud ( ply_path )
        points = np.asarray ( pcd.points )

        if len ( points ) == 0:
            print ( f"[WARN] PLY vide : {ply_path}" )
            return None

        orientation = self.compute_orientation ( points )
        json_path = self.get_orientation_json ( ply_path )

        with open ( json_path, "w" ) as f:
            json.dump ( orientation, f, indent = 4 )

        print ( f"[OK] Orientation ({orientation['method']}) → {json_path}" )
        return json_path

    def export_all(self, overwrite: bool = True):
        """Calcule et sauvegarde les orientations de tous les PLY du dossier."""
        ply_files = self.get_ply_files ()
        print ( f"Export orientation pour {len ( ply_files )} frames..." )
        for ply in ply_files:
            json_path = self.get_orientation_json ( ply )
            if not overwrite and os.path.exists ( json_path ):
                print ( f"[SKIP] Déjà existant : {json_path}" )
                continue
            self.export_one ( ply )
        print ( "Export terminé." )

    # ─────────────────────────────────────────────────────────────────────────
    # APPLICATION AU VISUALIZER OPEN3D
    # ─────────────────────────────────────────────────────────────────────────

    def apply_to_visualizer(self, vis: o3d.visualization.Visualizer, ply_path: str):
        """
        Applique l'orientation sauvegardée au ViewControl d'Open3D.
        À appeler APRÈS add_geometry() ET poll_events() + update_renderer().

        Le délai est nécessaire car Open3D initialise la caméra de manière
        asynchrone — sans poll_events() préalable, set_front() est ignoré.
        """
        json_path = self.get_orientation_json ( ply_path )

        if not os.path.exists ( json_path ):
            print ( f"[WARN] JSON manquant, calcul à la volée : {ply_path}" )
            self.export_one ( ply_path )

        with open ( json_path, "r" ) as f:
            orientation = json.load ( f )

        ctr = vis.get_view_control ()
        ctr.set_lookat ( orientation["centroid"] )
        ctr.set_front ( orientation["front_dir"] )
        ctr.set_up ( orientation["up_dir"] )
        ctr.set_zoom ( orientation.get ( "zoom", 0.55 ) )

    def apply_to_visualizer_robust(
            self,
            vis: o3d.visualization.Visualizer,
            ply_path: str,
            n_tries: int = 3,
    ):
        """
        Version robuste : applique l'orientation plusieurs fois de suite
        pour contourner le bug Open3D où set_front() est ignoré au premier appel.
        Utile pour draw_geometries() ou les fenêtres qui s'initialisent lentement.
        """
        import time

        json_path = self.get_orientation_json ( ply_path )
        if not os.path.exists ( json_path ):
            self.export_one ( ply_path )

        with open ( json_path, "r" ) as f:
            orientation = json.load ( f )

        for _ in range ( n_tries ):
            vis.poll_events ()
            vis.update_renderer ()
            ctr = vis.get_view_control ()
            ctr.set_lookat ( orientation["centroid"] )
            ctr.set_front ( orientation["front_dir"] )
            ctr.set_up ( orientation["up_dir"] )
            ctr.set_zoom ( orientation.get ( "zoom", 0.55 ) )
            time.sleep ( 0.02 )

    # ─────────────────────────────────────────────────────────────────────────
    # CHARGEMENT DU REPÈRE POUR LES MODULES SQUELETTE
    # ─────────────────────────────────────────────────────────────────────────

    def load_body_frame(self, ply_path: str):
        """
        Retourne (center, eU, eV, eD) depuis le JSON sauvegardé.
          eU = right_dir  (axe latéral gauche/droite)
          eV = up_dir     (axe vertical)
          eD = front_dir  (axe de profondeur, vers le capteur)

        Cache mémoire : le JSON n'est lu qu'une seule fois par chemin.
        Gestion d'erreur : si le JSON est corrompu ou incomplet, recalcul à la volée.
        """
        ply_path = os.path.abspath(ply_path)

        # Retourner depuis le cache si déjà chargé
        if ply_path in self._orientation_cache:
            orientation = self._orientation_cache[ply_path]
        else:
            json_path = self.get_orientation_json(ply_path)

            if not os.path.exists(json_path):
                print(f"[WARN] JSON manquant, calcul à la volée : {ply_path}")
                self.export_one(ply_path)

            try:
                with open(json_path, "r") as f:
                    orientation = json.load(f)
                # Vérifier que les champs requis sont présents
                for key in ("centroid", "right_dir", "up_dir", "front_dir"):
                    if key not in orientation:
                        raise KeyError(f"Champ manquant dans le JSON : {key}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] JSON d'orientation invalide ({e}), recalcul : {ply_path}")
                self.export_one(ply_path)
                with open(json_path, "r") as f:
                    orientation = json.load(f)

            self._orientation_cache[ply_path] = orientation

        center = np.array(orientation["centroid"],  dtype=np.float64)
        eU     = np.array(orientation["right_dir"], dtype=np.float64)
        eV     = np.array(orientation["up_dir"],    dtype=np.float64)
        eD     = np.array(orientation["front_dir"], dtype=np.float64)

        return center, eU, eV, eD

