# 🦾 Analyse des mouvements par fusion vision–LiDAR

> Projet Annuel – Master 1 Intelligence Artificielle & Sciences des Données en Santé  
> Université de Caen Normandie — Mai 2026  
> Encadré par : M. Youssef Chahir · M. Nolan Merry

**Équipe :** Khalil Ghanam · Danylo Sukach · Khanie Tsitana · Salma Benamar

---

## 📌 Description

Ce projet propose un pipeline complet d'**analyse quantitative du mouvement humain** à partir de séquences de nuages de points 3D acquises par un capteur LiDAR (RS-Helios-16P).

L'objectif est de transformer ces données brutes en une **représentation squelettique structurée**, exploitable pour la rééducation médicale, l'analyse sportive ou la reconstruction 3D d'avatars animés.

---

## 🎯 Objectifs

- Acquérir et prétraiter des données LiDAR au format PCAP/PLY
- Segmenter la silhouette humaine dans un environnement 3D bruité
- Reconstruire un squelette articulé via deux approches (L1-médiale & morphologique 2D)
- Fusionner les données LiDAR avec une caméra RGB via MediaPipe
- Visualiser le mouvement en temps réel via une interface PyQt6

---

## 🏗️ Architecture du pipeline

```
Fichier PCAP
    │
    ▼
Fichiers PLY (frame par frame)
    │
    ▼
Nettoyage + RANSAC + DBSCAN
    │
    ▼
Cluster humain PLY
    │
    ▼
JSON articulations (L1-médiale ou Morphologique 2D)
    │
    ▼
Lissage du signal (smooth_json.py)
    │
    ▼
Interface PyQt6 + Animation Unity (IK)
```

| Module         | Rôle                                      | Fichiers                          |
|----------------|-------------------------------------------|-----------------------------------|
| Prétraitement  | Filtrage, RANSAC, DBSCAN                  | `src/preprocessing/`              |
| Squelettisation| L1-médiale & Morphologique 2D             | `src/fusion/L1.py`, `dilate.py`   |
| Fusion RGB     | MediaPipe + squelette 3D                  | `src/fusion/MediaPipe.py`         |
| I/O            | Conversion CSV/PLY                        | `src/io/`                         |
| Visualisation  | Interface interactive PyQt6               | `src/visualization/`, `UNITY/interface.py` |
| Post-traitement| Lissage des articulations JSON            | `UNITY/smooth_json.py`            |

---

## 🛠️ Technologies utilisées

- **Python 3.10** — traitement des données, reconstruction 3D
- **Open3D** — manipulation des nuages de points
- **scikit-image** — squelettisation morphologique 2D
- **RANSAC / DBSCAN** — segmentation géométrique
- **MediaPipe** — détection de pose RGB
- **PyQt6** — interface interactive avec lecteur vidéo multi-flux
- **SciPy / Pandas** — lissage du signal (filtre Savitzky-Golay, interpolation polynomiale)
- **NumPy** — calculs vectoriels (angles, distances, articulations)
- **Unity** — visualisation et animation 3D de l'avatar *(projet disponible sur demande)*

---

## 🔬 Méthodologie

### 1. Collecte des données
- Capteur : LiDAR RS-Helios-16P (10 FPS, ~28 000 points/frame)
- Caméra RGB synchronisée
- Enregistrements en salle vide (BU et gymnase de l'université)

### 2. Prétraitement (`src/preprocessing/`)

| Étape | Détail |
|-------|--------|
| Filtrage | Suppression des NaN, ±∞, points hors distance [1m ; 4.5m] |
| Voxel downsampling | Grille ∈ [0.03 ; 0.05] m |
| Outliers | k=10 voisins, seuil α=5σ |
| RANSAC | Suppression des plans (sol, murs) |
| DBSCAN | ε=0.22m, min_pts=10 — sélection du cluster humain |

### 3. Reconstruction du squelette (`src/fusion/`)

**Approche 1 — L1-médiale (`L1.py`) — 3D**
- 120 contracteurs convergent vers la médiale géométrique
- Graphe MST + repère OBB → 15 articulations anatomiques
- Robustesse au bruit : ✅ élevée

**Approche 2 — Morphologique 2D (`dilate.py`)**
- Projection frontale 3D → image 512×512
- Fermeture morphologique (noyau 5×5) + squelettisation topologique (skimage)
- Rétroprojection 3D des articulations
- Aucune dépendance à la caméra RGB

### 4. Fusion RGB (`src/fusion/MediaPipe.py`)
- Détection de pose par MediaPipe sur flux vidéo
- Fusion avec le squelette 3D LiDAR pour enrichir les articulations

### 5. Lissage du signal (`UNITY/smooth_json.py`)
- Suppression des points anatomiquement impossibles (genoux au-dessus des épaules, longueurs d'os aberrantes)
- Interpolation polynomiale + filtre Savitzky-Golay sur chaque axe (x, y, z)
- Filtre médian final pour supprimer les pics résiduels

### 6. Interface de visualisation (`UNITY/interface.py`)
- Interface PyQt6 multi-flux (4 vidéos simultanées)
- Affichage en temps réel des angles articulaires (coude, genou, hanche)
- Calcul de distances et centre de masse corporel frame par frame

---

## 📁 Structure du projet

```
lidar-3d-segmentation/
├── Projet II/
│   └── src/
│       ├── preprocessing/
│       │   ├── Nettoyage.py       # Filtrage des nuages de points
│       │   ├── Ransac.py          # Suppression des plans
│       │   ├── Dbscan.py          # Clustering et sélection humain
│       │   ├── Slice.py
│       │   └── OrientationExporter.py
│       ├── fusion/
│       │   ├── L1.py              # Squelettisation L1-médiale 3D
│       │   ├── dilate.py          # Squelettisation morphologique 2D
│       │   ├── MediaPipe.py       # Fusion RGB + pose detection
│       │   ├── Squelette.py
│       │   └── TraiterMp4.py
│       ├── visualization/
│       │   └── Visualisation.py
│       ├── io/
│       │   └── csv_to_ply.py
│       ├── squelette/
│       │   ├── squelette.py
│       │   └── squelette_land_marker.py
│       ├── config.py
│       └── skeleton_def.py
└── UNITY/
    ├── interface.py               # Interface PyQt6 multi-flux
    └── smooth_json.py             # Lissage des articulations JSON
```

---

## ▶️ Installation et utilisation

### Prérequis

```bash
pip install open3d scikit-image numpy scipy pandas PyQt6 mediapipe
```

### Exécution du pipeline

```bash
# Étape 1 : Prétraitement (nettoyage + segmentation)
python "Projet II/src/preprocessing/Nettoyage.py"

# Étape 2 : Reconstruction du squelette (L1-médiale)
python "Projet II/src/fusion/L1.py"

# Étape 3 : Lissage des articulations
python UNITY/smooth_json.py

# Étape 4 : Lancer l'interface de visualisation
python UNITY/interface.py
```

---

## 👥 Équipe

| Nom | GitHub |
|-----|--------|
| Salma Benamar | [@saliza21](https://github.com/saliza21) |
| Khalil Ghanam |
| Danylo Sukach |
| Khanie Tsitana |

---

## 📄 

Projet annuel  réalisé à l'Université de Caen Normandie .  
Toute réutilisation doit mentionner les auteurs et l'établissement.
