# Football Video Detection

Projet Python pour détecter et analyser des événements/objets dans des vidéos de football (ex. ballon, joueurs, arbitre, zones du terrain, etc.).

> **Statut :** dépôt initial (README en cours d'initialisation).

## Objectifs

- Détecter des éléments clés dans une vidéo de match (ballon, joueurs, etc.)
- Suivre (tracking) ces éléments image par image
- Produire des métriques/visualisations (trajectoires, heatmaps, vitesses, zones d'occupation, etc.)
- Exporter une vidéo annotée (bounding boxes, IDs, trajectoires)

## Pré-requis

- Python 3.10+ (recommandé)
- `pip`

## Installation

```bash
# 1) Cloner
git clone https://github.com/Medhi34/football-video-detection.git
cd football-video-detection

# 2) (Optionnel) Créer un venv
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Installer les dépendances
pip install -r requirements.txt
```

> Si le fichier `requirements.txt` n'existe pas encore, tu peux l'ajouter plus tard quand les dépendances seront stabilisées.

## Utilisation

### Suivi des équipes (vidéo annotée)

```bash
cd src
python main.py --input data/videos/match.mp4 --output-video data/videos/res_final.mp4
```

### Extraction des statistiques du match

```bash
cd src
python main.py --input data/videos/match.mp4 --stats --stats-dir data/stats
```

Les fichiers générés dans `data/stats/` sont :

| Fichier | Description |
|---|---|
| `match_stats.json` | Statistiques agrégées (possession, nb joueurs…) |
| `match_stats_per_frame.csv` | Données frame par frame (position ballon, possession, effectif) |
| `heatmap_Equipe_1.jpg` | Heatmap OpenCV des positions de l'équipe 1 |
| `heatmap_Equipe_2.jpg` | Heatmap OpenCV des positions de l'équipe 2 |
| `heatmap_Equipe_1_plot.jpg` | Heatmap Matplotlib de l'équipe 1 |
| `heatmap_Equipe_2_plot.jpg` | Heatmap Matplotlib de l'équipe 2 |
| `possession_chart.jpg` | Graphique camembert de la possession |
| `ball_trajectory.jpg` | Trajectoire du ballon sur toute la vidéo |

## Structure de dossier (proposée)

```text
football-video-detection/
├── data/
│   ├── raw/
│   └── outputs/
├── models/
├── notebooks/
├── src/
└── README.md
```

## Données

- Place tes vidéos dans `data/raw/`
- Les sorties (vidéos annotées, CSV, etc.) iront dans `data/outputs/`

**Important :** évite de committer de grosses vidéos sur Git. Utilise plutôt Git LFS ou un stockage externe.

## Roadmap (idées)

- [x] Choisir/entraîner un détecteur (ex. YOLO)
- [x] Tracking multi-objets (clustering KMeans pour les équipes)
- [x] Exports : vidéo annotée + fichiers CSV/JSON + heatmaps + trajectoire ballon
- [x] Extraction des statistiques du match (possession, effectif, heatmaps)
- [ ] Homographie / repérage terrain pour projeter les positions
- [ ] Tests + CI

## Contribuer

Les contributions sont les bienvenues :

1. Fork le repo
2. Crée une branche (`git checkout -b feature/ma-feature`)
3. Commit (`git commit -m "Add: ..."`)
4. Push (`git push origin feature/ma-feature`)
5. Ouvre une Pull Request

## Licence

Aucune licence n'est définie pour l'instant. Ajoute un fichier `LICENSE` si tu souhaites préciser les conditions d'utilisation.
