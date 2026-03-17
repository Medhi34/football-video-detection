# match_stats.py
# Extraction des données statistiques d'un match de football

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
from team_tracker import TeamTracker


class MatchStats(TeamTracker):
    """
    Classe pour extraire les statistiques d'un match de football à partir d'une vidéo.

    Statistiques extraites :
    - Possession du ballon par équipe (%)
    - Nombre moyen de joueurs détectés par équipe
    - Trajectoire du ballon
    - Positions des joueurs (heatmaps)
    - Export JSON et CSV
    """

    def __init__(
        self,
        model_path='yolov10n.pt',
        video_path='football_video.mp4',
        output_final_path='data/videos/res_final.mp4',
        output_ball_filter_path='data/videos/res_ball_filter.mp4',
        stats_output_dir='data/stats',
    ):
        super().__init__(model_path, video_path, output_final_path, output_ball_filter_path)
        self.stats_output_dir = stats_output_dir

        # Statistiques collectées frame par frame
        self.ball_positions = []          # (x, y) ou None par frame
        self.possession_per_frame = []    # 0, 1 ou None par frame (équipe 0 ou 1)
        self.player_counts_per_frame = [] # [(nb_team0, nb_team1), ...] par frame
        self.team0_positions = []         # [(x_centre, y_centre), ...] pour toutes les frames
        self.team1_positions = []         # idem équipe 1

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _get_player_centers(self, boxes, labels):
        """Retourne les centres (x, y) des joueurs de chaque équipe."""
        centers_team0 = []
        centers_team1 = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if i < len(labels):
                if labels[i] == 0:
                    centers_team0.append((cx, cy))
                else:
                    centers_team1.append((cx, cy))
        return centers_team0, centers_team1

    def _nearest_team(self, ball_pos, centers_team0, centers_team1):
        """Retourne l'indice de l'équipe (0 ou 1) dont un joueur est le plus proche du ballon."""
        if ball_pos is None:
            return None
        bx, by = ball_pos

        def min_dist(centers):
            if not centers:
                return float('inf')
            return min(np.sqrt((cx - bx) ** 2 + (cy - by) ** 2) for cx, cy in centers)

        d0 = min_dist(centers_team0)
        d1 = min_dist(centers_team1)
        if d0 == float('inf') and d1 == float('inf'):
            return None
        return 0 if d0 <= d1 else 1

    def _detect_ball_position(self, frame, results):
        """Détecte la position du ballon dans la frame via YOLO (classe ball=1)."""
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == self.classes_to_detect['ball']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    return (cx, cy)
        return None

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def extract_stats(self):
        """
        Parcourt la vidéo, détecte les objets, classe les équipes et collecte
        toutes les statistiques. Appelle ensuite les méthodes d'export.

        Returns:
            dict: Dictionnaire récapitulatif des statistiques.
        """
        # --- Passe 1 : collecte des couleurs pour le clustering KMeans ---
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            self.process_frame_for_teams(frame, results)
        cap.release()

        if not any(self.all_color_data):
            print("Aucun joueur détecté dans la vidéo.")
            return {}

        labels_per_frame = self.adjust_labels()

        # --- Passe 2 : collecte des statistiques frame par frame ---
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            ball_pos = self._detect_ball_position(frame, results)
            self.ball_positions.append(ball_pos)

            labels = labels_per_frame[frame_index] if frame_index < len(labels_per_frame) else []
            boxes = self.all_player_boxes[frame_index] if frame_index < len(self.all_player_boxes) else []

            centers0, centers1 = self._get_player_centers(boxes, labels)
            self.team0_positions.extend(centers0)
            self.team1_positions.extend(centers1)
            self.player_counts_per_frame.append((len(centers0), len(centers1)))

            possession = self._nearest_team(ball_pos, centers0, centers1)
            self.possession_per_frame.append(possession)

            frame_index += 1

        cap.release()

        # --- Calcul des statistiques agrégées ---
        stats = self._compute_aggregated_stats()

        # --- Export ---
        os.makedirs(self.stats_output_dir, exist_ok=True)
        self._export_json(stats)
        self._export_csv()
        self._generate_heatmaps()
        self._generate_possession_chart(stats)
        self._generate_ball_trajectory()

        print(f"\n=== Statistiques du match ===")
        print(f"  Possession Équipe 1 : {stats['possession_team1_pct']:.1f}%")
        print(f"  Possession Équipe 2 : {stats['possession_team2_pct']:.1f}%")
        print(f"  Frames analysées    : {stats['total_frames']}")
        print(f"  Frames avec ballon  : {stats['frames_with_ball']}")
        print(f"  Joueurs moy. Éq. 1  : {stats['avg_players_team1']:.1f}")
        print(f"  Joueurs moy. Éq. 2  : {stats['avg_players_team2']:.1f}")
        print(f"  Fichiers exportés dans : {self.stats_output_dir}")

        return stats

    # ------------------------------------------------------------------
    # Calcul des statistiques agrégées
    # ------------------------------------------------------------------

    def _compute_aggregated_stats(self):
        total_frames = len(self.possession_per_frame)
        frames_with_ball = sum(1 for p in self.possession_per_frame if p is not None)
        # Internal KMeans labels are 0 and 1; we expose them as Team 1 and Team 2 (1-indexed)
        possession_team1_frames = sum(1 for p in self.possession_per_frame if p == 0)
        possession_team2_frames = sum(1 for p in self.possession_per_frame if p == 1)

        if frames_with_ball > 0:
            pct_team1 = possession_team1_frames / frames_with_ball * 100
            pct_team2 = possession_team2_frames / frames_with_ball * 100
        else:
            pct_team1 = pct_team2 = 0.0

        counts_team1 = [c[0] for c in self.player_counts_per_frame]
        counts_team2 = [c[1] for c in self.player_counts_per_frame]
        avg_team1 = float(np.mean(counts_team1)) if counts_team1 else 0.0
        avg_team2 = float(np.mean(counts_team2)) if counts_team2 else 0.0

        return {
            'total_frames': total_frames,
            'frames_with_ball': frames_with_ball,
            'possession_team1_frames': possession_team1_frames,
            'possession_team2_frames': possession_team2_frames,
            'possession_team1_pct': round(pct_team1, 2),
            'possession_team2_pct': round(pct_team2, 2),
            'avg_players_team1': round(avg_team1, 2),
            'avg_players_team2': round(avg_team2, 2),
        }

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def _export_json(self, stats):
        """Exporte les statistiques agrégées en JSON."""
        path = os.path.join(self.stats_output_dir, 'match_stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    def _export_csv(self):
        """Exporte les statistiques frame par frame en CSV.

        La colonne ``possession_team`` utilise les valeurs 1 et 2 (Équipe 1 et Équipe 2),
        cohérentes avec les clés du fichier JSON.
        """
        path = os.path.join(self.stats_output_dir, 'match_stats_per_frame.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'ball_x', 'ball_y', 'possession_team', 'players_team1', 'players_team2'])
            for i, (ball_pos, possession, counts) in enumerate(
                zip(self.ball_positions, self.possession_per_frame, self.player_counts_per_frame)
            ):
                bx, by = ball_pos if ball_pos else ('', '')
                # Internal KMeans label 0 → Équipe 1, label 1 → Équipe 2 (1-indexed, same as JSON)
                team = (possession + 1) if possession is not None else ''
                writer.writerow([i, bx, by, team, counts[0], counts[1]])

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def _generate_heatmaps(self):
        """Génère les heatmaps de position des joueurs de chaque équipe."""
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        for team_idx, (positions, label) in enumerate(
            [(self.team0_positions, 'Equipe_1'), (self.team1_positions, 'Equipe_2')]
        ):
            if not positions:
                continue

            heatmap = np.zeros((height, width), dtype=np.float32)
            for x, y in positions:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(heatmap, (x, y), 20, 1.0, -1)

            # Normaliser et appliquer colormap
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

            out_path = os.path.join(self.stats_output_dir, f'heatmap_{label}.jpg')
            cv2.imwrite(out_path, colored)

            # Version matplotlib avec colorbar
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(heatmap, cmap='hot', origin='upper')
            plt.colorbar(im, ax=ax, label='Densité de présence')
            ax.set_title(f'Heatmap de position – {label}')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.stats_output_dir, f'heatmap_{label}_plot.jpg'))
            plt.close()

    def _generate_possession_chart(self, stats):
        """Génère un graphique camembert de la possession du ballon."""
        labels = ['Équipe 1', 'Équipe 2']
        sizes = [stats['possession_team1_pct'], stats['possession_team2_pct']]
        colors = ['#3498db', '#e74c3c']

        if sum(sizes) == 0:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 13},
        )
        ax.set_title('Possession du ballon', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_output_dir, 'possession_chart.jpg'))
        plt.close()

    def _generate_ball_trajectory(self):
        """Génère une visualisation de la trajectoire du ballon."""
        valid_positions = [(i, pos) for i, pos in enumerate(self.ball_positions) if pos is not None]
        if not valid_positions:
            return

        frames_idx = [v[0] for v in valid_positions]
        xs = [v[1][0] for v in valid_positions]
        ys = [v[1][1] for v in valid_positions]

        fig, ax = plt.subplots(figsize=(12, 5))
        scatter = ax.scatter(xs, ys, c=frames_idx, cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Numéro de frame')
        ax.set_title('Trajectoire du ballon', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_output_dir, 'ball_trajectory.jpg'))
        plt.close()
