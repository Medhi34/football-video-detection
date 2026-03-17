import argparse
from team_tracker import TeamTracker
from match_stats import MatchStats


def main():
    parser = argparse.ArgumentParser(description='Football Video Detection & Stats Extraction')
    parser.add_argument('--input', default='data/videos/test_1.mp4', help='Chemin vers la vidéo d\'entrée')
    parser.add_argument('--model', default='yolov10n.pt', help='Chemin vers le modèle YOLO')
    parser.add_argument('--output-video', default='data/videos/res_final.mp4', help='Chemin vers la vidéo annotée de sortie')
    parser.add_argument('--stats', action='store_true', help='Extraire les statistiques du match')
    parser.add_argument('--stats-dir', default='data/stats', help='Dossier de sortie des statistiques')
    args = parser.parse_args()

    if args.stats:
        # Mode extraction de statistiques
        stats_extractor = MatchStats(
            model_path=args.model,
            video_path=args.input,
            output_final_path=args.output_video,
            stats_output_dir=args.stats_dir,
        )
        stats_extractor.extract_stats()
    else:
        # Mode suivi des équipes uniquement
        team_tracker = TeamTracker(model_path=args.model, video_path=args.input, output_final_path=args.output_video)
        team_tracker.track_teams_with_ball_lines_and_filter()


if __name__ == '__main__':
    main()
