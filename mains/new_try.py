from new_main import BadmintonAnalyzer

def main():
    # Paths to model weights and video
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "input/video.mov"
    
    # Create analyzer instance
    analyzer = BadmintonAnalyzer(
        shuttle_model_path=shuttle_model_path,
        court_model_path=court_model_path,
        video_path=video_path,
        conf_threshold=0.3,  # Adjust as needed
        frame_skip=2  # Process every 2nd frame for speed
    )
    
    # Process video and create shot dataset
    shot_data = analyzer.create_shot_dataset(save_output=True)
    
    print(f"\nDetected {len(shot_data)} shots in the video")
    print("Shot dataset has been saved to CSV and JSON files")

if __name__ == "__main__":
    main()