from shuttle_tracker import ShuttleTracker
import argparse

def main():
    parser = argparse.ArgumentParser(description='Badminton Shuttle Tracking and Shot Detection')
    parser.add_argument('--model', type=str, required=True, 
                      help='Path to YOLOv8 model for player, racket, and shuttle detection')
    parser.add_argument('--video', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--confidence', type=float, default=0.3,
                      help='Confidence threshold for detections (default: 0.3)')
    parser.add_argument('--frame-skip', type=int, default=5,
                      help='Number of frames to skip between processing (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                      help='Path for output video (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Initialize shuttle tracker
    tracker = ShuttleTracker(
        model_path=args.model,
        conf_threshold=args.confidence
    )
    
    # Process the video
    shot_data = tracker.process_video(
        video_path=args.video,
        frame_skip=args.frame_skip,
        save_output=True,
        output_path=args.output
    )
    
    print(f"Detected {len(shot_data)} shots in the video")
    print(shot_data)

if __name__ == "__main__":
    main()