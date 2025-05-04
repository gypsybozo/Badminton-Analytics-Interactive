# main.py
from badminton_analyzer import BadmintonAnalyzer # <-- Ensure this imports the refactored class
import pandas as pd
import sys
import traceback

if __name__ == "__main__":
    # --- Configuration ---
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "/Users/kriti.bharadwaj03/Badminton_Analysis/input/Srikanth_Momota.mp4" 
    SAVE_OUTPUT_VIDEO = True
    DRAW_POSE_ON_VIDEO = True # Control pose drawing

    # --- Analysis ---
    try:
        # Initialize the orchestrator
        analyzer = BadmintonAnalyzer(
            shuttle_model_path=shuttle_model_path,
            court_model_path=court_model_path,
            video_path=video_path,
            conf_threshold=0.3, # Adjust as needed
            frame_skip=1        # Adjust as needed (e.g., 2 or 3 for faster processing)
        )

        print("Starting badminton video analysis...", flush=True)

        # Run the main processing pipeline
        result = analyzer.process_video_with_shot_detection(
            save_output=SAVE_OUTPUT_VIDEO,
            draw_pose=DRAW_POSE_ON_VIDEO
        )

        if result is None:
             print("\nAnalysis returned None. Exiting.", flush=True)
             sys.exit(1)

        print(f"\nAnalysis complete!", flush=True)

        # --- Report Results ---
        confirmed_shots_dataset = result.get('dataset', [])
        num_confirmed_shots = len(confirmed_shots_dataset)
        confirmed_indices = result.get('shots_confirmed_indices', []) # Use the correct key

        max_rally = 0
        if confirmed_shots_dataset:
            try:
                 valid_rallies = [item['rally_id'] for item in confirmed_shots_dataset if isinstance(item.get('rally_id'), int)]
                 if valid_rallies: max_rally = max(valid_rallies)
            except Exception as e: print(f"Warning: Could not determine max rally ID: {e}", flush=True)

        print(f"\n--- Results Summary ---", flush=True)
        print(f"Total Processed Frames: {result.get('total_processed_frames', 'N/A')}", flush=True)
        print(f"Confirmed Shot Events (Triggers): {len(confirmed_indices)}", flush=True) # Number of triggers confirmed
        print(f"Entries in Shot Dataset: {num_confirmed_shots}", flush=True) # Should match above if no errors
        print(f"Rallies Identified: {max_rally}", flush=True)

        # Display dataset summary
        if confirmed_shots_dataset:
            dataset_df = pd.DataFrame(confirmed_shots_dataset)
            print("\nDataset summary:")
            if not dataset_df.empty and 'player_who_hit' in dataset_df.columns:
                p1_hits = len(dataset_df[dataset_df['player_who_hit'] == 1])
                p2_hits = len(dataset_df[dataset_df['player_who_hit'] == 2])
                unknown_hits = len(dataset_df[dataset_df['player_who_hit'] == 0])
                print(f"  Shots by player 1: {p1_hits}")
                print(f"  Shots by player 2: {p2_hits}")
                if unknown_hits > 0: print(f"  Shots with unknown player: {unknown_hits}")
            else: print("  Could not calculate hits per player.")

            if result.get('dataset_path'): print(f"\nDataset saved to: {result['dataset_path']}")
            else: print("\nDataset CSV was not saved.")
        else:
            print("\nDataset is empty.")

        # Report output file locations
        print(f"Shot images saved to '{analyzer.shots_output_dir}/' directory") # Access from analyzer instance
        print(f"Player shot images saved to '{analyzer.player_shots_dir}/' directory")
        if result.get('video_output_path'): print(f"Output video saved to: {result['video_output_path']}")
        else: print("Output video was not saved.")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required file was not found.", flush=True)
        print(e, flush=True)
        sys.exit(1)
    except KeyError as e:
         print(f"\nERROR: Missing expected key in analysis results: {e}", flush=True)
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred in main.py: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)