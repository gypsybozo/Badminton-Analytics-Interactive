# main.py
from badminton_analyzer import BadmintonAnalyzer
import pandas as pd
import sys # Import sys for detailed error handling

if __name__ == "__main__":
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "/Users/kriti.bharadwaj03/Badminton_Analysis/input/Srikanth_Momota.mp4" # Make sure this path is correct

    try:
        analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)

        print("Starting badminton video analysis...")
        print("Processing video to detect shots based on direction changes...")

        # Run direction-based shot detection with improved trajectory analysis
        result = analyzer.process_video_with_shot_detection(save_output=True)

        if result is None:
             print("\nAnalysis failed. Exiting.")
             sys.exit(1) # Exit if analysis function returned None

        print(f"\nAnalysis complete!")

        # --- MODIFIED LINE ---
        # Use the correct key 'shots_processed_indices' to get the list of shots
        num_shots = len(result.get('shots_processed_indices', [])) # Use .get for safety

        # Calculate max rally ID safely
        max_rally = 0
        if result.get('dataset'): # Check if dataset is not empty
            try:
                 max_rally = max(item['rally_id'] for item in result['dataset'] if isinstance(item.get('rally_id'), int))
            except ValueError: # Handles case where dataset is empty after filtering Nones
                 max_rally = 0 # Default to 0 if no valid rally_id found


        print(f"Detected {num_shots} shots across {max_rally} rallies")
        # --- END MODIFIED LINES ---


        # Display dataset summary if dataset exists and is not empty
        dataset = result.get('dataset', [])
        if dataset:
            dataset_df = pd.DataFrame(dataset)
            print("\nDataset summary:")
            print(f"Total shots in dataset: {len(dataset_df)}")
            if not dataset_df.empty and 'player_who_hit' in dataset_df.columns:
                # Count only valid player hits (1 or 2)
                p1_hits = len(dataset_df[dataset_df['player_who_hit'] == 1])
                p2_hits = len(dataset_df[dataset_df['player_who_hit'] == 2])
                unknown_hits = len(dataset_df[dataset_df['player_who_hit'] == 0])
                print(f"  Shots by player 1: {p1_hits}")
                print(f"  Shots by player 2: {p2_hits}")
                if unknown_hits > 0:
                    print(f"  Shots with unknown player: {unknown_hits}")
            else:
                 print("  Could not calculate hits per player (Dataset empty or 'player_who_hit' column missing).")

            if result.get('dataset_path'):
                print(f"\nDataset saved to {result['dataset_path']}")
        else:
            print("\nDataset is empty.")


        print(f"Shot images saved to '{analyzer.shots_output_dir}/' directory")
        print(f"Player shot images saved to '{analyzer.player_shots_dir}/' directory")
        if result.get('video_output_path'):
             print(f"Output video saved to {result['video_output_path']}")
        else:
             print("Output video was not saved.")


    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required file was not found.")
        print(e)
        sys.exit(1)
    except KeyError as e:
         print(f"\nERROR: Missing expected key in analysis results: {e}")
         print("This might indicate an issue during the analysis processing.")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred in main.py: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)