# data_manager.py
import os
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import traceback
from utils.constants import DEFAULT_SHOT_IMAGE_DIR, DEFAULT_PLAYER_SHOT_DIR, RACKET_CLASS_ID
from utils.geom import crop_image_safely, get_bbox_center

class DataManager:
    def __init__(self, shots_output_dir=DEFAULT_SHOT_IMAGE_DIR, player_shots_dir=DEFAULT_PLAYER_SHOT_DIR):
        self.shots_output_dir = shots_output_dir
        self.player_shots_dir = player_shots_dir
        print(f"  [DEBUG DM] Creating output directories ('{self.shots_output_dir}', '{self.player_shots_dir}')...", flush=True)
        try:
            os.makedirs(self.shots_output_dir, exist_ok=True)
            os.makedirs(self.player_shots_dir, exist_ok=True)
        except OSError as e:
             print(f"ERROR creating output directories: {e}", flush=True)
             # Decide if this is fatal or just a warning

    def save_shot_image(self, frame, shot_info, shot_data):
        """ Saves an image of the shot with relevant information overlaid """
        if frame is None:
             print("Warning: Cannot save shot image - frame is None.", flush=True)
             return
        try:
            shot_frame = frame.copy()
            racket_center = get_bbox_center(shot_info.get('racket_box'))
            shuttle_center = get_bbox_center(shot_info.get('shuttle_box'))

            # Draw line between racket and shuttle if both exist
            if racket_center and shuttle_center:
                 try: cv2.line(shot_frame, racket_center, shuttle_center, (0,255,255), 2)
                 except Exception as line_e: print(f"    [DEBUG DM] Error drawing line: {line_e}", flush=True)

            # Add text overlays
            player_hit_val = shot_data.get('player_who_hit', 0)
            player_hit_text = f"Player Hit: {player_hit_val}" if player_hit_val != 0 else "Player Hit: Unknown"
            conf_method = shot_data.get('confirmation_method', 'N/A')
            conf_frame = shot_data.get('confirmation_frame', 'N/A')
            info_text = [f"Rally: {shot_data.get('rally_id','X')}", f"Shot: {shot_data.get('shot_num','Y')}",
                         player_hit_text, f"Method: {conf_method}",
                         f"Orig Frame: {shot_data.get('frame_number','N/A')}", f"Conf Frame: {conf_frame}"]
            y_offset = 30
            for line in info_text:
                try: cv2.putText(shot_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2); y_offset += 20
                except Exception as text_e: print(f"    [DEBUG DM] Error drawing text '{line}': {text_e}", flush=True)


            # Save main shot image
            filename = f"shot_r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}.jpg"
            filepath = os.path.join(self.shots_output_dir, filename)
            # print(f"    [DEBUG DM] Saving shot image to: {filepath}", flush=True) # Can be verbose
            if not cv2.imwrite(filepath, shot_frame): print(f"Warning: Failed to save shot image: {filepath}", flush=True)

            # Save player crop
            player_box_to_crop = shot_info.get('player1_box') if player_hit_val == 1 else shot_info.get('player2_box')
            if player_box_to_crop:
                player_img = crop_image_safely(frame, player_box_to_crop) # Use util
                if player_img is not None and player_img.size > 0:
                    p_filename = f"player_r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}_p{player_hit_val}.jpg"
                    p_filepath = os.path.join(self.player_shots_dir, p_filename)
                    # print(f"    [DEBUG DM] Saving player crop to: {p_filepath}", flush=True) # Can be verbose
                    if not cv2.imwrite(p_filepath, player_img): print(f"Warning: Failed to save player crop: {p_filepath}", flush=True)

        except Exception as e:
             print(f"ERROR saving shot image for r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}: {e}\n{traceback.format_exc()}", flush=True)

    def save_dataset_to_csv(self, shot_dataset):
        """ Saves the final dataset to a CSV file. """
        csv_path = f'badminton_shot_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_success = False
        df = None

        if not shot_dataset:
            print("WARNING: No shots recorded, saving empty CSV.", flush=True)
            # Define columns explicitly for empty DataFrame
            cols = ['rally_id', 'shot_num', 'player_who_hit', 'player1_coords', 'player2_coords',
                    'shuttle_coords_impact', 'shot_played', 'stroke_hand', 'hitting_posture',
                    'confirmation_method', 'frame_number', 'confirmation_frame']
            try: df = pd.DataFrame(columns=cols)
            except Exception as e: print(f"ERROR creating empty DataFrame: {e}", flush=True)
        else:
            try:
                 df = pd.DataFrame(shot_dataset)
                 print(f"Total CONFIRMED shots recorded: {len(df)}", flush=True)
            except Exception as e: print(f"ERROR creating DataFrame from shot_dataset: {e}", flush=True)

        if df is not None:
            try:
                 df.to_csv(csv_path, index=False)
                 print(f"Dataset saved to {csv_path}", flush=True)
                 save_success = True
            except Exception as e:
                 print(f"ERROR saving dataset CSV ({csv_path}): {e}", flush=True)
                 csv_path = None # No valid path if save failed
        else:
             print("Skipping CSV save due to DataFrame creation error.", flush=True)
             csv_path = None

        return csv_path, save_success