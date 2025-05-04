# video_output.py
import cv2
import numpy as np
import time
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_styles
from utils.constants import RACKET_CLASS_ID # Needed for image saving part

class VideoOutputWriter:
    def __init__(self, output_path, effective_fps, frame_width, frame_height):
        print(f"  [DEBUG VO] Initializing VideoOutputWriter for {output_path}...", flush=True)
        self.output_path = output_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'avc1'
        self.out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        if not self.out.isOpened():
            print(f"ERROR: Failed to open VideoWriter in VideoOutputWriter for: {output_path}", flush=True)
            self.out = None
        print(f"  [DEBUG VO] VideoOutputWriter initialized. Is open: {self.is_opened()}", flush=True)
        # Store pose drawing styles
        self.pose_landmark_style = mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1) # Cyan dots
        self.pose_connection_style = mp_drawing.DrawingSpec(color=(255,0,255), thickness=1) # Magenta lines
        self.mp_pose = mp.solutions.pose # To access POSE_CONNECTIONS

    def is_opened(self):
        return self.out is not None and self.out.isOpened()

    def _draw_pose(self, image, landmarks, box):
        """ Draws pose on a copy of the image using overlay method """
        if landmarks is None or box is None: return image # Return original if no data
        try:
            x1, y1, x2, y2 = map(int, box)
            if y1 >= y2 or x1 >= x2: return image # Invalid box
            # Important: Crop from the image *to be drawn on*
            overlay_area = image[y1:y2, x1:x2]
            if overlay_area.size == 0: return image # Invalid crop

            # Make a copy of the crop to draw on, to avoid modifying the original if drawing fails
            overlay_copy = overlay_area.copy()

            mp_drawing.draw_landmarks(
                image=overlay_copy,
                landmark_list=landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_landmark_style,
                connection_drawing_spec=self.pose_connection_style)

            # Put the drawn-on copy back
            image[y1:y2, x1:x2] = overlay_copy
            return image
        except Exception as e:
            print(f"    [DEBUG VO Draw Pose] Error: {e}", flush=True)
            return image # Return original on error

    def write_frame(self, frame, frame_index, original_frame_num,
                      court_corners, interpolated_shuttle_box,
                      shuttle_detection_status, # Bool: True if detected, False if interpolated
                      p1_landmarks, p2_landmarks, p1_box, p2_box, # Pass poses and boxes
                      confirmed_shot_indices, shot_dataset, draw_pose=True): # Add draw_pose flag
        """Draws annotations and writes the frame."""
        if not self.is_opened() or frame is None: return

        frame_to_write = frame.copy()

        # --- Draw Pose Landmarks ---
        if draw_pose:
            frame_to_write = self._draw_pose(frame_to_write, p1_landmarks, p1_box)
            frame_to_write = self._draw_pose(frame_to_write, p2_landmarks, p2_box)

        # --- Draw Court Lines ---
        if court_corners:
             try: cv2.polylines(frame_to_write, [np.array(court_corners, dtype=np.int32)], True, (255, 255, 255), 2)
             except Exception as e: print(f"    [DEBUG VO] Error drawing court: {e}", flush=True)

        # --- Draw Shuttle Box ---
        if interpolated_shuttle_box and all(c is not None for c in interpolated_shuttle_box):
             try:
                 x1, y1, x2, y2 = map(int, interpolated_shuttle_box)
                 color = (0, 255, 0); label = "Shuttle"
                 if not shuttle_detection_status: label += " (Interp.)"
                 cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), color, 2)
                 cv2.putText(frame_to_write, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
             except Exception as e: print(f"    [DEBUG VO] Error drawing shuttle: {e}", flush=True)

        # --- Annotate Confirmed Shot Frames ---
        if frame_index in confirmed_shot_indices:
             try:
                 # Find shot corresponding to this *trigger* frame index
                 shot_data_list = [sd for sd in shot_dataset if sd.get('frame_number') == original_frame_num]
                 if shot_data_list:
                     sd = shot_data_list[0] # Assume one trigger per original frame num
                     text = f"SHOT! R{sd.get('rally_id','?')} S{sd.get('shot_num','?')} P{sd.get('player_who_hit','?')}"
                     ts, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                     tx = (self.frame_width - ts[0]) // 2; ty = 40
                     cv2.rectangle(frame_to_write, (tx-5, ty-ts[1]-5), (tx+ts[0]+5, ty+5), (0,0,0), -1)
                     cv2.putText(frame_to_write, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
             except Exception as e: print(f"    [DEBUG VO] Error drawing shot text: {e}", flush=True)

        # --- Frame Number ---
        try: cv2.putText(frame_to_write, f'Frame: {original_frame_num}', (self.frame_width-150, self.frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        except Exception as e: print(f"    [DEBUG VO] Error drawing frame num: {e}", flush=True)

        # --- Write ---
        try: self.out.write(frame_to_write)
        except Exception as e: print(f"    [DEBUG VO] Error writing frame {original_frame_num}: {e}", flush=True)

    def release(self):
        if self.is_opened():
            print("  [DEBUG VO] Releasing VideoOutputWriter...", flush=True)
            self.out.release()
            self.out = None
            print("  [DEBUG VO] VideoOutputWriter released.", flush=True)
        else:
             print("  [DEBUG VO] VideoOutputWriter already released or never opened.", flush=True)