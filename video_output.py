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
                      confirmed_shot_indices, shot_dataset, draw_pose,
                      current_detections, get_player_positions_func,
                      inverse_homography_matrix, real_world_test_points): 
        """Draws annotations and writes the frame."""
        if not self.is_opened() or frame is None: return

        frame_to_write = frame.copy()

        # --- Draw Pose Landmarks ---
        if draw_pose:
            frame_to_write = self._draw_pose(frame_to_write, p1_landmarks, p1_box)
            frame_to_write = self._draw_pose(frame_to_write, p2_landmarks, p2_box)

        # --- Draw Court Lines ---
        if court_corners:
            try: 
                corner_points = np.array(court_corners, dtype=np.int32)
                cv2.polylines(frame_to_write, [corner_points], isClosed=True, color=(0, 255, 255), thickness=2)
            except Exception as e: 
                print(f"    [DEBUG VO] Error drawing court: {e}", flush=True)
                
        # --- Draw Projected Real-World Test Points ---
        if inverse_homography_matrix is not None and real_world_test_points:
            print(f"  [DEBUG VO Frame {original_frame_num}] Drawing test points...") # Verbose
            for name, real_point_h in real_world_test_points.items():
                try:
                    image_point_h = np.dot(inverse_homography_matrix, real_point_h)
                    if abs(image_point_h[2]) > 1e-6:
                        image_point = (int(image_point_h[0] / image_point_h[2]),
                                       int(image_point_h[1] / image_point_h[2]))
                        # Draw Green Circles for Test Points
                        cv2.circle(frame_to_write, image_point, 8, (0, 255, 0), 2)
                        cv2.putText(frame_to_write, name[:4], (image_point[0]+10, image_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    else: print(f"    [DEBUG VO] Skipping {name} - Z near zero") # Verbose
                except Exception as e: print(f"    [DEBUG VO] Error projecting/drawing {name}: {e}", flush=True)
        # --- Draw Player Positions ---  
        if inverse_homography_matrix is not None and get_player_positions_func and current_detections:
            try:
                # Calculate real-world positions for *this specific frame*
                p_real, _, _ = get_player_positions_func(current_detections)
                p1_real, p2_real = p_real

                # Project P1
                if p1_real:
                    p1_real_h = np.array([p1_real[0], p1_real[1], 1], dtype=np.float32)
                    p1_img_h = np.dot(inverse_homography_matrix, p1_real_h)
                    if abs(p1_img_h[2]) > 1e-6:
                        p1_img_proj = (int(p1_img_h[0]/p1_img_h[2]), int(p1_img_h[1]/p1_img_h[2]))
                        # Draw Blue 'X' at projected real-world location
                        cv2.drawMarker(frame_to_write, p1_img_proj, (255, 100, 0), cv2.MARKER_CROSS, 20, 2)
                        # Optionally draw the original point used for homography (bottom-center)
                        if p1_box: 
                            cv2.circle(frame_to_write, ( (p1_box[0]+p1_box[2])//2, p1_box[3] ), 5, (255, 100, 0), -1)
                            cv2.putText(frame_to_write, f"P1({p1_real[0]:.1f},{p1_real[1]:.1f})", (p1_img_proj[0]+5, p1_img_proj[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,100,0), 1)
                # Project P2
                if p2_real:
                    p2_real_h = np.array([p2_real[0], p2_real[1], 1], dtype=np.float32)
                    p2_img_h = np.dot(inverse_homography_matrix, p2_real_h)
                    if abs(p2_img_h[2]) > 1e-6:
                        p2_img_proj = (int(p2_img_h[0]/p2_img_h[2]), int(p2_img_h[1]/p2_img_h[2]))
                        # Draw Red '+' at projected real-world location
                        cv2.drawMarker(frame_to_write, p2_img_proj, (0, 100, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                        # Optionally draw the original point used for homography (bottom-center)
                        if p2_box: 
                            cv2.circle(frame_to_write, ( (p2_box[0]+p2_box[2])//2, p2_box[3] ), 5, (0, 100, 255), -1)
                            cv2.putText(frame_to_write, f"P2({p2_real[0]:.1f},{p2_real[1]:.1f})", (p2_img_proj[0]+5, p2_img_proj[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,255), 1)
            except Exception as e: print(f"    [DEBUG VO] Error projecting/drawing player positions: {e}", flush=True)

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