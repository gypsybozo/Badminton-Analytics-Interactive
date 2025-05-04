# frame_processor.py
import cv2
import mediapipe as mp
import numpy as np
import time
import traceback
from utils.constants import PLAYER_CLASS_ID, RACKET_CLASS_ID, SHUTTLE_CLASS_ID, POSE_VISIBILITY_THRESHOLD
from utils.geom import crop_image_safely

class FrameProcessor:
    def __init__(self, yolo_model, class_labels):
        print("  [DEBUG] Initializing FrameProcessor...", flush=True)
        self.yolo_model = yolo_model
        self.class_labels = class_labels
        self.mp_pose = mp.solutions.pose
        try:
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False, model_complexity=1,
                smooth_landmarks=True, enable_segmentation=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except Exception as e:
            print(f"ERROR initializing MediaPipe Pose: {e}", flush=True)
            raise # Re-raise error if critical component fails
        print("  [DEBUG] FrameProcessor initialized.", flush=True)

    def _get_player_pose_data(self, player_image):
        """ Processes a player image with MediaPipe Pose """
        if player_image is None or player_image.size == 0: return None
        try:
            image_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose_estimator.process(image_rgb)
            image_rgb.flags.writeable = True
            return results.pose_landmarks
        except Exception as e:
            print(f"    [DEBUG POSE PROC] Error processing player pose: {e}\n{traceback.format_exc()}", flush=True)
            return None

    def get_wrist_keypoints(self, pose_landmarks, img_width, img_height):
        """ Extracts wrist keypoints (image coords relative to **input image dimensions**) """
        if pose_landmarks is None or img_width <= 0 or img_height <= 0: return None, None
        try:
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lw_coords, rw_coords = None, None
            if left_wrist.visibility > POSE_VISIBILITY_THRESHOLD:
                 lw_coords = (int(left_wrist.x * img_width), int(left_wrist.y * img_height))
            if right_wrist.visibility > POSE_VISIBILITY_THRESHOLD:
                 rw_coords = (int(right_wrist.x * img_width), int(right_wrist.y * img_height))
            return lw_coords, rw_coords
        except IndexError: # Handle cases where landmark might not exist
            print(f"    [DEBUG WRIST] Error: Pose landmark index out of bounds.", flush=True)
            return None, None
        except Exception as e:
            print(f"    [DEBUG WRIST] Error extracting wrist keypoints: {e}", flush=True)
            return None, None

    def process_frame(self, frame, frame_num, conf_threshold, player_boxes_prev):
        """Processes a single frame for detections and poses."""
        frame_process_start = time.time()
        print(f"--- [DEBUG FP] Processing Frame {frame_num} ---", flush=True)

        # 1. YOLO Detection
        yolo_start = time.time()
        detections = []
        shuttle_candidates = []
        yolo_results = None
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                raise ValueError("Invalid frame provided to YOLO model")
            yolo_results = self.yolo_model(frame, conf=conf_threshold, verbose=False)[0]
            yolo_dur = time.time() - yolo_start
            # print(f"  [DEBUG FP] YOLO Done. Duration: {yolo_dur:.4f}s", flush=True) # Verbose

            if yolo_results.boxes is not None and yolo_results.boxes.data is not None:
                for detection in yolo_results.boxes.data:
                    try:
                        # (Safety checks as before)
                        x1, y1, x2, y2, confidence, class_id_tensor = detection.cpu().split([1,1,1,1,1,1], dim=0)
                        class_id = int(class_id_tensor.item())
                        label = self.class_labels.get(class_id, "Unknown")
                        if class_id not in [PLAYER_CLASS_ID, RACKET_CLASS_ID, SHUTTLE_CLASS_ID]: continue
                        det_info = {
                            'box': [int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())],
                            'confidence': float(confidence.item()),
                            'class_id': class_id, 'label': label}
                        detections.append(det_info)
                        if class_id == SHUTTLE_CLASS_ID: shuttle_candidates.append(det_info)
                    except Exception as e: print(f"    [DEBUG FP] Error processing YOLO box data: {e}", flush=True)
        except Exception as e:
            print(f"  [DEBUG FP] ERROR during YOLO inference frame {frame_num}: {e}\n{traceback.format_exc()}", flush=True)
            return None, None, None, None # Indicate failure, return 4 Nones

        # 2. Pose Estimation
        p1_landmarks, p2_landmarks = None, None
        pose_total_dur = 0
        p1_box_passed, p2_box_passed = player_boxes_prev # Use boxes passed from Analyzer
        pose_total_duration = 0

        if p1_box_passed:
            pose_p1_start = time.time()
            player1_img = crop_image_safely(frame, p1_box_passed)
            if player1_img is not None:
                p1_landmarks = self._get_player_pose_data(player1_img)
            pose_p1_dur = time.time() - pose_p1_start
            pose_total_dur += pose_p1_dur
            # print(f"  [DEBUG FP] P1 Pose Done. Landmarks found: {p1_landmarks is not None}. Dur: {pose_p1_dur:.4f}s", flush=True) # Verbose

        if p2_box_passed:
            pose_p2_start = time.time()
            player2_img = crop_image_safely(frame, p2_box_passed)
            if player2_img is not None:
                p2_landmarks = self._get_player_pose_data(player2_img)
            pose_p2_dur = time.time() - pose_p2_start
            pose_total_dur += pose_p2_dur
            # print(f"  [DEBUG FP] P2 Pose Done. Landmarks found: {p2_landmarks is not None}. Dur: {pose_p2_dur:.4f}s", flush=True) # Verbose

        frame_process_dur = time.time() - frame_process_start
        print(f"--- [DEBUG FP] Finished Frame {frame_num}. Total Dur: {frame_process_dur:.4f}s (Pose Total: {pose_total_dur:.4f}s) ---", flush=True)

        return detections, shuttle_candidates, (p1_landmarks, p2_landmarks), pose_total_duration # Return pose duration too