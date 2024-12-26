import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
import json
from trackers.court import CourtDetector 


class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3, frame_skip=5):
        """
        Initialize the BadmintonAnalyzer with model and video paths
        
        Args:
            shuttle_model_path (str): Path to trained YOLOv8 model weights for shuttle detection
            court_model_path (str): Path to trained YOLOv8 model weights for court detection
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(shuttle_model_path)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.trajectory_data = []
        # Initialize court detector
        self.court_detector = CourtDetector(
            conf_threshold=conf_threshold,
            model_path=court_model_path
        )
        self.court_coords = None
        self.court_corners = None
        
        self.class_labels = {0: "Player", 1: "Racket", 2: "Shuttle"}
        self.prev_shuttle_pos = None
        self.max_shuttle_movement = 400  # Maximum pixel distance shuttle can move between frames
        
    def get_bbox_center(self, box):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def select_best_shuttle_detection(self, shuttle_detections):
        """
        Select the most likely shuttle detection based on previous position and confidence
        
        Args:
            shuttle_detections: List of dictionaries containing shuttle detections
                              Each detection should have 'box' and 'confidence' keys
        
        Returns:
            Best shuttle detection or None if no valid detection found
        """
        if not shuttle_detections:
            return None
        
        # If there's only one detection, select it automatically
        if len(shuttle_detections) == 1:
            best_detection = shuttle_detections[0]
            self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
            return best_detection
     
        # If this is the first detection, choose the highest confidence detection
        if self.prev_shuttle_pos is None:
            best_detection = max(shuttle_detections, key=lambda x: x['confidence'])
            self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
            return best_detection
        

        best_detection = None
        min_distance_score = float('inf')
        
        for detection in shuttle_detections:
            current_center = self.get_bbox_center(detection['box'])
            distance = self.calculate_distance(current_center, self.prev_shuttle_pos)
            
            # # Skip if movement is impossibly large
            if distance > self.max_shuttle_movement:
                continue
                
            # Create a score that balances distance and confidence
            # Lower score is better
            distance_score = distance * (1.0 - detection['confidence'])
            
            if distance_score < min_distance_score:
                min_distance_score = distance_score
                best_detection = detection
        
        if best_detection:
            self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
        else:
            # If no valid detection found, keep the previous position but return None
            pass
            
        return best_detection  
      
    def determine_player(self, player_position):
        """Determine if player is Player 1 (back) or 2 (front) based on y-coordinate"""
        if player_position[1] > 6.7:  # Middle of court is 13.4/2
            return 1
        return 2

    def process_video(self, save_output=True, output_path=None):
        cap = cv2.VideoCapture(self.video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = original_fps // self.frame_skip
        self.shot_data = []
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'output_video_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        frame_count = 0
        court_detected = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
                
            if not court_detected:
                self.court_coords = self.court_detector.detect_court_boundary(frame)
                if self.court_coords is not None:
                    self.court_coords = self.court_detector.sort_court_coords(self.court_coords)
                    self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
                    self.homography_matrix = self.court_detector.compute_homography(self.court_corners)
                    court_detected = True
            elif self.court_coords is not None:
                self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
            
            results = self.model(frame, conf=self.conf_threshold)[0]
            frame_detections = []
            shuttle_detections = []

            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                box = [int(x1), int(y1), int(x2), int(y2)]
                confidence = float(confidence)
                class_id = int(class_id)
                
                detection_dict = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / effective_fps,
                    'box': box,
                    'confidence': confidence,
                    'label': self.class_labels.get(class_id, "Unknown")
                }
                
                if class_id == 2:  # Shuttle
                    shuttle_detections.append(detection_dict)
                else:
                    frame_detections.append(detection_dict)
                    
                    # Draw non-shuttle detections
                    color = (0, 0, 0) if class_id == 0 else (255, 0, 0)  # Black for players, blue for rackets
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(frame, f'{self.class_labels.get(class_id)}: {confidence:.2f}', 
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process shuttle detections
            best_shuttle = self.select_best_shuttle_detection(shuttle_detections)
            if best_shuttle:
                frame_detections.append(best_shuttle)
                box = best_shuttle['box']
                # Draw shuttle detection in white
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 3)
                cv2.putText(frame, f'Shuttle: {best_shuttle["confidence"]:.2f}', 
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.trajectory_data.extend(frame_detections)

            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            cv2.imshow('Badminton Analysis', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)
    analyzer.process_video(save_output=True)
    