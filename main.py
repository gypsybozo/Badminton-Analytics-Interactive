import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
from trackers.court import CourtDetector 

class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3):
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
        self.trajectory_data = []
        
        # Initialize court detector
        self.court_detector = CourtDetector(
            conf_threshold=conf_threshold,
            model_path=court_model_path
        )
        self.court_coords = None
        self.court_corners = None
        
        self.class_labels = {0: "Player", 1: "Racket", 2: "Shuttle"}
        
    def process_video(self, save_output=True, output_path=None):
        """
        Process the video and track shuttle positions
        
        Args:
            save_output (bool): Whether to save the processed video
            output_path (str): Path to save the output video (optional)
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup video writer if saving output
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'output_video_{timestamp}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        court_detected = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect court in the first frames until successful
            if not court_detected:
                self.court_coords = self.court_detector.detect_court_boundary(frame)
                if self.court_coords is not None:
                    court_detected = True
                    self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
            elif self.court_coords is not None:
                self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
            
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            # Process detections
            frame_detections = []
            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                
                # Convert tensor values to integers
                box = [int(x1), int(y1), int(x2), int(y2)]
                confidence = float(confidence)
                class_id = int(class_id) 
                color = () 
                if class_id==0:
                    color = 0, 0 , 0 #player
                elif class_id == 1:
                    color = 255, 0, 0 #racket
                else:
                    color = 255, 255, 255 #shuttle
                
                label = self.class_labels.get(class_id, "Unknown")  
                
                frame_detections.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'box': box,
                    'confidence': confidence,
                    'label': label 
                })
                
                # Draw
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                cv2.putText(frame, f'{label}: {confidence:.2f}', 
                           (box[0], box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.trajectory_data.extend(frame_detections)
            
            #Frame number
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            # Display the frame
            cv2.imshow('Badminton Analysis', frame)
            
            frame_count += 1
        
        # Clean up
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    shuttle_model_path = "models/shuttle_player_racket/30epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)
    analyzer.process_video(save_output=True)