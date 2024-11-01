import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3zcYipvGszKYVhtJ8E5J"
)


class BadmintonAnalyzer:
    def __init__(self, model_path, video_path, conf_threshold=0.3):
        """
        Initialize the BadmintonAnalyzer with model and video paths
        
        Args:
            model_path (str): Path to trained YOLOv8 model weights
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.trajectory_data = []
        
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLOv8 inference on the frame
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            # Process detections
            frame_detections = []
            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                
                # Convert tensor values to integers
                box = [int(x1), int(y1), int(x2), int(y2)]
                confidence = float(confidence)
                
                # Store detection data
                frame_detections.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'box': box,
                    'confidence': confidence
                })
                
                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'Shuttle: {confidence:.2f}', 
                           (box[0], box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.trajectory_data.extend(frame_detections)
            
            # Display frame number
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            # Display the frame
            cv2.imshow('Badminton Analysis', frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        # Clean up
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
    def save_trajectory_data(self, output_path=None):
        """
        Save the trajectory data to a JSON file
        
        Args:
            output_path (str): Path to save the JSON file (optional)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'trajectory_data_{timestamp}.json'
            
        with open(output_path, 'w') as f:
            json.dump(self.trajectory_data, f, indent=4)
        
        print(f"Trajectory data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    model_path = "models/shuttle/v8_after_change/best.pt"
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(model_path, video_path)
    analyzer.process_video(save_output=True)
    analyzer.save_trajectory_data()