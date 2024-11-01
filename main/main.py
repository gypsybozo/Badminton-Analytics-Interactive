import cv2
import numpy as np
import base64
from datetime import datetime
import json
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

class EnhancedBadmintonAnalyzer:
    def __init__(self, video_path, conf_threshold=0.3):
        """
        Initialize the EnhancedBadmintonAnalyzer with video path
        
        Args:
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
        """
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.trajectory_data = []
        
        # Initialize both models
        self.shuttle_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="3zcYipvGszKYVhtJ8E5J"
        )
        self.yolo_model = YOLO('yolov8x.pt')
        
    def process_frame(self, frame):
        """
        Process a single frame using both models
        
        Args:
            frame: Input frame to process
            
        Returns:
            list: Detections for the frame
        """
        frame_detections = []
        
        # 1. Detect players using YOLOv8
        yolo_results = self.yolo_model(frame, classes=[0])  # class 0 is person
        
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame_detections.append({
                        'type': 'player',
                        'box': [x1, y1, x2, y2],
                        'confidence': conf
                    })
                    
                    # Draw player bounding box in blue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Player: {conf:.2f}', 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 2. Detect shuttlecock using Roboflow API
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        shuttle_result = self.shuttle_client.infer(img_base64, model_id="shuttlecock-cqzy3/1")
        
        if 'predictions' in shuttle_result:
            for detection in shuttle_result['predictions']:
                confidence = detection['confidence']
                if confidence >= self.conf_threshold:
                    x1, y1, width, height = detection['x'], detection['y'], detection['width'], detection['height']
                    x1 = int(x1 - width / 2)
                    y1 = int(y1 - height / 2)
                    x2 = int(x1 + width)
                    y2 = int(y1 + height)
                    
                    frame_detections.append({
                        'type': 'shuttle',
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
                    
                    # Draw shuttle bounding box in green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Shuttle: {confidence:.2f}', 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_detections
        
    def process_video(self, save_output=True, output_path=None):
        """
        Process the video and track both players and shuttle positions
        
        Args:
            save_output (bool): Whether to save the processed video
            output_path (str): Path to save the output video (optional)
        """
        cap = cv2.VideoCapture(self.video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'outputs/op_video_{timestamp}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with both models
            frame_detections = self.process_frame(frame)
            
            # Add frame information to detections
            for detection in frame_detections:
                detection.update({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps
                })
            
            self.trajectory_data.extend(frame_detections)
            
            # Display frame number
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            cv2.imshow('Enhanced Badminton Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
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
    video_path = "input/video.mov"
    
    analyzer = EnhancedBadmintonAnalyzer(video_path)
    analyzer.process_video(save_output=True)
    # analyzer.save_trajectory_data()