import cv2
from cv2 import legacy
import numpy as np
import base64
from datetime import datetime
import json
from inference_sdk import InferenceHTTPClient

# Set up the Roboflow API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3zcYipvGszKYVhtJ8E5J"
)

class BadmintonAnalyzer:
    def __init__(self, video_path, conf_threshold=0.3):
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.trajectory_data = []
        self.tracker = None
        self.tracking = False
        self.last_detection = None
        self.last_velocity = None
        self.search_window_size = 150  # Increased search window size
        
    def initialize_tracker(self, frame, bbox):
        """Initialize the tracker with a larger search area"""
        try:
            self.tracker = cv2.legacy.TrackerCSRT.create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT.create()
            
        # Expand the bounding box slightly to help with fast movement
        x, y, w, h = bbox
        w = int(w * 1.2)  # Make box 20% wider
        h = int(h * 1.2)  # Make box 20% taller
        expanded_bbox = (x, y, w, h)
        
        self.tracking = self.tracker.init(frame, expanded_bbox)
    
    def calculate_velocity(self, current_pos, last_pos, dt=1.0):
        """Calculate velocity between two points"""
        if last_pos is None:
            return None
        return (current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
    
    def predict_next_position(self, last_pos, velocity):
        """Predict next position based on velocity"""
        if last_pos is None or velocity is None:
            return None
        return (int(last_pos[0] + velocity[0]), int(last_pos[1] + velocity[1]))
    
    def get_search_region(self, frame, last_pos, velocity):
        """Get search region based on last position and velocity"""
        h, w = frame.shape[:2]
        if last_pos is None:
            return (0, 0, w, h)
            
        # Use velocity to adjust search window if available
        if velocity is not None:
            offset_x = int(velocity[0] * 1.5)  # Look ahead in direction of movement
            offset_y = int(velocity[1] * 1.5)
        else:
            offset_x = offset_y = 0
            
        center_x = last_pos[0] + offset_x
        center_y = last_pos[1] + offset_y
        
        # Create search window
        x1 = max(0, int(center_x - self.search_window_size))
        y1 = max(0, int(center_y - self.search_window_size))
        x2 = min(w, int(center_x + self.search_window_size))
        y2 = min(h, int(center_y + self.search_window_size))
        
        return (x1, y1, x2-x1, y2-y1)
    
    def process_video(self, save_output=True, output_path=None):
        cap = cv2.VideoCapture(self.video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'output_video_{timestamp}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        last_center = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detected = False
            
            # Convert frame to base64
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            # Run inference
            result = CLIENT.infer(img_base64, model_id="shuttlecock-cqzy3/1")
            
            # Process detections
            if 'predictions' in result:
                for detection in result['predictions']:
                    confidence = detection['confidence']
                    
                    if confidence >= self.conf_threshold:
                        # Calculate bounding box
                        x1, y1, width, height = detection['x'], detection['y'], detection['width'], detection['height']
                        x1 = int(x1 - width / 2)
                        y1 = int(y1 - height / 2)
                        x2 = int(x1 + width)
                        y2 = int(y1 + height)
                        bbox = (x1, y1, width, height)
                        
                        # Calculate center and velocity
                        current_center = (int(x1 + width/2), int(y1 + height/2))
                        self.last_velocity = self.calculate_velocity(current_center, last_center)
                        last_center = current_center
                        
                        # Initialize tracker with the detected bounding box
                        self.initialize_tracker(frame, bbox)
                        detected = True
                        self.last_detection = bbox
                        
                        # Store detection data
                        self.trajectory_data.append({
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'detection_type': 'direct'
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Shuttle: {confidence:.2f}', 
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
            
            # Use tracker when no detection
            if not detected and self.tracker is not None and self.tracking and self.last_detection is not None:
                # Get search region based on last known position and velocity
                if last_center is not None:
                    search_bbox = self.get_search_region(frame, last_center, self.last_velocity)
                    
                    # Update tracker
                    self.tracking, bbox = self.tracker.update(frame)
                    
                    if self.tracking:
                        x1, y1, width, height = map(int, bbox)
                        x2, y2 = x1 + width, y1 + height
                        current_center = (int(x1 + width/2), int(y1 + height/2))
                        
                        # Update velocity
                        self.last_velocity = self.calculate_velocity(current_center, last_center)
                        last_center = current_center
                        
                        # Store tracking data
                        self.trajectory_data.append({
                            'frame_number': frame_count,
                            'timestamp': frame_count / fps,
                            'box': [x1, y1, x2, y2],
                            'confidence': 'tracked',
                            'detection_type': 'tracked'
                        })
                        
                        # Draw tracking box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, 'Shuttle: Tracked', 
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        # Reset tracker if tracking fails
                        self.tracker = None
                        self.last_velocity = None
            
            # Display frame number
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            cv2.imshow('Badminton Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
    def save_trajectory_data(self, output_path=None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'trajectory_data_{timestamp}.json'
            
        with open(output_path, 'w') as f:
            json.dump(self.trajectory_data, f, indent=4)
        
        print(f"Trajectory data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(video_path)
    analyzer.process_video(save_output=True)
    analyzer.save_trajectory_data()