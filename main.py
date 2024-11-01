import cv2
from trackers.player import PlayerDetector
from trackers.shuttle import ShuttleDetector
from trackers.court import CourtDetector
from datetime import datetime
import json

class BadmintonAnalyzer:
    def __init__(self, video_path, conf_threshold=0.3):
        """
        Initialize the BadmintonAnalyzer with video path and configurations
        
        Args:
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
        """
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        
        # Initialize court detector first
        self.court_detector = CourtDetector(conf_threshold)
        
        # Initialize other detectors, passing court detector to player detector
        self.player_detector = PlayerDetector(conf_threshold, court_detector=self.court_detector)
        self.shuttle_detector = ShuttleDetector(conf_threshold)
        
        self.trajectory_data = []
        
    def process_video(self, save_output=True, output_path=None):
        """
        Process the video and track players, shuttle, and court
        
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
                output_path = f'outputs/badminton_analysis_{timestamp}.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        court_coords = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect court boundaries (first frame only)
            if court_coords is None:
                court_coords = self.court_detector.detect_court_boundary(frame)
                if court_coords is None:
                    print("Could not detect court boundaries.")
                    break
            
            # Draw court lines
            if court_coords:
                self.court_detector.draw_court_lines(frame, court_coords)
            
            # Detect players (pass court coordinates)
            player_detections = self.player_detector.detect_players(frame, court_coords)
            
            # Detect shuttlecock
            shuttle_detections = self.shuttle_detector.detect_shuttlecock(frame)
            
            # Combine detections
            frame_detections = player_detections + shuttle_detections
            
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
            
            cv2.imshow('Badminton Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        return self.trajectory_data
        
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

def main():
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(video_path)
    trajectory_data = analyzer.process_video(save_output=True)
    analyzer.save_trajectory_data()

if __name__ == "__main__":
    main()