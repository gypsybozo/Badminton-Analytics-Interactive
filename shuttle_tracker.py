from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle

class ShuttleTracker:
    def __init__(self, model_path, conf_threshold=0.3):
        """
        Initialize the ShuttleTracker with the YOLOv8 model path
        
        Args:
            model_path (str): Path to trained YOLOv8 model weights for object detection
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_labels = {0: "Player", 1: "Racket", 2: "Shuttle"}
        self.shuttle_trajectories = []
        self.shots_output_dir = "shot_images"
        
        # Create directory for shot images if it doesn't exist
        os.makedirs(self.shots_output_dir, exist_ok=True)
    
    def detect_frame(self, frame):
        """
        Detect objects in a single frame
        
        Args:
            frame: Video frame to process
            
        Returns:
            Dictionary of detections by class
        """
        results = self.model(frame, conf=self.conf_threshold)[0]
        
        # Organize detections by class
        detections = {
            "Player": [],
            "Racket": [],
            "Shuttle": []
        }
        
        for detection in results.boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection
            box = [int(x1), int(y1), int(x2), int(y2)]
            confidence = float(confidence)
            class_id = int(class_id)
            
            label = self.class_labels.get(class_id, "Unknown")
            
            detection_dict = {
                'box': box,
                'confidence': confidence,
                'center': self.get_bbox_center(box)
            }
            
            detections[label].append(detection_dict)
            
        return detections
    
    def get_bbox_center(self, box):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)
        
    def process_video(self, video_path, frame_skip=2, save_output=True, output_path=None):
        """
        Process the entire video to track the shuttle and detect shots
        
        Args:
            video_path (str): Path to input video file
            frame_skip (int): Number of frames to skip between processing
            save_output (bool): Whether to save the output video
            output_path (str): Path for output video
            
        Returns:
            DataFrame with shot data
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = original_fps // frame_skip
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'output_badminton_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        # Lists to store shuttle detections per frame
        shuttle_positions = []
        frames_list = []
        frame_count = 0
        
        print("Processing video frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Store original frame
            original_frame = frame.copy()
            frames_list.append(original_frame)
            
            # Detect objects in the frame
            detections = self.detect_frame(frame)
            
            # Store shuttle position (use first shuttle if multiple are detected)
            shuttle_pos = {}
            if detections["Shuttle"]:
                best_shuttle = max(detections["Shuttle"], key=lambda x: x['confidence'])
                shuttle_box = best_shuttle['box']
                shuttle_center = best_shuttle['center']
                shuttle_pos = {
                    2: shuttle_box,
                    'center': shuttle_center
                }
            
            shuttle_positions.append(shuttle_pos)
            
            # Draw detections on frame for visualization
            self.draw_detections(frame, detections)
            
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
        
        print("Video processing complete, analyzing shuttle trajectory...")
        
        # Interpolate missing shuttle positions
        interpolated_positions = self.interpolate_shuttle_positions(shuttle_positions)
        
        # Detect direction changes in shuttle trajectory
        direction_change_frames = self.detect_direction_changes(interpolated_positions)
        print(f"Detected {len(direction_change_frames)} direction changes")
        
        # Save direction change images
        self.save_direction_change_images(frames_list, direction_change_frames, interpolated_positions)
        
        # Create and save detailed data about direction changes
        direction_change_data = self.create_direction_change_data(direction_change_frames, effective_fps)
        
        # Save results as a CSV
        df = pd.DataFrame(direction_change_data)
        csv_filename = f"direction_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Direction change data saved to {csv_filename}")
        
        return df
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detections on the frame"""
        # Draw players (in black)
        for player in detections["Player"]:
            box = player['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 3)
            cv2.putText(frame, f'Player: {player["confidence"]:.2f}', 
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw rackets (in blue)
        for racket in detections["Racket"]:
            box = racket['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
            cv2.putText(frame, f'Racket: {racket["confidence"]:.2f}', 
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw shuttles (in white)
        for shuttle in detections["Shuttle"]:
            box = shuttle['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 3)
            cv2.putText(frame, f'Shuttle: {shuttle["confidence"]:.2f}', 
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def interpolate_shuttle_positions(self, shuttle_positions):
        """
        Interpolate missing shuttle positions using pandas
        
        Args:
            shuttle_positions: List of dictionaries with shuttle positions
            
        Returns:
            List of interpolated shuttle positions
        """
        # Create lists to store x, y coordinates for centers
        centers_x = []
        centers_y = []
        
        # Extract centers from shuttle positions
        for pos in shuttle_positions:
            if pos and 'center' in pos:
                centers_x.append(pos['center'][0])
                centers_y.append(pos['center'][1])
            else:
                centers_x.append(np.nan)
                centers_y.append(np.nan)
        
        # Create DataFrame with centers
        df = pd.DataFrame({
            'x': centers_x,
            'y': centers_y
        })
        
        # Interpolate missing values
        df_interp = df.interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining NaN values at start/end
        df_interp = df_interp.fillna(method='ffill').fillna(method='bfill')
        
        # Convert back to list of dictionaries with proper structure
        interpolated = []
        for i, row in df_interp.iterrows():
            center = (int(round(row['x'])), int(round(row['y'])))
            
            # Estimate a bounding box around the center point
            # Using fixed size of 20x20 pixels for simplicity
            box = [center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10]
            
            interpolated.append({
                2: box,  # Class ID for shuttle
                'center': center,
                'interpolated': i not in shuttle_positions or 2 not in shuttle_positions[i]
            })
                
        return interpolated
    
    def detect_direction_changes(self, shuttle_positions, window_size=3, angle_threshold=15, min_distance=10, min_frames_between=5):
        """
        Detect frames where the shuttle changes direction
        
        Args:
            shuttle_positions: List of dictionaries with shuttle positions
            window_size: Window size for smoothing
            angle_threshold: Minimum angle change to detect (in degrees)
            min_distance: Minimum distance shuttle needs to move to detect direction change
            min_frames_between: Minimum frames between consecutive direction changes
            
        Returns:
            List of frames where direction changes occur
        """
        centers = []
        for pos in shuttle_positions:
            if 'center' in pos:
                centers.append(pos['center'])
            else:
                centers.append((np.nan, np.nan))
        
        # Convert to numpy array for easier calculations
        centers = np.array(centers)
        
        # Calculate displacements between consecutive positions
        displacements = np.zeros((len(centers) - 1, 2))
        for i in range(len(centers) - 1):
            displacements[i] = [centers[i+1][0] - centers[i][0], centers[i+1][1] - centers[i][1]]
        
        # Calculate angles of movement (in radians)
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])
        
        # Calculate distances moved
        distances = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
        
        # Detect significant direction changes
        direction_changes = []
        last_change_frame = -min_frames_between  # Initialize to allow first detection
        
        for i in range(window_size, len(angles) - window_size):
            # Skip if shuttle didn't move enough
            if distances[i] < min_distance:
                continue
                
            # Get average angle before and after current point
            angles_before = angles[i-window_size:i]
            angles_after = angles[i+1:i+window_size+1]
            
            avg_angle_before = np.arctan2(np.sin(angles_before).mean(), np.cos(angles_before).mean())
            avg_angle_after = np.arctan2(np.sin(angles_after).mean(), np.cos(angles_after).mean())
            
            # Calculate absolute angle difference
            angle_diff = abs(avg_angle_after - avg_angle_before)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
                
            # Convert to degrees
            angle_diff_deg = angle_diff * 180 / np.pi
            
            # Check if angle change exceeds threshold and enough frames have passed
            if angle_diff_deg > angle_threshold and i - last_change_frame >= min_frames_between:
                direction_changes.append({
                    'frame': i,
                    'angle_change': angle_diff_deg,
                    'position': centers[i]
                })
                last_change_frame = i
        
        return direction_changes
    
    def save_direction_change_images(self, frames, direction_changes, shuttle_positions):
        """
        Save images of frames where shuttle changes direction
        
        Args:
            frames: List of video frames
            direction_changes: List of dictionaries with direction change info
            shuttle_positions: List of interpolated shuttle positions
        """
        for idx, change in enumerate(direction_changes):
            frame_idx = change['frame']
            
            # Skip if frame is out of range
            if frame_idx >= len(frames):
                continue
                
            # Get the frame
            frame = frames[frame_idx].copy()
            
            # Draw trail of shuttle positions (recent past positions)
            trail_length = 10
            for i in range(max(0, frame_idx - trail_length), frame_idx + 1):
                if i < len(shuttle_positions) and 'center' in shuttle_positions[i]:
                    center = shuttle_positions[i]['center']
                    # Color gradient from red (oldest) to yellow (newest)
                    progress = (i - (frame_idx - trail_length)) / trail_length
                    color = (0, int(255 * (1 - progress)), 255)  # BGR: Blue to Cyan
                    cv2.circle(frame, center, 4, color, -1)
            
            # Draw future trail to show direction change
            for i in range(frame_idx + 1, min(frame_idx + trail_length + 1, len(shuttle_positions))):
                if i < len(shuttle_positions) and 'center' in shuttle_positions[i]:
                    center = shuttle_positions[i]['center']
                    # Color gradient from yellow to green
                    progress = (i - frame_idx) / trail_length
                    color = (0, 255, int(255 * (1 - progress)))  # BGR: Yellow to Green
                    cv2.circle(frame, center, 4, color, -1)
            
            # Draw current shuttle position
            if 'center' in shuttle_positions[frame_idx]:
                center = shuttle_positions[frame_idx]['center']
                cv2.circle(frame, center, 8, (255, 255, 255), -1)
                cv2.circle(frame, center, 8, (0, 0, 0), 2)  # Black outline
            
            # Add annotation overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add text info
            cv2.putText(frame, f"Direction Change #{idx+1}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle Change: {change['angle_change']:.1f} degrees", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the image
            filename = f"direction_change_{idx+1:03d}_frame_{frame_idx}.jpg"
            cv2.imwrite(os.path.join(self.shots_output_dir, filename), frame)
    
    def create_direction_change_data(self, direction_changes, effective_fps):
        """
        Create structured data about each direction change
        
        Args:
            direction_changes: List of dictionaries with direction change info
            effective_fps: Effective frames per second after frame skipping
            
        Returns:
            List of dictionaries with direction change data
        """
        data = []
        
        for idx, change in enumerate(direction_changes):
            # Calculate time between consecutive direction changes
            time_since_prev = 0
            if idx > 0:
                time_since_prev = (change['frame'] - direction_changes[idx-1]['frame']) / effective_fps
            
            entry = {
                'change_id': idx + 1,
                'frame': change['frame'],
                'timestamp': change['frame'] / effective_fps,
                'time_since_prev_change': time_since_prev,
                'angle_change_degrees': change['angle_change'],
                'position_x': change['position'][0],
                'position_y': change['position'][1]
            }
            
            data.append(entry)
            
        return data