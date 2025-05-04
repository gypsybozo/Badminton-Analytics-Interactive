# shuttle_detector.py
import numpy as np

class ShuttleDetector:
    def __init__(self, max_shuttle_movement=400, min_shuttle_movement=10):
        """
        Initialize the ShuttleDetector
        
        Args:
            max_shuttle_movement: Maximum pixel distance shuttle can move between frames
            min_shuttle_movement: Minimum movement to consider shuttle as "moving"
        """
        self.prev_shuttle_pos = None
        self.max_shuttle_movement = max_shuttle_movement
        self.min_shuttle_movement = min_shuttle_movement
    
    def get_bbox_center(self, box):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def select_best_shuttle_detection(self, shuttle_detections):
        """
        Select the most likely shuttle detection based on movement from previous frame
        
        Args:
            shuttle_detections: List of dictionaries containing shuttle detections
                              Each detection should have 'box' and 'confidence' keys
        
        Returns:
            Best shuttle detection or None if no valid detection found
        """
        if not shuttle_detections:
            return None
            
        # If first detection, choose highest confidence detection
        if self.prev_shuttle_pos is None:
            best_detection = max(shuttle_detections, key=lambda x: x['confidence'])
            self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
            return best_detection
            
        moving_shuttles = []
        for detection in shuttle_detections:
            current_center = self.get_bbox_center(detection['box'])
            distance = self.calculate_distance(current_center, self.prev_shuttle_pos)
            
            # Skip if movement is impossibly large
            if distance > self.max_shuttle_movement:
                continue
                
            # Only consider shuttles that have moved more than minimum threshold
            if distance > self.min_shuttle_movement:
                moving_shuttles.append({
                    'detection': detection,
                    'distance': distance,
                    'center': current_center
                })
        
        # If we found moving shuttles, pick the one with highest confidence
        if moving_shuttles:
            best_candidate = max(moving_shuttles, 
                               key=lambda x: x['detection']['confidence'])
            self.prev_shuttle_pos = best_candidate['center']
            return best_candidate['detection']
        
        # If no moving shuttles found, fallback to highest confidence detection
        best_detection = max(shuttle_detections, key=lambda x: x['confidence'])
        self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
        return best_detection
    
    def reset(self):
        """Reset the tracker state"""
        self.prev_shuttle_pos = None