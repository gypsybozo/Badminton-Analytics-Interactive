import cv2 
from ultralytics import YOLO 
import numpy as np

class PlayerDetector:
    def __init__(self, conf_threshold=0.3, court_detector=None):
        """
        Initialize the PlayerDetector
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            court_detector (CourtDetector): Optional court detector to validate player positions
        """
        self.model = YOLO('yolov8x.pt')
        self.conf_threshold = conf_threshold
        self.court_detector = court_detector
        
    def is_point_inside_court(self, point, court_coords):
        """
        Check if a point is inside the court boundaries
        
        Args:
            point (tuple): (x, y) coordinates of the point
            court_coords (list): List of court boundary coordinates
        
        Returns:
            bool: True if point is inside court, False otherwise
        """
        if not court_coords:
            return True  # If no court coordinates, assume all players are inside
        
        # Use point-in-polygon algorithm (ray casting method)
        x, y = point
        inside = False
        
        for i in range(len(court_coords)):
            j = (i + 1) % len(court_coords)
            xi, yi = court_coords[i][0], court_coords[i][1]
            xj, yj = court_coords[j][0], court_coords[j][1]
            
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
        
        return inside
    
    def detect_players(self, frame, court_coords=None):
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame to process
            court_coords: Optional court boundary coordinates
        
        Returns:
            list: Player detections within court boundaries
        """
        # If court_coords not provided, use the court_detector if available
        if court_coords is None and self.court_detector:
            court_coords = self.court_detector.detect_court_boundary(frame)
        
        player_detections = []
        
        # Detect players
        results = self.model(frame, classes=[0])  # class 0 is person
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate the lower center point of the bounding box
                    lower_center_point = (int((x1 + x2) // 2), y2)  # Use y2 for the lower edge
                    
                    # Check if the lower part of the player is inside the court
                    if court_coords is None or self.is_point_inside_court(lower_center_point, court_coords):
                        player_detection = {
                            'type': 'player',
                            'box': [x1, y1, x2, y2],
                            'confidence': conf,
                            'center': lower_center_point
                        }
                        player_detections.append(player_detection)
                        
                        # Draw player bounding box in blue
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'Player: {conf:.2f}',
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return player_detections
