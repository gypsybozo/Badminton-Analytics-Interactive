import cv2
from ultralytics import YOLO
from pathlib import Path

class CourtDetector:
    def __init__(self, conf_threshold=0.3, model_path='models/court_detection/best.pt'):
        """
        Initialize the CourtDetector
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            model_path (str): Path to the court detection model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Court detection model not found at {model_path}")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect_court_boundary(self, frame):
        """
        Detect court boundary in the frame
        
        Args:
            frame: Input frame
        
        Returns:
            court_coords: Coordinates of the four corners of the court if detected, else None
        """
        court_coords = []
        results = self.model(frame, conf=self.conf_threshold)
        
        # Loop through detected objects to find court boundary
        for result in results:
            for box in result.boxes:
                # Assume the court boundary class ID is known, e.g., 0
                if box.cls[0] == 0:  # Change this ID based on your model's court boundary class ID
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # Return coordinates of the boundary as a tuple of points
                    court_coords.append((x1, y1, x2, y2))
        
        if len(court_coords) == 4:
            return court_coords  # Return the four bounding boxes
        return None  # Continue detection if fewer than four boxes are found
    
    def draw_court_lines(self, frame, court_coords):
        """
        Draw court lines on the frame
        
        Args:
            frame: Input frame to draw lines on
            court_coords: Coordinates of the court boundaries
        """
        # Draw lines connecting the court boundary points
        cv2.line(frame, (court_coords[0][2], court_coords[0][3]), (court_coords[1][0], court_coords[1][3]), (255, 255, 255), 5)
        cv2.line(frame, (court_coords[1][0], court_coords[1][3]), (court_coords[2][0], court_coords[2][1]), (255, 255, 255), 5)
        cv2.line(frame, (court_coords[2][0], court_coords[2][1]), (court_coords[3][2], court_coords[3][1]), (255, 255, 255), 5)
        cv2.line(frame, (court_coords[3][2], court_coords[3][1]), (court_coords[0][2], court_coords[0][3]), (255, 255, 255), 5)
        
        #save the actual corners
        actual_bounds=[]
        actual_bounds.append((court_coords[0][2], court_coords[0][3]))
        actual_bounds.append((court_coords[1][0], court_coords[1][3]))
        actual_bounds.append((court_coords[2][0], court_coords[2][1]))
        actual_bounds.append((court_coords[3][2], court_coords[3][1]))
        # print(actual_bounds)
        return actual_bounds