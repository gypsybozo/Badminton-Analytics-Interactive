import cv2
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, conf_threshold=0.3):
        """
        Initialize the PlayerDetector
        
        Args:
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO('yolov8x.pt')
        self.conf_threshold = conf_threshold
        
    def detect_players(self, frame):
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame to process
            
        Returns:
            list: Player detections
        """
        player_detections = []
        
        # Detect players
        results = self.model(frame, classes=[0])  # class 0 is person
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_detections.append({
                        'type': 'player',
                        'box': [x1, y1, x2, y2],
                        'confidence': conf
                    })
                    
                    # Draw player bounding box in blue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'Player: {conf:.2f}', 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return player_detections
    
    