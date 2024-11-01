import cv2
import base64
from inference_sdk import InferenceHTTPClient

class ShuttleDetector:
    def __init__(self, conf_threshold=0.3):
        """
        Initialize the ShuttleDetector
        
        Args:
            conf_threshold (float): Confidence threshold for detections
        """
        self.shuttle_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="3zcYipvGszKYVhtJ8E5J"
        )
        self.conf_threshold = conf_threshold
        
    def detect_shuttlecock(self, frame):
        """
        Detect shuttlecock in a single frame
        
        Args:
            frame: Input frame to process
            
        Returns:
            list: Shuttlecock detections
        """
        shuttle_detections = []
        
        # Encode frame for Roboflow API
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Detect shuttlecock
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
                    
                    shuttle_detections.append({
                        'type': 'shuttle',
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
                    
                    # Draw shuttle bounding box in green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Shuttle: {confidence:.2f}', 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return shuttle_detections