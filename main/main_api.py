import cv2
import numpy as np
import base64
from datetime import datetime
import json
from inference_sdk import InferenceHTTPClient

# Initialize the inference clients
CLIENT_HUMAN_RACKET = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3zcYipvGszKYVhtJ8E5J"
)

# CLIENT_SHUTTLE = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="3zcYipvGszKYVhtJ8E5J"  # Replace with your actual API key for shuttle detection
# )

def detect_humans_and_rackets(frame):
    """Detect humans and rackets in the frame."""
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    result = CLIENT_HUMAN_RACKET.infer(img_base64, model_id="badminton-crsqf/1")
    
    return result

# def detect_shuttles(frame):
#     """Detect shuttles in the frame."""
#     _, img_encoded = cv2.imencode('.jpg', frame)
#     img_base64 = base64.b64encode(img_encoded).decode('utf-8')

#     result = CLIENT_SHUTTLE.infer(img_base64, model_id="shuttlecock-cqzy3/1")
    
#     return result

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Detect humans and rackets
        human_racket_detections = detect_humans_and_rackets(frame)
        
        if 'predictions' in human_racket_detections:
            for detection in human_racket_detections['predictions']:
                x1, y1, x2, y2 = detection['x'], detection['y'], detection['width'], detection['height']
                confidence = detection['confidence']
                label = detection['class']  # Adjust according to your API response

                # Draw bounding box for humans and rackets
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}: {confidence:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Step 2: Detect shuttles
        # shuttle_detections = detect_shuttles(frame)
        
        # if 'predictions' in shuttle_detections:
        #     for detection in shuttle_detections['predictions']:
        #         x1, y1, x2, y2 = detection['x'], detection['y'], detection['width'], detection['height']
        #         confidence = detection['confidence']

        #         # Draw bounding box for shuttles
        #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        #         cv2.putText(frame, f'Shuttle: {confidence:.2f}', (int(x1), int(y1) - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Video Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = "input/video.mov"  # Path to your video file
    process_video(video_path)
