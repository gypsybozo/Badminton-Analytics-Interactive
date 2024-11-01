import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

def detect_court_boundary(model, frame, conf_threshold):
    """
    Detect court boundary in the frame using YOLO model
    
    Args:
        model: YOLO model
        frame: Input frame
        conf_threshold: Confidence threshold
    
    Returns:
        court_coords: Coordinates of the four corners of the court if detected, else None
    """
    court_coords = []
    results = model(frame, conf=conf_threshold)
    
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
def process_frame(model, frame, conf_threshold, court_coords):
    """
    Process a single frame with bounding box annotation and court line drawing
    
    Args:
        model: YOLO model
        frame: Input frame
        conf_threshold: Confidence threshold
        court_coords: Coordinates of the four corners of the court
    
    Returns:
        Annotated frame
    """
    # print(court_coords)
    # court_coords[0][0]
    cv2.line(frame, (court_coords[0][2],court_coords[0][3]), (court_coords[1][0],court_coords[1][3]), (255,255,255), 5)
    # side line right
    cv2.line(frame, (court_coords[1][0],court_coords[1][3]), (court_coords[2][0],court_coords[2][1]), (255,255,255), 5)
    #far back line
    cv2.line(frame, (court_coords[2][0],court_coords[2][1]), (court_coords[3][2],court_coords[3][1]), (255,255,255), 5)
    #side line left
    cv2.line(frame, (court_coords[3][2],court_coords[3][1]), (court_coords[0][2],court_coords[0][3]), (255,255,255), 5)

    return frame

def test_on_video(model_path, video_path, conf_threshold=0.3, save_video=True, max_workers=4):
    """
    Process a video to detect court boundaries and annotate frames.
    
    Args:
        model_path: Path to YOLO model
        video_path: Input video path
        conf_threshold: Detection confidence threshold
        save_video: Whether to save the output video
        max_workers: Number of threads for processing
    """
    if not Path(model_path).exists() or not Path(video_path).exists():
        raise FileNotFoundError("Model or video path is invalid")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        f'results_{Path(video_path).name}', 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        fps, 
        (width, height)
    ) if save_video else None

    court_coords = None  # Store court boundary coordinates

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect court boundary if not already done
            if court_coords is None:
                court_coords = detect_court_boundary(model, frame, conf_threshold)

            # Process frame in a separate thread
            future = executor.submit(process_frame, model, frame, conf_threshold, court_coords)
            
            try:
                annotated_frame = future.result()
                
                cv2.imshow('Detection', annotated_frame)
                if save_video and writer:
                    writer.write(annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame: {e}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    model_path = 'models/court_detection/best.pt'
    video_path = 'input/video.mov'
    test_on_video(model_path, video_path)

if __name__ == "__main__":
    main()
