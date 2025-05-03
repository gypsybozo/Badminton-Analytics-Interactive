# court.py
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

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

        for result in results:
            for box in result.boxes:
                if box.cls[0] == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    court_coords.append((x1, y1, x2, y2))

        # if len(court_coords) == 4:
        #     return court_coords  # Return the four bounding boxes
        return court_coords  # Continue detection if fewer than four boxes are found

    def sort_court_coords(self, court_coords):
        """
        Sort court coordinates in order: clockwise, top left first.

        Args:
            court_coords: List of (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            sorted_coords: List of ordered (x1, y1, x2, y2) coordinates.
        """
        centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for (x1, y1, x2, y2) in court_coords]

        boxes_with_centers = list(zip(court_coords, centers))

        # Sort by y-coordinate to separate top and bottom rows
        boxes_with_centers = sorted(boxes_with_centers, key=lambda box_center: box_center[1][1])

        # Split into top and bottom based on sorted y-coordinate
        top_two = sorted(boxes_with_centers[:2], key=lambda box_center: box_center[1][0])  # Sort top row by x
        bottom_two = sorted(boxes_with_centers[2:], key=lambda box_center: box_center[1][0])  # Sort bottom row by x

        sorted_coords = [top_two[0][0], top_two[1][0], bottom_two[1][0], bottom_two[0][0]]
        # print(sorted_coords)
        return sorted_coords

    def draw_court_lines(self, frame, court_coords):
        """
        Draw court lines on the frame

        Args:
            frame: Input frame to draw lines on
            court_coords: Coordinates of the court boundaries
        """

        # print(court_coords)
        top_left = (court_coords[0][2], court_coords[0][1])  # Top-left of the first box
        top_right = (court_coords[1][0], court_coords[1][1])  # Top-right of the second box
        bottom_right = (court_coords[2][0], court_coords[2][3])  # Bottom-right of the third box
        bottom_left = (court_coords[3][2], court_coords[3][3])  # Bottom-left of the fourth box

        # print(f"Top Left: {top_left}")
        # print(f"Top Right: {top_right}")
        # print(f"Bottom Right: {bottom_right}")
        # print(f"Bottom Left: {bottom_left}")

        cv2.line(frame, top_left, top_right, (255, 255, 255), 5)  # Top line
        cv2.line(frame, top_right, bottom_right, (255, 255, 255), 5)  # Right line
        cv2.line(frame, bottom_right, bottom_left, (255, 255, 255), 5)  # Bottom line
        cv2.line(frame, bottom_left, top_left, (255, 255, 255), 5)  # Left line

        # Save the actual corners
        actual_bounds = [top_left, top_right, bottom_right, bottom_left]

        return actual_bounds
    def compute_homography(self, court_coords):
        """
        Compute the homography matrix to map image coordinates to real-world coordinates.

        Args:
            court_coords: List of 4 corner coordinates in the image [(x, y), ...]

        Returns:
            homography_matrix: The transformation matrix
        """
        # Real-world coordinates of a standard badminton court in meters
        # Redefine with (0,0) at bottom-left to match typical coordinate systems
        real_world_coords = np.array([
            [0, 13.4],       # Top-left corner (x=0, y=13.4)
            [5.18, 13.4],    # Top-right corner (x=5.18, y=13.4)
            [5.18, 0],       # Bottom-right corner (x=5.18, y=0)
            [0, 0]           # Bottom-left corner (x=0, y=0)
        ], dtype=np.float32)

        # Image coordinates (court corners)
        image_coords = np.array(court_coords, dtype=np.float32)

        # Compute the homography matrix
        homography_matrix, _ = cv2.findHomography(image_coords, real_world_coords)
        return homography_matrix

    def translate_to_real_world(self, point, homography_matrix):
        """
        Translate a point from image coordinates to real-world coordinates.

        Args:
            point: (x, y) coordinate in the image
            homography_matrix: The transformation matrix

        Returns:
            real_world_point: Translated (x, y) coordinate in real-world space
        """
        # Convert the point to homogeneous coordinates
        image_point = np.array([[point[0], point[1], 1]], dtype=np.float32).T

        # Apply the homography transformation
        real_world_point = np.dot(homography_matrix, image_point)

        # Normalize to get the actual (x, y) in real-world space
        real_world_point /= real_world_point[2]
        return real_world_point[:2].flatten()

    def crop_to_court(self, frame, court_coords):
        """
        Crop the frame to include only the area inside the court.

        Args:
            frame: Input frame
            court_coords: Coordinates of the court boundaries

        Returns:
            cropped_frame: Frame cropped to the court area
        """
        # Define the polygon for the court area
        court_polygon = np.array(court_coords, dtype=np.int32)

        # Create a mask for the court area
        # mask = np.zeros_like(frame)
        # cv2.fillPoly(mask, [court_polygon], (255, 255, 255))

        # Apply the mask to the frame
        # masked_frame = cv2.bitwise_and(frame, mask)

        # Get the bounding box of the court area
        x, y, w, h = cv2.boundingRect(court_polygon)

        # Crop the frame to the bounding box
        cropped_frame = frame[:, x:x+w]

        return cropped_frame