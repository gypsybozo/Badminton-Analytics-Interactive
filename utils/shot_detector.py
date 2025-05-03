#shuttle detector
import cv2
import os
import math
import numpy as np

class ShotDetector:
    def __init__(self, buffer_size=2, distance_threshold=100, overlap_threshold=0.1):
        """
        Initialize enhanced shot detector
        
        Args:
            buffer_size (int): Number of frames to analyze for temporal detection
            distance_threshold (float): Maximum distance between racket and shuttle centers
            overlap_threshold (float): Minimum IoU threshold for detection
        """
        self.buffer_size = buffer_size
        self.distance_threshold = distance_threshold
        self.overlap_threshold = overlap_threshold
        self.frame_buffer = []
        self.last_shot_frame = -float('inf')
        self.min_frames_between_shots = 40  # Minimum frames between shots
        
    def calculate_center_distance(self, box1, box2):
        """Calculate Euclidean distance between centers of two boxes"""
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def detect_shot(self, frame_detections, frame_number):
        """
        Enhanced shot detection using multiple techniques
        
        Args:
            frame_detections (list): List of detections in current frame
            frame_number (int): Current frame number
            
        Returns:
            tuple: (bool, dict) indicating if shot detected and shot details
        """
        # Add current frame to buffer
        self.frame_buffer.append({
            'frame_number': frame_number,
            'detections': frame_detections
        })
        
        # Maintain buffer size
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Check if enough frames have passed since last shot
        if (frame_number - self.last_shot_frame) < self.min_frames_between_shots:
            return False, None
            
        # Get current frame detections
        current_rackets = [d for d in frame_detections if d['label'] == 'Racket']
        current_shuttles = [d for d in frame_detections if d['label'] == 'Shuttle']
        
        shot_detected = False
        shot_info = None
        
        # Technique 1: Check current frame for close proximity
        for racket in current_rackets:
            for shuttle in current_shuttles:
                distance = self.calculate_center_distance(racket['box'], shuttle['box'])
                if distance <= self.distance_threshold:
                    shot_detected = True
                    shot_info = {
                        'type': 'proximity',
                        'frame_number': frame_number,
                        'distance': distance,
                        'racket_box': racket['box'],
                        'shuttle_box': shuttle['box'],
                        'confidence': min(racket['confidence'], shuttle['confidence'])
                    }
                    self.last_shot_frame = frame_number
                    return True, shot_info
                
        
        # Technique 2: Check temporal overlap (racket position matches previous shuttle position)
        if not shot_detected and len(self.frame_buffer) >= 2:
            previous_shuttles = []
            for buf in self.frame_buffer[:]:  # Exclude current frame
                previous_shuttles.extend([d for d in buf['detections'] if d['label'] == 'Shuttle'])
                
            for racket in current_rackets:
                for shuttle in previous_shuttles:
                    iou = self.calculate_iou(racket['box'], shuttle['box'])
                    distance = self.calculate_center_distance(racket['box'], shuttle['box'])
                    if iou >= self.overlap_threshold or distance <= self.distance_threshold:
                        shot_detected = True
                        shot_info = {
                            'type': 'temporal_overlap or proximity',
                            'frame_number': frame_number,
                            'iou': iou,
                            'racket_box': racket['box'],
                            'shuttle_box': shuttle['box'],
                            'confidence': min(racket['confidence'], shuttle['confidence'])
                        }
                        self.last_shot_frame = frame_number
                        return True, shot_info
        
        # Technique 3: Check racket-shuttle overlap in current frame
        if not shot_detected:
            for racket in current_rackets:
                for shuttle in current_shuttles:
                    iou = self.calculate_iou(racket['box'], shuttle['box'])
                    if iou >= self.overlap_threshold:
                        shot_detected = True
                        shot_info = {
                            'type': 'direct_overlap',
                            'frame_number': frame_number,
                            'iou': iou,
                            'racket_box': racket['box'],
                            'shuttle_box': shuttle['box'],
                            'confidence': min(racket['confidence'], shuttle['confidence'])
                        }
                        self.last_shot_frame = frame_number
                        return True, shot_info
        
        # Technique 4: Check for sudden change in shuttle direction
        if not shot_detected and len(self.frame_buffer) >= 3:
            shuttle_positions = []
            for buf in self.frame_buffer[-3:]:
                shuttles = [d for d in buf['detections'] if d['label'] == 'Shuttle']
                if shuttles:
                    shuttle_positions.append({
                        'frame': buf['frame_number'],
                        'position': shuttles[0]['box'],
                        'confidence': shuttles[0]['confidence']
                    })
            
            if len(shuttle_positions) == 3:
                # Calculate direction changes
                v1 = (shuttle_positions[1]['position'][0] - shuttle_positions[0]['position'][0],
                      shuttle_positions[1]['position'][1] - shuttle_positions[0]['position'][1])
                v2 = (shuttle_positions[2]['position'][0] - shuttle_positions[1]['position'][0],
                      shuttle_positions[2]['position'][1] - shuttle_positions[1]['position'][1])
                
                # Check for significant direction change
                angle_change = abs(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
                if angle_change > math.pi/4:  # More than 45 degrees change
                    # Find closest racket to the direction change point
                    change_point = shuttle_positions[1]['position']
                    closest_racket = None
                    min_distance = float('inf')
                    
                    for racket in current_rackets:
                        dist = self.calculate_center_distance(change_point, racket['box'])
                        if dist < min_distance:
                            min_distance = dist
                            closest_racket = racket
                    
                    if closest_racket and min_distance < self.distance_threshold * 1.5:
                        shot_detected = True
                        shot_info = {
                            'type': 'direction_change',
                            'frame_number': frame_number,
                            'angle_change': angle_change,
                            'racket_box': closest_racket['box'],
                            'shuttle_box': shuttle_positions[2]['position'],
                            'confidence': min(closest_racket['confidence'], 
                                           shuttle_positions[2]['confidence'])
                        }
        
        if shot_detected:
            # Update last shot frame
            self.last_shot_frame = frame_number
            return True, shot_info
            
        return False, None
