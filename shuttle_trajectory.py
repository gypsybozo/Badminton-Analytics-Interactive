import pandas as pd
import numpy as np

class ShuttleTrajectoryAnalyzer:
    def __init__(self):
        """Initialize the Shuttle Trajectory Analyzer"""
        pass
        
    def interpolate_shuttle_positions(self, shuttle_detections):
        """
        Interpolate missing shuttle positions
        
        Args:
            shuttle_detections: List of dictionaries with shuttle detections
            
        Returns:
            List of interpolated shuttle positions with each being [x1, y1, x2, y2]
        """
        # Extract box coordinates, handling different input formats
        box_positions = []
        for detection in shuttle_detections:
            if detection:
                if isinstance(detection, dict):
                    if 'box' in detection:
                        box_positions.append(detection['box'])
                    elif 1 in detection:
                        box_positions.append(detection[1])
                    else:
                        box_positions.append([None, None, None, None])
                elif isinstance(detection, list) and len(detection) == 4:
                    box_positions.append(detection)
                else:
                    box_positions.append([None, None, None, None])
            else:
                box_positions.append([None, None, None, None])
        
        # Convert to DataFrame
        df_positions = pd.DataFrame(box_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_positions = df_positions.interpolate(method='linear')
        
        # Handle edge cases with both forward and backward fill
        df_positions = df_positions.bfill().ffill()
        
        # Convert back to list of box coordinates
        interpolated_positions = df_positions.values.tolist()
        
        return interpolated_positions

    def detect_direction_changes(self, shuttle_positions, window_size=3, 
                                     min_frames_between_changes=5, 
                                     angle_change_threshold=30.0,
                                     velocity_threshold=2.0,
                                     max_frames_between_hits=100):
        """
        Detect shuttle hits using changes in movement angle
        """
        df = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Calculate center points
        df['center_x'] = (df['x1'] + df['x2']) / 2
        df['center_y'] = (df['y1'] + df['y2']) / 2

        # Apply smoothing
        df['x_smooth'] = df['center_x'].rolling(window=window_size, min_periods=1, center=True).mean()
        df['y_smooth'] = df['center_y'].rolling(window=window_size, min_periods=1, center=True).mean()

        # Calculate velocity vectors
        df['dx'] = df['x_smooth'].diff()
        df['dy'] = df['y_smooth'].diff()
        
        # Calculate speed (magnitude of velocity)
        df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
        
        # Calculate angles (in degrees)
        df['angle'] = np.degrees(np.arctan2(df['dy'], df['dx']))
        
        # Calculate angle changes
        df['angle_change'] = df['angle'].diff().abs()
        # Handle angle wrapping around 360 degrees
        df.loc[df['angle_change'] > 180, 'angle_change'] = 360 - df['angle_change']
        
        df['potential_hit'] = 0
        df['confirmed_hit'] = 0

        # Detect potential hits based on angle changes
        for i in range(window_size, len(df) - window_size):
            # Only consider significant angle changes when the shuttle has sufficient speed
            if (df['angle_change'].iloc[i] > angle_change_threshold and 
                df['speed'].iloc[i] > velocity_threshold):
                df.loc[i, 'potential_hit'] = 1

        # Confirm hits with consistent direction after hit
        hits = []
        last_hit_frame = -max_frames_between_hits

        for i in df[df['potential_hit'] == 1].index:
            if i - last_hit_frame < min_frames_between_changes:
                continue

            # Get new angle
            new_angle = df['angle'].iloc[i]
            
            # Check for angle consistency
            consistent_frames = 0
            min_consistent_frames = min(15, len(df) - i - 1)
            
            for j in range(i+1, min(i + min_consistent_frames + 1, len(df))):
                angle_diff = abs(df['angle'].iloc[j] - new_angle)
                # Adjust for angle wrapping
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff < 20:  # Allow small deviations
                    consistent_frames += 1
                    
            if consistent_frames >= min_consistent_frames * 0.6:
                df.loc[i, 'confirmed_hit'] = 1
                hits.append(i)
                last_hit_frame = i

        # Final filtering
        filtered_hits = []
        if hits:
            filtered_hits = [hits[0]]
            for hit in hits[1:]:
                if hit - filtered_hits[-1] >= min_frames_between_changes:
                    filtered_hits.append(hit)

        return filtered_hits


    def analyze_shot_trajectory(self, shuttle_positions, hits):
        """
        Analyze shuttle trajectory to extract additional features about each shot
        
        Args:
            shuttle_positions: List of interpolated shuttle positions [x1, y1, x2, y2]
            hits: List of frame indices where hits occur
            
        Returns:
            DataFrame with shot analysis
        """
        if not hits or len(hits) < 2:
            return pd.DataFrame()  # Return empty DataFrame if no valid shots
            
        # Convert positions to DataFrame
        df = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Calculate center coordinates
        df['center_x'] = (df['x1'] + df['x2']) / 2
        df['center_y'] = (df['y1'] + df['y2']) / 2
        
        # Initialize shot analysis list
        shot_analysis = []
        
        for i in range(len(hits) - 1):
            start_frame = hits[i]
            end_frame = hits[i+1]
            
            # Extract trajectory segment
            trajectory = df.iloc[start_frame:end_frame+1]
            
            # Calculate shot metrics
            max_height = trajectory['center_y'].min()  # Y is inverted in image coordinates
            horizontal_distance = abs(trajectory['center_x'].iloc[-1] - trajectory['center_x'].iloc[0])
            
            # Determine shot type based on trajectory
            # This is a simplified example - would need to be customized based on court dimensions
            shot_type = "Unknown"
            if horizontal_distance > 300:  # Long shot
                if max_height < 100:  # Low trajectory
                    shot_type = "Drive"
                else:
                    shot_type = "Clear"
            else:  # Short shot
                if max_height < 150:
                    shot_type = "Drop"
                else:
                    shot_type = "Net Shot"
            
            shot_analysis.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': end_frame - start_frame,
                'max_height': max_height,
                'horizontal_distance': horizontal_distance,
                'shot_type': shot_type
            })
        
        return pd.DataFrame(shot_analysis)