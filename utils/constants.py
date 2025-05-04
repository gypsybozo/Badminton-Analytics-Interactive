# utils/constants.py

# Object Detection Class IDs (Adjust if your model differs)
PLAYER_CLASS_ID = 0
RACKET_CLASS_ID = 1
SHUTTLE_CLASS_ID = 2

# Court Dimensions
COURT_LENGTH_METERS = 13.4
COURT_WIDTH_METERS = 5.18
COURT_HALF_LENGTH_METERS = COURT_LENGTH_METERS / 2.0

# 3-Stage Shot Confirmation Thresholds (Tune these)
CONFIRMATION_WINDOW = 2         # Frames before/after trigger to check (idx-W..idx..idx+W)
RACKET_SHUTTLE_IOU_THRESHOLD = 0.5 # Min IoU for racket/shuttle confirmation
WRIST_SHUTTLE_PROXIMITY_THRESHOLD = 250 # Max pixel distance for wrist/shuttle confirmation

# Temporal Filtering
MIN_FRAMES_BETWEEN_CONFIRMED_SHOTS = 20 # Min frames between two confirmed shots
FRAMES_BETWEEN_RALLIES = 250           # Frame gap to start a new rally

# Trajectory Analysis Thresholds (Tune these)
TRAJ_ANGLE_CHANGE_THRESHOLD = 25.0  # Min angle change (degrees) for candidate shot
TRAJ_VELOCITY_THRESHOLD = 1.5       # Min speed (pixels/frame) for candidate shot
TRAJ_WINDOW_SIZE = 3                # Smoothing window for trajectory analysis
TRAJ_MIN_FRAMES_BETWEEN_CHANGES = 5 # Min frames between candidate triggers

# Pose Estimation
POSE_VISIBILITY_THRESHOLD = 0.3     # Min visibility for MediaPipe keypoints

# Default Output Dirs
DEFAULT_SHOT_IMAGE_DIR = "shot_images"
DEFAULT_PLAYER_SHOT_DIR = "player_shot_images"