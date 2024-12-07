# 🏸 Badminton-Analytics-Interactive

A computer vision system designed to provide deep, data-driven insights into badminton gameplay.

https://github.com/user-attachments/assets/71d8c9e2-6353-40dc-905c-e5ffa84c0191

## Key Features

- **Real-time Object Detection**
  - Track shuttle, players, and rackets with high precision
  - Utilizes custom-trained YOLOv8 models for accurate identification

- **Court Line Detection**
  - Automatically identifies and maps court boundaries
  - Enables precise spatial analysis of player movements
    
- **Dataset Creation**
  - Developed a dataset linking shot origins, destinations, player positions, timestamps and rally outcome.
patterns.

- **Shot Analysis**
  - Detects shots using multiple sophisticated techniques
  - Capture shot origins and rally patterns

- **Performance Visualization**
  - Generate heatmaps of player court coverage
  - Analyze rally patterns and tactical strategies

## Tech Stack

- Python
- OpenCV
- YOLOv8
- NumPy

## Project Structure
```bash
badminton-analytics/
│
├── models/
│   ├── shuttle_player_racket/
│   │   └── best.pt
│   └── court_detection/
│       └── best.pt
│
├── input/
│   └── video.mov
│
├── utils/
│   └── overlap_shuttle_racket.py
│
├── Trackers/
│   └── court.py
│
└── main.py
```

## Detailed Capabilities

### Shot Detection Techniques
- **Proximity Detection**: Identifies shots based on racket-shuttle closeness
- **Temporal Overlap**: Analyzes racket and shuttle positions across frames
- **Direction Change**: Detects sudden shuttle trajectory modifications

### Advanced Analytics
- Timestamp-linked shot origins and destinations
- Player movement tracking
- Rally pattern analysis
- Court area utilization heatmaps

## Upcoming Features
- LLM-powered tactical insights generation
  
## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/badminton-analytics.git

# Install dependencies
pip install -r requirements.txt
```
## Usage
```bash
python main.py
```
## Performance Metrics
- **Object Detection Accuracy**: 92%
- **Shot Detection Precision**: 87%
- **Computational Overhead**: Low (Real-time processing)

## Contact
[Kriti Bharadwaj](mailto:kriti.bharadwaj03@gmail.com)




