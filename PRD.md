# Constellation - Product Requirements Document (PRD)

## Executive Summary

**Constellation** is a real-time, interactive computer vision application that transforms human body movements into dynamic N-body particle simulations. The system uses advanced pose detection algorithms and custom physics engines to create a mesmerizing visual experience where thousands of particles gravitationally respond to user gestures and skeletal movements.

## Technical Architecture

### Core Technologies
- **Frontend Framework**: Streamlit web application framework
- **Computer Vision**: MediaPipe Pose Landmarker with OpenCV
- **Video Processing**: WebRTC integration via streamlit-webrtc
- **Physics Engine**: Custom N-body particle simulation with NumPy
- **Real-time Processing**: AV (PyAV) for video frame handling
- **Mathematical Computing**: NumPy for vector operations and physics calculations

### System Requirements
- **Python Version**: 3.8+
- **Operating System**: Cross-platform (Windows, macOS, Linux)
- **Hardware**: Webcam required, GPU acceleration recommended
- **Memory**: Minimum 4GB RAM (8GB recommended for optimal performance)

## Core Features & Implementation Details

### 1. Real-Time Biometric Tracking System

**Technical Implementation:**
- MediaPipe Pose Landmarker with 33 skeletal landmark detection
- Confidence threshold filtering (visibility > 0.3 for landmarks, > 0.5 for wrists)
- Video frame processing at 1280x720 resolution
- Horizontal image flip for mirror-like interaction
- Timestamp-based video processing for temporal consistency

**Key Parameters:**
```python
min_pose_detection_confidence=0.5
min_pose_presence_confidence=0.5
min_tracking_confidence=0.5
```

### 2. N-Body Physics Engine

**Particle System Architecture:**
- **Maximum Particles**: 500 concurrent particles
- **Particle Properties**: Position (x,y), Velocity (vx,vy), Active State, RGB Color
- **Initialization**: Random distribution across canvas with zero initial velocity
- **Default Color**: Cyan (0,255,255)

**Physics Calculations:**
- **Gravitational Attraction**: Particles attracted to nearest pose landmarks
- **Force Calculation**: `force = clip(0.15 * error, -10, 10)` where error = distance - 40px
- **Acceleration**: Vector-based acceleration toward target positions
- **Damping**: Velocity damping factor of 0.95 for smooth motion
- **Boundary Checking**: Particles constrained to canvas dimensions

### 3. Gesture Recognition System

**Black Hole Activation:**
- Trigger: Wrist distance < 10% of canvas width
- Implementation: Euclidean distance calculation between left and right wrist landmarks
- Effect: Creates singular gravity well at wrist center point
- Visual Feedback: Particle convergence to central point

**Wrist Tracking:**
- 10-frame rolling history buffer for smooth gesture detection
- Landmark indices: LEFT_WRIST=15, RIGHT_WRIST=16
- Real-time distance calculation for gesture recognition

### 4. Biometric Scanning System

**Scanning Animation:**
- Trigger: Initial person detection event
- Implementation: Horizontal scanning bar moving downward at 4px/frame
- Particle Activation: Particles activated when scan line passes their y-coordinate
- Visual Effect: Yellow scanning line (255,255,0) with 3px thickness
- Reset Condition: Scan completes when reaching canvas bottom

### 5. Real-Time Video Processing Pipeline

**Processing Steps:**
1. Frame capture from WebRTC stream
2. Resize to 1280x720 resolution
3. Horizontal flip for mirror interaction
4. BGR to RGB color space conversion
5. MediaPipe pose detection with timestamp
6. Particle physics simulation
7. Visual rendering and overlay
8. Frame output via AV VideoFrame

**Performance Optimizations:**
- NumPy vectorized operations for particle calculations
- Conditional rendering only for active particles
- Efficient distance calculations using broadcasting
- Memory-efficient particle state management

## Configuration Parameters

### Canvas Configuration
```python
WIDTH = 1280
HEIGHT = 720
MAX_PARTICLES = 500
```

### Physics Constants
```python
G_BASE = 5000  # Gravitational constant baseline
FORCE_MULTIPLIER = 0.15
DAMPING_FACTOR = 0.95
TARGET_DISTANCE = 40  # pixels
MIN_DISTANCE_SQ = 20  # for numerical stability
```

### Animation Parameters
```python
SCAN_SPEED = 4  # pixels per frame
PARTICLE_RADIUS = 2  # pixels
LANDMARK_RADIUS = 3  # pixels
```

## Dependencies & Installation

### Core Dependencies
```bash
streamlit>=1.28.0
streamlit-webrtc>=0.44.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
av>=10.0.0
```

### MediaPipe Model File
- **Required**: `pose_landmarker.task` (5.8MB)
- **Source**: Google MediaPipe pose detection model
- **Location**: Project root directory

## Application Structure

### Main Components

1. **Particle Initialization Module**
   - Function: `init_particles(n)`
   - Returns: position, velocity, active state, color arrays

2. **MediaPipe Configuration**
   - Pose landmarker setup with video running mode
   - Confidence thresholds configuration
   - Model asset path specification

3. **Video Processor Class** (`PoseProcessor`)
   - Inherits from `VideoProcessorBase`
   - Manages particle system state
   - Handles frame-by-frame processing
   - Implements physics simulation

4. **Streamlit Web Interface**
   - WebRTC streamer integration
   - Video processor factory pattern
   - Media stream constraints (video-only)

### State Management
- Person detection boolean flag
- Scanning animation state and position
- Wrist position history buffer
- Particle system arrays (position, velocity, active, colors)

## Performance Characteristics

### Frame Rate Targets
- **Minimum**: 30 FPS for smooth animation
- **Optimal**: 60 FPS for best user experience
- **Resolution**: 1280x720 at 30 FPS baseline

### Memory Usage
- **Particle System**: ~500 * 4 * 8 bytes = 16KB for state arrays
- **Video Frames**: ~3.7MB per frame (1280x720x3)
- **Total Memory**: ~50-100MB typical usage

### CPU Utilization
- **Pose Detection**: ~15-25% CPU (MediaPipe optimized)
- **Physics Simulation**: ~10-15% CPU (NumPy vectorized)
- **Video Processing**: ~20-30% CPU (OpenCV operations)

## User Experience Flow

### Initialization Sequence
1. Application launches and loads MediaPipe model
2. WebRTC camera access requested
3. Particle system initialized with 500 inactive particles
4. Video stream begins processing

### Interaction Flow
1. User enters camera view
2. System detects human pose
3. Scanning animation triggers particle activation
4. Particles gravitationally attract to skeletal landmarks
5. User gestures control particle behavior
6. Real-time visual feedback continues

### Gesture Controls
- **Body Movement**: Particles follow skeletal landmarks
- **Wrist Proximity**: Black hole effect activation
- **Natural Motion**: Smooth particle constellation formation

## Technical Limitations & Considerations

### Current Limitations
- Single user support (no multi-person tracking)
- Fixed particle count (500 particles maximum)
- No persistent configuration options
- Limited to 2D plane simulation

### Scalability Considerations
- Particle count limited by real-time performance requirements
- Pose detection accuracy dependent on lighting conditions
- WebRTC compatibility varies across browsers

### Environmental Dependencies
- Requires consistent lighting for optimal pose detection
- Camera quality affects tracking accuracy
- Background complexity impacts pose detection performance

## Deployment & Replication Guide

### Environment Setup
1. Create Python virtual environment (3.8+)
2. Install dependencies via pip
3. Download `pose_landmarker.task` model file
4. Ensure camera permissions are granted

### Running the Application
```bash
streamlit run constellation.py
```

### Browser Compatibility
- Chrome/Chromium: Full support
- Firefox: WebRTC support required
- Safari: Limited WebRTC compatibility
- Edge: Chromium-based versions supported

## Future Enhancement Opportunities

### Technical Improvements
- GPU acceleration for particle physics
- Multi-person pose tracking
- 3D particle simulation with depth perception
- Customizable particle counts and colors
- Audio-reactive particle behavior

### Feature Expansions
- Gesture library expansion
- Particle effect presets
- Recording and playback functionality
- Calibration and sensitivity settings
- Multi-camera support

## Security & Privacy Considerations

### Data Handling
- All video processing occurs locally
- No data transmitted to external servers
- Camera feed accessed only during active session
- No biometric data storage or persistence

### Permissions Required
- Camera access (mandatory)
- No microphone access required
- No file system access beyond model loading

## Conclusion

The Constellation application represents a sophisticated fusion of computer vision, physics simulation, and real-time graphics rendering. Its modular architecture and well-defined parameters make it highly replicable while maintaining the core interactive experience that transforms human movement into dynamic particle constellations.

The system's balance between performance optimization and visual fidelity creates an engaging user experience that demonstrates advanced Python programming capabilities in multimedia applications.
