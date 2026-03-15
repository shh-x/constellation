# Constellation 

**Constellation** is a high-performance, interactive Python desktop application that transforms a user into a dynamic N-Body stellar constellation. Utilizing computer vision and physics simulations, the application maps thousands of independent particles to your body's movements in real-time, creating a mesmerizing, gesture-driven digital reflection.

##  Features

- **Real-Time Biometric Tracking**: Uses MediaPipe and OpenCV to accurately capture human pose landmarks through a webcam feed.
- **N-Body Physics Engine**: A custom physics simulation where thousands of individual particles are gravitationally drawn to the user's skeletal structure, complete with spring forces, auras, and damping for smooth orbital motion.
- **Dynamic Skeletal Interpolation**: Fills the gaps between standard pose coordinates, mapping particles across limbs and core shapes for a complete "constellation" silhouette.
- **Gesture-Driven Events**: 
  -  **Black Hole**: Bring your wrists closely together to trigger a singular gravity well, dragging all particles into a white-hot core.
  - **Supernova**: Rapidly separate your wrists to trigger a violent repulsion wave and blinding flash, scattering the particles away.
- **Zenith/Nadir Color Shifting**: The constellation actively changes colors based on your vertical wrist positions—shifting from golden/white highs to deep purple/red lows.
- **Holographic Scanner**: Features an atmospheric, top-down biometric scanning sequence upon detection or manual trigger.
- **Automatic Performance Management**: Built-in optimization that dynamically reduces particle counts if the frame rate dips below 30 FPS, keeping the experience smooth.
- **Alpha-Blended Trailing**: Beautiful particle motion-blur powered by semi-transparent frame blending.

## Prerequisites

Ensure you have Python 3.8+ installed on your system. You will need the following dependencies:

```bash
pip install pygame opencv-python mediapipe numpy
