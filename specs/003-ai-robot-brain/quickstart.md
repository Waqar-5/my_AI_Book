# Quickstart: AI-Robot Brain (NVIDIA Isaac™)

## Overview
This quickstart guide will help you set up the environment and run the first example from the AI-Robot Brain module. This module teaches photorealistic simulation with Isaac Sim, hardware-accelerated perception with Isaac ROS, and path planning with Nav2 for humanoid robots.

## Prerequisites
- NVIDIA GPU (RTX series recommended) with latest drivers
- ROS 2 Humble Hawksbill
- Isaac Sim (2023.1 or newer)
- Isaac ROS packages
- Nav2 (Humble compatible)
- Python 3.10
- Basic knowledge of ROS 2 (from Module 1)
- Docusaurus development environment (Node.js, npm)

## Setting Up Your Environment

### 1. Install NVIDIA Isaac Sim
Follow the official installation guide for Isaac Sim:
- [Installation Guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation_guide/index.html)

### 2. Install Isaac ROS
Install the Isaac ROS packages compatible with your ROS 2 version:
```bash
sudo apt update
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-visual-slam
# Install other Isaac ROS packages as needed
```

### 3. Install Nav2
Install the navigation stack for ROS 2 Humble:
```bash
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

### 4. Verify Installations
```bash
# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Check Isaac ROS packages
ros2 pkg list | grep isaac_ros

# Check Nav2 packages
ros2 pkg list | grep nav2
```

### 5. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 6. Set Up Docusaurus Environment
```bash
cd frontend
npm install
```

## Running Your First Example

### 1. Navigate to the Examples
```bash
cd static/examples/ai-robot-brain
```

### 2. Run Isaac Sim Example
```bash
# Launch Isaac Sim with the basic scene
# This will depend on your specific Isaac Sim installation
```

### 3. Run Isaac ROS VSLAM Example
```bash
# Make sure ROS 2 is sourced
source /opt/ros/humble/setup.bash

# Launch the Isaac ROS VSLAM pipeline
ros2 launch isaac_ros_vslam.launch.py
```

You should see the VSLAM algorithm processing images and building a map.

## Module Structure
The AI-Robot Brain module is organized into three main chapters:

1. **Photorealistic Simulation with Isaac Sim**: Using Isaac Sim for synthetic data generation and high-fidelity environment modeling
2. **Hardware-Accelerated Perception with Isaac ROS**: Implementing VSLAM and sensor integration with Isaac ROS
3. **Path Planning with Nav2**: Navigation for bipedal humanoid robots using Nav2

Each chapter contains:
- Theoretical explanations
- Isaac Sim/ROS configuration examples
- Diagrams and visual aids
- Hands-on exercises

## Next Steps
- Read the first chapter: "Photorealistic Simulation with Isaac Sim"
- Set up the Isaac Sim environment
- Experiment with creating basic simulation scenes