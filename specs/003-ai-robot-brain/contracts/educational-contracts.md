# Educational Content Contracts: AI-Robot Brain (NVIDIA Isaac™)

## Overview
This document defines the expected interfaces and behaviors for the educational content in the AI-Robot Brain module. These "contracts" ensure consistency and clarity for learners working with Isaac Sim, Isaac ROS, and Nav2.

## Isaac Sim Contract

### Photorealistic Simulation
- **Input**: 3D environment model and humanoid robot description
- **Configuration**: Isaac Sim scene parameters with lighting and materials
- **Behavior**: Generates synthetic sensor data with photorealistic properties
- **Expected Output**: Sensor data indistinguishable from real sensor data
- **Performance**: Maintains real-time simulation (1x) on GPU-accelerated hardware
- **Example File**: `static/examples/ai-robot-brain/basic-scene.usd`

## Isaac ROS Contract

### Visual SLAM Implementation
- **Input**: Camera image stream from simulated or real robot
- **Behavior**: Estimates camera pose and reconstructs scene geometry in real-time
- **Expected Output**: Accurate map and pose estimates at >10Hz processing rate
- **Performance**: Real-time processing leveraging GPU acceleration
- **Example File**: `static/examples/ai-robot-brain/isaac-ros-vslam.launch.py`

### Sensor Integration
- **Input**: Multiple sensor streams (camera, LiDAR, IMU)
- **Behavior**: Fuses sensor data to create comprehensive perception of environment
- **Expected Output**: Fused perception data with reduced noise and improved accuracy
- **Performance**: Real-time processing on GPU-accelerated platform
- **Example File**: `static/examples/ai-robot-brain/sensor-fusion.launch.py`

## Nav2 Contract

### Bipedal Navigation
- **Input**: Environment map with obstacles and goal pose
- **Behavior**: Plans safe trajectory respecting humanoid robot constraints
- **Expected Output**: Valid path that accounts for bipedal locomotion requirements
- **Performance**: Path planning completes within 5 seconds
- **Example File**: `static/examples/ai-robot-brain/nav2-bipedal-config.yaml`

### Obstacle Avoidance
- **Input**: Dynamic obstacle information in real-time
- **Behavior**: Repath planning to avoid collisions while maintaining stability
- **Expected Output**: Updated trajectory that safely avoids new obstacles
- **Performance**: Real-time replanning during navigation
- **Example File**: `static/examples/ai-robot-brain/local-planner-config.yaml`

## Integration Contract

### Isaac Sim-ROS-Nav2 Pipeline
- **Message Types**: Standard ROS 2 message structures for sensor and navigation data
- **Topics**: Standardized naming conventions for perception and navigation topics
- **Data Format**: Expected formats for images, point clouds, transforms, and navigation goals
- **Synchronization**: Consistent timing between simulation, perception, and navigation

## Educational Content Contract

### Chapter Consistency
- **Learning Objectives**: Each chapter clearly states what the learner will be able to do
- **Prerequisites**: Each chapter states required knowledge (from Module 1 & 2)
- **Simulation Examples**: All Isaac Sim examples are runnable and produce expected outputs
- **ROS Examples**: All Isaac ROS and Nav2 configurations work with Humble
- **Diagrams**: All diagrams are clear, labeled, and referenced in text
- **Exercises**: Each chapter includes hands-on exercises with clear instructions

### Success Criteria
- All Isaac Sim examples run on GPU-accelerated systems
- Students can reproduce all examples in their local environment (where possible)
- Students can modify examples to create new simulation and navigation scenarios
- Students understand the concepts well enough to apply them to different robot platforms