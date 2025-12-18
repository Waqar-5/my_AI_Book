# Unity Integration Example

This directory contains a minimal runnable example demonstrating Unity-Gazebo integration.

## Files Included

- `RobotState.cs`: Definition of robot state structure
- `GazeboUnityBridge.cs`: Basic bridge implementation
- `RobotVisualizer.cs`: Component to visualize robot state in Unity
- `SimpleGazeboPlugin.cpp`: Simple Gazebo plugin to publish robot state
- `README.md`: This file

## How to Run

1. Build the Gazebo plugin with the provided C++ code
2. Run the Gazebo simulation with the plugin loaded
3. Start the Unity application to connect to the state broadcast
4. The Unity visual will update based on the Gazebo simulation state

## Purpose

This example demonstrates:
- Basic state broadcasting from Gazebo
- WebSocket connection to Unity
- Visualization of robot state in Unity
- Coordinate system conversion between systems