# Gazebo Physics Example

This directory contains a minimal runnable example demonstrating Gazebo physics simulation for humanoid robots.

## Files Included

- `simple_humanoid.urdf.xacro`: A simplified humanoid robot model
- `humanoid_world.world`: A basic simulation environment
- `gazebo_launch.py`: A ROS 2 launch file to start the simulation
- `README.md`: This file

## How to Run

1. Make sure you have Gazebo and ROS 2 installed with the necessary packages
2. Launch the simulation with: `ros2 launch gazebo_launch.py`
3. The humanoid robot model will appear in the simulation environment
4. Use the provided controller to move the robot's joints

## Purpose

This example demonstrates:
- Basic humanoid robot model setup in Gazebo
- Physics simulation with realistic gravity and collisions
- Joint control using ROS 2 interfaces
- Environment modeling with obstacles