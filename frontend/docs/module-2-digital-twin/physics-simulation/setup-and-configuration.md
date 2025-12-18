# Gazebo Setup and Configuration for Humanoid Robots

## Introduction

This chapter covers the setup and configuration of Gazebo for physics-based simulation of humanoid robots. Gazebo is a powerful robotics simulator that provides accurate physics simulation, realistic rendering, and support for various sensors and robot models. This guide will walk you through the installation, basic configuration, and setup of humanoid robot models in Gazebo.

## Prerequisites

Before starting with Gazebo simulation, ensure that you have:

1. Gazebo installed on your system (version 11 or higher recommended)
2. A URDF (Unified Robot Description Format) model of your humanoid robot
3. Basic understanding of ROS 2 (Robot Operating System 2) for communication between nodes
4. A Linux environment (Ubuntu 20.04 LTS or 22.04 LTS recommended) or WSL2 for Windows

## Installing Gazebo

### For Ubuntu Users

```bash
# Add the OSRF repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo add-apt-repository ppa:openrobotics/gazebo
sudo apt update

# Install Gazebo
sudo apt install gazebo
```

### For Docker Users

If you prefer a containerized setup:

```bash
# Pull the official Gazebo image with ROS 2 support
docker pull osrf/ros:gazebo-ros-base-humble

# Run with display support
xhost +local:docker
docker run -it \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --network="host" \
  osrf/ros:gazebo-ros-base-humble
```

## Basic Gazebo Configuration

Once installed, you can run Gazebo to test the basic functionality:

```bash
gazebo
```

This should launch the Gazebo interface with a default empty world. The interface consists of:

1. **Menu Bar**: Contains File, Edit, View, World, Plugins, and Help menus
2. **Toolbar**: Provides quick access to common operations like inserting objects, positioning, and simulation controls
3. **3D Visualization**: The main window showing the simulated world
4. **Scene Tree**: Shows all objects in the current world
5. **Layer Tabs**: Provides access to different views and tools

## Configuring Your Environment

### Creating a Workspace

Create a new ROS 2 workspace for your humanoid robot simulation:

```bash
# Create the workspace directory
mkdir -p ~/humanoid_sim_ws/src
cd ~/humanoid_sim_ws

# Build the workspace (even if empty)
colcon build --packages-select
source install/setup.bash
```

### Setting up a Custom World

You can create a custom world file to define your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include the default outdoor environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Add lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Optional: Add a simple obstacle -->
    <model name="obstacle_box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.3 0.3 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

Save this as `humanoid_world.world` in your workspace. You can launch your custom world with:

```bash
gazebo --verbose humanoid_world.world
```

## Humanoid Robot Model Setup

### URDF Model Requirements

For humanoid robot simulation in Gazebo, your URDF model should include:

1. Proper joint definitions with appropriate limits and dynamics
2. Collision and visual geometries for all links
3. Gazebo-specific plugins for joint control and sensor simulation
4. Appropriate inertial properties for realistic physics simulation

### Sample URDF Structure

A basic humanoid robot model might look like:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Add more joints and links for arms, legs, head, etc. -->
</robot>
```

## Gazebo-Specific Extensions

To make your URDF work with Gazebo, add the following extensions:

1. **Gazebo Material and Properties**:
```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
</gazebo>
```

2. **Joint Control Plugins**:
```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/humanoid</namespace>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>joint_name</joint_name>
  </plugin>
</gazebo>
```

## Running Your First Simulation

To launch your humanoid robot in Gazebo:

```bash
# Launch Gazebo with your custom world
gazebo --verbose humanoid_world.world &

# Load your robot model (this requires a ROS 2 launch file)
ros2 launch your_robot_description.launch.py
```

## Troubleshooting Common Issues

1. **Gazebo crashes on startup**: Ensure your graphics drivers are up to date and you have sufficient VRAM.
2. **Robot falls through the ground**: Check that your robot has appropriate collision geometries and inertial properties.
3. **Joints don't respond**: Verify that joint controllers are properly configured and running.

## Summary

In this section, we covered the essential setup and configuration for using Gazebo with humanoid robots. We installed Gazebo, configured a basic environment, and prepared a URDF model for simulation. The next section will dive deeper into the physics properties that govern realistic robot behavior in Gazebo.