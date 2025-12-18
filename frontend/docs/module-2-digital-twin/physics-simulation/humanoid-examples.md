# Humanoid Robot Examples and Best Practices

## Introduction

This chapter provides practical examples of humanoid robot models for Gazebo simulation and demonstrates best practices for modeling, configuring, and simulating humanoid robots. We'll examine complete robot examples and discuss design patterns that lead to successful simulation outcomes.

## Popular Humanoid Robot Models

### Atlas (Boston Dynamics)

The Atlas robot is a well-known example of a humanoid robot designed for complex tasks. While the exact specifications are proprietary, we can create a simplified model for educational purposes.

### HRP-2 (AIST)

HRP-2 is a research humanoid robot with a well-documented kinematic structure that's ideal for educational purposes.

### NAO (SoftBank Robotics)

NAO is a smaller humanoid often used in education and research, making it an excellent model for beginners.

## Complete Robot Model Example: Simple Humanoid

Let's create a complete, functional humanoid model with 18 degrees of freedom, appropriate for basic simulation:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Include Gazebo plugins -->
  <gazebo>
    <plugin name="ground_truth" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>ground_truth/state</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>map</frameName>
      <xyzOffsets>0 0 0</xyzOffsets>
      <rpyOffsets>0 0 0</rpyOffsets>
    </plugin>
  </gazebo>

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.1 0.07 0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="15" velocity="1"/>
    <dynamics damping="1.0" friction="0.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.3 0.3 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_yaw" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
    <dynamics damping="0.8" friction="0.0"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.007" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.3 0.3 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Arm -->
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.1 -0.07 0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="15" velocity="1"/>
    <dynamics damping="1.0" friction="0.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_shoulder_yaw" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
    <dynamics damping="0.8" friction="0.0"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.007" ixy="0" ixz="0" iyy="0.007" iyz="0" izz="0.003"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.2"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="-0.05 0.08 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
    <dynamics damping="2.0" friction="0.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.3 0.8 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="0.2" effort="40" velocity="1"/>
    <dynamics damping="1.5" friction="0.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
      <material name="green">
        <color rgba="0.3 0.8 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_pitch" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="-0.05 -0.08 -0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
    <dynamics damping="2.0" friction="0.0"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="0.2" effort="40" velocity="1"/>
    <dynamics damping="1.5" friction="0.0"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
      <material name="yellow">
        <color rgba="0.8 0.8 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.4 0.4 0.4 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo Materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_upper_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="left_lower_arm">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="right_upper_arm">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_lower_arm">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="left_thigh">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_shin">
    <material>Gazebo/Green</material>
  </gazebo>

  <gazebo reference="left_foot">
    <material>Gazebo/Grey</material>
  </gazebo>

  <gazebo reference="right_thigh">
    <material>Gazebo/Yellow</material>
  </gazebo>

  <gazebo reference="right_shin">
    <material>Gazebo/Yellow</material>
  </gazebo>

  <gazebo reference="right_foot">
    <material>Gazebo/Grey</material>
  </gazebo>
</robot>
```

## Loading the Robot in Gazebo

To use this model with Gazebo, you'll need a launch file:

```xml
<launch>
  <!-- Set the robot description parameter -->
  <param name="robot_description" command="$(find xacro)/xacro $(find your_robot_package)/urdf/simple_humanoid.urdf.xacro" />

  <!-- Publish robot state -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" respawn="false" output="screen">
    <param name="publish_frequency" value="30.0"/>
  </node>

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-param robot_description -urdf -model simple_humanoid -x 0 -y 0 -z 1" respawn="false" output="screen" />

  <!-- Launch Gazebo with a world file -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_robot_package)/worlds/humanoid_world.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>
</launch>
```

## Joint Control and Actuation

For realistic simulation, you need to implement joint controllers. Here's an example using ROS 2 and the ros2_control framework:

### Controller Configuration

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    forward_position_controller:
      type: position_controllers/JointGroupPositionController

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

forward_position_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch
      - left_shoulder_yaw
      - right_shoulder_pitch
      - right_shoulder_yaw
      - left_hip_pitch
      - right_hip_pitch
      - left_knee
      - right_knee

joint_trajectory_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch
      - left_shoulder_yaw
      - right_shoulder_pitch
      - right_shoulder_yaw
      - left_hip_pitch
      - right_hip_pitch
      - left_knee
      - right_knee
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
```

### URDF with ros2_control Integration

Add this to your URDF file to interface with the ros2_control system:

```xml
<!-- ros2_control interface -->
<ros2_control name="GazeboSystem" type="system">
  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>
  <joint name="left_shoulder_pitch">
    <command_interface name="position">
      <param name="min">-1.57</param>
      <param name="max">1.57</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>
  <!-- Add similar blocks for other joints -->
</ros2_control>

<!-- Transmission blocks for each joint -->
<transmission name="trans_left_shoulder_pitch">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_pitch">
    <hardwareInterface>hardware_interface/PositionJoint</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_pitch_motor">
    <hardwareInterface>hardware_interface/PositionJoint</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
<!-- Add similar blocks for other joints -->
```

## Simulation Best Practices

### 1. Proper Inertial Properties

The most common issue with humanoid simulations is inaccurate inertial properties. To ensure stability:

```xml
<inertial>
  <mass value="1.0"/>  <!-- Use realistic values -->
  <!-- Calculate inertia tensor properly -->
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" 
           iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

### 2. Realistic Joint Limits and Dynamics

Set appropriate limits based on real hardware:

```xml
<joint name="joint_name" type="revolute">
  <!-- ... -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="1.0" friction="0.1"/>
</joint>
```

### 3. Proper Center of Mass

Position masses to achieve a low center of mass for stability:

- Place heavier components toward the torso/core
- Ensure the center of mass is within the support polygon during stance phases

### 4. Collision Geometry Optimization

Balance accuracy with performance:

- Use simple shapes for collision detection
- Ensure collision geometry encompasses visual geometry
- Set appropriate friction and restitution parameters

### 5. Physics Parameter Tuning

Adjust Gazebo physics parameters for stability:

- Smaller `max_step_size` (0.001) for better accuracy
- Higher `real_time_update_rate` (1000) for responsiveness
- Proper ERP and CFM values for constraint stability

## Example: Walking Pattern Simulation

Here's a simple Python script to command the humanoid to move its joints in a coordinated pattern:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Create publisher for joint trajectory commands
        self.pub = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Define joint names based on our robot model
        self.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_yaw', 
            'right_shoulder_pitch', 'right_shoulder_yaw',
            'left_hip_pitch', 'right_hip_pitch', 
            'left_knee', 'right_knee'
        ]
        
        # Timer to send the command
        self.timer = self.create_timer(2.0, self.send_wave_pattern)
        
    def send_wave_pattern(self):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        # Create trajectory points for a wave pattern
        point = JointTrajectoryPoint()
        
        # Move left arm up and right arm down
        positions = [0.5, 0.0, -0.5, 0.0, 0, 0, 0, 0]
        point.positions = positions
        
        # Set time from start (2 seconds)
        point.time_from_start = Duration(sec=2)
        
        msg.points.append(point)
        
        # Publish the trajectory
        self.pub.publish(msg)
        self.get_logger().info('Published joint trajectory')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Validation and Testing

### Static Balance Check

1. Load your robot model into Gazebo
2. Verify that the robot maintains balance without active control
3. Check that joint limits prevent unrealistic poses
4. Confirm that collisions work properly with the environment

### Dynamic Behavior Testing

1. Test basic movements like arm waving
2. Validate that the center of mass remains stable during motion
3. Check for joint limit violations or self-collisions
4. Verify that the robot behaves predictably under external forces

### Performance Validation

1. Monitor simulation real-time factor (>0.9 for real-time performance)
2. Ensure consistent frame rates in visualization
3. Check for any "exploding" or unstable joint behaviors

## Common Issues and Solutions

### 1. Robot Falls Down or Explodes

**Cause**: Incorrect inertial properties or physics parameters
**Solution**: 
- Verify mass and inertia values are positive and reasonable
- Adjust physics solver parameters (reduce step size, increase iterations)
- Check that the center of mass is appropriately positioned

### 2. Joints Don't Move Smoothly

**Cause**: Control loop issues or inadequate actuator parameters
**Solution**:
- Ensure controllers are properly loaded and running
- Check URDF transmission blocks are correctly defined
- Verify command interface is properly configured

### 3. Self-Collision Issues

**Cause**: Collision geometries intersecting at default positions
**Solution**:
- Adjust initial joint positions to avoid collisions
- Refine collision geometry to be more accurate
- Implement collision avoidance in control algorithms

## Summary

Creating effective humanoid robot models for Gazebo requires attention to physics accuracy, proper mass distribution, and realistic joint constraints. The examples provided in this chapter offer a solid foundation for developing your own humanoid robot models. Remember to validate your models through systematic testing and adjust parameters to achieve stable, realistic behavior in simulation.