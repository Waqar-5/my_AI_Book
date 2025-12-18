---
sidebar_position: 3
---

# Robot Structure and Description with URDF

## Introduction

The Unified Robot Description Format (URDF) is an XML specification for describing robots in ROS. It provides a standardized way to represent robot models including their physical structure, kinematic properties, and visual appearance. URDF is essential for humanoid robots as it defines the robot's structure for simulation, visualization, and kinematic analysis in tools like Isaac Sim and MoveIt.

## Understanding URDF Structure

URDF describes a robot as a collection of rigid bodies (links) connected by joints that move relative to each other. This creates a kinematic chain representing the robot's mechanical structure with:

- **Links**: Rigid body elements with mass, visual, and collision properties
- **Joints**: Connections between links that define motion constraints
- **Materials**: Visual appearance properties
- **Transmissions**: Actuator interfaces for joints

## Links: The Building Blocks

A **link** is a rigid body element in a URDF robot model. Each link has properties like mass, visual representation, and collision properties:

```xml
<link name="link_name">
  <inertial>
    <mass value="0.1" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.1 0.1 0.1" />
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1" />
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.1 0.1 0.1" />
    </geometry>
  </collision>
</link>
```

### Components of a Link

1. **Inertial**: Physical properties for dynamics simulation
   - `mass`: Mass of the link in kg
   - `origin`: Center of mass relative to link frame
   - `inertia`: 3x3 inertia tensor values

2. **Visual**: How the link appears in visualization
   - `origin`: Visual origin relative to link frame
   - `geometry`: Shape definition (box, cylinder, sphere, mesh)
   - `material`: Color and appearance properties

3. **Collision**: How the link interacts in collision detection
   - `origin`: Collision origin relative to link frame
   - `geometry`: Collision shape definition

## Joints: Connecting Links

Joints define how links connect and move relative to each other. For humanoid robots, different joint types are crucial for realistic movement:

### Joint Types

1. **Fixed**: No movement between parent and child links
2. **Revolute**: Rotational movement with limits (like a hinge)
3. **Continuous**: Revolute joint without limits (like a wheel)
4. **Prismatic**: Linear sliding movement with limits
5. **Planar**: Movement on a plane
6. **Floating**: 6-DOF movement (rarely used in humanoid robots)

### Joint Definition

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link_name" />
  <child link="child_link_name" />
  <origin xyz="0 0 0.1" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <limit lower="-1.57" upper="1.57" effort="10" velocity="3.14" />
  <dynamics damping="0.1" friction="0.0" />
</joint>
```

### Components of a Joint

1. **parent/child**: Links connected by this joint
2. **origin**: Position and orientation of joint relative to parent link
3. **axis**: Direction of rotation/translation
4. **limit**: Motion constraints (for revolute/prismatic joints)
5. **dynamics**: Joint dynamics like damping and friction

## Modeling Humanoid Robots with URDF

Humanoid robots have specific structural requirements that differ from wheeled or other robot types:

### Key Characteristics

1. **Bipedal Locomotion**: Two legs with feet for walking
2. **Arms and Manipulation**: Arms with degrees of freedom for manipulation
3. **Balance Requirements**: Center of mass considerations for stability
4. **Human-like Proportions**: Limbs proportioned similarly to humans

### Humanoid Link Structure

A typical humanoid robot model might have these components:

1. **Base Link**: Root of the kinematic tree (often pelvis or base of support)
2. **Torso**: Links for spine/trunk segments
3. **Head**: Neck and head links with sensors
4. **Arms**: Shoulder, upper arm, lower arm, and hand links
5. **Legs**: Hip, thigh, shin, and foot links

## URDF for Simulation and Control

URDF models serve multiple purposes in simulation and control:

### Isaac Sim Integration

In Isaac Sim, URDF models are converted to the underlying physics representation. Isaac Sim enhances URDF with:

- Visual materials and textures
- Physics properties
- Sensors and actuators
- Collision properties

### Kinematic Analysis

URDF enables kinematic analysis including:
- Forward kinematics (given joint angles, where is the end effector?)
- Inverse kinematics (given desired end effector position, what joint angles are needed?)
- Jacobian computation for motion planning

## Sensors and Actuation Basics

In addition to mechanical structure, URDF can describe sensors and actuators:

### Adding Sensors

```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.05 0.05 0.05" />
    </geometry>
  </visual>
</link>

<joint name="head_to_camera" type="fixed">
  <parent link="head" />
  <child link="camera_link" />
  <origin xyz="0.05 0 0" />
</joint>

<gazebo reference="camera_link">
  <sensor type="camera" name="head_camera">
    <pose>0 0 0 0 0 0</pose>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### Transmission Definitions

```xml
<transmission name="left_elbow_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Best Practices for Humanoid URDF Modeling

### 1. Kinematic Tree Structure
- Establish a clear parent-child hierarchy
- Usually starts with pelvis/base_link as the root
- Ensure all parts are connected to form a valid tree structure

### 2. Mass and Inertia Properties
- Accurate mass properties are crucial for physics simulation
- Use reasonable physical approximations (cylinders for limbs, boxes for torso)
- Consider using tools like `SW_ISS` to compute inertial properties

### 3. Joint Limits and Dynamics
- Set realistic limits based on human or robotic specifications
- Properly configure effort and velocity limits
- Consider range of motion for humanoid locomotion

### 4. Coordinate Frame Conventions
- Use standard ROS coordinate conventions (x-forward, y-left, z-up)
- Consistent frame orientations simplify control and perception
- Clearly define attachment points for sensors and accessories

## Validation and Tools

### Checking URDF Validity
```bash
# Validate URDF syntax
check_urdf robot.urdf

# View the kinematic tree
urdf_to_graphviz robot.urdf
```

### Visualizing URDF
```bash
# Use RViz to visualize your robot model
# Or use free tools like urdf-viewer
```

## Integration with Navigation Stacks (Nav2)

URDF models play an important role in navigation:

- Define robot footprint for costmap inflation
- Specify sensor mounting positions for spatial reasoning
- Inform path planning with robot dimensions and kinematic constraints
- Enable accurate collision checking

For humanoid robots, special attention is needed for:
- Footprint calculation considering bipedal stance
- Local planning constraints for balance
- Step size limitations in traversable space calculations

## Summary

URDF is fundamental to representing humanoid robots in ROS-based systems. Well-structured URDF files enable accurate simulation, visualization, and kinematic analysis. When creating URDF models for humanoid robots, pay special attention to mechanical structure, mass properties, and joint constraints to ensure realistic behavior in simulation and proper integration with control algorithms.

## Learning Objectives

After completing this chapter, you should be able to:

1. Understand URDF as the standard format for robot description in ROS
2. Create links with appropriate visual, collision, and inertial properties
3. Define joints with appropriate types and limits for humanoid robots
4. Model a basic humanoid robot structure with links and joints
5. Describe how sensors and actuators fit into the URDF framework

## Success Criteria

- You can interpret and modify a humanoid URDF file (from the module success criteria)
- You understand the relationship between links and joints in robot modeling
- You can create a proper URDF representation with appropriate links and joints for a simple robot