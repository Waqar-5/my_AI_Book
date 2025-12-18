# Gravity, Collisions, and Rigid-Body Dynamics

## Introduction

This chapter delves into the physics fundamentals that govern how humanoid robots behave in Gazebo simulation. Understanding gravity, collision detection, and rigid-body dynamics is essential for creating realistic digital twins that accurately reflect the behavior of physical robots.

## Physics Engine Configuration

Gazebo uses the Open Dynamics Engine (ODE) by default, though it also supports other physics engines like Bullet and DART. The physics configuration significantly impacts simulation realism and stability.

### Setting Physics Parameters

The physics engine can be configured in your world file with parameters such as:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Key Parameters Explained

- **max_step_size**: The largest time increment the simulator will take between physics calculations. Smaller values increase accuracy but decrease performance.
- **real_time_factor**: The ratio of simulation time to real time (1.0 means real-time).
- **gravity**: The gravitational acceleration vector (default is 9.8 m/s² in the negative Z direction).
- **iters**: Number of iterations for the constraint solver (more iterations improve stability).
- **erp**: Error reduction parameter (controls constraint error correction, higher values reduce errors but may cause oscillations).
- **cfm**: Constraint force mixing parameter (adds compliance to constraints, preventing singularities).

## Gravity Simulation

Gravity is fundamental to realistic humanoid robot simulation. In Earth-like conditions, the gravitational acceleration is approximately 9.8 m/s², pulling all objects toward the center of the Earth.

### Applying Gravity to Robot Models

All links in your URDF model will be affected by gravity according to their mass and inertial properties. The inertial properties should be defined in the `<inertial>` tag of each link:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" 
             iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
  ...
</link>
```

### Gravity Considerations for Humanoid Robots

When modeling humanoid robots, consider:

1. **Mass Distribution**: Humanoid robots should have appropriate mass distribution to match real-world counterparts.
2. **Center of Mass**: The center of mass affects stability and balance algorithms.
3. **Gravity Compensation**: For walking robots, control algorithms may need to account for gravitational forces.

## Collision Detection

Collision detection is crucial for preventing robots from passing through obstacles or self-colliding. Gazebo provides several collision shapes to balance accuracy and computational efficiency.

### Collision Shapes

Common collision shapes include:

- **Box**: Defined by width, height, and depth
- **Cylinder**: Defined by radius and length
- **Sphere**: Defined by radius
- **Mesh**: Complex shapes based on 3D models (performance intensive)

Example collision definition:
```xml
<collision name="collision">
  <geometry>
    <box>
      <size>0.1 0.1 0.1</size>
    </box>
  </geometry>
</collision>
```

### Collision Properties

You can specify material properties for collisions:

```xml
<collision name="collision">
  <geometry>
    <box>
      <size>0.1 0.1 0.1</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>1.0</mu2>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1000000000000</kp>
        <kd>1</kd>
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Contact Sensors

To detect when collisions occur, you can add contact sensors to your URDF:

```xml
<gazebo>
  <plugin name="contact_sensor" filename="libgazebo_ros_contact.so">
    <ros>
      <namespace>/contact_sensor</namespace>
      <remapping>~/contacts:=/my_robot/contacts</remapping>
    </ros>
    <frame_name>world</frame_name>
    <contact_properties>
      <collision_group>all</collision_group>
      <collision_object>ground_plane_collision</collision_object>
    </contact_properties>
  </plugin>
</gazebo>
```

## Rigid-Body Dynamics

Rigid-body dynamics govern how objects move and interact in response to applied forces. Understanding these principles is essential for realistic humanoid simulation.

### Key Concepts

- **Degrees of Freedom (DOF)**: The number of independent movements a body can have. A rigid body in 3D space has 6 DOF (3 translations + 3 rotations).
- **Inertia Tensor**: Describes how mass is distributed in a body and how it resists rotational motion.
- **Force and Torque**: External forces cause linear acceleration, while torques cause angular acceleration.
- **Joint Constraints**: Joints limit the relative motion between connected links.

### Joint Dynamics

Joints in Gazebo can have various dynamic properties:

- **Joint Limits**: Position, velocity, and effort limits
- **Spring Damping**: Simulates spring-like behavior in joints
- **Friction**: Static and dynamic friction effects
- **Detents**: Simulated "clicks" or positions where joints prefer to rest

Example joint definition with dynamics:
```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Center of Mass and Stability

For humanoid robots, the center of mass (COM) is critical for stability:

- **Stable Balance**: The COM must remain within the support polygon (the area bounded by contact points with the ground).
- **Walking Dynamics**: During walking, the COM trajectory follows specific patterns to maintain balance.
- **Control Strategies**: Various control methods like Zero Moment Point (ZMP) or Linear Inverted Pendulum Mode (LIPM) help maintain balance.

## Practical Example: Humanoid Balance

Let's look at a practical example of how to configure a simple humanoid model to stand stably:

```xml
<robot name="stable_humanoid">
  <!-- Base link with significant mass to lower overall COM -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso with appropriate mass distribution -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>
  
  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Hip joint for leg movement -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0.05 -0.08 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="100" velocity="1"/>
    <dynamics damping="1.0" friction="0.0"/>
  </joint>
  
  <link name="left_thigh">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Continue for other links... -->
</robot>
```

## Tuning Simulation Parameters

Different robot designs may require different physics parameters. Here are some tuning guidelines:

### Stability Issues
- If the robot is unstable or "explodes":
  - Decrease max_step_size
  - Increase solver iterations
  - Adjust ERP and CFM values

### Performance Issues
- If the simulation is slow:
  - Increase max_step_size (to 0.01 or 0.005)
  - Reduce solver iterations
  - Simplify collision shapes

### Penetration Issues
- If objects pass through each other:
  - Decrease max_step_size
  - Increase contact surface layer
  - Increase constraint stiffness (kp)

## Best Practices

1. **Start Simple**: Begin with basic shapes and gradually add complexity to your models.
2. **Realistic Masses**: Use realistic mass values based on actual robot specifications.
3. **Consistent Units**: Use SI units (meters, kilograms, seconds) throughout your models.
4. **Test Incrementally**: Test with simple scenarios before attempting complex behaviors.
5. **Monitor Performance**: Keep an eye on real-time factor to ensure simulation runs efficiently.

## Summary

Understanding gravity, collisions, and rigid-body dynamics is fundamental to creating realistic humanoid simulations in Gazebo. Proper configuration of these elements ensures that your digital twin accurately reflects the behavior of the physical robot, which is essential for effective testing, validation, and training of robot behaviors.