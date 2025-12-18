# Environment and World Modeling for Humanoids

## Introduction

This chapter focuses on creating and configuring simulation environments for humanoid robots in Gazebo. Realistic environments are crucial for testing and validating humanoid robot capabilities, especially for tasks involving navigation, manipulation, and human-robot interaction. We'll explore how to create diverse and challenging environments that effectively test humanoid capabilities.

## Environment Fundamentals

### Types of Environments

When designing environments for humanoid robots, consider different categories:

1. **Simple Testing Environments**: Basic open spaces for fundamental movement and balance testing
2. **Challenging Terrain**: Uneven surfaces, stairs, ramps, and obstacles to test locomotion
3. **Functional Spaces**: Rooms with furniture, doors, and interactive elements for task execution
4. **Human-Centric Environments**: Spaces designed for human-robot interaction scenarios

### Key Environmental Elements

Effective humanoid environments should address:

- **Traversable Surfaces**: Different materials and textures affecting friction
- **Obstacles**: Static and dynamic obstacles for navigation challenges
- **Interactive Elements**: Objects to manipulate, doors to open, switches to press
- **Perceptual Challenges**: Lighting conditions, textures, and visual complexity
- **Acoustic Properties**: For simulation of sound-based perception

## Creating Basic Environments

### Using Built-in Models

Gazebo comes with a standard set of models that can be used to construct environments:

- **ground_plane**: Infinite flat ground plane
- **sun**: Default lighting source
- **shapes**: Basic geometric shapes (box, cylinder, sphere)
- **house**: Simple house model
- **table**: Various table configurations

### Constructing Custom Environments

You can create custom environments by defining them in SDF (Simulation Description Format) files.

### Basic Indoor Room Example

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_room">
    <!-- Include default ground plane and lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room walls -->
    <model name="wall_1">
      <pose>0 5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>5 0 1.5 0 0 1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add furniture -->
    <include>
      <name>table_1</name>
      <pose>2 -2 0 0 0 0</pose>
      <uri>model://table</uri>
    </include>

    <model name="chair_1">
      <pose>2.5 -1.5 0 0 0 0.5</pose>
      <include>
        <uri>model://chair</uri>
      </include>
    </model>

    <!-- Add objects for manipulation -->
    <model name="box_1">
      <pose>1.5 -1.8 0.8 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.5 0.9 1</ambient>
            <diffuse>0.1 0.6 1.0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>0.5</mass>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Advanced Environmental Features

### Terrain Generation

For outdoor environments or challenging terrain:

1. **Heightmaps**: Use image files to define complex terrain elevations
2. **Procedural generation**: Create random or parameterized terrain
3. **Obstacle placement**: Strategically place rocks, logs, or other natural obstacles

Example of a heightmap terrain:
```xml
<model name="terrain">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://terrain.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### Dynamic Elements

Consider adding elements that change over time:

- **Moving obstacles**: Platforms, doors, or mobile robots
- **Interactive objects**: Buttons, switches, drawers
- **Weather effects**: Rain, wind (simulated effects)
- **Time-varying lighting**: Day/night cycles

### Sensory Challenges

Create environments that test robot perception:

- **Low-texture surfaces**: White walls, uniform floors
- **Repetitive patterns**: That may confuse localization algorithms
- **Glass or transparent elements**: That are difficult to detect
- **Highly reflective surfaces**: That create specular reflections
- **Moving elements**: That affect static vs dynamic object detection

## Humanoid-Specific Environmental Considerations

### Scale and Proportions

Humanoid robots have specific dimensional requirements:

- **Doorways**: Standard height (2.0-2.1m) and width (0.8-0.9m)
- **Ceiling height**: At least 2.4m for full-range humanoid movement
- **Furniture dimensions**: Tables (0.75m), chairs (0.45m), etc.
- **Stair dimensions**: Rise (~0.17m) and run (~0.28m) per step

### Locomotion Challenges

Design environments that challenge different aspects of humanoid locomotion:

- **Narrow passages**: Testing navigation between obstacles
- **Uneven terrain**: Testing balance and adaptive walking
- **Stairs and ramps**: Testing multi-level navigation
- **Slippery surfaces**: Testing with varying friction coefficients
- **Sloped surfaces**: Testing stability on inclines

### Manipulation Environments

For humanoid manipulation tasks:

- **Kitchen-like environments**: With appliances, utensils, and food items
- **Workstations**: With tools, containers, and assembly tasks
- **Storage areas**: Shelves, drawers, and cabinets to access
- **Assembly areas**: With parts and fixtures for construction tasks

## Environmental Modeling Best Practices

### Performance Optimization

1. **Simplify collision geometry**: Use basic shapes for collision detection instead of complex meshes
2. **Reduce polygon count**: For visual geometry where detailed rendering isn't needed
3. **Use static models**: For unchanging elements to reduce physics computation
4. **Limit dynamic objects**: Excessive dynamic objects can reduce simulation performance

### Realism vs. Computational Efficiency

Balance environmental complexity with simulation performance:

- Use simpler models for distant objects
- Implement level-of-detail (LOD) where appropriate
- Consider pre-computing complex environmental effects
- Validate that simplifications don't impact task performance

### Repeatability and Variability

- **Deterministic environments**: For consistent testing and validation
- **Stochastic elements**: For testing robustness to environmental variations
- **Parameterized environments**: For generating variations systematically
- **Modular design**: To easily combine environment components

## Environmental Testing Scenarios

### Validation Environments

Create specific environments to test robot capabilities:

1. **Balance Testing Arena**: Flat surface with marked boundaries
2. **Navigation Maze**: With varying complexity levels
3. **Stair Climbing Setup**: Multiple stair configurations
4. **Manipulation Workbench**: With various objects and tools
5. **Interaction Space**: With humans (simulated) and social scenarios

### Benchmark Environments

Standard environments that allow comparison of performance:

- **Rescue scenarios**: Disaster response environments
- **Home environments**: Standard household layouts
- **Industrial settings**: Factory floors or warehouses
- **Public spaces**: Hallways, offices, or retail environments

## Example: Multi-room Home Environment

Here's a complete example of a multi-room environment designed for humanoid testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_room_house">
    <!-- Include default ground plane and lighting -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Living room -->
    <model name="living_room_wall_1">
      <pose>-3 0 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>6 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>6 0.2 3</size></box>
          </geometry>
          <material><ambient>0.8 0.7 0.6 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Kitchen area -->
    <model name="kitchen_wall_1">
      <pose>3 0 1.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>6 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>6 0.2 3</size></box>
          </geometry>
          <material><ambient>0.6 0.8 0.7 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Doorway between rooms -->
    <model name="doorway">
      <pose>0 2.5 1.5 0 0 1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1.2 0.2 2.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.2 0.2 2.5</size></box>
          </geometry>
          <material><ambient>0.4 0.3 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Kitchen counter with objects -->
    <include>
      <name>counter</name>
      <pose>4 1.5 0.8 0 0 0</pose>
      <uri>model://table</uri>
    </include>

    <!-- Objects for manipulation -->
    <model name="cup_1">
      <pose>3.8 1.5 1.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>0.05</radius><length>0.1</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.05</radius><length>0.1</length></cylinder></geometry>
        </visual>
        <inertial><mass>0.2</mass><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.000025"/></inertial>
      </link>
    </model>

    <!-- Person model (simplified) -->
    <model name="person_1">
      <pose>-2 -2 0 0 0 0</pose>
      <link name="body">
        <collision name="collision">
          <geometry><cylinder><radius>0.2</radius><length>1.7</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.2</radius><length>1.7</length></cylinder></geometry>
          <material><ambient>0.9 0.5 0.5 1</ambient></material>
        </visual>
        <inertial><mass>70</mass><inertia ixx="5" ixy="0" ixz="0" iyy="5" iyz="0" izz="3"/></inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Validation and Quality Assurance

### Environmental Validation

- **Physical realism**: Check that materials and interactions are realistic
- **Task feasibility**: Ensure the environment enables the intended tasks
- **Safety**: Verify that the environment doesn't create unrealistic hazards
- **Consistency**: Ensure environmental properties are uniform across areas

### Testing Procedures

1. **Robot navigation**: Test basic movement throughout the environment
2. **Sensory perception**: Validate that sensors can properly perceive environmental elements
3. **Physics interaction**: Ensure objects behave as expected when interacted with
4. **Performance**: Verify that the simulation maintains adequate real-time factor

## Summary

Creating effective environments for humanoid robots requires balancing realism with computational efficiency. Good environments should challenge different aspects of humanoid capabilities while remaining performant and repeatable. The examples and principles in this chapter provide a foundation for developing diverse and effective simulation environments for your humanoid robots.