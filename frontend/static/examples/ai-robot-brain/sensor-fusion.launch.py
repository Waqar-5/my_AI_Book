# Isaac ROS Sensor Fusion Example
# File: sensor-fusion.launch.py

"""
Launch file for Isaac ROS sensor fusion pipeline

This launch file sets up the sensor fusion pipeline using Isaac ROS components:
- Camera data processing
- LiDAR data processing  
- IMU data processing
- Data fusion for comprehensive environment understanding
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for Isaac ROS sensor fusion pipeline"""
    
    # Launch arguments
    config_dir = LaunchConfiguration('config_dir')
    config_dir_cmd = DeclareLaunchArgument(
        'config_dir',
        default_value='/etc/isaac_ros/sensor_fusion',
        description='Configuration directory'
    )

    # Isaac ROS Image Pipeline
    image_pipeline_node = Node(
        package='isaac_ros_image_pipeline',
        executable='image_rectify_node',
        name='image_rectify_node',
        parameters=[config_dir]
    )

    # Isaac ROS LiDAR Processing
    lidar_processor_node = Node(
        package='isaac_ros_pointcloud_utils',
        executable='pointcloud_to_laserscan_node',
        name='lidar_processor',
        parameters=[config_dir]
    )

    # Isaac ROS IMU Filter
    imu_filter_node = Node(
        package='isaac_ros_imu_filter',
        executable='imu_filter_node',
        name='imu_filter_node',
        parameters=[
            config_dir,
            {'use_mag': False},
            {'publish_tf': False}
        ]
    )

    # Isaac ROS Sensor Fusion Node
    fusion_node = Node(
        package='isaac_ros_modular_multisense_fusion',
        executable='multisense_fusion_node', 
        name='sensor_fusion_node',
        parameters=[config_dir],
        remappings=[
            ('image_raw', 'camera/image_rect'),
            ('points_raw', 'lidar/points_processed'),
            ('imu_raw', 'imu/data_filtered'),
            ('fused_perception', 'perception/fused_output')
        ]
    )

    # Create launch description and add actions
    ld = LaunchDescription()

    ld.add_action(config_dir_cmd)
    ld.add_action(image_pipeline_node)
    ld.add_action(lidar_processor_node)
    ld.add_action(imu_filter_node)
    ld.add_action(fusion_node)

    return ld

# This launch file would be executed with:
# ros2 launch isaac_ros_sensor_fusion sensor-fusion.launch.py