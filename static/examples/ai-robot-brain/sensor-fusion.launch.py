# Isaac ROS Sensor Fusion Launch File
# File: sensor-fusion.launch.py

# This is a conceptual launch file that would configure Isaac ROS sensor fusion
# In a real implementation, this would be a Python ROS 2 launch file

"""
Launch file for Isaac ROS sensor fusion pipeline

This launch file sets up the sensor fusion pipeline using Isaac ROS components:
- Camera data processing
- LiDAR data processing
- IMU data processing
- Multi-sensor data fusion
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
        executable='image_pipeline',
        name='image_pipeline',
        parameters=[
            config_dir,
            {'rectification_width': 640},
            {'rectification_height': 480},
            {'enable_stereo': False}
        ],
        remappings=[
            ('image_raw', 'camera/image_raw'),
            ('camera_info', 'camera/camera_info'),
            ('image_rect', 'camera/image_rect')
        ]
    )

    # Isaac ROS LiDAR Processor
    lidar_processor_node = Node(
        package='isaac_ros_lidar_processing',
        executable='lidar_processor',
        name='lidar_processor',
        parameters=[config_dir],
        remappings=[
            ('point_cloud_in', 'lidar/points_raw'),
            ('point_cloud_out', 'lidar/points_processed')
        ]
    )

    # Isaac ROS IMU Filter
    imu_filter_node = Node(
        package='isaac_ros_imu_filter',
        executable='imu_filter',
        name='imu_filter',
        parameters=[config_dir],
        remappings=[
            ('imu_raw', 'imu/data_raw'),
            ('imu_filtered', 'imu/data_filtered')
        ]
    )

    # Isaac ROS Sensor Fusion Node
    fusion_node = Node(
        package='isaac_ros_sensor_fusion',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        parameters=[config_dir],
        remappings=[
            ('camera_data', 'camera/image_rect'),
            ('lidar_data', 'lidar/points_processed'),
            ('imu_data', 'imu/data_filtered'),
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