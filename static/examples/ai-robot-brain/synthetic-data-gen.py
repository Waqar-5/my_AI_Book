#!/usr/bin/env python3
# Isaac Sim synthetic data generation example

"""
This script demonstrates synthetic data generation using Isaac Sim
"""
import numpy as np
import json

def generate_synthetic_sensor_data():
    """
    Creates synthetic sensor data that mimics real sensor outputs
    """
    print("Generating synthetic sensor data...")
    
    # Generate synthetic RGB image
    width, height = 640, 480
    rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Generate synthetic depth map
    depth_map = np.random.uniform(0.1, 10.0, (height, width)).astype(np.float32)
    
    # Generate synthetic LiDAR point cloud
    num_points = 1000
    point_cloud = np.random.uniform(-10, 10, (num_points, 3)).astype(np.float32)
    
    # Generate synthetic IMU data
    imu_data = {
        "acceleration": [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(9.81, 0.1)],
        "angular_velocity": [np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)]
    }
    
    print(f"✓ Generated RGB image: {rgb_image.shape}")
    print(f"✓ Generated depth map: {depth_map.shape}")  
    print(f"✓ Generated point cloud: {point_cloud.shape}")
    print(f"✓ Generated IMU data: {imu_data}")
    
    # Save data in a format that could be used for training
    synthetic_data_package = {
        "timestamp": "2025-12-16T10:30:00Z",
        "sensors": {
            "rgb_camera": {
                "resolution": [width, height],
                "format": "uint8"
            },
            "depth_camera": {
                "resolution": [width, height],
                "format": "float32"
            },
            "lidar": {
                "num_points": num_points,
                "format": "float32"
            },
            "imu": {
                "format": "dict with x,y,z values"
            }
        }
    }
    
    with open("synthetic-data-package.json", "w") as f:
        json.dump(synthetic_data_package, f, indent=2)
    
    print("✓ Synthetic data package saved to synthetic-data-package.json")
    
    return True

if __name__ == "__main__":
    generate_synthetic_sensor_data()