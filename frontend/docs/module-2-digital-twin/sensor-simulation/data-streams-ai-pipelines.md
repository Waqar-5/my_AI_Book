# Data Streams and AI Perception Pipelines

## Introduction

This chapter explores creating and processing realistic sensor data streams from simulated sensors (LiDAR, depth cameras, IMUs) in digital twin environments for AI perception pipelines. We'll cover how to generate, process, and validate sensor data that closely mimics real-world conditions, enabling effective training and testing of AI perception systems for humanoid robots.

## Sensor Data Pipeline Architecture

### Overview of the Pipeline

The sensor data pipeline in digital twins typically consists of:

1. **Sensor Simulation**: Gazebo generating realistic sensor data
2. **Data Transport**: ROS 2 topics/messages transferring data
3. **Preprocessing**: Filtering, calibration, and formatting
4. **AI Processing**: Perception algorithms consuming the data
5. **Validation**: Comparing results with ground truth

```
Gazebo Sensors → ROS 2 Topics → Preprocessing → AI Perception → Validation
```

### Message Types and Formats

Common ROS 2 message types for sensor data:

- **sensor_msgs/LaserScan**: LiDAR data
- **sensor_msgs/Image**: Camera images  
- **sensor_msgs/PointCloud2**: Point cloud data
- **sensor_msgs/Imu**: IMU data
- **sensor_msgs/CameraInfo**: Camera calibration parameters

## Multi-Sensor Data Integration

### Synchronizing Sensor Streams

When working with multiple sensors, synchronization is critical:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters

class MultiSensorFusionNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')
        
        # Create subscribers for different sensors
        lidar_sub = message_filters.Subscriber(self, LaserScan, '/humanoid_robot/lidar/scan')
        camera_sub = message_filters.Subscriber(self, Image, '/humanoid_robot/camera/rgb/image_raw')
        imu_sub = message_filters.Subscriber(self, Imu, '/humanoid_robot/imu/data')
        
        # Synchronize sensor data with approximate time sync
        # This handles slight timing differences between sensors
        ats = ApproximateTimeSynchronizer(
            [lidar_sub, camera_sub, imu_sub], 
            queue_size=10, 
            slop=0.1  # 100ms tolerance
        )
        ats.registerCallback(self.sensor_callback)
        
    def sensor_callback(self, lidar_msg, camera_msg, imu_msg):
        """Process synchronized sensor data"""
        self.get_logger().info(f'Received synchronized data: '
                              f'Lidar={len(lidar_msg.ranges)} points, '
                              f'Camera={camera_msg.width}x{camera_msg.height}, '
                              f'IMU orientation valid={imu_msg.orientation.w != 0}')
        
        # Process with AI perception algorithms
        self.process_perception(lidar_msg, camera_msg, imu_msg)
    
    def process_perception(self, lidar_msg, camera_msg, imu_msg):
        """AI perception pipeline processing"""
        # Implementation of perception algorithms
        # 1. Object detection using LiDAR and camera data
        # 2. State estimation using IMU data
        # 3. Fusion of information from multiple sensors
        pass

def main(args=None):
    rclpy.init(args=args)
    node = MultiSensorFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Calibration and Transformation

Properly calibrated sensors are essential for accurate AI processing:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros
from geometry_msgs.msg import TransformStamped

class SensorCalibrator:
    def __init__(self):
        # Example transformation matrix from robot base to LiDAR
        self.lidar_to_base = np.array([
            [1, 0, 0, 0.1],   # x offset
            [0, 1, 0, 0.0],   # y offset
            [0, 0, 1, 1.0],   # z offset (height)
            [0, 0, 0, 1]
        ])
        
        # Example transformation matrix from robot base to camera
        self.camera_to_base = np.array([
            [1, 0, 0, 0.15],  # x offset
            [0, 1, 0, 0.0],   # y offset
            [0, 0, 1, 1.2],   # z offset (height)
            [0, 0, 0, 1]
        ])
    
    def transform_lidar_points(self, points, base_pose):
        """Transform LiDAR points to world coordinates"""
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation: world = base * lidar_to_base * points
        world_points = (base_pose @ self.lidar_to_base @ points_homo.T).T
        
        return world_points[:, :3]  # Remove homogeneous coordinate
    
    def fuse_sensor_data(self, lidar_data, camera_data, imu_data, robot_pose):
        """Fuse data from multiple sensors"""
        # Transform sensor data to consistent coordinate frame
        lidar_world = self.transform_lidar_points(lidar_data, robot_pose)
        
        # Apply IMU-based motion compensation to LiDAR data
        # (if the scan was collected over time)
        compensated_lidar = self.compensate_motion(lidar_world, imu_data, robot_pose)
        
        # Process fused data through perception pipeline
        return self.run_perception_pipeline(compensated_lidar, camera_data)
    
    def compensate_motion(self, lidar_points, imu_data, base_pose):
        """Apply motion compensation to LiDAR points"""
        # Motion compensation implementation
        # This would adjust points based on robot motion during scan
        return lidar_points  # Simplified implementation
    
    def run_perception_pipeline(self, lidar_data, camera_data):
        """Run perception algorithms on fused data"""
        # 1. Object detection on LiDAR data
        lidar_objects = self.lidar_object_detection(lidar_data)
        
        # 2. Object detection on camera data
        camera_objects = self.camera_object_detection(camera_data)
        
        # 3. Data association and fusion
        fused_objects = self.associate_and_fuse(lidar_objects, camera_objects)
        
        return fused_objects
    
    def lidar_object_detection(self, points):
        """Run object detection on point cloud data"""
        # Implementation of point cloud object detection
        # Could use clustering, deep learning, or geometric methods
        return []  # Simplified implementation
    
    def camera_object_detection(self, image):
        """Run object detection on camera image"""
        # Implementation of 2D object detection
        # Could use CNNs like YOLO, SSD, or classical methods
        return []  # Simplified implementation
    
    def associate_and_fuse(self, lidar_objects, camera_objects):
        """Associate and fuse detections from different sensors"""
        # Implementation of data association and fusion
        return []  # Simplified implementation
```

## LiDAR Processing Pipeline

### Point Cloud Generation and Processing

Convert LiDAR scans to point clouds for AI processing:

```python
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import struct

class LiDARProcessor:
    def __init__(self):
        # LiDAR parameters
        self.min_angle = 0.0
        self.max_angle = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0
        self.scan_time = 0.0
    
    def scan_to_pointcloud(self, scan_msg):
        """Convert LaserScan to PointCloud2 message"""
        # Update parameters if this is the first scan
        if self.min_angle == 0:
            self.min_angle = scan_msg.angle_min
            self.max_angle = scan_msg.angle_max
            self.angle_increment = scan_msg.angle_increment
            self.range_min = scan_msg.range_min
            self.range_max = scan_msg.range_max
            self.scan_time = scan_msg.scan_time
        
        # Generate point cloud from scan data
        points = []
        angle = self.min_angle
        
        for range_val in scan_msg.ranges:
            # Skip invalid ranges
            if range_val < self.range_min or range_val > self.range_max or np.isnan(range_val):
                angle += self.angle_increment
                continue
            
            # Calculate 2D point in LiDAR frame
            x = range_val * np.cos(angle)
            y = range_val * np.sin(angle)
            z = 0.0  # Assuming 2D scan
            
            points.append([x, y, z])
            angle += self.angle_increment
        
        # Convert to PointCloud2 message
        return self.create_pointcloud2(scan_msg.header, points)
    
    def create_pointcloud2(self, header, points):
        """Create PointCloud2 message from list of points"""
        # Create fields for PointCloud2
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Pack the points into binary data
        data = []
        for point in points:
            data.append(struct.pack('fff', point[0], point[1], point[2]))
        
        # Create and return PointCloud2 message
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12  # 3 * 4 bytes per float
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = b''.join(data)
        
        return cloud
    
    def preprocess_pointcloud(self, pointcloud):
        """Preprocess point cloud for AI perception"""
        # 1. Remove ground plane
        filtered_points = self.remove_ground_plane(pointcloud)
        
        # 2. Downsample if needed
        downsampled_points = self.voxel_filter(filtered_points, voxel_size=0.1)
        
        # 3. Remove outliers
        cleaned_points = self.remove_outliers(downsampled_points)
        
        return cleaned_points
    
    def remove_ground_plane(self, points, threshold=0.1):
        """Remove ground plane using RANSAC algorithm"""
        # Simplified ground removal - in practice would use RANSAC
        # This example just removes points below a certain Z threshold
        ground_z = np.median(points[:, 2])  # Estimate ground plane Z
        filtered_points = points[points[:, 2] > ground_z + threshold]
        return filtered_points
    
    def voxel_filter(self, points, voxel_size=0.1):
        """Downsample points using voxel grid filtering"""
        # Create voxel grid
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        
        # Compute voxel coordinates
        voxel_coords = np.floor((points - min_vals) / voxel_size).astype(int)
        
        # Create unique keys for each voxel
        unique_coords, indices = np.unique(voxel_coords, axis=0, return_index=True)
        
        return points[indices]
    
    def remove_outliers(self, points, k=20, threshold=2.0):
        """Remove outliers using statistical analysis"""
        # This would typically use a k-nearest neighbors approach
        # For simplicity, using a basic z-score approach
        if len(points) < k:
            return points
        
        # Calculate statistics per dimension
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        
        # Remove points that are more than 'threshold' standard deviations away
        distances = np.abs(points - mean) / (std + 1e-6)  # Add small value to avoid division by 0
        valid_indices = np.all(distances < threshold, axis=1)
        
        return points[valid_indices]
    
    def run_object_detection(self, pointcloud):
        """Run AI object detection on point cloud"""
        # 1. Preprocess
        processed_cloud = self.preprocess_pointcloud(pointcloud)
        
        # 2. Extract features
        features = self.extract_features(processed_cloud)
        
        # 3. Run detection model (placeholder)
        detections = self.detect_objects_ml(processed_cloud, features)
        
        return detections
    
    def extract_features(self, pointcloud):
        """Extract geometric features from point cloud"""
        # Calculate various geometric features
        features = {
            'centroid': np.mean(pointcloud, axis=0),
            'std_deviation': np.std(pointcloud, axis=0),
            'min_coords': np.min(pointcloud, axis=0),
            'max_coords': np.max(pointcloud, axis=0),
            'volume': self.estimate_volume(pointcloud),
            'density': self.estimate_density(pointcloud)
        }
        return features
    
    def estimate_volume(self, points):
        """Estimate volume using bounding box"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return np.prod(max_coords - min_coords)  # Simplified volume calculation
    
    def estimate_density(self, points):
        """Estimate density as points per unit volume"""
        volume = self.estimate_volume(points)
        if volume == 0:
            return 0
        return len(points) / volume
    
    def detect_objects_ml(self, points, features):
        """Placeholder for ML-based object detection"""
        # In a real implementation, this would call a trained model
        # For example, using PointNet, PointNet++, or other 3D deep learning models
        return []
```

## Depth Camera Processing Pipeline

### RGB-D Data Processing

Processing RGB-D data for AI perception:

```python
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class RGBDProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None
        
    def process_rgbd_data(self, rgb_image_msg, depth_image_msg, camera_info_msg):
        """Process RGB-D data for AI perception"""
        # Convert ROS messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        
        # Get camera parameters
        if self.camera_matrix is None:
            self.camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(camera_info_msg.d)
        
        # 1. Object detection on RGB image
        rgb_detections = self.detect_objects_2d(rgb_image)
        
        # 2. Extract 3D positions from depth
        objects_3d = self.annotate_3d_positions(rgb_detections, depth_image, self.camera_matrix)
        
        # 3. Semantic segmentation
        semantic_mask = self.run_semantic_segmentation(rgb_image)
        
        # 4. Combine 2D detection with 3D position
        combined_results = self.combine_2d_3d(rgb_detections, objects_3d, semantic_mask)
        
        return combined_results
    
    def detect_objects_2d(self, image):
        """Run 2D object detection on RGB image"""
        # Example using OpenCV's DNN module with pre-trained model
        # Or integrate with ROS perception packages
        
        # Placeholder implementation
        # This would typically use models like YOLO, SSD, or Faster R-CNN
        detections = []
        
        # Convert image for network input
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=1.0/255.0, 
            size=(416, 416), 
            mean=(0, 0, 0),
            swapRB=True, 
            crop=False
        )
        
        # For this example, just return empty detections
        # In practice, run through a neural network
        return detections
    
    def annotate_3d_positions(self, detections, depth_image, camera_matrix):
        """Annotate 2D detections with 3D positions"""
        objects_3d = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']  # Bounding box
            
            # Calculate center of bounding box
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # Get depth at center point (average over small area)
            roi_depth = depth_image[center_y-5:center_y+5, center_x-5:center_x+5]
            valid_depths = roi_depth[np.isfinite(roi_depth)]
            
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                
                # Convert 2D image coordinates to 3D world coordinates
                point_3d = self.image_to_world(center_x, center_y, avg_depth, camera_matrix)
                
                detection['position_3d'] = point_3d
                objects_3d.append(detection)
        
        return objects_3d
    
    def image_to_world(self, u, v, depth, camera_matrix):
        """Convert image coordinates to world coordinates using depth"""
        # Camera intrinsic parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return [x, y, z]
    
    def run_semantic_segmentation(self, image):
        """Run semantic segmentation on RGB image"""
        # Placeholder for semantic segmentation
        # This would use models like DeepLab, PSPNet, or Mask R-CNN
        
        # For this example, return a dummy segmentation mask
        height, width = image.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)
    
    def combine_2d_3d(self, detections_2d, detections_3d, semantic_mask):
        """Combine 2D and 3D detection information"""
        combined_results = []
        
        for i, det_2d in enumerate(detections_2d):
            if i < len(detections_3d):
                combined = {
                    'class': det_2d['class'],
                    'bbox_2d': det_2d['bbox'],
                    'position_3d': detections_3d[i]['position_3d'],
                    'confidence': det_2d['confidence'],
                    'semantic_mask': semantic_mask
                }
                combined_results.append(combined)
        
        return combined_results
    
    def create_3d_point_cloud(self, rgb_image, depth_image, camera_info):
        """Create 3D point cloud from RGB-D data"""
        height, width = depth_image.shape
        
        # Get camera parameters
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y
        fx = camera_info.k[0]  # Focal length x
        fy = camera_info.k[4]  # Focal length y
        
        points = []
        colors = []
        
        # Downsample the image for performance
        step = 2  # Use every 2nd pixel
        
        for v in range(0, height, step):
            for u in range(0, width, step):
                z = depth_image[v, u]
                
                # Skip invalid pixels
                if z == 0 or np.isnan(z) or np.isinf(z):
                    continue
                
                # Calculate 3D coordinates
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # Get color
                color = rgb_image[v, u] if len(rgb_image.shape) == 3 else [rgb_image[v, u]] * 3
                
                points.append([x, y, z])
                colors.append(color)
        
        return np.array(points), np.array(colors)
```

## IMU Processing Pipeline

### Inertial Data Processing

Processing IMU data for AI perception and state estimation:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu
import math

class IMUProcessor:
    def __init__(self):
        # State estimation
        self.orientation = R.from_quat([0, 0, 0, 1])  # Identity quaternion
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.linear_acceleration = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, 0.0])
        
        # Time tracking
        self.last_time = None
        
        # Parameters for filtering
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.acc_bias = np.array([0.0, 0.0, 0.0])
        
        # Filtering parameters
        self.complementary_filter_alpha = 0.98  # High weight to gyro
        self.gravity = np.array([0, 0, 9.81])
    
    def process_imu_data(self, imu_msg):
        """Process IMU message and update state estimates"""
        # Convert ROS message to numpy arrays
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y
        az = imu_msg.linear_acceleration.z
        linear_acc = np.array([ax, ay, az])
        
        wx = imu_msg.angular_velocity.x
        wy = imu_msg.angular_velocity.y
        wz = imu_msg.angular_velocity.z
        angular_vel = np.array([wx, wy, wz])
        
        # Extract orientation from message if available
        orientation_msg = imu_msg.orientation
        orientation_quat = np.array([
            orientation_msg.x,
            orientation_msg.y, 
            orientation_msg.z,
            orientation_msg.w
        ])
        
        # Get timestamp
        current_time = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec / 1e9
        
        # If this is the first message, initialize
        if self.last_time is None:
            self.last_time = current_time
            if np.linalg.norm(orientation_quat) > 0:
                self.orientation = R.from_quat(orientation_quat)
            return
            
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update state using sensor fusion
        self.update_orientation(linear_acc, angular_vel, dt)
        self.update_position(linear_acc, dt)
        self.update_velocity(linear_acc, dt)
    
    def update_orientation(self, linear_acc, angular_vel, dt):
        """Update orientation estimate using gyroscope and accelerometer"""
        # Method 1: Integrate gyroscope readings
        # Create a small rotation from angular velocity
        angle = np.linalg.norm(angular_vel) * dt
        if angle > 1e-6:  # Avoid division by zero
            axis = angular_vel / np.linalg.norm(angular_vel)
            # Create rotation vector
            rot_vec = axis * angle
            # Create small rotation
            small_rotation = R.from_rotvec(rot_vec)
            # Update orientation
            self.orientation = small_rotation * self.orientation
        
        # Method 2: Use accelerometer for gravity reference (complementary filter)
        # This helps correct drift in roll and pitch
        if np.linalg.norm(linear_acc) > 1e-6:  # Avoid zero norm
            # Normalize accelerometer reading (remove self-motion)
            acc_norm = linear_acc / np.linalg.norm(linear_acc)
            
            # Calculate roll and pitch from accelerometer
            pitch = math.atan2(-acc_norm[0], math.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            roll = math.atan2(acc_norm[1], acc_norm[2])
            
            # Create quaternion from roll and pitch (ignoring yaw from accelerometer)
            acc_quat = R.from_euler('xyz', [roll, pitch, 0])
            
            # Apply complementary filter
            # Convert rotations to quaternions for blending
            current_quat = self.orientation.as_quat()
            acc_quat_vec = acc_quat.as_quat()
            
            # Blend: high weight to gyro (for fast changes), low weight to acc (for drift correction)
            blended_quat = (self.complementary_filter_alpha * current_quat + 
                           (1 - self.complementary_filter_alpha) * acc_quat_vec)
            blended_quat = blended_quat / np.linalg.norm(blended_quat)
            
            self.orientation = R.from_quat(blended_quat)
    
    def update_position(self, linear_acc, dt):
        """Update position by integrating acceleration"""
        # Transform acceleration from body frame to world frame
        world_acc = self.orientation.apply(linear_acc)
        
        # Remove gravity
        world_acc -= self.gravity
        
        # Update velocity: v = v0 + a * dt
        self.velocity += world_acc * dt
        
        # Update position: p = p0 + v * dt
        self.position += self.velocity * dt
    
    def update_velocity(self, linear_acc, dt):
        """Update velocity by integrating acceleration"""
        # This is also done in update_position, but separated for clarity
        world_acc = self.orientation.apply(linear_acc) - self.gravity
        self.velocity += world_acc * dt
    
    def get_robot_state(self):
        """Return current robot state estimate"""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation.as_quat(),
            'angular_velocity': self.angular_velocity,
            'linear_acceleration': self.linear_acceleration
        }
    
    def calibrate_sensor(self, imu_data_buffer):
        """Calibrate sensor biases using a buffer of static measurements"""
        # Calculate bias from static measurements
        # For gyroscope, bias is the average reading when stationary
        gyro_readings = np.array([data.angular_velocity for data in imu_data_buffer])
        self.gyro_bias = np.mean(gyro_readings, axis=0)
        
        # For accelerometer, bias is difference from gravity
        acc_readings = np.array([data.linear_acceleration for data in imu_data_buffer])
        self.acc_bias = np.mean(acc_readings, axis=0) - np.array([0, 0, 9.81])
    
    def fusion_with_other_sensors(self, lidar_pose, camera_pose, imu_weight=0.7):
        """Fuse IMU pose estimate with other sensors"""
        # Get pose from IMU integration
        imu_pose = {
            'position': self.position,
            'orientation': self.orientation.as_quat()
        }
        
        # Fusion could use various techniques:
        # 1. Kalman Filter
        # 2. Particle Filter  
        # 3. Complementary Filter (weighted average)
        
        # Simple weighted fusion (in practice, use proper filtering)
        fused_position = (imu_weight * imu_pose['position'] + 
                         (1 - imu_weight) * lidar_pose['position'])
        # Orientation fusion would require special handling of quaternions
        
        return {
            'position': fused_position,
            'orientation': imu_pose['orientation']  # Simplified
        }
```

## AI Perception Pipeline Patterns

### Modular Perception Architecture

Create a modular architecture that allows different sensors and algorithms to be swapped:

```python
from abc import ABC, abstractmethod
import numpy as np

class SensorData(ABC):
    """Base class for sensor data"""
    def __init__(self, timestamp):
        self.timestamp = timestamp

class LiDARData(SensorData):
    def __init__(self, timestamp, points):
        super().__init__(timestamp)
        self.points = points

class CameraData(SensorData):
    def __init__(self, timestamp, image, camera_info):
        super().__init__(timestamp)
        self.image = image
        self.camera_info = camera_info

class IMUData(SensorData):
    def __init__(self, timestamp, linear_acceleration, angular_velocity):
        super().__init__(timestamp)
        self.linear_acceleration = linear_acceleration
        self.angular_velocity = angular_velocity

class PerceptionAlgorithm(ABC):
    """Base class for perception algorithms"""
    
    @abstractmethod
    def process(self, sensor_data):
        """Process sensor data and return results"""
        pass

class LiDARObjectDetector(PerceptionAlgorithm):
    def process(self, lidar_data):
        """Detect objects in LiDAR point cloud"""
        # Implementation of LiDAR object detection
        # This could use geometric methods, clustering, or deep learning
        return []

class CameraObjectDetector(PerceptionAlgorithm):
    def process(self, camera_data):
        """Detect objects in camera image"""
        # Implementation of 2D object detection
        # This could use CNNs like YOLO, SSD, etc.
        return []

class SensorFusion:
    """Fusion module to combine data from different sensors"""
    def __init__(self):
        self.lidar_detector = LiDARObjectDetector()
        self.camera_detector = CameraObjectDetector()
    
    def fuse_data(self, lidar_data, camera_data):
        """Fuse data from multiple sensors"""
        # Run individual detectors
        lidar_objects = self.lidar_detector.process(lidar_data)
        camera_objects = self.camera_detector.process(camera_data)
        
        # Apply sensor fusion techniques (data association, etc.)
        fused_objects = self.associate_detections(lidar_objects, camera_objects)
        
        return fused_objects
    
    def associate_detections(self, lidar_objects, camera_objects):
        """Associate detections from different sensors"""
        # Implementation of data association
        # Match 3D LiDAR objects with 2D camera objects
        return []

class AIPipeline:
    """Main AI perception pipeline"""
    def __init__(self):
        self.fusion = SensorFusion()
        
        # Buffer for temporal fusion
        self.temporal_buffer = {}
        
    def run_pipeline(self, sensor_data_list):
        """Run the complete AI perception pipeline"""
        results = {}
        
        # Process each sensor type
        lidar_data = [data for data in sensor_data_list if isinstance(data, LiDARData)]
        camera_data = [data for data in sensor_data_list if isinstance(data, CameraData)]
        imu_data = [data for data in sensor_data_list if isinstance(data, IMUData)]
        
        # For simplicity, just process the latest available data
        if lidar_data and camera_data:
            latest_lidar = lidar_data[-1]
            latest_camera = camera_data[-1]
            
            # Fuse sensor data
            fused_results = self.fusion.fuse_data(latest_lidar, latest_camera)
            results['fused_objects'] = fused_results
        
        # Add temporal fusion if needed
        results['temporal_analysis'] = self.temporal_analysis(sensor_data_list)
        
        return results
    
    def temporal_analysis(self, sensor_data_list):
        """Analyze temporal patterns in sensor data"""
        # Implementation of temporal analysis
        # Could detect movement patterns, predict future states, etc.
        return []
```

## Performance Optimization

### Efficient Data Processing

Optimize the pipeline for real-time performance:

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedAIPipeline:
    def __init__(self, num_threads=4):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # Pre-allocated buffers for performance
        self.buffer_size = 100
        self.lidar_buffer = [None] * self.buffer_size
        self.camera_buffer = [None] * self.buffer_size
        
    def start_pipeline(self):
        """Start the processing pipeline"""
        self.is_running = True
        
        # Start processing threads
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()
        
    def stop_pipeline(self):
        """Stop the processing pipeline"""
        self.is_running = False
        self.processing_thread.join()
        self.executor.shutdown(wait=True)
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            # Batch process multiple sensor readings
            batch_data = self._collect_batch()
            
            if batch_data:
                # Process batch asynchronously
                future = self.executor.submit(self._process_batch, batch_data)
                
                # Add result to queue for downstream processing
                future.add_done_callback(
                    lambda f: self.result_queue.put(f.result())
                )
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def _collect_batch(self):
        """Collect a batch of sensor data for processing"""
        # Implementation to collect synchronized sensor data
        return []
    
    def _process_batch(self, batch_data):
        """Process a batch of sensor data"""
        # Process each sensor modality
        results = {}
        
        # Run perception algorithms in parallel
        futures = []
        
        for sensor_type, data in batch_data.items():
            if sensor_type == 'lidar':
                future = self.executor.submit(self._process_lidar, data)
                futures.append(('lidar', future))
            elif sensor_type == 'camera':
                future = self.executor.submit(self._process_camera, data)
                futures.append(('camera', future))
            # Add other sensor types as needed
        
        # Collect results
        for name, future in futures:
            try:
                results[name] = future.result(timeout=1.0)
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        # Fuse results
        fused_result = self._fuse_results(results)
        return fused_result
    
    def _process_lidar(self, lidar_data):
        """Process LiDAR data"""
        # Optimized LiDAR processing
        return []
    
    def _process_camera(self, camera_data):
        """Process camera data"""
        # Optimized camera processing
        return []
    
    def _fuse_results(self, results):
        """Fuse results from different sensors"""
        # Implementation of sensor fusion
        return results
```

## Quality Assurance and Validation

### Data Quality Metrics

Evaluate the quality of sensor data and perception outputs:

```python
import numpy as np

class QualityAssessment:
    def __init__(self):
        self.metrics = {}
        
    def assess_lidar_quality(self, pointcloud, ground_truth=None):
        """Assess quality of LiDAR data"""
        metrics = {
            'point_density': self.calculate_point_density(pointcloud),
            'coverage': self.calculate_fov_coverage(pointcloud),
            'noise_level': self.estimate_noise_level(pointcloud),
            'outlier_ratio': self.calculate_outlier_ratio(pointcloud)
        }
        
        if ground_truth:
            metrics['accuracy'] = self.calculate_accuracy(pointcloud, ground_truth)
        
        return metrics
    
    def calculate_point_density(self, points, region_size=1.0):
        """Calculate point density in a region"""
        if len(points) == 0:
            return 0
        
        # Simple approach: calculate density in bounding box
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        
        volume = np.prod(max_bounds - min_bounds)
        if volume == 0:
            return float('inf')
        
        return len(points) / volume
    
    def calculate_fov_coverage(self, points):
        """Calculate field of view coverage"""
        # Simplified implementation - in practice this would
        # calculate coverage based on expected sensor FoV
        if len(points) == 0:
            return 0
        
        # Calculate coverage as ratio of occupied angle space
        angles = np.arctan2(points[:, 1], points[:, 0])  # Azimuth angles
        unique_angles = np.unique(np.round(angles, decimals=2))  # Discretize angles
        
        # Assume 360-degree coverage possible
        possible_angles = np.linspace(-np.pi, np.pi, 360)  # 1-degree resolution
        
        coverage = len(unique_angles) / len(possible_angles)
        return min(coverage, 1.0)  # Cap at 1.0
    
    def estimate_noise_level(self, points):
        """Estimate noise level in point cloud"""
        if len(points) < 2:
            return 0
        
        # Calculate distances to nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Average distance to nearest neighbor (excluding self)
        avg_distance = np.mean(distances[:, 1])
        return avg_distance
    
    def calculate_outlier_ratio(self, points, threshold=2.0):
        """Calculate ratio of outlier points"""
        if len(points) < 2:
            return 0
        
        # Use statistical approach to identify outliers
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        
        # Calculate z-scores
        z_scores = np.abs((points - mean) / (std + 1e-6))  # Avoid division by zero
        outliers = np.any(z_scores > threshold, axis=1)
        
        return np.sum(outliers) / len(points)
    
    def calculate_accuracy(self, measured, ground_truth):
        """Calculate accuracy against ground truth"""
        if len(measured) == 0 or len(ground_truth) == 0:
            return 0
        
        # Simple distance-based accuracy
        distances = np.linalg.norm(measured[:, :3] - ground_truth[:, :3], axis=1)
        mae = np.mean(distances)  # Mean absolute error
        rmse = np.sqrt(np.mean(distances**2))  # Root mean square error
        
        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy_at_10cm': np.mean(distances < 0.1),  # Accuracy within 10cm
            'accuracy_at_1m': np.mean(distances < 1.0)  # Accuracy within 1m
        }
    
    def assess_camera_quality(self, image):
        """Assess quality of camera image"""
        metrics = {
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'sharpness': self.calculate_sharpness(image),
            'noise_level': self.estimate_image_noise(image)
        }
        return metrics
    
    def calculate_brightness(self, image):
        """Calculate image brightness"""
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.mean(gray)
    
    def calculate_contrast(self, image):
        """Calculate image contrast"""
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.std(gray)
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def estimate_image_noise(self, image):
        """Estimate image noise level"""
        # Simple approach using wavelet decomposition or noise estimation
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use the difference between original and smoothed image
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        noise = np.std(gray - smoothed)
        return noise
```

## Summary

Creating realistic sensor data streams for AI perception pipelines in digital twins involves multiple components working together: sensor simulation with realistic noise models, proper data transport and synchronization, efficient preprocessing algorithms, and comprehensive validation systems. The quality of the simulated data directly affects the effectiveness of AI training and testing.

The techniques covered in this chapter provide a foundation for building robust perception pipelines that can effectively process data from multiple sensors (LiDAR, cameras, IMUs) to enable humanoid robots to understand and interact with their environment. Proper sensor fusion and temporal consistency ensure that AI systems receive coherent, accurate information about the world.

The next chapter will cover perception accuracy validation, ensuring that the simulated sensor data and perception algorithms produce results that are suitable for training AI systems for deployment on physical robots.