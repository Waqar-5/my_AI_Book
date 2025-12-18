# Perception and Navigation in Autonomous Humanoid Systems

## Introduction

This chapter focuses on the integration of perception and navigation systems in Vision-Language-Action (VLA) architectures for autonomous humanoid robots. The combination of these two systems enables robots to understand their environment and navigate safely through complex spaces to perform tasks based on voice commands.

## Perception for Navigation

### Environment Sensing

For effective navigation, humanoid robots require multiple perception modalities working in concert:

1. **Visual Perception**: Optical sensors provide rich environmental information
2. **Range Sensing**: LiDAR and depth sensors provide accurate distance measurements
3. **Localization**: Determining robot position within the environment
4. **Mapping**: Creating and updating representations of unknown or changing environments

### Object Detection and Classification

Robots need to identify and classify objects in their environment to navigate safely:

```python
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

class PerceptionNavigator:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Subscribe to camera and depth data
        self.image_sub = rospy.Subscriber('/humanoid_robot/camera/rgb/image_raw', 
                                         Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/humanoid_robot/camera/depth/image_raw',
                                         Image, self.depth_callback)
        
        self.latest_image = None
        self.latest_depth = None
        self.object_detector = self.initialize_object_detector()
        
    def initialize_object_detector(self):
        """
        Initialize object detection model (e.g., YOLO, SSD, etc.)
        """
        # In practice, this would load a pre-trained model
        # For example, using a model that can detect obstacles, humans, doors, etc.
        return None  # Placeholder
    
    def image_callback(self, image_msg):
        """
        Process incoming RGB image for object detection
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.latest_image = cv_image
            self.process_image_for_navigation(cv_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def depth_callback(self, depth_msg):
        """
        Process incoming depth image for distance measurements
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.latest_depth = depth_image
            self.process_depth_for_navigation(depth_image)
        except Exception as e:
            rospy.logerr(f"Error processing depth: {e}")
    
    def process_image_for_navigation(self, image):
        """
        Detect objects relevant for navigation in the image
        """
        # Detect obstacles, humans, doors, stairs, etc.
        detections = self.detect_objects(image)
        
        # Filter for objects relevant to navigation
        navigation_objects = self.filter_navigation_objects(detections)
        
        # Create obstacle map from detections
        obstacle_map = self.create_obstacle_map(navigation_objects)
        
        # Update navigation system with obstacle information
        self.update_navigation_system(obstacle_map)
        
        return navigation_objects
    
    def process_depth_for_navigation(self, depth_image):
        """
        Process depth information for navigation
        """
        # Identify surfaces, steps, obstacles at different heights
        traversable_areas = self.identify_traversable_areas(depth_image)
        obstacles = self.identify_depth_obstacles(depth_image)
        
        # Update navigation system with depth information
        self.update_navigation_with_depth(traversable_areas, obstacles)
        
        return traversable_areas, obstacles
    
    def detect_objects(self, image):
        """
        Detect objects in the image relevant for navigation
        """
        # In a real implementation, this would use a deep learning model
        # For this example, we'll simulate object detection
        # with mock data
        
        # Simulate detection of common navigation-relevant objects
        detections = []
        
        # Example: Detect humans in the scene
        human_boxes = self.simulate_human_detection(image)
        for box in human_boxes:
            detections.append({
                'label': 'person',
                'bbox': box,
                'confidence': 0.85,
                'distance': self.estimate_distance_from_depth(box, self.latest_depth)
            })
        
        # Example: Detect obstacles
        obstacle_boxes = self.simulate_obstacle_detection(image)
        for box in obstacle_boxes:
            detections.append({
                'label': 'obstacle',
                'bbox': box,
                'confidence': 0.92,
                'distance': self.estimate_distance_from_depth(box, self.latest_depth)
            })
        
        return detections
    
    def filter_navigation_objects(self, detections):
        """
        Filter objects that are relevant for navigation
        """
        navigation_objects = []
        
        for detection in detections:
            if detection['label'] in ['person', 'obstacle', 'door', 'stair', 'furniture']:
                navigation_objects.append(detection)
        
        return navigation_objects
    
    def create_obstacle_map(self, navigation_objects):
        """
        Create a 2D obstacle map from object detections
        """
        # This would create a map representation suitable for navigation
        # In practice, this might be a costmap or occupancy grid
        obstacle_map = np.zeros((200, 200))  # 20mx20m map at 10cm resolution
        
        for obj in navigation_objects:
            # Convert bbox to map coordinates
            center_x, center_y = obj['bbox']['center']
            distance = obj['distance']
            
            # Calculate position relative to robot
            # This is a simplified version - real implementation would use TF
            map_x = int(center_x / 0.1)  # Convert to map coordinates
            map_y = int(center_y / 0.1)
            
            # Mark as obstacle if it's close and substantial
            if distance < 3.0:  # Only obstacles within 3 meters
                obstacle_map[map_x, map_y] = 1.0
        
        return obstacle_map
    
    def identify_traversable_areas(self, depth_image):
        """
        Identify areas that are safe and traversable based on depth data
        """
        traversable_areas = []
        
        # Identify flat surfaces suitable for walking
        # Check for sufficient clearance above ground
        # Check for obstacles in the path
        
        # Simple example: find areas with appropriate height range
        ground_height = self.estimate_ground_height(depth_image)
        height_threshold = 0.2  # Minimum clearance for robot
        max_height = 2.0       # Maximum height for ceiling
        height_map = depth_image - ground_height
        
        # Identify areas within acceptable height range
        traversable_mask = np.logical_and(height_map > height_threshold, height_map < max_height)
        
        return traversable_mask
    
    def identify_depth_obstacles(self, depth_image):
        """
        Identify obstacles from depth data
        """
        # Identify surfaces too close to the robot
        min_obstacle_distance = 0.3  # 30cm minimum clearance
        obstacle_mask = depth_image < min_obstacle_distance
        
        # Use connected components to identify obstacle regions
        import scipy.ndimage as ndi
        labeled_obstacles, num_obstacles = ndi.label(obstacle_mask)
        
        obstacle_regions = []
        for i in range(1, num_obstacles + 1):
            coords = np.where(labeled_obstacles == i)
            if len(coords[0]) > 10:  # Only consider regions with minimum size
                center = (np.mean(coords[0]), np.mean(coords[1]))
                min_depth = np.min(depth_image[coords])
                obstacle_regions.append({
                    'center': center,
                    'size': len(coords[0]),
                    'min_distance': min_depth
                })
        
        return obstacle_regions
    
    def estimate_distance_from_depth(self, bbox, depth_image):
        """
        Estimate distance to an object based on its bounding box
        """
        if depth_image is None:
            return float('inf')
        
        x1, y1, x2, y2 = bbox['coordinates']
        # Get depth at center of bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Make sure coordinates are within image bounds
        center_x = max(0, min(depth_image.shape[1]-1, center_x))
        center_y = max(0, min(depth_image.shape[0]-1, center_y))
        
        distance = depth_image[center_y, center_x]
        return float(distance) if not np.isnan(distance) and not np.isinf(distance) else float('inf')
    
    def update_navigation_system(self, obstacle_map):
        """
        Update the navigation system with obstacle information
        """
        # In a real system, this would update the costmap for navigation
        # through a ROS service call or action
        pass
    
    def update_navigation_with_depth(self, traversable_areas, obstacles):
        """
        Update navigation system with depth-derived information
        """
        # In a real system, this would update navigation parameters based on
        # traversable areas and obstacle information
        pass
```

## Navigation Planning and Execution

### Path Planning Algorithms

Humanoid robots require sophisticated path planning that considers their physical characteristics:

```python
class HumanoidPathPlanner:
    def __init__(self):
        self.planning_algorithms = {
            'global': 'navfn',  # Global planner for finding route
            'local': 'teb_local_planner'  # Local planner for obstacle avoidance
        }
        
        # Humanoid-specific parameters
        self.footprint = [
            (-0.3, -0.2), (0.3, -0.2),   # Approximate humanoid footprint
            (0.3, 0.2), (-0.3, 0.2)
        ]
        self.inflation_radius = 0.5  # Safety margin around robot
        
    def plan_path(self, start_pose, goal_pose, costmap):
        """
        Plan a path from start pose to goal pose using the costmap
        """
        # Use a global planner to find an initial path
        global_path = self.global_plan(start_pose, goal_pose, costmap)
        
        if not global_path:
            return None
        
        # Optimize the path for humanoid characteristics
        optimized_path = self.humanoid_optimize_path(global_path, costmap)
        
        return optimized_path
    
    def global_plan(self, start_pose, goal_pose, costmap):
        """
        Use a global planner to find an initial path
        """
        # This would interface with ROS navigation stack
        # For example, using navfn, global_planner, or other global planners
        
        # Simulated path planning
        path = self.simulate_global_path(start_pose, goal_pose, costmap)
        return path
    
    def humanoid_optimize_path(self, path, costmap):
        """
        Optimize path for humanoid-specific constraints
        """
        optimized_path = []
        
        for i, waypoint in enumerate(path):
            # Ensure sufficient space for humanoid turning
            if self.has_sufficient_space(waypoint, costmap):
                optimized_path.append(waypoint)
            else:
                # Find alternative path around constrained area
                adjusted_waypoint = self.adjust_waypoint_for_clearance(waypoint, costmap)
                if adjusted_waypoint:
                    optimized_path.append(adjusted_waypoint)
        
        return optimized_path
    
    def has_sufficient_space(self, waypoint, costmap):
        """
        Check if there's sufficient space for humanoid at waypoint
        """
        # Check for sufficient clearance around waypoint
        # This would check the costmap for obstacles near the waypoint
        x, y = int(waypoint[0]), int(waypoint[1])
        
        # Check surrounding area for obstacles
        clearance_check_radius = int(self.inflation_radius / costmap.resolution)
        
        if x < clearance_check_radius or y < clearance_check_radius:
            return False
        
        if x >= costmap.width - clearance_check_radius or y >= costmap.height - clearance_check_radius:
            return False
        
        # Check costmap for obstacles in the vicinity
        neighborhood = costmap.costs[
            y - clearance_check_radius:y + clearance_check_radius,
            x - clearance_check_radius:x + clearance_check_radius
        ]
        
        # If any point in the neighborhood has high cost (obstacle), return False
        return not np.any(neighborhood > 50)  # Assuming costs > 50 are obstacles
    
    def adjust_waypoint_for_clearance(self, waypoint, costmap):
        """
        Find an alternative waypoint with sufficient clearance
        """
        # Search in nearby positions for a clear area
        search_radius = 20  # Search in a 20-cell radius
        x, y = int(waypoint[0]), int(waypoint[1])
        
        for r in range(1, search_radius):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) == r or abs(dy) == r:  # Only outer ring
                        nx, ny = x + dx, y + dy
                        
                        # Check bounds
                        if 0 <= nx < costmap.width and 0 <= ny < costmap.height:
                            # Check for sufficient space
                            test_waypoint = (nx, ny)
                            if self.has_sufficient_space(test_waypoint, costmap):
                                return test_waypoint
        
        return None  # No clear area found
```

### Navigation Execution

Execute the planned path while monitoring the environment:

```python
class NavigationController:
    def __init__(self):
        self.current_goal = None
        self.navigation_active = False
        self.path_following = False
        
        # Subscribe to navigation state
        self.state_sub = rospy.Subscriber('/humanoid_robot/move_base/status', 
                                         actionlib_msgs.msg.GoalStatusArray,
                                         self.navigation_state_callback)
        
        # Create action client for navigation
        self.nav_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.nav_client.wait_for_server()
    
    def navigate_to_pose(self, target_pose, callback=None):
        """
        Navigate to a specific pose in the environment
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = target_pose
        
        self.current_goal = goal
        self.navigation_active = True
        
        # Send goal to navigation server
        self.nav_client.send_goal(
            goal,
            done_cb=lambda status, result: self.navigation_done_callback(status, result, callback),
            active_cb=self.navigation_active_callback,
            feedback_cb=self.navigation_feedback_callback
        )
    
    def navigate_with_avoidance(self, target_pose, dynamic_obstacles=None):
        """
        Navigate to target while avoiding dynamic obstacles
        """
        # Incorporate dynamic obstacle information into navigation
        if dynamic_obstacles:
            self.update_dynamic_costmap(dynamic_obstacles)
        
        # Navigate as usual
        self.navigate_to_pose(target_pose)
    
    def update_dynamic_costmap(self, obstacles):
        """
        Update the costmap with dynamic obstacle information
        """
        # In a real system, this would update the navigation costmap
        # with information about moving obstacles
        
        # Example: Temporarily increase cost of cells near dynamic obstacles
        for obstacle in obstacles:
            if obstacle['type'] == 'moving':
                self.increase_local_cost(obstacle['position'], obstacle['velocity'])
    
    def navigation_done_callback(self, status, result, callback):
        """
        Handle completion of navigation goal
        """
        self.navigation_active = False
        rospy.loginfo(f"Navigation completed with status: {status}")
        
        if callback:
            callback(status, result)
    
    def navigation_active_callback(self):
        """
        Handle navigation goal becoming active
        """
        rospy.loginfo("Navigation goal is active")
        self.path_following = True
    
    def navigation_feedback_callback(self, feedback):
        """
        Handle navigation feedback (position updates)
        """
        # Monitor progress and adjust if needed
        current_pose = feedback.base_position
        
        # Check if we're making adequate progress
        if not self.making_adequate_progress(current_pose):
            rospy.logwarn("Navigation progress is inadequate, considering replanning")
            # Trigger replanning logic if needed
            # self.trigger_replanning()
    
    def making_adequate_progress(self, current_pose):
        """
        Check if the robot is making adequate progress toward the goal
        """
        # Calculate distance to goal
        if self.current_goal:
            goal_pose = self.current_goal.target_pose.pose
            distance_to_goal = self.calculate_distance(current_pose, goal_pose)
            
            # Compare with expected progress
            # Implementation would track progress over time
            return distance_to_goal < self.expected_remaining_distance
            
        return True  # Default to true if no goal set
```

## Integration with VLA System

### Voice Command Integration

Connect navigation to voice commands for autonomous movement:

```python
class VoiceNavigationIntegrator:
    def __init__(self, navigation_controller, perception_navigator):
        self.nav_controller = navigation_controller
        self.perception_navigator = perception_navigator
        self.location_map = {}  # Map of named locations
        
        # Initialize known locations
        self.initialize_known_locations()
    
    def initialize_known_locations(self):
        """
        Initialize known named locations in the environment
        """
        # This would typically come from a saved map or be learned during operation
        self.location_map = {
            'kitchen': (-2.5, 1.0, 0.0),  # (x, y, theta)
            'living_room': (1.0, 2.0, 1.57),
            'bedroom': (3.5, -1.0, 3.14),
            'entrance': (0.0, -3.0, 0.0),
            'office': (-1.5, -2.0, -1.57)
        }
    
    def process_navigation_command(self, command_text, entities):
        """
        Process a voice command for navigation
        """
        # Extract target location from entities
        target_location = None
        if 'locations' in entities:
            # Find the most appropriate location
            for loc in entities['locations']:
                if loc in self.location_map:
                    target_location = loc
                    break
        elif 'objects' in entities and entities['objects']:
            # Try to navigate to an object if detected
            target_location = self.find_object_location(entities['objects'][0])
        
        if target_location:
            target_pose = self.get_pose_for_location(target_location)
            if target_pose:
                rospy.loginfo(f"Navigating to {target_location}")
                self.nav_controller.navigate_to_pose(target_pose)
                return True
            else:
                rospy.logwarn(f"Unknown location: {target_location}")
                return False
        else:
            rospy.logwarn(f"Could not determine target location from command: {command_text}")
            return False
    
    def get_pose_for_location(self, location_name):
        """
        Get the pose for a named location
        """
        if location_name in self.location_map:
            x, y, theta = self.location_map[location_name]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            
            # Convert angle to quaternion
            from tf.transformations import quaternion_from_euler
            quat = quaternion_from_euler(0, 0, theta)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            return pose
        else:
            return None
    
    def find_object_location(self, object_name):
        """
        Find the location of a specific object if it's detected
        """
        # This would interface with perception system to locate objects
        # In a real implementation, this would query object detection results
        pass
    
    def update_known_locations(self, new_location_name, pose):
        """
        Update or add a new known location
        """
        self.location_map[new_location_name] = (
            pose.position.x, 
            pose.position.y, 
            self.quaternion_to_angle(pose.orientation)
        )
    
    def quaternion_to_angle(self, quat):
        """
        Convert quaternion to Euler angle (yaw)
        """
        import tf.transformations as tft
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return euler[2]  # Yaw component
```

## Safety Considerations

### Safe Navigation Practices

Implement safety mechanisms for humanoid navigation:

```python
class SafeNavigationSystem:
    def __init__(self, navigation_controller, perception_navigator):
        self.nav_controller = navigation_controller
        self.perception_navigator = perception_navigator
        self.safety_enabled = True
        self.emergency_stop_active = False
        
        # Safety parameters
        self.min_front_clearance = 0.5  # Minimum front clearance in meters
        self.max_speed = 0.5  # Maximum navigation speed in m/s
        self.stop_distance = 0.3  # Distance to stop from obstacles
        
        # Subscribe to safety-related topics
        self.safety_sub = rospy.Subscriber('/humanoid_robot/emergency_stop', 
                                          std_msgs.Bool, 
                                          self.emergency_stop_callback)
    
    def safe_navigate_to_pose(self, target_pose):
        """
        Navigate with safety checks and emergency handling
        """
        if not self.safety_check_before_navigation():
            rospy.logerr("Safety check failed before navigation")
            return False
        
        # Implement safe navigation logic
        result = self.execute_safe_navigation(target_pose)
        
        return result
    
    def safety_check_before_navigation(self):
        """
        Perform safety checks before starting navigation
        """
        # Check robot state (batteries, sensors, etc.)
        robot_status_ok = self.check_robot_status()
        
        # Check environment for safety issues
        environment_safe = self.check_environment_safety()
        
        return robot_status_ok and environment_safe
    
    def check_robot_status(self):
        """
        Check if robot systems are operational
        """
        # Check battery level
        battery_ok = self.check_battery_level()
        
        # Check sensor health
        sensors_ok = self.check_sensor_health()
        
        # Check actuator status
        actuators_ok = self.check_actuator_status()
        
        return battery_ok and sensors_ok and actuators_ok
    
    def check_environment_safety(self):
        """
        Check if environment is safe for navigation
        """
        # Use perception system to check for imminent dangers
        current_obstacles = self.perception_navigator.get_current_obstacles()
        
        # Check for unsafe obstacles in or near navigation path
        for obstacle in current_obstacles:
            if self.is_unsafe_obstacle(obstacle):
                rospy.logwarn(f"Unsafe obstacle detected: {obstacle}")
                return False
        
        return True
    
    def is_unsafe_obstacle(self, obstacle):
        """
        Determine if an obstacle poses a safety risk
        """
        if obstacle['min_distance'] < self.stop_distance:
            # Obstacle is too close
            return True
        
        if obstacle['label'] == 'person' and obstacle['min_distance'] < 1.0:
            # Too close to person
            return True
        
        return False
    
    def execute_safe_navigation(self, target_pose):
        """
        Execute navigation with continuous safety monitoring
        """
        # Start navigation
        self.nav_controller.navigate_to_pose(
            target_pose, 
            callback=self.safe_navigation_callback
        )
        
        # Start monitoring thread
        import threading
        monitoring_thread = threading.Thread(target=self.safety_monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        return True
    
    def safety_monitoring_loop(self):
        """
        Continuously monitor for safety during navigation
        """
        rate = rospy.Rate(10)  # Monitor at 10Hz
        
        while self.nav_controller.navigation_active and self.safety_enabled:
            if self.emergency_stop_active:
                # Emergency stop triggered
                self.perform_emergency_stop()
                break
            
            if self.safety_conditions_changed():
                # Reassess safety
                if not self.continue_with_navigation():
                    # Stop navigation if unsafe
                    self.nav_controller.nav_client.cancel_all_goals()
                    break
            
            rate.sleep()
    
    def safety_conditions_changed(self):
        """
        Check if safety conditions have changed during navigation
        """
        # Check for new obstacles in path
        current_obstacles = self.perception_navigator.get_current_obstacles()
        
        for obstacle in current_obstacles:
            if self.is_unsafe_obstacle(obstacle):
                return True
        
        return False
    
    def continue_with_navigation(self):
        """
        Determine if navigation should continue based on current conditions
        """
        safe = self.safety_check_before_navigation()
        if not safe:
            rospy.logwarn("Safety conditions compromised during navigation")
        
        return safe
    
    def perform_emergency_stop(self):
        """
        Perform emergency stop procedure
        """
        rospy.logerr("EMERGENCY STOP - Halting all robot motion")
        self.nav_controller.nav_client.cancel_all_goals()
        
        # Send stop command to base
        from geometry_msgs.msg import Twist
        cmd_vel_pub = rospy.Publisher('/humanoid_robot/cmd_vel', Twist, queue_size=1)
        stop_cmd = Twist()
        cmd_vel_pub.publish(stop_cmd)
    
    def emergency_stop_callback(self, msg):
        """
        Handle emergency stop message
        """
        self.emergency_stop_active = msg.data
```

## Performance Optimization

### Efficient Processing

Optimize perception and navigation for real-time performance:

```python
class EfficientPerceptionNavigator:
    def __init__(self):
        self.perception_rate = 5.0  # Hz
        self.navigation_rate = 10.0  # Hz
        self.max_processing_time = 0.1  # seconds per processing cycle
        
        # Create rate-limited processing
        self.perception_timer = rospy.Timer(
            rospy.Duration(1.0/self.perception_rate), 
            self.process_perception_limited
        )
        self.navigation_timer = rospy.Timer(
            rospy.Duration(1.0/self.navigation_rate),
            self.process_navigation_limited
        )
        
        # Process queues to prevent backlogs
        self.perception_queue = collections.deque(maxlen=3)  # Only newest 3 images
        self.navigation_queue = collections.deque(maxlen=5)  # Only newest 5 navigation updates
    
    def process_perception_limited(self, event):
        """
        Process perception data with rate limiting
        """
        start_time = time.time()
        
        # Process latest perception data
        if self.latest_image is not None:
            self.process_image_for_navigation(self.latest_image)
        
        if self.latest_depth is not None:
            self.process_depth_for_navigation(self.latest_depth)
        
        elapsed = time.time() - start_time
        if elapsed > self.max_processing_time:
            rospy.logwarn(f"Perception processing exceeded time budget: {elapsed:.3f}s > {self.max_processing_time:.3f}s")
    
    def process_navigation_limited(self, event):
        """
        Process navigation updates with rate limiting
        """
        start_time = time.time()
        
        # Process navigation updates
        self.update_navigation_path()
        
        elapsed = time.time() - start_time
        if elapsed > self.max_processing_time:
            rospy.logwarn(f"Navigation processing exceeded time budget: {elapsed:.3f}s > {self.max_processing_time:.3f}s")
    
    def update_navigation_path(self):
        """
        Update navigation path based on new perception data
        """
        # This would check if new obstacle information requires replanning
        pass

class AdaptiveResolutionSystem:
    def __init__(self, perception_navigator):
        self.perception_navigator = perception_navigator
        
        # Adaptive parameters
        self.base_resolution = 640  # Base resolution
        self.current_resolution = 640
        self.low_res_threshold = 10.0  # Distance for low resolution (meters)
        self.high_res_threshold = 2.0  # Distance for high resolution (meters)
        
        self.adaptation_enabled = True
    
    def adapt_perception_resolution(self, distance_to_target):
        """
        Adapt perception resolution based on distance to target
        """
        if not self.adaptation_enabled:
            return self.base_resolution
        
        if distance_to_target > self.low_res_threshold:
            # Far from target, use lower resolution for performance
            new_resolution = max(320, self.base_resolution // 2)
        elif distance_to_target > self.high_res_threshold:
            # Mid-range, use medium resolution
            new_resolution = self.base_resolution
        else:
            # Close to target, use high resolution for precision
            new_resolution = min(1280, self.base_resolution * 2)
        
        if new_resolution != self.current_resolution:
            self.current_resolution = new_resolution
            self.update_perception_settings(resolution=self.current_resolution)
            rospy.loginfo(f"Adjusted perception resolution to: {self.current_resolution}")
        
        return self.current_resolution
    
    def update_perception_settings(self, **kwargs):
        """
        Update perception system settings
        """
        # Update camera settings, processing parameters, etc.
        pass
```

## Testing and Validation

### Navigation System Testing

Validate the perception and navigation system performance:

```python
class NavigationSystemTester:
    def __init__(self, navigation_controller, perception_navigator):
        self.nav_controller = navigation_controller
        self.perception_navigator = perception_navigator
        self.test_results = []
        
    def run_navigation_tests(self):
        """
        Run comprehensive navigation tests
        """
        test_scenarios = [
            {
                "name": "straight_line_navigation",
                "start": (0, 0, 0),
                "goal": (5, 0, 0),
                "expected_time": 10.0,  # seconds
                "expected_path_length": 5.0
            },
            {
                "name": "obstacle_avoidance",
                "start": (0, 0, 0),
                "goal": (5, 5, 0),
                "expected_time": 15.0,
                "expected_path_length": 7.0,
                "obstacles": [(2, 1, 0.5)]  # x, y, radius
            },
            {
                "name": "narrow_passage",
                "start": (-3, 0, 0),
                "goal": (3, 0, 0),
                "expected_time": 20.0,
                "expected_path_length": 6.5,
                "obstacles": [
                    (-1, -1, 2),  # Wall on left
                    (-1, 1, 2),   # Wall on right
                    # Narrow passage centered at (0,0)
                ]
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            result = self.run_single_navigation_test(scenario)
            results.append(result)
        
        return results
    
    def run_single_navigation_test(self, scenario):
        """
        Run a single navigation test scenario
        """
        # Set up environment for test
        self.setup_test_environment(scenario)
        
        # Start navigation timer
        start_time = rospy.Time.now()
        
        # Create goal pose
        goal_pose = self.create_pose(scenario["goal"])
        
        # Execute navigation
        success = self.nav_controller.safe_navigate_to_pose(goal_pose)
        
        if success:
            # Wait for navigation to complete
            timeout = rospy.Duration(scenario["expected_time"] * 3)  # 3x expected time for safety
            timeout_time = rospy.Time.now() + timeout
            
            while (rospy.Time.now() < timeout_time and 
                   self.nav_controller.nav_client.get_state() == GoalStatus.ACTIVE):
                rospy.sleep(0.1)
            
            # Check final result
            final_state = self.nav_controller.nav_client.get_state()
            completion_time = (rospy.Time.now() - start_time).to_sec()
            
            # Determine success/failure
            navigation_success = final_state == GoalStatus.SUCCEEDED
            within_expected_time = completion_time <= scenario["expected_time"] * 2.0  # 2x tolerance
            
            result = {
                "scenario_name": scenario["name"],
                "success": navigation_success,
                "completion_time": completion_time,
                "expected_time": scenario["expected_time"],
                "path_length": self.get_actual_path_length(),  # Implementation needed
                "expected_path_length": scenario["expected_path_length"],
                "within_time_limit": within_expected_time
            }
        else:
            result = {
                "scenario_name": scenario["name"],
                "success": False,
                "completion_time": None,
                "expected_time": scenario["expected_time"],
                "path_length": 0,
                "expected_path_length": scenario["expected_path_length"],
                "within_time_limit": False,
                "error": "Failed to initiate navigation"
            }
        
        # Clean up test environment
        self.cleanup_test_environment(scenario)
        
        # Store result
        self.test_results.append(result)
        
        return result
    
    def setup_test_environment(self, scenario):
        """
        Set up the test environment (e.g., spawn obstacles in simulation)
        """
        # In Gazebo simulation, this would spawn test objects
        # This is a placeholder for the actual implementation
        pass
    
    def cleanup_test_environment(self, scenario):
        """
        Clean up the test environment
        """
        # In Gazebo simulation, this would remove test objects
        # This is a placeholder for the actual implementation
        pass
    
    def create_pose(self, pose_tuple):
        """
        Create a Pose object from a (x, y, theta) tuple
        """
        x, y, theta = pose_tuple
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        
        # Convert angle to quaternion
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def get_actual_path_length(self):
        """
        Calculate actual path length from odometry data
        This would require tracking the robot's path during navigation
        """
        # Implementation would track path using odometry or tf
        return 0.0  # Placeholder
    
    def generate_test_report(self):
        """
        Generate a comprehensive test report
        """
        if not self.test_results:
            return "No tests have been run yet."
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r["success"])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        report = f"""
        Navigation System Test Report
        ===========================
        
        Overall Results:
        - Total Tests: {total_tests}
        - Successful: {successful_tests}
        - Failed: {total_tests - successful_tests}
        - Success Rate: {success_rate:.1%}
        
        Detailed Results:
        """
        
        for result in self.test_results:
            status = "PASS" if result["success"] else "FAIL"
            time_status = "✓" if result.get("within_time_limit", False) else "✗"
            report += f"  {result['scenario_name']}: {status} ({result.get('completion_time', 'N/A')}s) [{time_status}]\n"
        
        return report
```

## Summary

This chapter covered the integration of perception and navigation systems in Vision-Language-Action architectures for autonomous humanoid robots. We explored:

1. **Perception for Navigation**: How visual and depth information enables safe navigation
2. **Object Detection**: Identifying navigation-relevant objects and obstacles
3. **Path Planning**: Algorithms optimized for humanoid robot characteristics  
4. **Navigation Execution**: Safely executing planned paths while monitoring the environment
5. **VLA Integration**: Connecting navigation to voice commands for autonomous movement
6. **Safety Considerations**: Mechanisms to ensure safe navigation in dynamic environments
7. **Performance Optimization**: Techniques for efficient real-time processing
8. **Testing and Validation**: Methods for verifying navigation system correctness

The perception and navigation system forms a critical component of the VLA architecture, enabling humanoid robots to understand their environment and navigate safely to perform tasks requested through voice commands. Proper integration of these systems ensures that the robot can move effectively while perceiving and responding to its surroundings.

The next chapter will cover manipulation and control systems that allow humanoid robots to interact with objects in their environment.