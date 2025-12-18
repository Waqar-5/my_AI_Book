---
sidebar_position: 2
---

# Python Agents and Robot Control with rclpy

## Introduction

The Python client library for ROS 2 (rclpy) enables developers to create ROS 2 nodes in Python. This is particularly valuable for AI/ML applications since Python is the dominant language in artificial intelligence and machine learning. This chapter explores how to write ROS 2 nodes in Python and connect AI/LLM-driven agents to ROS controllers.

## Understanding rclpy

**rclpy** is the Python library that provides the interface between Python applications and the ROS 2 middleware. It allows Python developers to create nodes, publish and subscribe to topics, provide and use services, and manage parameters.

The architecture allows us to leverage Python's rich ecosystem of AI libraries (TensorFlow, PyTorch, scikit-learn, etc.) while maintaining the distributed, modular nature of ROS 2 for robot control.

## Creating a Basic ROS 2 Node in Python

Let's start with the basic structure of a ROS 2 node using rclpy:

```python
import rclpy
from rclpy.node import Node

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.get_logger().info('Robot Control Node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This pattern is fundamental to all ROS 2 Python nodes:
1. Initialize the rclpy library
2. Create a node (inheriting from `rclpy.node.Node`)
3. Run the node using `rclpy.spin()`
4. Clean up resources when done

## Publishers and Subscribers

### Publishers

Publishers send messages to topics. In humanoid robots, publishers might send motor commands, sensor commands, or status updates.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        # Simulate some joint positions (e.g., for a humanoid robot)
        joint_positions = [np.sin(self.i * 0.1 + offset) for offset in [0, 0.5, 1.0, 1.5]]
        msg.data = joint_positions
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing joint commands: {msg.data}')
        self.i += 1
```

### Subscribers

Subscribers receive messages from topics. In humanoid robots, subscribers might receive sensor data, AI commands, or system status.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received joint states: {len(msg.position)} joints')
        # Process joint positions here
        for i, position in enumerate(msg.position):
            self.get_logger().info(f'Joint {i}: {position:.3f}')
```

## Services and Actions

### Services

Services provide synchronous request-response communication. In humanoid robots, services might be used for calibration, configuration, or emergency commands.

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool

class EmergencyStopService(Node):
    def __init__(self):
        super().__init__('emergency_stop_service')
        self.srv = self.create_service(
            SetBool, 
            'emergency_stop', 
            self.emergency_stop_callback)

    def emergency_stop_callback(self, request, response):
        if request.data:  # Emergency stop activated
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            # Implement actual emergency stop logic here
            response.success = True
            response.message = 'Emergency stop activated'
        else:  # Emergency stop released
            self.get_logger().info('Emergency stop released')
            response.success = True
            response.message = 'Emergency stop released'
        return response
```

## Connecting AI/LLM Agents to ROS Controllers

One of the most exciting applications is connecting AI/LLM agents to ROS controllers for high-level robot behavior. Here's a pattern for achieving this:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import openai  # Example for LLM integration

class AIBridgeNode(Node):
    def __init__(self):
        super().__init__('ai_bridge_node')
        
        # Publisher for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for user commands or observations
        self.command_subscriber = self.create_subscription(
            String, 
            '/user_commands', 
            self.user_command_callback, 
            10)
        
        # Initialize AI model (simplified example)
        self.ai_model = self.initialize_ai_model()
        
    def initialize_ai_model(self):
        # In practice, this would connect to your AI/LLM service
        self.get_logger().info('AI model initialized')
        return True
    
    def user_command_callback(self, msg):
        user_command = msg.data
        self.get_logger().info(f'Received command: {user_command}')
        
        # Process command with AI
        robot_action = self.process_with_ai(user_command)
        
        # Execute robot action
        if robot_action == 'move_forward':
            self.send_move_command(0.5, 0.0)  # Move forward at 0.5 m/s
        elif robot_action == 'turn_left':
            self.send_move_command(0.0, 0.3)  # Turn left at 0.3 rad/s
        # Add more actions as needed
    
    def process_with_ai(self, user_command):
        # This is a simplified example - in practice, this would call your LLM
        # and interpret the user's natural language command
        if 'forward' in user_command.lower():
            return 'move_forward'
        elif 'left' in user_command.lower():
            return 'turn_left'
        else:
            return 'stop'
    
    def send_move_command(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist_msg)
        self.get_logger().info(f'Sent move command: linear={linear_vel}, angular={angular_vel}')
```

## Complete Example: AI Agent for Navigation

Here's a more complete example that demonstrates connecting an AI agent to ROS navigation controllers:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math

class AINavigationNode(Node):
    def __init__(self):
        super().__init__('ai_navigation_node')
        
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.command_subscriber = self.create_subscription(
            String, '/navigation_commands', self.command_callback, 10)
        
        # State variables
        self.obstacle_distance = float('inf')
        self.target_direction = 0.0
        self.has_target = False
        
    def laser_callback(self, msg):
        # Process laser scan to detect obstacles
        if len(msg.ranges) > 0:
            # Get the front distance (simplified)
            front_index = len(msg.ranges) // 2
            self.obstacle_distance = msg.ranges[front_index]
    
    def command_callback(self, msg):
        # Process navigation command from AI agent
        command_parts = msg.data.split()
        if len(command_parts) >= 2:
            try:
                x, y = float(command_parts[0]), float(command_parts[1])
                self.target_direction = math.atan2(y, x)
                self.has_target = True
                self.get_logger().info(f'Set navigation target to ({x}, {y})')
            except ValueError:
                self.get_logger().error('Invalid navigation command format')
    
    def navigate(self):
        if not self.has_target:
            return
            
        msg = Twist()
        
        # Simple obstacle avoidance
        safe_distance = 0.5  # meters
        if self.obstacle_distance < safe_distance:
            # Stop or turn to avoid obstacle
            msg.linear.x = 0.0
            msg.angular.z = 0.5  # Turn right
        else:
            # Navigate toward target
            msg.linear.x = 0.3  # Move forward
            msg.angular.z = self.target_direction  # Turn toward target
            
        self.cmd_vel_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ai_nav_node = AINavigationNode()
    
    # Run navigation logic at 10Hz
    timer = ai_nav_node.create_timer(0.1, ai_nav_node.navigate)
    
    rclpy.spin(ai_nav_node)
    
    ai_nav_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Python Node Development

1. **Error Handling**: Always include try-catch blocks around critical operations
2. **Resource Management**: Properly close publishers/subscribers and clean up resources
3. **Logging**: Use logging to track node behavior for debugging
4. **Parameter Configuration**: Use ROS parameters for runtime configuration
5. **Testing**: Create unit tests for your nodes using `pytest`

## Summary

Python nodes with rclpy provide a powerful way to integrate AI and machine learning capabilities into humanoid robot systems. By connecting AI/LLM agents to ROS controllers, we can create sophisticated, natural-language-driven robot behaviors. The modular architecture of ROS 2 allows these AI components to work seamlessly with traditional robot controllers, sensors, and actuators.

## Learning Objectives

After completing this chapter, you should be able to:

1. Create a basic ROS 2 node in Python using rclpy
2. Implement publishers and subscribers in Python nodes
3. Create and use services for synchronous communication in robotics applications
4. Design a bridge between AI/LLM agents and ROS controllers
5. Understand the integration patterns for connecting external AI systems to ROS

## Success Criteria

- You can create and run basic ROS 2 Python nodes (from the module success criteria)
- You understand how AI agents interface with ROS 2 controllers
- You can implement a simple AI bridge node that processes commands and sends robot actions