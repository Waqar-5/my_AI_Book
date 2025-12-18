#!/usr/bin/env python3

# AI Bridge Node - Connecting AI/LLM agents to ROS controllers
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import math

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
        self.get_logger().info('AI Bridge Node initialized')
        
    def user_command_callback(self, msg):
        user_command = msg.data
        self.get_logger().info(f'Received command: {user_command}')
        
        # Process command with AI simulation
        robot_action = self.process_with_ai(user_command)
        
        # Execute robot action
        if robot_action == 'move_forward':
            self.send_move_command(0.5, 0.0)  # Move forward at 0.5 m/s
        elif robot_action == 'turn_left':
            self.send_move_command(0.0, 0.3)  # Turn left at 0.3 rad/s
        elif robot_action == 'turn_right':
            self.send_move_command(0.0, -0.3)  # Turn right at 0.3 rad/s
        else:
            self.send_move_command(0.0, 0.0)  # Stop
            
    def process_with_ai(self, user_command):
        # This is a simplified example - in practice, this would call your LLM
        # and interpret the user's natural language command
        command_lower = user_command.lower()
        if 'forward' in command_lower or 'go' in command_lower:
            return 'move_forward'
        elif 'left' in command_lower:
            return 'turn_left'
        elif 'right' in command_lower:
            return 'turn_right'
        else:
            return 'stop'
    
    def send_move_command(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist_msg)
        self.get_logger().info(f'Sent move command: linear={linear_vel}, angular={angular_vel}')

def main(args=None):
    rclpy.init(args=args)
    ai_bridge_node = AIBridgeNode()
    rclpy.spin(ai_bridge_node)
    ai_bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()