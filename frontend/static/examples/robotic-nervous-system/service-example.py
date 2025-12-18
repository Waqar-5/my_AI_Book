#!/usr/bin/env python3

# Service example for ROS 2
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
        self.is_emergency_stopped = False

    def emergency_stop_callback(self, request, response):
        if request.data:  # Emergency stop activated
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            self.is_emergency_stopped = True
            response.success = True
            response.message = 'Emergency stop activated'
        else:  # Emergency stop released
            self.get_logger().info('Emergency stop released')
            self.is_emergency_stopped = False
            response.success = True
            response.message = 'Emergency stop released'
        return response

def main(args=None):
    rclpy.init(args=args)
    emergency_stop_service = EmergencyStopService()
    rclpy.spin(emergency_stop_service)
    emergency_stop_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()