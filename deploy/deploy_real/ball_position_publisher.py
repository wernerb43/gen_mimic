#!/usr/bin/env python3
"""
Simple ROS2 publisher for ball position.
Publishes Float32MultiArray messages to /ball_position topic.

Usage:
    python ball_position_publisher.py [--frequency 30]

The script publishes ball position as [x, y, z] coordinates.
You can modify the hardcoded positions or add interactive input as needed.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time
import sys


class BallPositionPublisher(Node):
    def __init__(self):
        super().__init__('ball_position_publisher')
        
        # Get frequency parameter
        self.declare_parameter('frequency', 30)  # Hz
        frequency = self.get_parameter('frequency').value
        self.publish_period = 1.0 / frequency
        
        # Create publisher
        self.publisher = self.create_publisher(Float32MultiArray, '/ball_position', 10)
        
        # Initial ball position (x, y, z in meters)
        # Modify these values as needed
        self.ball_position = np.array([0.4, -0.7, 0.0], dtype=np.float32)
        
        self.get_logger().info(
            f"Ball Position Publisher started at {frequency} Hz\n"
            f"Initial position: {self.ball_position}\n"
            f"Publishing to /ball_position topic"
        )
        
        # Create a timer to publish at regular intervals
        self.timer = self.create_timer(self.publish_period, self.publish_callback)
    
    def publish_callback(self):
        """Publish the current ball position"""
        msg = Float32MultiArray()
        msg.data = self.ball_position.tolist()
        self.publisher.publish(msg)
        self.get_logger().debug(f"Published ball position: {self.ball_position}")
    
    def set_ball_position(self, x, y, z):
        """Update the ball position"""
        self.ball_position = np.array([x, y, z], dtype=np.float32)
        self.get_logger().info(f"Ball position updated to: {self.ball_position}")


def main():
    # Parse command line arguments
    frequency = 30
    if len(sys.argv) > 1:
        if sys.argv[1] == '--frequency' and len(sys.argv) > 2:
            try:
                frequency = int(sys.argv[2])
            except ValueError:
                print("Warning: Invalid frequency value, using default 30 Hz")
    
    rclpy.init()
    
    # Create and run the publisher node
    publisher_node = BallPositionPublisher()
    
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        publisher_node.get_logger().info("Shutting down ball position publisher...")
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
