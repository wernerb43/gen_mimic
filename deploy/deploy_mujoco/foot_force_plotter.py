# Copyright (c) 2025 Lizhi Yang AMBER LAB

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64


class FootForcePlotter(Node):
    def __init__(self):
        super().__init__("foot_force_plotter")

        # Subscribers
        self.foot_forces_sub = self.create_subscription(Float32MultiArray, "foot_forces", self.foot_forces_callback, 10)
        self.time_sub = self.create_subscription(Float64, "sim_time", self.time_callback, 10)
        self.ankle_heights_sub = self.create_subscription(
            Float32MultiArray, "ankle_heights", self.ankle_heights_callback, 10
        )

        # Data storage
        self.max_points = 1000
        self.times = deque(maxlen=self.max_points)
        self.left_forces = {
            "x": deque(maxlen=self.max_points),
            "y": deque(maxlen=self.max_points),
            "z": deque(maxlen=self.max_points),
        }
        self.right_forces = {
            "x": deque(maxlen=self.max_points),
            "y": deque(maxlen=self.max_points),
            "z": deque(maxlen=self.max_points),
        }
        self.ankle_heights = {
            "left": deque(maxlen=self.max_points),
            "right": deque(maxlen=self.max_points),
        }

        self.current_time = 0.0

        # Setup matplotlib
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Robot Foot Z-Forces and Ankle Heights")

        # Configure subplots
        self.axes[0, 0].set_title("Left Foot Z-Force")
        self.axes[0, 0].set_ylabel("Z-Force (N)")
        self.axes[0, 0].grid(True)

        self.axes[0, 1].set_title("Right Foot Z-Force")
        self.axes[0, 1].set_ylabel("Z-Force (N)")
        self.axes[0, 1].grid(True)

        self.axes[1, 0].set_title("Z-Force Comparison")
        self.axes[1, 0].set_ylabel("Z-Force (N)")
        self.axes[1, 0].set_xlabel("Time (s)")
        self.axes[1, 0].legend(["Left Foot", "Right Foot"])
        self.axes[1, 0].grid(True)

        self.axes[1, 1].set_title("Ankle Heights")
        self.axes[1, 1].set_ylabel("Height (m)")
        self.axes[1, 1].set_xlabel("Time (s)")
        self.axes[1, 1].legend(["Left Ankle", "Right Ankle"])
        self.axes[1, 1].grid(True)

        # Create timer for plot updates (30 Hz)
        self.plot_timer = self.create_timer(0.033, self.update_plot)

    def foot_forces_callback(self, msg):
        """Callback for foot forces data"""
        if len(msg.data) >= 6:
            # Left foot forces (first 3 elements)
            self.left_forces["x"].append(msg.data[0])
            self.left_forces["y"].append(msg.data[1])
            self.left_forces["z"].append(msg.data[2])

            # Right foot forces (next 3 elements)
            self.right_forces["x"].append(msg.data[3])
            self.right_forces["y"].append(msg.data[4])
            self.right_forces["z"].append(msg.data[5])

            # Add current time
            self.times.append(self.current_time)

    def time_callback(self, msg):
        """Callback for simulation time"""
        self.current_time = msg.data

    def ankle_heights_callback(self, msg):
        """Callback for ankle heights data"""
        if len(msg.data) >= 2:
            self.ankle_heights["left"].append(msg.data[0])
            self.ankle_heights["right"].append(msg.data[1])

    def update_plot(self):
        """Update plots using ROS timer on main thread"""
        try:
            if len(self.times) > 1:
                # Ensure all arrays have the same length
                min_length = min(
                    len(self.times),
                    len(self.left_forces["z"]),
                    len(self.right_forces["z"]),
                    len(self.ankle_heights["left"]),
                    len(self.ankle_heights["right"]),
                )

                if min_length < 2:
                    return

                # Get synchronized arrays
                times_array = np.array(list(self.times)[-min_length:])
                left_z = np.array(list(self.left_forces["z"])[-min_length:])
                right_z = np.array(list(self.right_forces["z"])[-min_length:])
                left_height = np.array(list(self.ankle_heights["left"])[-min_length:])
                right_height = np.array(list(self.ankle_heights["right"])[-min_length:])

                # Filter data for 2-second rolling window for ankle heights
                current_time = times_array[-1]
                time_window = 2.0  # 2 seconds
                window_mask = times_array >= (current_time - time_window)

                times_window = times_array[window_mask]
                left_height_window = left_height[window_mask]
                right_height_window = right_height[window_mask]

                # Clear all subplots
                for ax in self.axes.flat:
                    ax.clear()

                # Left foot Z-force
                self.axes[0, 0].plot(times_array, left_z, "b-")
                self.axes[0, 0].set_title("Left Foot Z-Force")
                self.axes[0, 0].set_ylabel("Z-Force (N)")
                self.axes[0, 0].grid(True)

                # Right foot Z-force
                self.axes[0, 1].plot(times_array, right_z, "r-")
                self.axes[0, 1].set_title("Right Foot Z-Force")
                self.axes[0, 1].set_ylabel("Z-Force (N)")
                self.axes[0, 1].grid(True)

                # Z-force comparison
                self.axes[1, 0].plot(times_array, left_z, "b-", label="Left Foot")
                self.axes[1, 0].plot(times_array, right_z, "r-", label="Right Foot")
                self.axes[1, 0].set_title("Z-Force Comparison")
                self.axes[1, 0].set_ylabel("Z-Force (N)")
                self.axes[1, 0].set_xlabel("Time (s)")
                self.axes[1, 0].legend()
                self.axes[1, 0].grid(True)

                # Ankle heights with 2-second rolling window
                self.axes[1, 1].plot(times_window, left_height_window, "b-", label="Left Ankle")
                self.axes[1, 1].plot(times_window, right_height_window, "r-", label="Right Ankle")
                self.axes[1, 1].set_title("Ankle Heights (2s window)")
                self.axes[1, 1].set_ylabel("Height (m)")
                self.axes[1, 1].set_xlabel("Time (s)")
                self.axes[1, 1].set_xlim(current_time - time_window, current_time)
                self.axes[1, 1].legend()
                self.axes[1, 1].grid(True)

                plt.tight_layout()
                plt.pause(0.001)  # Minimal pause for plot update

        except Exception as e:
            self.get_logger().error(f"Error updating plot: {e}")


def main(args=None):
    rclpy.init(args=args)
    plotter = FootForcePlotter()

    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
