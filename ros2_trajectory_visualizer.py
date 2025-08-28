#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time


class TrajectoryVisualizerNode(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer_node')

        # Subscribe to SLAM pose topic
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            'droid_slam/pose', 
            self.pose_callback, 
            10
        )

        # Store trajectory data
        self.trajectory_points = []
        self.trajectory_lock = threading.Lock()

        # Initialize 3D trajectory visualization
        self._init_3d_trajectory_plot()

        # Create a timer to periodically update the plot
        self.update_timer = self.create_timer(0.1, self.update_plot)  # 10Hz update frequency

        self.get_logger().info('Trajectory Visualizer Node started, subscribing to droid_slam/pose')

    def pose_callback(self, msg: PoseStamped):
        """Callback function for receiving pose messages"""
        # Extract x, y, z coordinates
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Add to trajectory points list
        with self.trajectory_lock:
            self.trajectory_points.append([x, y, z])

            # Save all trajectory points without limit

        self.get_logger().debug(f'Received pose: x={x:.3f}, y={y:.3f}, z={z:.3f}')

    def _init_3d_trajectory_plot(self):
        """Initialize 3D trajectory visualization (same as droid_visualizer.py)"""
        # Set matplotlib to non-blocking mode
        plt.ion()

        # Create 3D plot
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('DROID-SLAM Real-time Trajectory (3D) - ROS2 Subscriber')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Initialize trajectory line and point
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')

        # Set equal aspect ratio for axes
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.legend()
        
        plt.show(block=False)

    def update_plot(self):
        """Update 3D trajectory plot periodically (same as droid_visualizer.py)"""
        try:
            with self.trajectory_lock:
                trajectory_points_copy = self.trajectory_points.copy()

            # Update trajectory line
            if len(trajectory_points_copy) > 1:
                x_data = [p[0] for p in trajectory_points_copy]
                y_data = [p[1] for p in trajectory_points_copy]
                z_data = [p[2] for p in trajectory_points_copy]
                
                self.trajectory_line.set_data_3d(x_data, y_data, z_data)

                # Update current position point
                self.current_point.set_data_3d([x_data[-1]], [y_data[-1]], [z_data[-1]])

                # Dynamically adjust axis limits to keep the same scale (same as modified droid_visualizer.py)
                margin = 0.5  # Margin

                # Compute data range for each axis
                x_min, x_max = min(x_data) - margin, max(x_data) + margin
                y_min, y_max = min(y_data) - margin, max(y_data) + margin
                z_min, z_max = min(z_data) - margin, max(z_data) + margin

                # Find the maximum data range
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min
                max_range = max(x_range, y_range, z_range)

                # Compute center point for each axis
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                z_center = (z_min + z_max) / 2

                # Set all axes limits using the maximum range to keep the same scale
                half_range = max_range / 2
                self.ax.set_xlim(x_center - half_range, x_center + half_range)
                self.ax.set_ylim(y_center - half_range, y_center + half_range)
                self.ax.set_zlim(z_center - half_range, z_center + half_range)

            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f"Error updating 3D trajectory plot: {e}")

    def cleanup(self):
        """Clean up resources"""
        try:
            plt.close(self.fig)
            self.get_logger().info('Matplotlib figure closed')
        except Exception as e:
            self.get_logger().error(f'Error closing matplotlib figure: {e}')


def main():
    rclpy.init()

    # Create trajectory visualizer node
    visualizer_node = TrajectoryVisualizerNode()
    
    try:
        print("Starting Trajectory Visualizer Node...")
        print("Subscribing to topic: droid_slam/pose")
        print("Press Ctrl+C to stop...")

        # Run the node
        rclpy.spin(visualizer_node)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping visualizer...")
    
    finally:
        # Clean up resources
        visualizer_node.cleanup()
        visualizer_node.destroy_node()
        rclpy.shutdown()
        print("Trajectory Visualizer Node stopped")


if __name__ == '__main__':
    main()
