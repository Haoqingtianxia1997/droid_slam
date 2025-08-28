#!/usr/bin/env python3
"""
Use Open3D to visualize PLY point cloud files and display the world coordinate system
for comparing DROID-SLAM reconstructed point clouds with the simulation environment.
"""

import open3d as o3d
import numpy as np
import os
import glob

def load_and_visualize_ply(ply_file_path):
    """Load PLY file and visualize with Open3D"""

    # Check if file exists
    if not os.path.exists(ply_file_path):
        print(f"File does not exist: {ply_file_path}")
        return

    print(f"Loading point cloud file: {ply_file_path}")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    if len(pcd.points) == 0:
        print("Point cloud file is empty or not in the correct format")
        return

    print(f"Successfully loaded point cloud with {len(pcd.points)} points")

    # Create world coordinate system (axes)
    # Coordinate frame size can be adjusted based on point cloud extent
    coord_frame_size = 1.0  # Coordinate frame length (meters)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_frame_size, 
        origin=[0, 0, 0]
    )

    # Print point cloud statistics
    print(f"Point cloud bounding box:")
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"  Min: {bbox.min_bound}")
    print(f"  Max: {bbox.max_bound}")
    print(f"  Center: {bbox.get_center()}")

    # Compute point cloud center for view adjustment
    center = pcd.get_center()
    print(f"Point cloud center: {center}")

    # Optional: Perform some preprocessing on the point cloud
    # Remove outliers
    # pcd, outlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # print(f"After removing outliers, {len(pcd.points)} points remain")

    # Filter point cloud based on Z-axis height (exclude points above 1.0 and below -0.95)
    points = np.asarray(pcd.points)
    # z_filter = (points[:, 2] >= -0.95) & (points[:, 2] <= 1.0)

    # # If point cloud has color information, it also needs to be filtered
    # if len(pcd.colors) > 0:
    #     colors = np.asarray(pcd.colors)
    #     filtered_colors = colors[z_filter]
    #     pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # # If point cloud has normals, it also needs to be filtered
    # if len(pcd.normals) > 0:
    #     normals = np.asarray(pcd.normals)
    #     filtered_normals = normals[z_filter]
    #     pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    # # Apply Z-axis filtering
    # filtered_points = points[z_filter]
    filtered_points = points
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    print(f"After filtering Z-axis height (-0.95 <= z <= 1.0), {len(pcd.points)} points remain")

    # Estimate normals (for better visualization)
    pcd.estimate_normals()

    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="DROID-SLAM Point Cloud Visualization", width=1200, height=800)

    # Add point cloud and coordinate frame to the scene
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # Increase point size

    vis.add_geometry(world_frame)

    # Set render options
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background

    # Set view control
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, 1.0])   # Camera direction
    view_control.set_lookat(center)            # Look at point cloud center
    view_control.set_up([0.0, 1.0, 0.0])      # Up direction
    view_control.set_zoom(0.5)

    print("\nVisualization control instructions:")
    print("- Left mouse drag: Rotate view")
    print("- Right mouse drag: Pan view")
    print("- Mouse wheel: Zoom")
    print("- Press 'R' key: Reset view")
    print("- Press 'ESC' or close window: Exit")
    print("\nCoordinate system description:")
    print("- Red axis: X-axis")
    print("- Green axis: Y-axis")
    print("- Blue axis: Z-axis")

    # Run visualization
    vis.run()
    vis.destroy_window()

def find_latest_ply_file(search_dir="."):
    """Find the latest PLY file"""
    ply_files = glob.glob(os.path.join(search_dir, "accumulated_pointcloud_*.ply"))
    if not ply_files:
        ply_files = glob.glob(os.path.join(search_dir, "*.ply"))
    
    if not ply_files:
        return None

    # Sort by modification time and return the latest file
    latest_file = max(ply_files, key=os.path.getmtime)
    return latest_file

def main():
    import sys
    
    if len(sys.argv) > 1:
        # If a file path argument is provided
        ply_file = sys.argv[1]
    else:
        # Automatically find the latest PLY file
        ply_file = find_latest_ply_file()
        if ply_file is None:
            print("No PLY file found. Please ensure:")
            print("1. The DROID-SLAM visualizer is running and has saved point cloud files")
            print("2. Or manually specify the PLY file path: python visualize_ply_with_open3d.py <ply_file_path>")
            return
        else:
            print(f"Automatically found the latest PLY file: {ply_file}")

    load_and_visualize_ply(ply_file)

if __name__ == "__main__":
    main()
