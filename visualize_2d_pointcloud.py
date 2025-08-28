#!/usr/bin/env python3
"""
2D scatter point visualization
for 2D point cloud data from DROID-SLAM
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN

def read_ply_file(filepath):
    """Read PLY file and return point cloud data"""
    points = []
    colors = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header information
        header_end = 0
        vertex_count = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('end_header'):
                header_end = i + 1
                break

        # Read vertex data
        for i in range(header_end, header_end + vertex_count):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 6:  # x, y, z, r, g, b
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    
                    points.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
        
        return np.array(points), np.array(colors)
        
    except Exception as e:
        print(f"Failed to read PLY file: {e}")
        return None, None

def plot_2d_pointcloud(points, colors, title="2D Point Cloud", save_path=None, 
                       filter_outliers=True, eps=0.1, min_samples=5):
    """Draw 2D scatter plot, optionally using DBSCAN to filter outliers
    
    Args:
        points: Point cloud data (N, 3)
        colors: Color data (N, 3)
        title: Plot title
        save_path: Save path
        filter_outliers: Whether to use DBSCAN to filter outliers
        eps: DBSCAN's neighborhood radius parameter
        min_samples: DBSCAN's minimum samples parameter
    """
    if points is None or len(points) == 0:
        print("No point cloud data to display")
        return
    
    original_point_count = len(points)
    filtered_points = points.copy()
    filtered_colors = colors.copy()

    # Use DBSCAN to filter outliers
    if filter_outliers and len(points) > min_samples:
        print(f"Using DBSCAN to filter outliers (eps={eps}, min_samples={min_samples})...")

        # Only cluster on xy coordinates
        xy_points = points[:, :2]

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(xy_points)

        # Filter out outliers (label -1)
        inlier_mask = cluster_labels != -1
        filtered_points = points[inlier_mask]
        filtered_colors = colors[inlier_mask]
        
        outlier_count = original_point_count - len(filtered_points)
        print(f"Original point count: {original_point_count}, Filtered point count: {len(filtered_points)}, Outlier count: {outlier_count}")

    # Create figure
    plt.figure(figsize=(12, 10))

    # Draw 2D scatter plot (only use x,y coordinates)
    scatter = plt.scatter(filtered_points[:, 0], filtered_points[:, 1], 
                         c=filtered_colors, s=1, alpha=0.6, edgecolors='none')
    
    plt.title(title, fontsize=14)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Keep aspect ratio of x and y axes equal

    # Add statistics information
    x_range = np.max(filtered_points[:, 0]) - np.min(filtered_points[:, 0])
    y_range = np.max(filtered_points[:, 1]) - np.min(filtered_points[:, 1])
    info_text = f"Points: {len(filtered_points)}"
    if filter_outliers:
        info_text += f" (Original: {original_point_count})"
    info_text += f"\nX Range: {x_range:.2f}m\nY Range: {y_range:.2f}m"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D scatter plot saved to: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize DROID-SLAM 2D point cloud data')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to the PLY file to visualize')

    args = parser.parse_args()

    # Visualize the specified file
    print(f"Visualizing specified file: {args.file}")
    points, colors = read_ply_file(args.file)
    
    if points is not None:
        # If color data is empty, create default colors
        if colors is None or len(colors) == 0:
            colors = np.zeros((len(points), 3))
            
        filename = os.path.basename(args.file)
        plot_2d_pointcloud(points, colors, 
                          title=f"2D Point Cloud - {filename}",
                          save_path=f"2d_pointcloud_{filename.replace('.ply', '.png')}",
                          filter_outliers=False,
                          eps=0.1,
                          min_samples=5)
    else:
        print("Failed to read point cloud data")

if __name__ == "__main__":
    main()
