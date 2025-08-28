import moderngl
import numpy as np
from lietorch import SE3

import torch
import droid_backends
import moderngl_window
import moderngl
from moderngl_window.opengl.vao import VAO

import numpy as np
from .camera import OrbitDragCameraWindow
from align import align_pose_fragements

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import threading
import warnings

# Filter matplotlib warnings
warnings.filterwarnings("ignore", message=".*fixed x limits to fulfill fixed data aspect.*")

# Add sklearn for DBSCAN filtering
from sklearn.cluster import DBSCAN

# Add morphological operations
from scipy.ndimage import binary_erosion, binary_dilation

# Add path planning related
import heapq
from typing import List, Tuple, Optional

# Add UDP communication related modules
import socket
import json
import time

CAM_POINTS = 0.05 * np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
).astype("f4")

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)

CAM_SEGMENTS = []
for i, j in CAM_LINES:
    CAM_SEGMENTS.append(CAM_POINTS[i])
    CAM_SEGMENTS.append(CAM_POINTS[j])

CAM_SEGMENTS = np.stack(CAM_SEGMENTS, axis=0)


def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion (w,x,y,z) to rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Construct rotation matrix
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    return R


def quaternion_multiply(q1, q2):
    """Quaternion multiplication q1 * q2, quaternion format is [w,x,y,z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def merge_depths_and_poses(depth_video1, depth_video2):
    t1 = depth_video1.counter.value
    t2 = depth_video2.counter.value

    poses1 = depth_video1.poses[:max(t1, t2)].clone()
    poses2 = depth_video2.poses[:max(t1, t2)].clone()

    disps1 = depth_video1.disps[:max(t1, t2)].clone()
    disps2 = depth_video2.disps[:max(t1, t2)].clone()

    if t2 <= 0:
        return poses1, disps1
    
    if t2 >= t1:
        return poses2, disps2
    
    dP, s = align_pose_fragements(
        poses1[max(0, t2-16): t2],
        poses2[max(0, t2-16): t2],
    )

    poses1[..., :3] *= s

    poses2[t2:] = (dP * SE3(poses1[t2:])).data
    disps2[t2:] = disps1[t2:] / s

    return poses2, disps2


class DroidVisualizer(OrbitDragCameraWindow):
    def __init_point_cloud_accumulation(self):
        """Initialize point cloud accumulation related variables"""
        self.last_save_time = time.time()
        self.save_interval = 15.0  # Save every 15 seconds
        self.save_counter = 0

        # Store 2D point cloud data for the current frame for real-time visualization
        self.current_frame_2d_points = None
        self.current_frame_2d_colors = None

        # Occupancy Grid related variables (dynamic size)
        self.occupancy_grid = None  # Will be created dynamically based on point cloud range
        self.grid_origin = None     # Grid origin (dynamically calculated)
        self.grid_size = None       # Grid size (dynamically calculated)

        # Frontier exploration related variables
        self.explored_grid = None   # Explored area grid
        self.current_exploration_sector = None  # Current frame's exploration sector

        # Path planning related variables
        self.latest_robot_position = None  # Latest robot position

    def accumulate_point_cloud(self):
        """Accumulate point cloud data for the current frame"""
        try:
            # Get point cloud and color data from GPU buffer
            points = self.pts_buffer.read()
            colors = self.clr_buffer.read()
            valid = self.valid_buffer.read()

            # Convert to numpy arrays
            num_points = len(points) // (3 * 4)  # float32
            points_np = np.frombuffer(points, dtype=np.float32).reshape(num_points, 3)
            colors_np = np.frombuffer(colors, dtype=np.float32).reshape(num_points, 3)
            valid_np = np.frombuffer(valid, dtype=np.float32)

            # Keep only valid points
            mask = valid_np > 0
            valid_points = points_np[mask]
            valid_colors = colors_np[mask]
            
            if len(valid_points) > 0:
                # Apply camera-to-world rotation transformation
                if DroidVisualizer._first_angle_data is not None:
                    torso_quat_data = DroidVisualizer._first_angle_data.get('torso_to_world_quat', {})
                    torso_to_world_quat = np.array([
                        torso_quat_data.get('w', 1.0),
                        torso_quat_data.get('x', 0.0),
                        torso_quat_data.get('y', 0.0),
                        torso_quat_data.get('z', 0.0)
                    ])
                    # torso_to_world_quat = np.array([1.0, 0.0, 0.0, 0.0])

                    # Camera to torso transformation (based on the provided quaternion)
                    camera_to_torso_quat = np.array([0.45451948, -0.54167522, 0.54167522, -0.45451948])

                    # Compute camera to world coordinate quaternion: camera_to_world = torso_to_world * camera_to_torso
                    camera_to_world_quat = quaternion_multiply(torso_to_world_quat, camera_to_torso_quat)

                    # Convert quaternion to rotation matrix
                    R_camera_to_world = quaternion_to_rotation_matrix(
                        camera_to_world_quat[0], camera_to_world_quat[1], 
                        camera_to_world_quat[2], camera_to_world_quat[3]
                    )

                    # Apply rotation transformation to point cloud
                    valid_points = (R_camera_to_world @ valid_points.T).T

                # 1. Filter point cloud by z-axis height (keep points between -0.8 and 0.5)
                z_mask = (valid_points[:, 2] >= -0.8) & (valid_points[:, 2] <= 0.5)
                filtered_points = valid_points[z_mask]
                filtered_colors = valid_colors[z_mask]
                
                if len(filtered_points) > 0:
                    # 2. Project to xy plane (eliminate z-axis)
                    xy_points = filtered_points[:, :2]  # Keep only x,y coordinates

                    # 3. Grid downsampling
                    grid_size = 0.05  # 5cm grid size, adjustable as needed

                    # Compute grid indices
                    grid_indices = np.floor(xy_points / grid_size).astype(int)

                    # Use a dictionary to store points for each grid cell
                    grid_dict = {}
                    for i, (grid_x, grid_y) in enumerate(grid_indices):
                        grid_key = (grid_x, grid_y)
                        if grid_key not in grid_dict:
                            grid_dict[grid_key] = []
                        grid_dict[grid_key].append(i)

                    # Compute representative points for each grid cell
                    downsampled_points = []
                    downsampled_colors = []
                    
                    for grid_key, point_indices in grid_dict.items():
                        # Compute grid center coordinates (only xy coordinates, z is set to 0)
                        grid_center_x = (grid_key[0] + 0.5) * grid_size
                        grid_center_y = (grid_key[1] + 0.5) * grid_size

                        # Compute average color for all points in this grid cell
                        grid_colors = filtered_colors[point_indices]
                        avg_color = np.mean(grid_colors, axis=0)

                        # Create representative point (use grid center's xy coordinates, z is set to 0)
                        representative_point = np.array([grid_center_x, grid_center_y, 0.0])
                        
                        downsampled_points.append(representative_point)
                        downsampled_colors.append(avg_color)
                    
                    if len(downsampled_points) > 0:
                        downsampled_points = np.array(downsampled_points)
                        downsampled_colors = np.array(downsampled_colors)

                        # Store current frame's downsampled point cloud for 2D real-time visualization
                        self.current_frame_2d_points = downsampled_points
                        self.current_frame_2d_colors = downsampled_colors

                        # print(f"Downsampling: Original point count {len(filtered_points)} -> Downsampled {len(downsampled_points)}")

        except Exception as e:
            print(f"Accumulating point cloud failed: {e}")

    def save_accumulated_point_cloud(self, filename_prefix="accumulated_pointcloud"):
        """Save accumulated point cloud as PLY file"""
        try:
            if len(self.current_frame_2d_points) == 0:
                print("No point cloud data to save. Please accumulate point cloud first.")
                return

            # Merge all accumulated point clouds (coordinate transformations have already been applied during accumulation)
            all_points = self.current_frame_2d_points
            all_colors = self.current_frame_2d_colors

            # Remove duplicates: use point coordinates as unique identifiers
            # Round coordinates to grid precision to ensure consistency
            grid_size = 0.05  # Keep consistent with grid_size in accumulate_point_cloud
            rounded_points = np.round(all_points / grid_size) * grid_size

            # Use a simpler and safer deduplication method: based on string representation
            # Convert each point to a string as a unique key
            point_strings = [f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}" for p in rounded_points]

            # Create deduplication dictionary
            unique_dict = {}
            for i, point_str in enumerate(point_strings):
                if point_str not in unique_dict:
                    unique_dict[point_str] = i

            # Get unique point indices
            unique_indices = list(unique_dict.values())

            # Get deduplicated points and colors
            unique_points = all_points[unique_indices]
            unique_colors = all_colors[unique_indices]

            # Convert colors to 0-255
            unique_colors = (unique_colors * 255).astype(np.uint8)

            print(f"point counts before {len(all_points)}, after deduplication: {len(unique_points)}")

            filename = f"{filename_prefix}.ply"

            # Write PLY file
            with open(filename, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {unique_points.shape[0]}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(unique_points, unique_colors):
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
            
            print(f"Cumulated points saved to {filename}, All points: {unique_points.shape[0]}")
            self.save_counter += 1
            
        except Exception as e:
            print(f"Failed to save accumulated point cloud: {e}")

    def create_occupancy_grid_from_points(self, points, robot_position=None):
        """Create dynamic occupancy grid from point cloud data, including frontier exploration functionality

        Grid value definitions:
        - 0: unexplored area (frontier) - black
        - 25: explored free area - white
        - 50: safety redundancy area - yellow
        - 75: current exploration sector - green
        - 100: occupied area - red
        """
        if points is None or len(points) == 0:
            return

        # Save robot position for path planning
        if robot_position is not None:
            self.latest_robot_position = robot_position

        # Use static grid range or dynamically computed range
        if self._use_static_grid_range:
            # Use predefined static range
            x_min = self._static_grid_x_min
            x_max = self._static_grid_x_max
            y_min = self._static_grid_y_min
            y_max = self._static_grid_y_max
        else:
            # Dynamically compute point cloud boundaries
            margin = 1.0  # Expand boundary by 1 meter
            x_min = np.min(points[:, 0]) - margin
            x_max = np.max(points[:, 0]) + margin
            y_min = np.min(points[:, 1]) - margin
            y_max = np.max(points[:, 1]) + margin

        # Check if reinitialization is needed
        if self._use_static_grid_range:
            # Static grid mode: initialize only on first creation
            need_reinit = (
                self.grid_origin is None or 
                self.grid_size is None
            )
        else:
            # Dynamic grid mode: reinitialize when range expands
            need_reinit = (
                self.grid_origin is None or 
                self.grid_size is None or
                x_min < self.grid_origin[0] or
                y_min < self.grid_origin[1] or
                x_max > self.grid_origin[0] + self.grid_size[1] * self._occupancy_grid_resolution or
                y_max > self.grid_origin[1] + self.grid_size[0] * self._occupancy_grid_resolution
            )

        # Save old exploration state (before grid reinitialization)
        old_explored_grid = None
        old_grid_origin = None
        old_grid_size = None
        
        if need_reinit:
            # Save old state for migration
            if self.explored_grid is not None:
                old_explored_grid = self.explored_grid.copy()
                old_grid_origin = self.grid_origin.copy()
                old_grid_size = self.grid_size

            # Update grid origin and size
            self.grid_origin = np.array([x_min, y_min])

            # Compute grid size (round up)
            grid_width = int(np.ceil((x_max - x_min) / self._occupancy_grid_resolution))
            grid_height = int(np.ceil((y_max - y_min) / self._occupancy_grid_resolution))
            self.grid_size = (grid_height, grid_width)  # (rows, cols)

            # Create new explored area grid
            self.explored_grid = np.zeros(self.grid_size, dtype=np.uint8)

            # Migrate old exploration state to new grid
            if old_explored_grid is not None and old_grid_origin is not None:
                self._migrate_exploration_state(old_explored_grid, old_grid_origin, old_grid_size)

        # Create temporary obstacle grid (only contains current frame's obstacle information)
        temp_obstacle_grid = np.zeros(self.grid_size, dtype=np.uint8)

        # Reset current exploration sector markers
        self.current_exploration_sector = np.zeros(self.grid_size, dtype=np.uint8)

        # Compute number of grid pixels corresponding to the occupancy radius for each point
        radius_pixels = int(np.ceil(self._occupancy_point_radius / self._occupancy_grid_resolution))

        # Mark occupied area around each point (stored in temporary grid)
        for point in points:
            # Convert world coordinates to grid coordinates
            grid_x = int((point[0] - self.grid_origin[0]) / self._occupancy_grid_resolution)
            grid_y = int((point[1] - self.grid_origin[1]) / self._occupancy_grid_resolution)

            # Check if point is within grid bounds
            if (grid_x < 0 or grid_x >= self.grid_size[1] or
                grid_y < 0 or grid_y >= self.grid_size[0]):
                continue  # Skip out-of-bounds points

            # Mark occupied area around the point
            for dy in range(-radius_pixels, radius_pixels + 1):
                for dx in range(-radius_pixels, radius_pixels + 1):
                    # Compute distance from point center
                    distance = np.sqrt(dx*dx + dy*dy) * self._occupancy_grid_resolution
                    
                    # If within occupancy radius
                    if distance <= self._occupancy_point_radius:
                        new_grid_x = grid_x + dx
                        new_grid_y = grid_y + dy

                        # Check if within grid bounds
                        if (0 <= new_grid_x < self.grid_size[1] and
                            0 <= new_grid_y < self.grid_size[0]):
                            # Mark as occupied (100 indicates fully occupied)
                            temp_obstacle_grid[new_grid_y, new_grid_x] = 100

        # Apply morphological operations to temporary obstacle grid
        if self._enable_morphology and np.any(temp_obstacle_grid > 50):
            try:
                # Convert occupancy grid to binary image (0 or 1)
                binary_grid = (temp_obstacle_grid > 50).astype(bool)

                # Create structuring element (morphological operation kernel)
                kernel_size = self._morphology_kernel_size
                kernel = np.ones((kernel_size, kernel_size), dtype=bool)

                # Erode first (remove small noise points and thin connections)
                if self._erosion_iterations > 0:
                    for _ in range(self._erosion_iterations):
                        binary_grid = binary_erosion(binary_grid, structure=kernel)

                # Then dilate (restore size and smooth edges)
                if self._dilation_iterations > 0:
                    for _ in range(self._dilation_iterations):
                        binary_grid = binary_dilation(binary_grid, structure=kernel)

                # Convert binary result back to temporary obstacle grid
                temp_obstacle_grid = (binary_grid * 100).astype(np.uint8)
                
            except Exception as e:
                print(f"Warning: Morphological operations failed: {e}")

        # Add safety margin to temporary obstacle grid
        temp_safety_grid = np.zeros(self.grid_size, dtype=np.uint8)
        if self._enable_safety_margin and np.any(temp_obstacle_grid > 50):
            try:
                # Get current obstacle area
                obstacle_mask = (temp_obstacle_grid > 50)

                # Compute pixel count corresponding to safety margin distance
                safety_margin_pixels = int(np.ceil(self._safety_margin_distance / self._occupancy_grid_resolution))

                # Create circular structuring element for dilation
                y_coords, x_coords = np.ogrid[:2*safety_margin_pixels+1, :2*safety_margin_pixels+1]
                center = safety_margin_pixels
                circular_mask = ((x_coords - center)**2 + (y_coords - center)**2) <= safety_margin_pixels**2

                # Dilate obstacle area to generate safety margin
                safety_margin_mask = binary_dilation(obstacle_mask, structure=circular_mask)

                # Create safety margin grid (excluding obstacle area itself)
                safety_only_mask = safety_margin_mask & (~obstacle_mask)
                temp_safety_grid[safety_only_mask] = 50  # Safety margin area
                
            except Exception as e:
                print(f"Warning: Safety margin generation failed: {e}")

        # Apply frontier exploration functionality (after obstacle and safety areas are determined)
        if self._enable_frontier_exploration and robot_position is not None:
            self._update_frontier_exploration(robot_position, temp_obstacle_grid, temp_safety_grid)

        # Finally merge all layers into the final occupancy grid
        self._merge_exploration_layers(temp_obstacle_grid, temp_safety_grid)
    
    def _migrate_exploration_state(self, old_explored_grid, old_grid_origin, old_grid_size):
        """Migrate exploration state from old grid to new grid"""
        try:
            # Compute coordinate mapping between new grid and old grid
            for old_y in range(old_grid_size[0]):
                for old_x in range(old_grid_size[1]):
                    if old_explored_grid[old_y, old_x] == 1:  # If the position is explored in the old grid
                        # Convert old grid coordinates to world coordinates
                        world_x = old_grid_origin[0] + old_x * self._occupancy_grid_resolution
                        world_y = old_grid_origin[1] + old_y * self._occupancy_grid_resolution

                        # Convert world coordinates to new grid coordinates
                        new_x = int((world_x - self.grid_origin[0]) / self._occupancy_grid_resolution)
                        new_y = int((world_y - self.grid_origin[1]) / self._occupancy_grid_resolution)

                        # Check if new coordinates are within the new grid bounds
                        if (0 <= new_x < self.grid_size[1] and
                            0 <= new_y < self.grid_size[0]):
                            self.explored_grid[new_y, new_x] = 1
                            
        except Exception as e:
            print(f"Warning: Failed to migrate exploration state: {e}")
    
    def _update_frontier_exploration(self, robot_position, obstacle_grid, safety_grid):
        """Update frontier exploration state based on robot position and grid information.

        Args:
            robot_position: Robot pose in world coordinates [x, y, z]
            obstacle_grid: Obstacle grid
            safety_grid: Safety margin grid
        """
        try:
            if self.grid_origin is None or self.grid_size is None:
                return

            # Robot position in grid
            robot_grid_x = int((robot_position[0] - self.grid_origin[0]) / self._occupancy_grid_resolution)
            robot_grid_y = int((robot_position[1] - self.grid_origin[1]) / self._occupancy_grid_resolution)

            # Check if robot position is within grid bounds
            if (robot_grid_x < 0 or robot_grid_x >= self.grid_size[1] or
                robot_grid_y < 0 or robot_grid_y >= self.grid_size[0]):
                return

            # Get robot heading
            robot_heading = 0.0  # Default heading (radians), 0 means facing positive x direction

            # Try to get robot heading from angle data
            if DroidVisualizer._latest_angle_data is not None:
                # Compute yaw angle from torso quaternion
                torso_quat_data = DroidVisualizer._latest_angle_data.get('torso_to_world_quat', {})
                if torso_quat_data:
                    w = torso_quat_data.get('w', 1.0)
                    x = torso_quat_data.get('x', 0.0)
                    y = torso_quat_data.get('y', 0.0)
                    z = torso_quat_data.get('z', 0.0)

                    # Compute yaw angle from quaternion(suppose standard ZYX Euler angles)
                    # yaw = arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                    robot_heading = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

            # Compute exploration sector angles
            sector_angle_rad = np.deg2rad(self._exploration_sector_angle)
            start_angle = robot_heading - sector_angle_rad / 2
            end_angle = robot_heading + sector_angle_rad / 2

            # Compute the number of pixels corresponding to the exploration radius
            radius_pixels = int(self._exploration_sector_radius / self._occupancy_grid_resolution)

            # Generate rays
            ray_angles = np.arange(start_angle, end_angle, np.deg2rad(self._exploration_ray_resolution))
            
            for angle in ray_angles:
                # Compute ray direction
                dx = np.cos(angle)
                dy = np.sin(angle)

                # Step along the ray direction
                for step in range(1, radius_pixels + 1):
                    # Compute the grid coordinates of the current point
                    current_x = robot_grid_x + int(step * dx)
                    current_y = robot_grid_y + int(step * dy)

                    # Check if out of grid bounds
                    if (current_x < 0 or current_x >= self.grid_size[1] or
                        current_y < 0 or current_y >= self.grid_size[0]):
                        break

                    # Check if encountered obstacle or safety margin
                    if (obstacle_grid[current_y, current_x] >= 100 or 
                        safety_grid[current_y, current_x] >= 50):  # Obstacle or safety margin
                        break

                    # Mark current exploration area
                    self.current_exploration_sector[current_y, current_x] = 1

                    # Mark this point as explored
                    self.explored_grid[current_y, current_x] = 1
            
        except Exception as e:
            print(f"Warning: Frontier exploration update failed: {e}")
    
    def _merge_exploration_layers(self, obstacle_grid, safety_grid):
        """merge all exploration layers into the final occupancy grid

        Args:
            obstacle_grid: Obstacle grid
            safety_grid: Safety margin grid

        Grid value definitions:
        - 0: unexplored (frontier) - black
        - 25: explored free - white
        - 50: safety margin - yellow
        - 75: current frame exploration sector - green
        - 100: occupied (obstacle) - red
        """
        try:
            # Get various masks
            obstacle_mask = (obstacle_grid >= 100)
            safety_margin_mask = (safety_grid >= 50)
            current_exploration_mask = (self.current_exploration_sector == 1)
            explored_mask = (self.explored_grid == 1)

            # Rebuild occupancy grid, prioritizing from low to high
            # 1. Default to unexplored (0)
            final_grid = np.zeros(self.grid_size, dtype=np.uint8)

            # 2. Explored free space (25)
            final_grid[explored_mask] = 25

            # 3. Safety margin (50) - Cover explored areas
            final_grid[safety_margin_mask] = 50

            # 4. Current frame exploration sector (75) - Only show in non-obstacle, non-safety areas
            exploration_free_mask = current_exploration_mask & (~obstacle_mask) & (~safety_margin_mask)
            final_grid[exploration_free_mask] = 75

            # 5. Occupied (obstacle) (100) - Highest priority, covers all other areas
            final_grid[obstacle_mask] = 100
            
            self.occupancy_grid = final_grid

            # If path planning is enabled and a goal position is set, attempt path planning
            if (DroidVisualizer._path_planning_enabled and 
                DroidVisualizer._goal_position is not None and 
                hasattr(self, 'latest_robot_position') and 
                self.latest_robot_position is not None):
                # Calculate distance
                distance = self.heuristic(self.latest_robot_position, DroidVisualizer._goal_position)
                if distance > DroidVisualizer._path_planning_distance_threshold:
                    self.plan_global_path(self.latest_robot_position)
                    self._visualize_path_in_grid()
                else:
                    DroidVisualizer._path_planning_enabled = False

        except Exception as e:
            print(f"Warning: Failed to merge exploration layers: {e}")
    
    def check_auto_save(self):
        """Check if auto-save is needed"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_accumulated_point_cloud()
            self.last_save_time = current_time
    
    def world_to_grid(self, world_pos: List[float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        if self.grid_origin is None:
            return None, None
        grid_x = int((world_pos[0] - self.grid_origin[0]) / self._occupancy_grid_resolution)
        grid_y = int((world_pos[1] - self.grid_origin[1]) / self._occupancy_grid_resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> List[float]:
        """Convert grid coordinates to world coordinates"""
        if self.grid_origin is None:
            return [0, 0]
        world_x = self.grid_origin[0] + grid_pos[0] * self._occupancy_grid_resolution
        world_y = self.grid_origin[1] + grid_pos[1] * self._occupancy_grid_resolution
        return [world_x, world_y]
    
    def is_valid_grid_pos(self, pos: Tuple[int, int]) -> bool:
        """Check if grid position is valid"""
        x, y = pos
        if self.grid_size is None:
            return False
        return 0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]
    
    def is_free_space(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is free space"""
        if not self.is_valid_grid_pos(grid_pos) or self.occupancy_grid is None:
            return False
        x, y = grid_pos
        # Only explored free space (25) and current exploration area (75) are considered passable
        return self.occupancy_grid[y, x] in [25, 75]
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get passable neighboring nodes in 8 directions and their movement costs"""
        x, y = pos
        neighbors = []

        # 8-directional offsets
        directions = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                    (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),   (1, 1, 1.414)
        ]
        
        for dx, dy, cost in directions:
            new_pos = (x + dx, y + dy)
            if self.is_free_space(new_pos):
                neighbors.append((new_pos, cost))
        
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """A* algorithm heuristic function (Euclidean distance)"""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    # def find_nearest_free_space(self, grid_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    #     """Find the nearest free space point to the specified grid position

    #     Use BFS (Breadth-First Search) to ensure the nearest free space point is found.
    #     The search will traverse the entire grid map until a free space is found.

    #     Args:
    #         grid_pos: Target grid position (x, y)
    #
    #     Returns:
    #         The grid coordinates of the nearest free space point, or None if no free space exists in the entire map
    #     """
    #     if self.occupancy_grid is None:
    #         return None
    #
    #     x, y = grid_pos
    #
    #     # If the current position is already free space, return it directly
    #     if self.is_free_space(grid_pos):
    #         return grid_pos
    #
    #     # Record the occupancy type of the current position
    #     if self.is_valid_grid_pos(grid_pos):
    #         occupancy_value = self.occupancy_grid[y, x]
    #         if occupancy_value == 100:
    #             print(f"Position ({x}, {y}) is in obstacle area, searching for nearest free space...")
    #         elif occupancy_value == 50:
    #             print(f"Position ({x}, {y}) is in safety margin area, searching for nearest free space...")
    #         elif occupancy_value == 0:
    #             print(f"Position ({x}, {y}) is in unexplored area, searching for nearest free space...")
    #         else:
    #             print(f"Position ({x}, {y}) has occupancy value {occupancy_value}, searching for nearest free space...")

    #     # Use BFS (Breadth-First Search) to find the nearest free space without setting a search radius limit
    #     from collections import deque
        
    #     queue = deque([(x, y, 0)])  # (x, y, distance)
    #     visited = set()
    #     visited.add((x, y))

    #     # 8-directional offsets, prioritized by distance
    #     directions = [
    #         (-1, 0), (1, 0), (0, -1), (0, 1),  # 4-connectivity
    #         (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal connectivity
    #     ]
        
    #     while queue:
    #         curr_x, curr_y, distance = queue.popleft()

    #         # Check 8-directional neighbors
    #         for dx, dy in directions:
    #             new_x, new_y = curr_x + dx, curr_y + dy
    #             new_pos = (new_x, new_y)
                
    #             if new_pos in visited:
    #                 continue
                    
    #             if not self.is_valid_grid_pos(new_pos):
    #                 continue
                    
    #             visited.add(new_pos)

    #             # If a free space is found, return the position
    #             if self.is_free_space(new_pos):
    #                 world_pos = self.grid_to_world(new_pos)
    #                 print(f"Found nearest free space at grid ({new_x}, {new_y}), world ({world_pos[0]:.2f}, {world_pos[1]:.2f}), distance {distance + 1} cells")
    #                 return new_pos

    #             # Continue searching: allow passing through unexplored (0) and safety (50) areas, but not through obstacles (100)
    #             # This allows bypassing obstacles to find free space
    #             occupancy_val = self.occupancy_grid[new_y, new_x]
    #             if occupancy_val != 100:  # Not an obstacle
    #                 queue.append((new_x, new_y, distance + 1))
        
    #     print(f"Warning: No free space found in the entire occupancy grid from position ({x}, {y})")
    #     return None
    
    
    def find_nearest_free_space(self, grid_pos):
        """
        Returns the grid coordinates of the nearest free space point (considering only free==25 or 75).
        Does not consider reachability; returns None if no free space exists in the entire map.
        """
        if self.occupancy_grid is None or not self.is_valid_grid_pos(grid_pos):
            return None

        x0, y0 = grid_pos

        # free：25=Explored, 75=Current
        free_mask = (self.occupancy_grid == 25) | (self.occupancy_grid == 75)

        # If already on free space, return it
        if free_mask[y0, x0]:
            return (x0, y0)

        ys, xs = np.where(free_mask)
        if xs.size == 0:
            return None  # No free space found in the entire map

        # Compute squared distances to all free cells and take the minimum
        d2 = (xs - x0) ** 2 + (ys - y0) ** 2
        i = int(np.argmin(d2))
        return (int(xs[i]), int(ys[i]))

    def astar_path_planning(self, start_world: List[float], goal_world: List[float]) -> Optional[List[List[float]]]:
        """A* path planning algorithm that automatically handles cases where the start/goal is in non-free space.

        When the start or goal is in the following areas, it will automatically find the nearest free space point as a substitute:
        - Obstacle area (value 100)
        - Safety redundancy area (value 50)
        - Unexplored area (value 0)
        - Other non-free areas

        Only explored free areas (25) and currently explored areas (75) are considered as plannable free space.
        When searching for substitute points, the entire grid map will be searched to ensure the nearest free space point is found.

        Args:
            start_world: [x, y]
            goal_world: [x, y]

        Returns:
            A list of path points (world coordinates), or None if no path exists
        """
        if self.occupancy_grid is None or self.grid_origin is None:
            return None

        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_world)
        goal_grid = self.world_to_grid(goal_world)
        
        if start_grid[0] is None or goal_grid[0] is None:
            return None

        # record original start and goal (for logging and final path construction)
        original_start_grid = start_grid
        original_goal_grid = goal_grid
        original_start_world = start_world.copy()
        original_goal_world = goal_world.copy()

        # Check if the start position is in free space, if not (including unexplored areas) find the nearest free space point
        if not self.is_free_space(start_grid):
            print(f"Start position {start_world} (grid: {start_grid}) is not in free space (value: {self.occupancy_grid[start_grid[1], start_grid[0]]})")

            # Find the nearest free space point as the new start point
            new_start_grid = self.find_nearest_free_space(start_grid)
            if new_start_grid is None:
                print("Cannot find a valid free space near start position for path planning")
                return None
                
            start_grid = new_start_grid
            start_world = self.grid_to_world(start_grid)
            print(f"Using nearest free space as start: {start_world} (grid: {start_grid})")

        # Check if the goal position is in free space, if not (including unexplored areas) find the nearest free space point
        if not self.is_free_space(goal_grid):
            print(f"Goal position {goal_world} (grid: {goal_grid}) is not in free space (value: {self.occupancy_grid[goal_grid[1], goal_grid[0]]})")

            # Find the nearest free space point as the new goal point
            new_goal_grid = self.find_nearest_free_space(goal_grid)
            if new_goal_grid is None:
                print("Cannot find a valid free space near goal position for path planning")
                return None
                
            goal_grid = new_goal_grid
            goal_world = self.grid_to_world(goal_grid)
            print(f"Using nearest free space as goal: {goal_world} (grid: {goal_grid})")

        # If start and goal are the same, return single point path
        if start_grid == goal_grid:
            return [start_world, goal_world] if start_world != goal_world else [start_world]

        # A* algorithm main body
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # Reconstruct path
                path_grid = []
                while current in came_from:
                    path_grid.append(current)
                    current = came_from[current]
                path_grid.append(start_grid)
                path_grid.reverse()

                # Convert to world coordinates
                path_world = [self.grid_to_world(pos) for pos in path_grid]

                # If alternative start was used, add original start to path beginning
                if original_start_grid != start_grid:
                    path_world.insert(0, original_start_world)
                    print(f"Added original start point {original_start_world} to path beginning")

                # If alternative goal was used, add original goal to path end
                if original_goal_grid != goal_grid:
                    path_world.append(original_goal_world)
                    print(f"Added original goal point {original_goal_world} to path end")
                    
                print(f"A* path planning completed with {len(path_world)} waypoints")
                return path_world
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_grid)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

        print("A* path planning failed: no path found between start and goal positions")
    
    def plan_global_path(self, robot_position: List[float]) -> Optional[List[List[float]]]:
        """Execute global path planning"""
        if not DroidVisualizer._path_planning_enabled or DroidVisualizer._goal_position is None:
            return None
        
        try:
            path = self.astar_path_planning(
                start_world=[robot_position[0], robot_position[1]], 
                goal_world=[DroidVisualizer._goal_position[0], DroidVisualizer._goal_position[1]]
            )
            
            if path is not None:
                DroidVisualizer._planned_path = path
                print(f"Global path planned successfully with {len(path)} waypoints")

                # Send path data to ROS2 node via UDP
                self._send_planned_path_udp(path)
            else:
                print("Failed to find a valid path to goal")
                DroidVisualizer._planned_path = None
            
            return path
        except Exception as e:
            print(f"Error in path planning: {e}")
            return None
    
    def _visualize_path_in_grid(self):
        """Visualize the planned path in the occupancy grid"""
        if (DroidVisualizer._planned_path is None or 
            self.occupancy_grid is None or 
            self.grid_origin is None):
            return
        
        try:
            # Mark the path in the occupancy grid (use value 200 to indicate path)
            for point in DroidVisualizer._planned_path:
                grid_x, grid_y = self.world_to_grid(point)
                if grid_x is not None and self.is_valid_grid_pos((grid_x, grid_y)):
                    # Only show path in free space
                    if self.occupancy_grid[grid_y, grid_x] in [25, 75]:  # Free area
                        self.occupancy_grid[grid_y, grid_x] = 200  # Path marker

            # Mark the goal position (use value 250)
            if DroidVisualizer._goal_position is not None:
                goal_grid = self.world_to_grid([DroidVisualizer._goal_position[0], DroidVisualizer._goal_position[1]])
                if goal_grid[0] is not None and self.is_valid_grid_pos(goal_grid):
                    # Draw a small circle around the goal position
                    gx, gy = goal_grid
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if dx*dx + dy*dy <= 4:  # Circular area
                                nx, ny = gx + dx, gy + dy
                                if self.is_valid_grid_pos((nx, ny)):
                                    self.occupancy_grid[ny, nx] = 250  # Goal marker

        except Exception as e:
            print(f"Error visualizing path: {e}")
    title = "Droid Visualizer"
    _depth_video1 = None
    _depth_video2 = None

    _refresh_rate = 5
    _filter_threshold = 0.02
    _filter_count = 2

    # Add class variable to store trajectory data
    _latest_trajectory_data = None
    
    # UDP related
    _udp_socket = None
    _udp_host = "127.0.0.1"
    _udp_port = 12346

    # Angle data UDP receiver
    _angle_udp_socket = None
    _angle_udp_port = 12347
    _latest_angle_data = None
    _first_angle_data = None
    _first_recieved_angle_data = False

    # Global path planning related
    _goal_position = None
    _planned_path = None
    _path_planning_enabled = False
    _path_planning_distance_threshold = 2.0  # Default value is 2.0 meters

    # 2D point cloud visualization related
    _enable_2d_plot = True
    _2d_filter_outliers = False
    _2d_eps = 0.1
    _2d_min_samples = 5
    _2d_point_size = 3.0

    # Occupancy Grid config
    _enable_occupancy_grid = True
    _occupancy_grid_resolution = 0.05  # 5cm/pixel
    _occupancy_point_radius = 0.15  # occupancy radius around each point

    # Morphological operation config
    _enable_morphology = True        # Enable morphological operations (erosion + dilation)
    _erosion_iterations = 1          # Erosion iterations
    _dilation_iterations = 1         # Dilation iterations
    _morphology_kernel_size = 6     # Morphological operation kernel size

    # Safety margin config
    _enable_safety_margin = True     # Enable safety margin
    _safety_margin_distance = 1.0   # Safety margin distance (meters)

    # Frontier exploration config
    _enable_frontier_exploration = True  # Enable frontier exploration
    _exploration_sector_angle = 60       # Exploration sector angle (degrees)
    _exploration_sector_radius = 7.0     # Exploration sector radius (meters)
    _exploration_ray_resolution = 0.02   # Ray resolution (degrees)

    # Static grid range config
    _static_grid_x_min = -25.0           # Static grid X direction min (meters)
    _static_grid_x_max = 15.0            # Static grid X direction max (meters)
    _static_grid_y_min = -20.0           # Static grid Y direction min (meters)
    _static_grid_y_max = 20.0            # Static grid Y direction max (meters)
    _use_static_grid_range = True        # Enable static grid range

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = False

        # Initialize point cloud accumulation
        self.__init_point_cloud_accumulation()

        # Initialize UDP socket
        self._init_udp_socket()

        # Initialize angle UDP receiver
        self._init_angle_udp_receiver()

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec3 in_color0;
                in float in_alpha0;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                out vec3 color;
                out float alpha;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                    color = in_color0;
                    alpha = in_alpha0;
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec3 color;
                in float alpha;

                void main()
                {
                    if (alpha <= 0)
                        discard;

                    fragColor = vec4(color, alpha);
                }
            """,
        )

        self.cam_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                out vec4 fragColor;
                uniform vec3 color;

                void main()
                {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        n, h, w = self._depth_video1.disps.shape
        max_num_points = n * h * w

        # Upload buffer to GPU
        valid = np.zeros((max_num_points,), dtype="f4")
        points = np.zeros((max_num_points, 3), dtype="f4")
        colors = np.zeros((max_num_points, 3), dtype="f4")

        self.valid_buffer = self.ctx.buffer(valid.tobytes())
        self.pts_buffer = self.ctx.buffer(points.tobytes())
        self.clr_buffer = self.ctx.buffer(colors.tobytes())

        self.vao = VAO("geometry_frustrum", mode=moderngl.LINES)
        self.cam_prog["color"].value = (0, 0, 0)

        # cam_segments = CAM_SEGMENTS.repeat(n, 1).astype(np.float32)
        # print(cam_segments.shape)
        cam_segments = CAM_SEGMENTS.astype("f4")
        cam_segments = np.tile(cam_segments, (n, 1))


        self.count = 0

        # Create a vertex array manually
        self.points = self.ctx.vertex_array(
            self.prog,
            [
                (self.pts_buffer, "3f", "in_position"),
                (self.clr_buffer, "3f", "in_color0"),
                (self.valid_buffer, "f", "in_alpha0"),
            ],
        )

        self.cam_buffer = self.ctx.buffer(cam_segments.tobytes())
        self.cams = self.ctx.vertex_array(
            self.cam_prog,
            [
                (self.cam_buffer, "3f", "in_position"),
            ],
        )

        self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 1.0

        # Initialize 3D trajectory visualization
        self._init_3d_trajectory_plot()

        # Initialize 2D point cloud real-time visualization
        if self._enable_2d_plot:
            try:
                self._init_2d_pointcloud_plot()
            except Exception as e:
                print(f"Warning: Failed to initialize 2D visualization: {e}")
                self._enable_2d_plot = False  # Disable 2D visualization to avoid subsequent errors

        # Initialize Occupancy Grid visualization
        if self._enable_occupancy_grid:
            try:
                self._init_occupancy_grid_plot()
            except Exception as e:
                print(f"Warning: Failed to initialize occupancy grid visualization: {e}")
                self._enable_occupancy_grid = False

    def _init_udp_socket(self):
        """Initialize UDP socket for sending trajectory data"""
        try:
            DroidVisualizer._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        except Exception as e:

            DroidVisualizer._udp_socket = None
    
    def _init_angle_udp_receiver(self):
        """Initialize angle UDP receiver"""
        try:
            DroidVisualizer._angle_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            DroidVisualizer._angle_udp_socket.bind((DroidVisualizer._udp_host, DroidVisualizer._angle_udp_port))
            DroidVisualizer._angle_udp_socket.settimeout(0.001)  # Non-blocking receive

            # Start angle data receiving thread
            import threading
            self.angle_thread = threading.Thread(target=self._angle_receiver_thread, daemon=True)
            self.angle_thread.start()
            print(f"Angle UDP receiver started on port {DroidVisualizer._angle_udp_port}")
            
        except Exception as e:
            print(f"Failed to initialize angle UDP receiver: {e}")
            DroidVisualizer._angle_udp_socket = None
    
    def _angle_receiver_thread(self):
        """Quaternion data receiving thread"""
        while True:
            try:
                if DroidVisualizer._angle_udp_socket:
                    data, addr = DroidVisualizer._angle_udp_socket.recvfrom(1024)
                    json_data = data.decode('utf-8')
                    received_data = json.loads(json_data)

                    # Process different types of data
                    if 'type' in received_data and received_data['type'] == 'goal_position':
                        # Process goal position data
                        goal_pos = received_data['goal_position']
                        DroidVisualizer._goal_position = [goal_pos['x'], goal_pos['y'], goal_pos['z']]
                        DroidVisualizer._path_planning_enabled = True
                        print(f"Received goal position for path planning: {DroidVisualizer._goal_position}")
                    else:
                        # Process quaternion data
                        if not DroidVisualizer._first_recieved_angle_data:
                            DroidVisualizer._first_angle_data = received_data
                            DroidVisualizer._first_recieved_angle_data = True
                            
                        DroidVisualizer._latest_angle_data = received_data
            except socket.timeout:
                pass
            except Exception as e:
                # print(f"Error in angle receiver thread: {e}")
                pass
            time.sleep(0.001)

    def _send_trajectory_data_udp(self, trajectory_data):
        """Send trajectory data via UDP"""
        if DroidVisualizer._udp_socket is None:
            return
            
        try:
            # Serialize data to JSON
            json_data = json.dumps(trajectory_data)
            data_bytes = json_data.encode('utf-8')

            # Send data
            DroidVisualizer._udp_socket.sendto(data_bytes, (DroidVisualizer._udp_host, DroidVisualizer._udp_port))

        except Exception as e:
            print(f"Failed to send trajectory data via UDP: {e}")
    
    def _send_planned_path_udp(self, path):
        """Send planned path data to ROS2 node via UDP"""
        if DroidVisualizer._udp_socket is None:
            return
            
        try:
            # Build path data packet
            path_data = {
                'type': 'planned_path',
                'path': path,  # List of waypoints, each point is in [x, y] format
                'timestamp': time.time(),
                'path_length': len(path)
            }

            # Serialize data to JSON
            json_data = json.dumps(path_data)
            data_bytes = json_data.encode('utf-8')

            # Send data to new port (12348) for path transmission
            path_udp_port = 12348
            DroidVisualizer._udp_socket.sendto(data_bytes, (DroidVisualizer._udp_host, path_udp_port))
            print(f"Sent planned path via UDP with {len(path)} waypoints to port {path_udp_port}")

        except Exception as e:
            print(f"Failed to send planned path via UDP: {e}")

    def _init_3d_trajectory_plot(self):
        """Initialize 3d trajectory plot"""
        # Set matplotlib to non-blocking mode
        plt.ion()

        # Create 3D plot
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('DROID-SLAM real time trajectory(3D)')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Initialize trajectory line and point
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')

        # Set equal aspect ratio for axes
        self.ax.set_box_aspect([1,1,1])
        self.ax.legend()

        # Store trajectory data
        self.trajectory_points = []
        
        plt.show(block=False)

    def _update_3d_trajectory_from_cam_pts(self, cam_pts, frame_count):
        """Update 3D trajectory plot from camera-transformed point cloud data"""
        try:
            if cam_pts.shape[0] > 0:
                # cam_pts is of shape [frame_count * 20, 3]
                # where frame_count is the number of frames, 20 is the number of segment endpoints per camera frame
                points_per_frame = 20  # CAM_SEGMENTS has 20 points (10 segments × 2 endpoints)
                
                if cam_pts.shape[0] >= points_per_frame * frame_count:
                    # Reshape data to [frame_count, points_per_frame, 3]
                    cam_pts_reshaped = cam_pts[:frame_count * points_per_frame].reshape(frame_count, points_per_frame, 3)

                    # Compute the center point of the camera for each frame as a trajectory point
                    camera_positions = []
                    for frame_idx in range(frame_count):
                        # Compute the average position of all 20 points in the current frame as the camera center
                        camera_center_slam = np.mean(cam_pts_reshaped[frame_idx, :, :], axis=0)  # Camera position in SLAM coordinate system

        
                        if DroidVisualizer._first_angle_data is not None:
                            torso_quat_data = DroidVisualizer._first_angle_data.get('torso_to_world_quat', {})
                            torso_to_world_quat = np.array([
                                torso_quat_data.get('w', 1.0),
                                torso_quat_data.get('x', 0.0),
                                torso_quat_data.get('y', 0.0),
                                torso_quat_data.get('z', 0.0)
                            ])
                            # torso_to_world_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion (no rotation)

                        # Camera to torso transformation (based on the provided quaternion)
                        camera_to_torso_quat = np.array([0.45451948, -0.54167522, 0.54167522, -0.45451948])

                        # Compute the quaternion from camera to world coordinates: camera_to_world = torso_to_world * camera_to_torso
                        camera_to_world_quat = quaternion_multiply(torso_to_world_quat, camera_to_torso_quat)
                        # Convert quaternion to rotation matrix
                        R_camera_to_world = quaternion_to_rotation_matrix(
                            camera_to_world_quat[0], camera_to_world_quat[1], 
                            camera_to_world_quat[2], camera_to_world_quat[3]
                        )

                        # Critical: camera_center_slam is the position in SLAM coordinates
                        # Transform the camera position from SLAM coordinates to world coordinates
                        camera_position_world = R_camera_to_world @ camera_center_slam
                        
                        camera_positions.append(camera_position_world)
                    
                    camera_positions = np.array(camera_positions)

                    # Update trajectory data
                    self.trajectory_points = camera_positions.tolist()

                    # Store the latest trajectory data in a class variable (for external access)
                    if len(self.trajectory_points) > 0:
                        # Extract x and z coordinates (x corresponds to real x, z corresponds to real y in 3D plot)
                        x_coords = [p[0] for p in self.trajectory_points]
                        y_coords = [p[1] for p in self.trajectory_points]
                        z_coords = [p[2] for p in self.trajectory_points]
                        trajectory_data = {
                            'x_coords': x_coords,
                            'y_coords': y_coords,
                            'z_coords': z_coords,
                            'frame_count': frame_count,
                            'latest_position': {
                                'x': x_coords[-1],     
                                'y': y_coords[-1],      
                                'z': z_coords[-1]      
                            },
                            'timestamp': time.time()
                        }
                        
                        DroidVisualizer._latest_trajectory_data = trajectory_data

                        # Send trajectory data via UDP
                        self._send_trajectory_data_udp(trajectory_data)

                    # Update trajectory line
                    if len(self.trajectory_points) > 1:
                        x_data = [p[0] for p in self.trajectory_points]
                        y_data = [p[1] for p in self.trajectory_points]
                        z_data = [p[2] for p in self.trajectory_points]
                        
                        self.trajectory_line.set_data_3d(x_data, y_data, z_data)

                        # Update current position point
                        self.current_point.set_data_3d([x_data[-1]], [y_data[-1]], [z_data[-1]])

                        # Dynamically adjust axis limits to keep the same scale
                        margin = 0.5  # Margin

                        # Compute the data range for each axis
                        x_min, x_max = min(x_data) - margin, max(x_data) + margin
                        y_min, y_max = min(y_data) - margin, max(y_data) + margin
                        z_min, z_max = min(z_data) - margin, max(z_data) + margin

                        # Find the maximum data range
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        max_range = max(x_range, y_range, z_range)

                        # Compute the center point for each axis
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        z_center = (z_min + z_max) / 2

                        # Use the maximum range to set the limits for all axes, keeping the same scale
                        half_range = max_range / 2
                        self.ax.set_xlim(x_center - half_range, x_center + half_range)
                        self.ax.set_ylim(y_center - half_range, y_center + half_range)
                        self.ax.set_zlim(z_center - half_range, z_center + half_range)

                    # Refresh the figure
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    
        except Exception as e:
            print(f"Error when updating 3D trajectory from cam_pts: {e}")

    def _init_2d_pointcloud_plot(self):
        """Initialize 2D point cloud real-time visualization"""
        try:
            # Set matplotlib to non-blocking mode
            plt.ion()

            # Create 2D plot
            self.fig_2d = plt.figure(figsize=(12, 10))
            self.ax_2d = self.fig_2d.add_subplot(111)
            self.ax_2d.set_title('DROID-SLAM Real-time 2D Point Cloud', fontsize=14)
            self.ax_2d.set_xlabel('X (m)', fontsize=12)
            self.ax_2d.set_ylabel('Y (m)', fontsize=12)
            self.ax_2d.grid(True, alpha=0.3)
            self.ax_2d.set_aspect('equal', adjustable='box')  # Use adjustable='box' instead of axis('equal')

            # Initialize scatter plot
            self.pointcloud_scatter = self.ax_2d.scatter([], [], s=self._2d_point_size, alpha=0.6, edgecolors='none', c='blue', label='Point Cloud')

            # Initialize camera position point
            self.camera_position_scatter = self.ax_2d.scatter([], [], s=50, c='red', marker='o', label='Camera Position')

            # Add legend
            self.ax_2d.legend()

            # Initialize info text
            self.info_text_2d = self.ax_2d.text(0.02, 0.98, '', transform=self.ax_2d.transAxes,
                                               verticalalignment='top',
                                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Store current 2D point cloud data
            self.current_2d_points = None
            self.current_2d_colors = None
            
            plt.show(block=False)
            print("2D point cloud visualization initialized")
            
        except Exception as e:
            print(f"Failed to initialize 2D point cloud plot: {e}")

    def _update_2d_pointcloud(self, points, colors, camera_position=None):
        """Update real-time 2D point cloud visualization

        Args:
            points: Point cloud data (N, 3), using x,y coordinates
            colors: Color data (N, 3)
            camera_position: Camera position in world coordinates (3,)
        """
        try:
            if not self._enable_2d_plot or points is None or len(points) == 0:
                return

            # Store original data
            original_point_count = len(points)
            filtered_points = points.copy()
            filtered_colors = colors.copy()

            # Use DBSCAN to filter outliers (optional)
            if self._2d_filter_outliers and len(points) > self._2d_min_samples:
                # Only cluster xy coordinates
                xy_points = points[:, :2]

                # Apply DBSCAN clustering
                dbscan = DBSCAN(eps=self._2d_eps, min_samples=self._2d_min_samples)
                cluster_labels = dbscan.fit_predict(xy_points)

                # Filter out outliers (points with label -1)
                inlier_mask = cluster_labels != -1
                filtered_points = points[inlier_mask]
                filtered_colors = colors[inlier_mask]

                # If DBSCAN filtered out all points, fall back to original data
                if len(filtered_points) == 0:
                    print("Warning: DBSCAN filtered out all points, using original data")
                    filtered_points = points.copy()
                    filtered_colors = colors.copy()

            # Store current data
            self.current_2d_points = filtered_points
            self.current_2d_colors = filtered_colors

            # Update scatter plot data (only use x,y coordinates)
            xy_coords = filtered_points[:, :2]

            # Update scatter plot
            self.pointcloud_scatter.set_offsets(xy_coords)
            if len(filtered_colors) > 0:
                self.pointcloud_scatter.set_color(filtered_colors)

            # Update camera position
            if camera_position is not None:
                self.camera_position_scatter.set_offsets([[camera_position[0], camera_position[1]]])

            # Update statistics
            outlier_count = original_point_count - len(filtered_points)
            x_range = np.max(xy_coords[:, 0]) - np.min(xy_coords[:, 0]) if len(xy_coords) > 0 else 0
            y_range = np.max(xy_coords[:, 1]) - np.min(xy_coords[:, 1]) if len(xy_coords) > 0 else 0
            
            info_text = f"Points: {len(filtered_points)}"
            if self._2d_filter_outliers and outlier_count > 0:
                info_text += f" (Original: {original_point_count})"
            info_text += f"\nX Range: {x_range:.2f}m\nY Range: {y_range:.2f}m"
            if self._2d_filter_outliers:
                info_text += f"\nDBSCAN: eps={self._2d_eps}, min_samples={self._2d_min_samples}"
            
            self.info_text_2d.set_text(info_text)

            # Dynamically adjust axis limits
            if len(xy_coords) > 0:
                margin = 0.5
                x_min, x_max = np.min(xy_coords[:, 0]) - margin, np.max(xy_coords[:, 0]) + margin
                y_min, y_max = np.min(xy_coords[:, 1]) - margin, np.max(xy_coords[:, 1]) + margin

                # If camera position is available, ensure it is within the view
                if camera_position is not None:
                    x_min = min(x_min, camera_position[0] - margin)
                    x_max = max(x_max, camera_position[0] + margin)
                    y_min = min(y_min, camera_position[1] - margin)
                    y_max = max(y_max, camera_position[1] + margin)

                # Compute max range for equal aspect ratio
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)

                # Compute center point
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                # Use max range to set equal aspect ratio
                half_range = max_range / 2
                self.ax_2d.set_xlim(x_center - half_range, x_center + half_range)
                self.ax_2d.set_ylim(y_center - half_range, y_center + half_range)

            # Refresh the plot
            self.fig_2d.canvas.draw()
            self.fig_2d.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating 2D point cloud: {e}")

    def _init_occupancy_grid_plot(self):
        """Initialize Occupancy Grid visualization"""
        try:
            # Set matplotlib to non-blocking mode
            plt.ion()

            # Create Occupancy Grid figure
            self.fig_occupancy = plt.figure(figsize=(10, 10))
            self.ax_occupancy = self.fig_occupancy.add_subplot(111)
            self.ax_occupancy.set_title('DROID-SLAM Real-time Occupancy Grid Map', fontsize=14)
            self.ax_occupancy.set_xlabel('X (m)', fontsize=12)
            self.ax_occupancy.set_ylabel('Y (m)', fontsize=12)
            self.ax_occupancy.set_aspect('equal', adjustable='box')

            # Initialize empty occupancy grid image (will be set on first update)
            self.occupancy_image = None

            # Add grid
            # self.ax_occupancy.grid(True, alpha=0.3)
            self.ax_occupancy.grid(False)

            # Initialize robot position scatter
            self.robot_position_scatter = self.ax_occupancy.scatter(
                [], [], s=100, c='red', marker='o', label='Robot Position', zorder=5
            )

            # Add frontier exploration visualization (optional)
            if self._enable_frontier_exploration:
                pass
            
            self.ax_occupancy.legend()

            # Initialize statistics text
            self.info_text_occupancy = self.ax_occupancy.text(
                0.02, 0.98, '', transform=self.ax_occupancy.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            plt.show(block=False)
            print("Occupancy grid visualization initialized")
            
        except Exception as e:
            print(f"Failed to initialize occupancy grid plot: {e}")

    def _update_occupancy_grid_plot(self, points, robot_position=None):
        """Update Occupancy Grid visualization (supports frontier exploration)

        Args:
            points: Point cloud data (N, 3), will use x,y coordinates
            robot_position: Robot position in world coordinates (3,)
        """
        try:
            if not self._enable_occupancy_grid or points is None or len(points) == 0:
                return

            # Create dynamic occupancy grid from point cloud (including frontier exploration)
            self.create_occupancy_grid_from_points(points, robot_position)

            # If the grid has not been created yet, exit
            if self.occupancy_grid is None or self.grid_origin is None:
                return

            # Compute the world coordinate range of the grid
            x_extent = [self.grid_origin[0], 
                       self.grid_origin[0] + self.grid_size[1] * self._occupancy_grid_resolution]
            y_extent = [self.grid_origin[1], 
                       self.grid_origin[1] + self.grid_size[0] * self._occupancy_grid_resolution]

            # If this is the first time creating the image or the grid size has changed
            if (self.occupancy_image is None or
                self.occupancy_image.get_array().shape != self.occupancy_grid.shape):

                # Clear previous image
                if self.occupancy_image is not None:
                    self.occupancy_image.remove()

                # Create custom colormap for frontier exploration

                # Define colors:
                # 0: Unexplored (frontier) - Black
                # 25: Explored Free - White
                # 50: Safety - Yellow
                # 75: Current Exploration - Green
                # 100: Occupied - Red
                # 200: Path - Blue
                # 250: Goal - Magenta
                colors = ['black', 'white', 'yellow', 'lightgreen', 'red', 'blue', 'magenta']
                values = [0, 25, 50, 75, 100, 200, 250]

                # Create custom color map
                cmap = mcolors.ListedColormap(colors)
                norm = mcolors.BoundaryNorm([-0.5, 12.5, 37.5, 62.5, 87.5, 150, 225, 250.5], cmap.N)

                # Create new occupancy grid image
                self.occupancy_image = self.ax_occupancy.imshow(
                    self.occupancy_grid, 
                    cmap=cmap,
                    norm=norm,
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    interpolation='none',
                    resample=False
                )

                # Re-add colorbar (if not already present)
                if not hasattr(self, 'occupancy_colorbar') or self.occupancy_colorbar is None:
                    self.occupancy_colorbar = plt.colorbar(
                        self.occupancy_image, ax=self.ax_occupancy,
                        ticks=[0, 25, 50, 75, 100, 200, 250],
                        format='%d'
                    )
                    # Set colorbar labels
                    self.occupancy_colorbar.ax.set_yticklabels([
                        'Frontier', 'Explored', 'Safety', 'Current', 'Occupied', 'Path', 'Goal'
                    ])

            else:
                # Update existing image data and extent
                self.occupancy_image.set_array(self.occupancy_grid)
                self.occupancy_image.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])

            # Dynamically adjust axis limits
            if self._use_static_grid_range:
                # Use static grid range
                margin = 0.5
                self.ax_occupancy.set_xlim(self._static_grid_x_min - margin, self._static_grid_x_max + margin)
                self.ax_occupancy.set_ylim(self._static_grid_y_min - margin, self._static_grid_y_max + margin)
            else:
                # Dynamically adjust axis limits (similar to 2D point cloud)
                margin = 0.5
                self.ax_occupancy.set_xlim(x_extent[0] - margin, x_extent[1] + margin)
                self.ax_occupancy.set_ylim(y_extent[0] - margin, y_extent[1] + margin)

            # Update robot position
            if robot_position is not None:
                self.robot_position_scatter.set_offsets([[robot_position[0], robot_position[1]]])

            # # Update statistics (support frontier exploration)
            # frontier_cells = np.sum(self.occupancy_grid == 0)       # Unexplored (frontier)
            # explored_cells = np.sum(self.occupancy_grid == 25)      # Explored (free)
            # safety_margin_cells = np.sum(self.occupancy_grid == 50) # Safety (buffer)
            # current_exploration_cells = np.sum(self.occupancy_grid == 75)  # Current (exploration)
            # occupied_cells = np.sum(self.occupancy_grid == 100)     # Occupied (obstacle)
            # total_cells = self.occupancy_grid.size

            # # Calculate percentages
            # frontier_ratio = frontier_cells / total_cells * 100
            # explored_ratio = explored_cells / total_cells * 100
            # safety_ratio = safety_margin_cells / total_cells * 100
            # current_exploration_ratio = current_exploration_cells / total_cells * 100
            # occupied_ratio = occupied_cells / total_cells * 100

            # # Construct information text
            # info_text = f"Frontier(Black): {frontier_cells} ({frontier_ratio:.1f}%)"
            # info_text += f"\nExplored(White): {explored_cells} ({explored_ratio:.1f}%)"
            # info_text += f"\nSafety(Yellow): {safety_margin_cells} ({safety_ratio:.1f}%)"
            # info_text += f"\nCurrent(Green): {current_exploration_cells} ({current_exploration_ratio:.1f}%)"
            # info_text += f"\nOccupied(Red): {occupied_cells} ({occupied_ratio:.1f}%)"
            # info_text += f"\nTotal: {total_cells}"
            # info_text += f"\nResolution: {self._occupancy_grid_resolution}m/pixel"
            # info_text += f"\nGrid Size: {self.grid_size[1]}x{self.grid_size[0]}"

            # # Display grid range information
            # if self._use_static_grid_range:
            #     info_text += f"\nStatic Range: X[{self._static_grid_x_min},{self._static_grid_x_max}] Y[{self._static_grid_y_min},{self._static_grid_y_max}]"
            #     info_text += f"\nCoverage: {self._static_grid_x_max-self._static_grid_x_min:.1f}m x {self._static_grid_y_max-self._static_grid_y_min:.1f}m"
            # else:
            #     info_text += f"\nDynamic Range: {self.grid_size[1]*self._occupancy_grid_resolution:.1f}m x {self.grid_size[0]*self._occupancy_grid_resolution:.1f}m"

            # # Add exploration parameters information
            # if self._enable_frontier_exploration:
            #     info_text += f"\nExploration Sector: {self._exploration_sector_angle}°"
            #     info_text += f"\nExploration Radius: {self._exploration_sector_radius}m"
            #     info_text += f"\nRay Resolution: {self._exploration_ray_resolution}°"
            # else:
            #     info_text += f"\nFrontier Exploration: Disabled"

            # # Add morphology information
            # if self._enable_morphology:
            #     info_text += f"\nMorphology: E{self._erosion_iterations}→D{self._dilation_iterations} (K={self._morphology_kernel_size})"
            # else:
            #     info_text += f"\nMorphology: Disabled"

            # # Add safety margin information
            # if self._enable_safety_margin:
            #     info_text += f"\nSafety Margin: {self._safety_margin_distance}m"
            # else:
            #     info_text += f"\nSafety Margin: Disabled"
            
            # self.info_text_occupancy.set_text(info_text)

            # Refresh the figure
            self.fig_occupancy.canvas.draw()
            self.fig_occupancy.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating occupancy grid plot: {e}")

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.ctx.point_size = 0.2

        self.prog["m_proj"].write(self.camera.projection.matrix)
        self.prog["m_cam"].write(self.camera.matrix)

        self.cam_prog["m_proj"].write(self.camera.projection.matrix)
        self.cam_prog["m_cam"].write(self.camera.matrix)

        t = self._depth_video1.counter.value

        if t > 12 and self.count % self._refresh_rate == 0:
            images = self._depth_video1.images[:t, :, 4::8, 4::8]
            intrinsics = self._depth_video1.intrinsics

            if self._depth_video2 is not None:
                poses, disps = merge_depths_and_poses(self._depth_video1, self._depth_video2)
                poses = poses[:t]
                disps = disps[:t]
            else:
                disps = self._depth_video1.disps[:t]
                poses = self._depth_video1.poses[:t]

            # Check if poses and disps contain valid data
            if poses.numel() == 0:
                print("Warning: poses tensor is empty!")
                self.count += 1
                return
            if disps.numel() == 0:
                print("Warning: disps tensor is empty!")
                self.count += 1
                return

            # 4x4 homogenous matrix
            cam_pts = torch.from_numpy(CAM_SEGMENTS).cuda()
            cam_pts = SE3(poses[:, None]).inv() * cam_pts[None]
            cam_pts = cam_pts.reshape(-1, 3).cpu().numpy()

            self._update_3d_trajectory_from_cam_pts(cam_pts, t)

            self.cam_buffer.write(cam_pts)

            index = torch.arange(t, device="cuda")
            thresh = self._filter_threshold * torch.ones_like(disps.mean(dim=[1, 2]))

            points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
            colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

            counts = droid_backends.depth_filter(
                poses, disps, intrinsics[0], index, thresh
            )

            # Ensure all tensors have matching dimensions
            min_size = min(counts.shape[0], disps.shape[0], points.shape[0], colors.shape[0])
            counts = counts[:min_size]
            disps_for_mask = disps[:min_size]
            points = points[:min_size]
            colors = colors[:min_size]
            
            mask = (counts >= self._filter_count) & (disps_for_mask > 0.25 * disps_for_mask.mean())

            valid = mask.float()

            # wasteful (gpu -> cpu -> gpu)
            self.pts_buffer.write(points.contiguous().cpu().numpy())
            self.clr_buffer.write(colors.contiguous().cpu().numpy())
            self.valid_buffer.write(valid.contiguous().cpu().numpy())

            # Accumulate point cloud data
            self.accumulate_point_cloud()

            # Check if auto-save is needed
            self.check_auto_save()

            # Update 2D point cloud visualization
            if self._enable_2d_plot:
                try:
                    # Directly use the downsampled point cloud data processed in accumulate_point_cloud
                    current_points = getattr(self, 'current_frame_2d_points', None)
                    current_colors = getattr(self, 'current_frame_2d_colors', None)

                    # Get current camera position (from the last camera position)
                    camera_position = None
                    if cam_pts.shape[0] > 0 and DroidVisualizer._latest_trajectory_data is not None:
                        latest_pos = DroidVisualizer._latest_trajectory_data.get('latest_position')
                        if latest_pos:
                            camera_position = np.array([latest_pos['x'], latest_pos['y'], latest_pos['z']])

                    # Update 2D visualization
                    self._update_2d_pointcloud(current_points, current_colors, camera_position)
                except Exception as e:
                    # 2D visualization errors should not affect the main SLAM process
                    print(f"Warning: 2D visualization update failed: {e}")
                    pass

            # Update Occupancy Grid visualization (following the same approach as 2D point cloud)
            if self._enable_occupancy_grid:
                try:
                    # Directly use the downsampled point cloud data processed in accumulate_point_cloud
                    current_points = getattr(self, 'current_frame_2d_points', None)

                    # Get current robot position (from the last camera position)
                    robot_position = None
                    if cam_pts.shape[0] > 0 and DroidVisualizer._latest_trajectory_data is not None:
                        latest_pos = DroidVisualizer._latest_trajectory_data.get('latest_position')
                        if latest_pos:
                            robot_position = np.array([latest_pos['x'], latest_pos['y'], latest_pos['z']])

                    # Update occupancy grid visualization
                    self._update_occupancy_grid_plot(current_points, robot_position)
                except Exception as e:
                    # Occupancy grid visualization errors should not affect the main SLAM process
                    print(f"Warning: Occupancy grid visualization update failed: {e}")
                    pass

        self.count += 1
        self.points.render(mode=moderngl.POINTS)
        self.cams.render(mode=moderngl.LINES)


def visualization_fn(depth_video1, depth_video2):
    config = DroidVisualizer
    config._depth_video1 = depth_video1
    config._depth_video2 = depth_video2

    try:
        # run visualizer
        moderngl_window.run_window_config(config, args=["-r", "True"])
    finally:
        # clear UDP socket
        if DroidVisualizer._udp_socket:
            try:
                DroidVisualizer._udp_socket.close()
                print("UDP socket closed")
            except Exception as e:
                print(f"Error closing UDP socket: {e}")

        # clear angle UDP socket
        if DroidVisualizer._angle_udp_socket:
            try:
                DroidVisualizer._angle_udp_socket.close()
                print("Angle UDP socket closed")
            except Exception as e:
                print(f"Error closing angle UDP socket: {e}")

