#!/usr/bin/env python3
"""
DROID-SLAM轨迹数据读取和可视化工具
用于读取DROID-SLAM保存的轨迹数据并进行可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy.spatial.transform import Rotation


def load_trajectory(filename):
    """加载轨迹数据"""
    try:
        data = np.load(filename)
        trajectory = {
            'timestamps': data['timestamps'],
            'poses': data['poses'],  # [N, 7] - [tx, ty, tz, qx, qy, qz, qw]
            'frame_count': data['frame_count']
        }
        print(f"Loaded trajectory with {trajectory['frame_count']} frames")
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None


def extract_positions_and_orientations(poses):
    """从位姿数据中提取位置和方向信息"""
    positions = poses[:, :3]  # [tx, ty, tz]
    quaternions = poses[:, 3:]  # [qx, qy, qz, qw]
    
    # 转换四元数为欧拉角 (roll, pitch, yaw)
    rotations = Rotation.from_quat(quaternions)
    euler_angles = rotations.as_euler('xyz', degrees=True)
    
    return positions, quaternions, euler_angles


def plot_trajectory_3d(positions, title="DROID-SLAM Trajectory"):
    """绘制3D轨迹"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               color='green', s=100, label='Start', marker='o')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               color='red', s=100, label='End', marker='s')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # 设置相等的坐标轴比例
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def plot_trajectory_2d(positions, title="DROID-SLAM Trajectory (Top View)"):
    """绘制2D轨迹（俯视图）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制轨迹线
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], 
               color='green', s=100, label='Start', marker='o')
    ax.scatter(positions[-1, 0], positions[-1, 1], 
               color='red', s=100, label='End', marker='s')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    return fig


def plot_pose_components(timestamps, positions, euler_angles):
    """绘制位置和姿态分量随时间的变化"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 位置分量
    position_labels = ['X', 'Y', 'Z']
    for i in range(3):
        axes[0, i].plot(timestamps, positions[:, i], 'b-', linewidth=2)
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel(f'{position_labels[i]} Position (m)')
        axes[0, i].set_title(f'{position_labels[i]} Position vs Time')
        axes[0, i].grid(True, alpha=0.3)
    
    # 姿态分量（欧拉角）
    angle_labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axes[1, i].plot(timestamps, euler_angles[:, i], 'r-', linewidth=2)
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel(f'{angle_labels[i]} (deg)')
        axes[1, i].set_title(f'{angle_labels[i]} vs Time')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_statistics(positions, timestamps):
    """计算轨迹统计信息"""
    # 总距离
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = np.sum(distances)
    
    # 平均速度
    total_time = timestamps[-1] - timestamps[0]
    avg_speed = total_distance / total_time if total_time > 0 else 0
    
    # 直线距离
    straight_distance = np.linalg.norm(positions[-1] - positions[0])
    
    # 轨迹范围
    pos_range = {
        'x': (positions[:, 0].min(), positions[:, 0].max()),
        'y': (positions[:, 1].min(), positions[:, 1].max()),
        'z': (positions[:, 2].min(), positions[:, 2].max())
    }
    
    stats = {
        'total_distance': total_distance,
        'straight_distance': straight_distance,
        'avg_speed': avg_speed,
        'total_time': total_time,
        'frame_count': len(positions),
        'position_range': pos_range
    }
    
    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n=== Trajectory Statistics ===")
    print(f"Frame count: {stats['frame_count']}")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Total distance: {stats['total_distance']:.2f} meters")
    print(f"Straight-line distance: {stats['straight_distance']:.2f} meters")
    print(f"Average speed: {stats['avg_speed']:.2f} m/s")
    print(f"Position range:")
    print(f"  X: {stats['position_range']['x'][0]:.2f} to {stats['position_range']['x'][1]:.2f} meters")
    print(f"  Y: {stats['position_range']['y'][0]:.2f} to {stats['position_range']['y'][1]:.2f} meters")
    print(f"  Z: {stats['position_range']['z'][0]:.2f} to {stats['position_range']['z'][1]:.2f} meters")


def main():
    parser = argparse.ArgumentParser(description='DROID-SLAM Trajectory Visualization Tool')
    parser.add_argument('trajectory_file', help='Path to trajectory .npz file')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output_dir', default='.', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # 加载轨迹数据
    trajectory = load_trajectory(args.trajectory_file)
    if trajectory is None:
        return
    
    # 提取位置和方向信息
    positions, quaternions, euler_angles = extract_positions_and_orientations(trajectory['poses'])
    
    # 计算并打印统计信息
    stats = calculate_statistics(positions, trajectory['timestamps'])
    print_statistics(stats)
    
    # 绘制3D轨迹
    fig1 = plot_trajectory_3d(positions)
    if args.save_plots:
        fig1.savefig(f"{args.output_dir}/trajectory_3d.png", dpi=300, bbox_inches='tight')
    
    # 绘制2D轨迹
    fig2 = plot_trajectory_2d(positions)
    if args.save_plots:
        fig2.savefig(f"{args.output_dir}/trajectory_2d.png", dpi=300, bbox_inches='tight')
    
    # 绘制位姿分量
    fig3 = plot_pose_components(trajectory['timestamps'], positions, euler_angles)
    if args.save_plots:
        fig3.savefig(f"{args.output_dir}/pose_components.png", dpi=300, bbox_inches='tight')
    
    if args.save_plots:
        print(f"\nPlots saved to {args.output_dir}/")
    else:
        print("\nShowing plots...")
        plt.show()


if __name__ == '__main__':
    main()
