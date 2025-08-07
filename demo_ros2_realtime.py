import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import time
import argparse
import threading
import queue
from collections import deque

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

class DroidSlamNode(Node):
    def __init__(self, rgb_topic, depth_topic, calib_file, args):
        super().__init__('droid_slam_node')
        
        self.bridge = CvBridge()
        self.args = args
        self.droid = None
        self.frame_count = 0
        
        # Create pose publishers (only if enabled)
        if args.publish_pose:
            self.pose_pub = self.create_publisher(PoseStamped, 'droid_slam/pose', 10)
            self.odom_pub = self.create_publisher(Odometry, 'droid_slam/odometry', 10)
            
            # TF broadcaster for publishing transform
            try:
                from tf2_ros import TransformBroadcaster
                self.tf_broadcaster = TransformBroadcaster(self)
                self.publish_tf = True
                self.get_logger().info("TF broadcaster initialized successfully")
            except ImportError:
                self.publish_tf = False
        else:
            self.pose_pub = None
            self.odom_pub = None
            self.publish_tf = False

        
        # Store previous pose for velocity calculation
        self.previous_pose = None
        self.previous_time = None
        
        # intrinsics
        calib = np.loadtxt(calib_file, delimiter=" ")
        fx, fy, cx, cy = calib[:4]
        self.K = np.eye(3)
        self.K[0,0] = fx
        self.K[0,2] = cx
        self.K[1,1] = fy
        self.K[1,2] = cy
        
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.01  # 允许10ms的时间差
        )
        self.ts.registerCallback(self.synchronized_callback)
    
    #TODO: video frame wrong
    def publish_camera_pose(self, frame_id, timestamp, processed_frame_id=None):
        """发布相机位姿到ROS2话题"""
        if self.droid is None or not self.args.publish_pose:
            return
            
        # 获取当前帧数
        if hasattr(self.droid, "video2"):
            video = self.droid.video2
        else:
            video = self.droid.video
            
        current_frame = video.counter.value - 1  # 当前帧索引
        
        # 如果有处理帧ID，使用它作为参考
        if processed_frame_id is not None and processed_frame_id > 0:
            # 使用处理帧ID-1作为当前帧（因为我们刚刚处理了这一帧）
            frame_to_use = min(processed_frame_id - 1, current_frame)
            
            # 如果video counter还没有更新，可能需要等待一下
            if current_frame < 0 and processed_frame_id > 0:
                # 直接使用 processed_frame_id - 1，但要确保不小于0
                frame_to_use = max(0, processed_frame_id - 1)
                self.get_logger().warning(f"Video counter not updated yet, using processed_frame_id-1: {frame_to_use}")
        else:
            frame_to_use = current_frame
            
        print(f"Using frame_to_use: {frame_to_use}")
        
        if frame_to_use < 0:
            self.get_logger().warning(f"Frame index is negative: {frame_to_use}, skipping pose publication")
            return
            
        # 获取位姿数据 [tx, ty, tz, qx, qy, qz, qw]
        try:
            pose_data = video.poses[frame_to_use].cpu().numpy()
            print("==================================================", pose_data)
        except IndexError as e:
            self.get_logger().warning(f"Cannot access pose data for frame {frame_to_use}: {e}")
            return
        # 创建ROS时间戳
        ros_time = self.get_clock().now().to_msg()
        current_time = time.time()
        
        # 计算速度 (如果有前一帧数据)
        linear_vel = [0.0, 0.0, 0.0]
        angular_vel = [0.0, 0.0, 0.0]
        
        if self.previous_pose is not None and self.previous_time is not None:
            dt = current_time - self.previous_time
            if dt > 0:
                # 计算线性速度
                linear_vel[0] = (pose_data[0] - self.previous_pose[0]) / dt
                linear_vel[1] = (pose_data[1] - self.previous_pose[1]) / dt
                linear_vel[2] = (pose_data[2] - self.previous_pose[2]) / dt
                
                # 简单的角速度估计（基于四元数差异）
                # 这里可以做更精确的计算，但作为示例这样就足够了
        
        # 更新前一帧数据
        self.previous_pose = pose_data.copy()
        self.previous_time = current_time
        
        # 发布PoseStamped消息
        if self.pose_pub is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = ros_time
            pose_msg.header.frame_id = self.args.map_frame_id
            pose_msg.pose.position.x = float(pose_data[0])
            pose_msg.pose.position.y = float(pose_data[1])
            pose_msg.pose.position.z = float(pose_data[2])
            pose_msg.pose.orientation.x = float(pose_data[3])
            pose_msg.pose.orientation.y = float(pose_data[4])
            pose_msg.pose.orientation.z = float(pose_data[5])
            pose_msg.pose.orientation.w = float(pose_data[6])
            self.pose_pub.publish(pose_msg)
        
        # 发布Odometry消息
        if self.odom_pub is not None:
            odom_msg = Odometry()
            odom_msg.header.stamp = ros_time
            odom_msg.header.frame_id = self.args.map_frame_id
            odom_msg.child_frame_id = frame_id
            odom_msg.pose.pose.position.x = float(pose_data[0])
            odom_msg.pose.pose.position.y = float(pose_data[1])
            odom_msg.pose.pose.position.z = float(pose_data[2])
            odom_msg.pose.pose.orientation.x = float(pose_data[3])
            odom_msg.pose.pose.orientation.y = float(pose_data[4])
            odom_msg.pose.pose.orientation.z = float(pose_data[5])
            odom_msg.pose.pose.orientation.w = float(pose_data[6])
            
            # 添加速度信息
            odom_msg.twist.twist.linear.x = linear_vel[0]
            odom_msg.twist.twist.linear.y = linear_vel[1]
            odom_msg.twist.twist.linear.z = linear_vel[2]
            odom_msg.twist.twist.angular.x = angular_vel[0]
            odom_msg.twist.twist.angular.y = angular_vel[1]
            odom_msg.twist.twist.angular.z = angular_vel[2]
            self.odom_pub.publish(odom_msg)
        
        # 发布TF变换
        if self.publish_tf:
            try:
                t = TransformStamped()
                t.header.stamp = ros_time
                t.header.frame_id = self.args.map_frame_id
                t.child_frame_id = frame_id
                t.transform.translation.x = float(pose_data[0])
                t.transform.translation.y = float(pose_data[1])
                t.transform.translation.z = float(pose_data[2])
                t.transform.rotation.x = float(pose_data[3])
                t.transform.rotation.y = float(pose_data[4])
                t.transform.rotation.z = float(pose_data[5])
                t.transform.rotation.w = float(pose_data[6])
                self.tf_broadcaster.sendTransform(t)
            except Exception as e:
                self.get_logger().warning(f"Failed to publish TF: {str(e)}")
        
        # # 记录位姿信息（降低频率以避免日志过多）
        # if frame_to_use % 1 == 0:  # 每1帧打印一次
        #     self.get_logger().info(f'Published pose for frame {frame_to_use}: '
        #                           f'pos=({pose_data[0]:.3f}, {pose_data[1]:.3f}, {pose_data[2]:.3f}), '
        #                           f'quat=({pose_data[3]:.3f}, {pose_data[4]:.3f}, {pose_data[5]:.3f}, {pose_data[6]:.3f}), '
        #                           f'vel=({linear_vel[0]:.3f}, {linear_vel[1]:.3f}, {linear_vel[2]:.3f})')

    def get_trajectory_data(self):
        """获取完整的轨迹数据"""
        if self.droid is None:
            return None
            
        # 获取视频数据
        if hasattr(self.droid, "video2"):
            video = self.droid.video2
        else:
            video = self.droid.video
            
        t = video.counter.value
        if t <= 0:
            return None
            
        trajectory_data = {
            "timestamps": video.tstamp[:t].cpu().numpy(),
            "poses": video.poses[:t].cpu().numpy(),  # [N, 7] - [tx, ty, tz, qx, qy, qz, qw]
            "frame_count": t
        }
        
        return trajectory_data

    def save_trajectory_to_file(self, filename):
        """保存轨迹到文件"""
        trajectory_data = self.get_trajectory_data()
        if trajectory_data is None:
            self.get_logger().warning("No trajectory data to save")
            return False
            
        try:
            np.savez(filename, 
                    timestamps=trajectory_data["timestamps"],
                    poses=trajectory_data["poses"],
                    frame_count=trajectory_data["frame_count"])
            self.get_logger().info(f"Trajectory saved to {filename}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory: {str(e)}")
            return False

    def visualize_synchronized_images(self, rgb_image, depth_image, rgb_stamp, depth_stamp):
        """可视化同步的RGB和深度图像"""
        # 计算时间戳差异
        rgb_time = rgb_stamp.sec + rgb_stamp.nanosec * 1e-9
        depth_time = depth_stamp.sec + depth_stamp.nanosec * 1e-9
        time_diff = abs(rgb_time - depth_time) * 1000  # 转换为毫秒
        
        # 创建显示窗口
        display_rgb = cv2.resize(rgb_image, (640, 480))
        
        # 处理深度图像用于显示
        if depth_image is not None:
            # 归一化深度图像到0-255范围用于显示
            if depth_image.max() > depth_image.min():
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                # 显示深度值范围
                valid_depth = depth_image[depth_image > 0]
                if len(valid_depth) > 0:
                    depth_range_text = f"Depth: {valid_depth.min():.1f}-{valid_depth.max():.1f}"
                else:
                    depth_range_text = "No valid depth"
            else:
                depth_colored = np.zeros((*depth_image.shape, 3), dtype=np.uint8)
                depth_range_text = "Invalid depth range"
                
            display_depth = cv2.resize(depth_colored, (640, 480))
            
            # 在图像上添加时间戳信息
            cv2.putText(display_rgb, f'RGB: {rgb_time:.3f}s', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_depth, f'Depth: {depth_time:.3f}s', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_depth, f'Time diff: {time_diff:.1f}ms', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_depth, depth_range_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_depth, f'Scale: {self.args.depth_scale}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 水平拼接显示
            combined = np.hstack([display_rgb, display_depth])
            cv2.imshow('Synchronized RGB-Depth (ApproximateTimeSynchronizer)', combined)
        else:
            cv2.putText(display_rgb, f'RGB: {rgb_time:.3f}s (No Depth)', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Synchronized RGB-Depth (ApproximateTimeSynchronizer)', display_rgb)
        
        cv2.waitKey(1)

    def synchronized_callback(self, rgb_msg, depth_msg):
        try:
            # Convert ROS message to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            # 可视化RGB和深度图像以验证同步效果
            # self.visualize_synchronized_images(rgb_image, depth_image, rgb_msg.header.stamp, depth_msg.header.stamp)
            
            # Process image data
            processed_data = self.process_frame(rgb_image, depth_image)
            
            if processed_data is not None:
                t, image_tensor, depth_tensor, intrinsics = processed_data
                
                # Show image if visualization is enabled
                if not self.args.disable_vis:
                    show_image(image_tensor[0])

                # Initialize DROID (on first frame)
                if self.droid is None:
                    self.args.image_size = [image_tensor.shape[2], image_tensor.shape[3]]
                    self.droid = DroidAsync(self.args) if self.args.asynchronous else Droid(self.args)

                # Track current frame
                self.droid.track(t, image_tensor, depth=depth_tensor, intrinsics=intrinsics)

                # Publish camera pose after tracking (with a small delay to ensure processing is complete)
                self.publish_camera_pose(self.args.camera_frame_id, rgb_msg.header.stamp, processed_frame_id=t)

            else:
                self.get_logger().warning('process_frame returned None')

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def process_frame(self, rgb_image, depth_image):
        """Process RGB and depth images, returning the format needed by DROID-SLAM"""
        
            
        h0, w0, _ = rgb_image.shape

        # Adjust resolution for higher density point cloud
        h1 = int(h0 * np.sqrt((720 * 540) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((720 * 540) / (h0 * w0)))

        rgb_image = cv2.resize(rgb_image, (w1, h1))
        rgb_image = rgb_image[:h1-h1%8, :w1-w1%8]
        rgb_tensor = torch.as_tensor(rgb_image).permute(2, 0, 1)

        # Process depth image
        if depth_image is not None:
            
            # Depth image processing
            depth = torch.as_tensor(depth_image.astype(np.float32) / self.args.depth_scale)

            # Apply depth range limits
            depth = torch.clamp(depth, self.args.min_depth, self.args.max_depth)

            # Filter invalid depth values
            depth = torch.where(depth > 0, depth, torch.tensor(0.0))

            # Adjust depth image size
            depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
            depth = depth[:h1-h1%8, :w1-w1%8]

        else:
            depth = None

        # Adjust intrinsics
        intrinsics = torch.as_tensor([self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        self.frame_count += 1
        return self.frame_count, rgb_tensor[None], depth, intrinsics

    def terminate_slam(self):
        """Terminate SLAM and save results"""
        if self.droid is not None:
            self.get_logger().info('Terminating DROID-SLAM...')
            # Note: Since this is a real-time stream, we cannot pass image_stream to terminate
            # Need to check if DROID's terminate method supports no-argument calls
            try:
                traj_est = self.droid.terminate()
                self.get_logger().info('DROID-SLAM terminated successfully')
                return traj_est
            except Exception as e:
                self.get_logger().error(f'Error terminating DROID-SLAM: {str(e)}')
                return None
        return None

def save_reconstruction(droid, save_path):
    """Save reconstruction results"""
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)

def main():
    parser = argparse.ArgumentParser(description='DROID-SLAM with ROS2 input')

    # ROS2 specific arguments
    parser.add_argument("--rgb_topic", type=str, required=True, help="ROS2 RGB image topic")
    parser.add_argument("--depth_topic", type=str, required=True, help="ROS2 depth image topic")
    parser.add_argument("--calib", type=str, required=True, help="path to calibration file")

    # Depth processing arguments
    parser.add_argument("--depth_scale", type=float, default=1.0, help="depth scale factor")
    parser.add_argument("--min_depth", type=float, default=0.1, help="minimum valid depth in meters")
    parser.add_argument("--max_depth", type=float, default=100000.0, help="maximum valid depth in meters")

    # DROID-SLAM arguments
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[480, 640])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames within this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal suppression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    
    # Pose publishing arguments
    parser.add_argument("--publish_pose", action="store_true", help="enable pose publishing to ROS topics")
    parser.add_argument("--camera_frame_id", type=str, default="camera_link", help="frame ID for camera in TF tree")
    parser.add_argument("--map_frame_id", type=str, default="map", help="frame ID for map/world coordinate system")
    parser.add_argument("--trajectory_output", type=str, help="path to save trajectory data (numpy format)")
    
    args = parser.parse_args()

    # Set DROID arguments
    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    # Enable upsampling if high-resolution depth is needed
    if args.reconstruction_path is not None:
        args.upsample = True

    # Initialize ROS2
    rclpy.init()

    # Create DROID-SLAM node
    slam_node = DroidSlamNode(args.rgb_topic, args.depth_topic, args.calib, args)
    
    try:
        print(f"Starting DROID-SLAM, subscribing to topics:")
        print(f"  RGB: {args.rgb_topic}")
        print(f"  Depth: {args.depth_topic}")
        print("Press Ctrl+C to stop...")

        # Run the node
        rclpy.spin(slam_node)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping DROID-SLAM...")

    finally:
        # Terminate SLAM and save results
        traj_est = slam_node.terminate_slam()
        
        # Save trajectory data if requested
        if args.trajectory_output is not None:
            slam_node.save_trajectory_to_file(args.trajectory_output)
        
        if args.reconstruction_path is not None and slam_node.droid is not None:
            save_reconstruction(slam_node.droid, args.reconstruction_path)
            print(f"Reconstruction results saved to: {args.reconstruction_path}")

        # Clean up ROS2 resources
        slam_node.destroy_node()
        rclpy.shutdown()
        print("DROID-SLAM stopped")

if __name__ == '__main__':
    main()
