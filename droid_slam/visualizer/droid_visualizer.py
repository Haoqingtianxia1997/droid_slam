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

# 添加 matplotlib 用于 3D 轨迹可视化
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import warnings

# 过滤matplotlib的坐标轴警告
warnings.filterwarnings("ignore", message=".*fixed x limits to fulfill fixed data aspect.*")

# 添加sklearn用于DBSCAN过滤
from sklearn.cluster import DBSCAN

# 添加形态学操作
from scipy.ndimage import binary_erosion, binary_dilation

# 添加UDP传输相关模块
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
    """将四元数(w,x,y,z)转换为旋转矩阵"""
    # 归一化四元数
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 构建旋转矩阵
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])
    return R


def quaternion_multiply(q1, q2):
    """四元数乘法 q1 * q2，四元数格式为[w,x,y,z]"""
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
        """初始化点云累积相关变量"""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.last_save_time = time.time()
        self.save_interval = 15.0  # 15秒保存一次
        self.save_counter = 0
        
        # 存储当前帧的2D点云数据，用于实时可视化
        self.current_frame_2d_points = None
        self.current_frame_2d_colors = None
        
        # Occupancy Grid相关变量（动态大小）
        self.occupancy_grid = None  # 将根据点云范围动态创建
        self.grid_origin = None     # 网格原点（动态计算）
        self.grid_size = None       # 网格尺寸（动态计算）
    
    def accumulate_point_cloud(self):
        """累积当前帧的点云数据"""
        try:
            # 从GPU buffer获取点云和颜色数据
            points = self.pts_buffer.read()
            colors = self.clr_buffer.read()
            valid = self.valid_buffer.read()
            
            # 转为numpy数组
            num_points = len(points) // (3 * 4)  # float32
            points_np = np.frombuffer(points, dtype=np.float32).reshape(num_points, 3)
            colors_np = np.frombuffer(colors, dtype=np.float32).reshape(num_points, 3)
            valid_np = np.frombuffer(valid, dtype=np.float32)
            
            # 只保留有效点
            mask = valid_np > 0
            valid_points = points_np[mask]
            valid_colors = colors_np[mask]
            
            if len(valid_points) > 0:
                # 应用相机到世界坐标系的旋转变换
                if DroidVisualizer._latest_angle_data is not None:
                    torso_quat_data = DroidVisualizer._latest_angle_data.get('torso_to_world_quat', {})
                    torso_to_world_quat = np.array([
                        torso_quat_data.get('w', 1.0),
                        torso_quat_data.get('x', 0.0),
                        torso_quat_data.get('y', 0.0),
                        torso_quat_data.get('z', 0.0)
                    ])
                    
                    # 相机到torso的变换（根据您提供的四元数）
                    camera_to_torso_quat = np.array([0.45451948, -0.54167522, 0.54167522, -0.45451948])
                    
                    # 计算相机到世界坐标系的四元数：camera_to_world = torso_to_world * camera_to_torso
                    camera_to_world_quat = quaternion_multiply(torso_to_world_quat, camera_to_torso_quat)
                    
                    # 将四元数转换为旋转矩阵
                    R_camera_to_world = quaternion_to_rotation_matrix(
                        camera_to_world_quat[0], camera_to_world_quat[1], 
                        camera_to_world_quat[2], camera_to_world_quat[3]
                    )
                    
                    # 对点云应用旋转变换
                    valid_points = (R_camera_to_world @ valid_points.T).T

                # 1. 按z轴高度过滤点云（保留-0.8到0.5之间的点）
                z_mask = (valid_points[:, 2] >= -0.8) & (valid_points[:, 2] <= 0.5)
                filtered_points = valid_points[z_mask]
                filtered_colors = valid_colors[z_mask]
                
                if len(filtered_points) > 0:
                    # 2. 投影到xy平面（消除z轴）
                    xy_points = filtered_points[:, :2]  # 只取x,y坐标
                    
                    # 3. 网格降采样
                    grid_size = 0.05  # 5cm网格大小，可以根据需要调整
                    
                    # 计算网格索引
                    grid_indices = np.floor(xy_points / grid_size).astype(int)
                    
                    # 使用字典存储每个网格的点
                    grid_dict = {}
                    for i, (grid_x, grid_y) in enumerate(grid_indices):
                        grid_key = (grid_x, grid_y)
                        if grid_key not in grid_dict:
                            grid_dict[grid_key] = []
                        grid_dict[grid_key].append(i)
                    
                    # 对每个网格计算代表点
                    downsampled_points = []
                    downsampled_colors = []
                    
                    for grid_key, point_indices in grid_dict.items():
                        # 计算网格中心点坐标（只有xy坐标，z设为0）
                        grid_center_x = (grid_key[0] + 0.5) * grid_size
                        grid_center_y = (grid_key[1] + 0.5) * grid_size
                        
                        # 计算该网格内所有点的平均颜色
                        grid_colors = filtered_colors[point_indices]
                        avg_color = np.mean(grid_colors, axis=0)
                        
                        # 创建代表点（使用网格中心的xy坐标，z坐标设为0）
                        representative_point = np.array([grid_center_x, grid_center_y, 0.0])
                        
                        downsampled_points.append(representative_point)
                        downsampled_colors.append(avg_color)
                    
                    if len(downsampled_points) > 0:
                        downsampled_points = np.array(downsampled_points)
                        downsampled_colors = np.array(downsampled_colors)
                        
                        # 存储当前帧的降采样点云用于2D实时可视化
                        self.current_frame_2d_points = downsampled_points
                        self.current_frame_2d_colors = downsampled_colors
                        
                        # print(f"降采样：原始点数 {len(filtered_points)} -> 降采样后 {len(downsampled_points)}")
                
        except Exception as e:
            print(f"累积点云失败: {e}")
    
    def save_accumulated_point_cloud(self, filename_prefix="accumulated_pointcloud"):
        """保存累积的点云为PLY文件"""
        try:
            if len(self.current_frame_2d_points) == 0:
                print("没有可保存的点云数据。请先累积点云。")
                return
                
            # 合并所有累积的点云（已经在累积时应用了坐标变换）
            all_points = self.current_frame_2d_points
            all_colors = self.current_frame_2d_colors
            
            # 去重：使用点的坐标作为唯一标识
            # 将坐标四舍五入到网格精度来确保一致性
            grid_size = 0.05  # 与accumulate_point_cloud中的grid_size保持一致
            rounded_points = np.round(all_points / grid_size) * grid_size
            
            # 使用更简单安全的去重方法：基于字符串表示
            # 将每个点转换为字符串作为唯一键
            point_strings = [f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}" for p in rounded_points]
            
            # 创建去重字典
            unique_dict = {}
            for i, point_str in enumerate(point_strings):
                if point_str not in unique_dict:
                    unique_dict[point_str] = i
            
            # 获取唯一点的索引
            unique_indices = list(unique_dict.values())
            
            # 获取去重后的点和颜色
            unique_points = all_points[unique_indices]
            unique_colors = all_colors[unique_indices]
            
            # 颜色转为0-255
            unique_colors = (unique_colors * 255).astype(np.uint8)
            
            print(f"去重前点数: {len(all_points)}, 去重后点数: {len(unique_points)}")
            
            # 生成带时间戳的文件名
            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.ply"
            
            # 写PLY文件
            with open(filename, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {unique_points.shape[0]}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(unique_points, unique_colors):
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
            
            print(f"累积点云已保存到 {filename}, 总点数: {unique_points.shape[0]}")
            self.save_counter += 1
            
        except Exception as e:
            print(f"保存累积点云失败: {e}")
    
    def create_occupancy_grid_from_points(self, points):
        """从点云数据创建动态occupancy grid，每个点周围的区域都被标记为占用"""
        if points is None or len(points) == 0:
            return
        
        # 计算点云的边界范围
        margin = 1.0  # 边界扩展1米
        x_min = np.min(points[:, 0]) - margin
        x_max = np.max(points[:, 0]) + margin
        y_min = np.min(points[:, 1]) - margin
        y_max = np.max(points[:, 1]) + margin
        
        # 更新网格原点和尺寸
        self.grid_origin = np.array([x_min, y_min])
        
        # 计算网格尺寸（向上取整）
        grid_width = int(np.ceil((x_max - x_min) / self._occupancy_grid_resolution))
        grid_height = int(np.ceil((y_max - y_min) / self._occupancy_grid_resolution))
        self.grid_size = (grid_height, grid_width)  # (rows, cols)
        
        # 创建新的占用网格（全部初始化为自由空间）
        self.occupancy_grid = np.zeros(self.grid_size, dtype=np.uint8)
        
        # 计算每个点的占用半径对应的网格像素数
        radius_pixels = int(np.ceil(self._occupancy_point_radius / self._occupancy_grid_resolution))
        
        # 为每个点标记周围的占用区域
        for point in points:
            # 世界坐标到网格坐标的转换
            grid_x = int((point[0] - self.grid_origin[0]) / self._occupancy_grid_resolution)
            grid_y = int((point[1] - self.grid_origin[1]) / self._occupancy_grid_resolution)
            
            # 在点周围的区域内标记占用
            for dy in range(-radius_pixels, radius_pixels + 1):
                for dx in range(-radius_pixels, radius_pixels + 1):
                    # 计算距离点中心的距离
                    distance = np.sqrt(dx*dx + dy*dy) * self._occupancy_grid_resolution
                    
                    # 如果在占用半径内
                    if distance <= self._occupancy_point_radius:
                        new_grid_x = grid_x + dx
                        new_grid_y = grid_y + dy
                        
                        # 检查是否在网格范围内
                        if (0 <= new_grid_x < self.grid_size[1] and 
                            0 <= new_grid_y < self.grid_size[0]):
                            # 标记为占用（100表示完全占用）
                            self.occupancy_grid[new_grid_y, new_grid_x] = 100
        
        # 应用形态学操作（先腐蚀再膨胀）
        if self._enable_morphology and np.any(self.occupancy_grid > 0):
            try:
                # 将占用网格转换为二值图像（0或1）
                binary_grid = (self.occupancy_grid > 50).astype(bool)
                
                # 创建结构元素（形态学操作核）
                kernel_size = self._morphology_kernel_size
                kernel = np.ones((kernel_size, kernel_size), dtype=bool)
                
                # 先腐蚀（去除小噪声点和细小连接）
                if self._erosion_iterations > 0:
                    for _ in range(self._erosion_iterations):
                        binary_grid = binary_erosion(binary_grid, structure=kernel)
                
                # 再膨胀（恢复大小并平滑边缘）
                if self._dilation_iterations > 0:
                    for _ in range(self._dilation_iterations):
                        binary_grid = binary_dilation(binary_grid, structure=kernel)
                
                # 将二值结果转换回占用网格
                self.occupancy_grid = (binary_grid * 100).astype(np.uint8)
                
            except Exception as e:
                print(f"Warning: Morphological operations failed: {e}")
                # 如果形态学操作失败，继续使用原始网格
        
        # 添加安全冗余区域
        if self._enable_safety_margin and np.any(self.occupancy_grid > 50):
            try:
                # 获取当前的障碍物区域
                obstacle_mask = (self.occupancy_grid > 50)
                
                # 计算安全冗余距离对应的像素数
                safety_margin_pixels = int(np.ceil(self._safety_margin_distance / self._occupancy_grid_resolution))
                
                # 创建圆形结构元素用于膨胀操作
                y_coords, x_coords = np.ogrid[:2*safety_margin_pixels+1, :2*safety_margin_pixels+1]
                center = safety_margin_pixels
                circular_mask = ((x_coords - center)**2 + (y_coords - center)**2) <= safety_margin_pixels**2
                
                # 对障碍物区域进行膨胀，生成安全冗余区域
                safety_margin_mask = binary_dilation(obstacle_mask, structure=circular_mask)
                
                # 创建新的占用网格，包含三种状态：
                # 100 = 完全占用（障碍物）
                # 50 = 安全冗余区域（浅色）
                # 0 = 自由空间
                new_occupancy_grid = np.zeros_like(self.occupancy_grid)
                new_occupancy_grid[safety_margin_mask] = 50  # 安全冗余区域
                new_occupancy_grid[obstacle_mask] = 100      # 障碍物区域（覆盖冗余区域）
                
                self.occupancy_grid = new_occupancy_grid
                
            except Exception as e:
                print(f"Warning: Safety margin generation failed: {e}")
                # 如果冗余区域生成失败，继续使用现有网格
    
    def check_auto_save(self):
        """检查是否需要自动保存"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_accumulated_point_cloud()
            self.last_save_time = current_time
    title = "Droid Visualizer"
    _depth_video1 = None
    _depth_video2 = None

    _refresh_rate = 5
    _filter_threshold = 0.02
    _filter_count = 2
    
    # 添加类变量来存储轨迹数据
    _latest_trajectory_data = None
    
    # UDP相关配置
    _udp_socket = None
    _udp_host = "127.0.0.1"
    _udp_port = 12346
    
    # 角度数据UDP接收相关
    _angle_udp_socket = None
    _angle_udp_port = 12347
    _latest_angle_data = None

    # 2D点云实时可视化配置
    _enable_2d_plot = True
    _2d_filter_outliers = False
    _2d_eps = 0.1
    _2d_min_samples = 5
    _2d_point_size = 3.0

    # Occupancy Grid配置
    _enable_occupancy_grid = True
    _occupancy_grid_resolution = 0.05  # 5cm/pixel
    _occupancy_point_radius = 0.15  # 每个点周围的占用半径（米）
    
    # 形态学操作配置
    _enable_morphology = True        # 启用形态学操作（腐蚀+膨胀）
    _erosion_iterations = 1          # 腐蚀迭代次数
    _dilation_iterations = 1         # 膨胀迭代次数
    _morphology_kernel_size = 7     # 形态学操作核大小
    
    # 冗余区域配置
    _enable_safety_margin = True     # 启用安全冗余区域
    _safety_margin_distance = 0.3   # 安全冗余距离（米）

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = False
        
        # 初始化点云累积功能
        self.__init_point_cloud_accumulation()
        
        # 初始化UDP socket
        self._init_udp_socket()
        
        # 初始化角度UDP接收器
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

        # 初始化 3D 轨迹可视化
        self._init_3d_trajectory_plot()

        # 初始化 2D 点云实时可视化
        if self._enable_2d_plot:
            try:
                self._init_2d_pointcloud_plot()
            except Exception as e:
                print(f"Warning: Failed to initialize 2D visualization: {e}")
                self._enable_2d_plot = False  # 禁用2D可视化以避免后续错误

        # 初始化 Occupancy Grid 可视化
        if self._enable_occupancy_grid:
            try:
                self._init_occupancy_grid_plot()
            except Exception as e:
                print(f"Warning: Failed to initialize occupancy grid visualization: {e}")
                self._enable_occupancy_grid = False

    def _init_udp_socket(self):
        """初始化UDP socket用于发送轨迹数据"""
        try:
            DroidVisualizer._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        except Exception as e:

            DroidVisualizer._udp_socket = None
    
    def _init_angle_udp_receiver(self):
        """初始化角度UDP接收器"""
        try:
            DroidVisualizer._angle_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            DroidVisualizer._angle_udp_socket.bind((DroidVisualizer._udp_host, DroidVisualizer._angle_udp_port))
            DroidVisualizer._angle_udp_socket.settimeout(0.001)  # 非阻塞接收
            
            # 启动角度数据接收线程
            import threading
            self.angle_thread = threading.Thread(target=self._angle_receiver_thread, daemon=True)
            self.angle_thread.start()
            print(f"Angle UDP receiver started on port {DroidVisualizer._angle_udp_port}")
            
        except Exception as e:
            print(f"Failed to initialize angle UDP receiver: {e}")
            DroidVisualizer._angle_udp_socket = None
    
    def _angle_receiver_thread(self):
        """四元数数据接收线程"""
        while True:
            try:
                if DroidVisualizer._angle_udp_socket:
                    data, addr = DroidVisualizer._angle_udp_socket.recvfrom(1024)
                    json_data = data.decode('utf-8')
                    quaternion_data = json.loads(json_data)
                    DroidVisualizer._latest_angle_data = quaternion_data
                    # print(f"Received quaternion data: {quaternion_data}")
            except socket.timeout:
                pass
            except Exception as e:
                # print(f"Error in angle receiver thread: {e}")
                pass
            time.sleep(0.001)

    def _send_trajectory_data_udp(self, trajectory_data):
        """通过UDP发送轨迹数据"""
        if DroidVisualizer._udp_socket is None:
            return
            
        try:
            # 将数据序列化为JSON
            json_data = json.dumps(trajectory_data)
            data_bytes = json_data.encode('utf-8')
            
            # 发送数据
            DroidVisualizer._udp_socket.sendto(data_bytes, (DroidVisualizer._udp_host, DroidVisualizer._udp_port))

        except Exception as e:
            print(f"Failed to send trajectory data via UDP: {e}")

    def _init_3d_trajectory_plot(self):
        """初始化 3D 轨迹可视化"""
        # 设置 matplotlib 非阻塞模式
        plt.ion()
        
        # 创建 3D 图
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title('DROID-SLAM real time trajectory(3D)')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 初始化轨迹线和点
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Current Position')

        # 设置坐标轴等比例
        self.ax.set_box_aspect([1,1,1])
        self.ax.legend()
        
        # 存储轨迹数据
        self.trajectory_points = []
        
        plt.show(block=False)

    def _update_3d_trajectory_from_cam_pts(self, cam_pts, frame_count):
        """从相机变换后的点云数据更新 3D 轨迹图"""
        try:
            if cam_pts.shape[0] > 0:
                # cam_pts 是 [frame_count * 20, 3] 的形状
                # 其中 frame_count 是帧数，20 是每个相机框架的线段端点数
                points_per_frame = 20  # CAM_SEGMENTS 有20个点（10条线段×2个端点）
                
                if cam_pts.shape[0] >= points_per_frame * frame_count:
                    # 重新组织数据为 [frame_count, points_per_frame, 3]
                    cam_pts_reshaped = cam_pts[:frame_count * points_per_frame].reshape(frame_count, points_per_frame, 3)
                    
                    # 计算每帧相机的中心点作为轨迹点
                    camera_positions = []
                    for frame_idx in range(frame_count):
                        # 计算当前帧所有20个点的平均位置作为相机中心
                        camera_center_slam = np.mean(cam_pts_reshaped[frame_idx, :, :], axis=0)  # SLAM坐标系中的相机位置
                        
        
                        if DroidVisualizer._latest_angle_data is not None:
                            torso_quat_data = DroidVisualizer._latest_angle_data.get('torso_to_world_quat', {})
                            torso_to_world_quat = np.array([
                                torso_quat_data.get('w', 1.0),
                                torso_quat_data.get('x', 0.0),
                                torso_quat_data.get('y', 0.0),
                                torso_quat_data.get('z', 0.0)
                            ])
                        
                        # 相机到torso的变换（根据您提供的四元数）
                        camera_to_torso_quat = np.array([0.45451948, -0.54167522, 0.54167522, -0.45451948])
                        
                        # 计算相机到世界坐标系的四元数：camera_to_world = torso_to_world * camera_to_torso
                        camera_to_world_quat = quaternion_multiply(torso_to_world_quat, camera_to_torso_quat)
                        # 将四元数转换为旋转矩阵
                        R_camera_to_world = quaternion_to_rotation_matrix(
                            camera_to_world_quat[0], camera_to_world_quat[1], 
                            camera_to_world_quat[2], camera_to_world_quat[3]
                        )
                        
                        # 关键修改：camera_center_slam是SLAM相机坐标系中的位置，需要正确变换
                        # 将SLAM相机坐标系中的相机位置变换到世界坐标系
                        camera_position_world = R_camera_to_world @ camera_center_slam
                        
                        camera_positions.append(camera_position_world)
                    
                    camera_positions = np.array(camera_positions)
                    
                    # 更新轨迹数据
                    self.trajectory_points = camera_positions.tolist()
                    
                    # 存储最新的轨迹数据到类变量（供外部访问）
                    if len(self.trajectory_points) > 0:
                        # 提取x和z坐标（3D plot中的x对应真实x，z对应真实y）
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
                        
                        # 通过UDP发送轨迹数据
                        self._send_trajectory_data_udp(trajectory_data)
                    
                    # 更新轨迹线
                    if len(self.trajectory_points) > 1:
                        x_data = [p[0] for p in self.trajectory_points]
                        y_data = [p[1] for p in self.trajectory_points]
                        z_data = [p[2] for p in self.trajectory_points]
                        
                        self.trajectory_line.set_data_3d(x_data, y_data, z_data)
                        
                        # 更新当前位置点
                        self.current_point.set_data_3d([x_data[-1]], [y_data[-1]], [z_data[-1]])
                        
                        # 动态调整坐标轴范围，保持各轴分度值相同
                        margin = 0.5  # 边距
                        
                        # 计算每个轴的数据范围
                        x_min, x_max = min(x_data) - margin, max(x_data) + margin
                        y_min, y_max = min(y_data) - margin, max(y_data) + margin
                        z_min, z_max = min(z_data) - margin, max(z_data) + margin
                        
                        # 找到最大的数据范围
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        max_range = max(x_range, y_range, z_range)
                        
                        # 计算每个轴的中心点
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        z_center = (z_min + z_max) / 2
                        
                        # 使用最大范围设置所有轴的范围，保持相同的分度值
                        half_range = max_range / 2
                        self.ax.set_xlim(x_center - half_range, x_center + half_range)
                        self.ax.set_ylim(y_center - half_range, y_center + half_range)
                        self.ax.set_zlim(z_center - half_range, z_center + half_range)
                    
                    # 刷新图形
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    
        except Exception as e:
            print(f"Error when updating 3D trajectory from cam_pts: {e}")

    def _init_2d_pointcloud_plot(self):
        """初始化 2D 点云实时可视化"""
        try:
            # 设置 matplotlib 非阻塞模式
            plt.ion()
            
            # 创建 2D 图
            self.fig_2d = plt.figure(figsize=(12, 10))
            self.ax_2d = self.fig_2d.add_subplot(111)
            self.ax_2d.set_title('DROID-SLAM Real-time 2D Point Cloud', fontsize=14)
            self.ax_2d.set_xlabel('X (m)', fontsize=12)
            self.ax_2d.set_ylabel('Y (m)', fontsize=12)
            self.ax_2d.grid(True, alpha=0.3)
            self.ax_2d.set_aspect('equal', adjustable='box')  # 使用adjustable='box'而不是axis('equal')
            
            # 初始化散点图
            self.pointcloud_scatter = self.ax_2d.scatter([], [], s=self._2d_point_size, alpha=0.6, edgecolors='none', c='blue', label='Point Cloud')
            
            # 初始化相机位置点
            self.camera_position_scatter = self.ax_2d.scatter([], [], s=50, c='red', marker='o', label='Camera Position')
            
            # 添加图例
            self.ax_2d.legend()
            
            # 初始化统计信息文本
            self.info_text_2d = self.ax_2d.text(0.02, 0.98, '', transform=self.ax_2d.transAxes, 
                                               verticalalignment='top', 
                                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 存储当前的2D点云数据
            self.current_2d_points = None
            self.current_2d_colors = None
            
            plt.show(block=False)
            print("2D point cloud visualization initialized")
            
        except Exception as e:
            print(f"Failed to initialize 2D point cloud plot: {e}")

    def _update_2d_pointcloud(self, points, colors, camera_position=None):
        """更新实时2D点云可视化
        
        Args:
            points: 点云数据 (N, 3)，将使用x,y坐标
            colors: 颜色数据 (N, 3)
            camera_position: 相机在世界坐标系中的位置 (3,)
        """
        try:
            if not self._enable_2d_plot or points is None or len(points) == 0:
                return
            
            # 保存原始数据
            original_point_count = len(points)
            filtered_points = points.copy()
            filtered_colors = colors.copy()
            
            # 使用DBSCAN过滤离群点（可选）
            if self._2d_filter_outliers and len(points) > self._2d_min_samples:
                # 只对xy坐标进行聚类
                xy_points = points[:, :2]
                
                # 应用DBSCAN聚类
                dbscan = DBSCAN(eps=self._2d_eps, min_samples=self._2d_min_samples)
                cluster_labels = dbscan.fit_predict(xy_points)
                
                # 过滤掉离群点（标签为-1的点）
                inlier_mask = cluster_labels != -1
                filtered_points = points[inlier_mask]
                filtered_colors = colors[inlier_mask]
                
                # 如果DBSCAN过滤掉了所有点，则回退到原始数据
                if len(filtered_points) == 0:
                    print("Warning: DBSCAN filtered out all points, using original data")
                    filtered_points = points.copy()
                    filtered_colors = colors.copy()
            
            # 存储当前数据
            self.current_2d_points = filtered_points
            self.current_2d_colors = filtered_colors
            
            # 更新散点图数据（只使用x,y坐标）
            xy_coords = filtered_points[:, :2]
            
            # 更新散点图
            self.pointcloud_scatter.set_offsets(xy_coords)
            if len(filtered_colors) > 0:
                self.pointcloud_scatter.set_color(filtered_colors)
            
            # 更新相机位置
            if camera_position is not None:
                self.camera_position_scatter.set_offsets([[camera_position[0], camera_position[1]]])
            
            # 更新统计信息
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
            
            # 动态调整坐标轴范围
            if len(xy_coords) > 0:
                margin = 0.5
                x_min, x_max = np.min(xy_coords[:, 0]) - margin, np.max(xy_coords[:, 0]) + margin
                y_min, y_max = np.min(xy_coords[:, 1]) - margin, np.max(xy_coords[:, 1]) + margin
                
                # 如果有相机位置，确保相机位置也在视野内
                if camera_position is not None:
                    x_min = min(x_min, camera_position[0] - margin)
                    x_max = max(x_max, camera_position[0] + margin)
                    y_min = min(y_min, camera_position[1] - margin)
                    y_max = max(y_max, camera_position[1] + margin)
                
                # 计算最大范围以保持等比例
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)
                
                # 计算中心点
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # 使用最大范围设置等比例坐标轴
                half_range = max_range / 2
                self.ax_2d.set_xlim(x_center - half_range, x_center + half_range)
                self.ax_2d.set_ylim(y_center - half_range, y_center + half_range)
            
            # 刷新图形
            self.fig_2d.canvas.draw()
            self.fig_2d.canvas.flush_events()
            
        except Exception as e:
            print(f"Error updating 2D point cloud: {e}")

    def _init_occupancy_grid_plot(self):
        """初始化 Occupancy Grid 可视化"""
        try:
            # 设置 matplotlib 非阻塞模式
            plt.ion()
            
            # 创建 Occupancy Grid 图
            self.fig_occupancy = plt.figure(figsize=(10, 10))
            self.ax_occupancy = self.fig_occupancy.add_subplot(111)
            self.ax_occupancy.set_title('DROID-SLAM Real-time Occupancy Grid Map', fontsize=14)
            self.ax_occupancy.set_xlabel('X (m)', fontsize=12)
            self.ax_occupancy.set_ylabel('Y (m)', fontsize=12)
            self.ax_occupancy.set_aspect('equal', adjustable='box')
            
            # 初始化为空的占用网格图像（将在第一次更新时设置）
            self.occupancy_image = None
            
            # 添加网格
            self.ax_occupancy.grid(True, alpha=0.3)
            
            # 初始化机器人位置点
            self.robot_position_scatter = self.ax_occupancy.scatter(
                [], [], s=100, c='red', marker='o', label='Robot Position', zorder=5
            )
            
            self.ax_occupancy.legend()
            
            # 初始化统计信息文本
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
        """更新Occupancy Grid可视化（动态范围，仿照2D点云的更新模式）
        
        Args:
            points: 点云数据 (N, 3)，将使用x,y坐标
            robot_position: 机器人在世界坐标系中的位置 (3,)
        """
        try:
            if not self._enable_occupancy_grid or points is None or len(points) == 0:
                return
            
            # 从点云创建动态occupancy grid
            self.create_occupancy_grid_from_points(points)
            
            # 如果网格还没有创建，退出
            if self.occupancy_grid is None or self.grid_origin is None:
                return
            
            # 计算网格的世界坐标范围
            x_extent = [self.grid_origin[0], 
                       self.grid_origin[0] + self.grid_size[1] * self._occupancy_grid_resolution]
            y_extent = [self.grid_origin[1], 
                       self.grid_origin[1] + self.grid_size[0] * self._occupancy_grid_resolution]
            
            # 如果这是第一次创建图像或网格尺寸发生变化
            if (self.occupancy_image is None or 
                self.occupancy_image.get_array().shape != self.occupancy_grid.shape):
                
                # 清除之前的图像
                if self.occupancy_image is not None:
                    self.occupancy_image.remove()
                
                # 创建新的占用网格图像
                self.occupancy_image = self.ax_occupancy.imshow(
                    self.occupancy_grid, 
                    cmap='RdYlBu_r',  # 使用红-黄-蓝反向色彩映射：红色=障碍物，黄色=冗余区域，蓝色=自由空间
                    origin='lower',
                    extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                    vmin=0, vmax=100
                )
                
                # 重新添加colorbar（如果还没有）
                if not hasattr(self, 'occupancy_colorbar') or self.occupancy_colorbar is None:
                    self.occupancy_colorbar = plt.colorbar(self.occupancy_image, ax=self.ax_occupancy)
                    self.occupancy_colorbar.set_label('Occupancy (0=free, 50=safety margin, 100=occupied)', fontsize=12)
            else:
                # 更新现有图像的数据和范围
                self.occupancy_image.set_array(self.occupancy_grid)
                self.occupancy_image.set_extent([x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
            
            # 动态调整坐标轴范围（类似2D点云）
            margin = 0.5
            self.ax_occupancy.set_xlim(x_extent[0] - margin, x_extent[1] + margin)
            self.ax_occupancy.set_ylim(y_extent[0] - margin, y_extent[1] + margin)
            
            # 更新机器人位置
            if robot_position is not None:
                self.robot_position_scatter.set_offsets([[robot_position[0], robot_position[1]]])
            
            # 更新统计信息
            occupied_cells = np.sum(self.occupancy_grid == 100)      # 完全占用
            safety_margin_cells = np.sum(self.occupancy_grid == 50)  # 安全冗余区域
            free_cells = np.sum(self.occupancy_grid == 0)            # 自由空间
            total_cells = self.occupancy_grid.size
            
            occupied_ratio = occupied_cells / total_cells * 100
            safety_ratio = safety_margin_cells / total_cells * 100
            free_ratio = free_cells / total_cells * 100
            
            info_text = f"Occupied: {occupied_cells} ({occupied_ratio:.1f}%)"
            info_text += f"\nSafety Margin: {safety_margin_cells} ({safety_ratio:.1f}%)"
            info_text += f"\nFree: {free_cells} ({free_ratio:.1f}%)"
            info_text += f"\nTotal: {total_cells}"
            info_text += f"\nResolution: {self._occupancy_grid_resolution}m/pixel"
            info_text += f"\nGrid Size: {self.grid_size[1]}x{self.grid_size[0]}"
            info_text += f"\nCoverage: {self.grid_size[1]*self._occupancy_grid_resolution:.1f}m x {self.grid_size[0]*self._occupancy_grid_resolution:.1f}m"
            info_text += f"\nPoint Radius: {self._occupancy_point_radius}m"
            
            # 添加形态学操作信息
            if self._enable_morphology:
                info_text += f"\nMorphology: E{self._erosion_iterations}→D{self._dilation_iterations} (K={self._morphology_kernel_size})"
            else:
                info_text += f"\nMorphology: Disabled"
            
            # 添加安全冗余区域信息
            if self._enable_safety_margin:
                info_text += f"\nSafety Margin: {self._safety_margin_distance}m"
            else:
                info_text += f"\nSafety Margin: Disabled"
            
            self.info_text_occupancy.set_text(info_text)
            
            # 刷新图形
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
            
            # 检查poses和disps是否包含有效数据
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
            
            # 确保所有张量的尺寸匹配
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
            
            # 累积点云数据
            self.accumulate_point_cloud()
            
            # 检查是否需要自动保存
            self.check_auto_save()

            # 更新2D点云可视化
            if self._enable_2d_plot:
                try:
                    # 直接使用accumulate_point_cloud中处理好的降采样点云数据
                    current_points = getattr(self, 'current_frame_2d_points', None)
                    current_colors = getattr(self, 'current_frame_2d_colors', None)
                    
                    # 获取当前相机位置（从最后一个相机位置）
                    camera_position = None
                    if cam_pts.shape[0] > 0 and DroidVisualizer._latest_trajectory_data is not None:
                        latest_pos = DroidVisualizer._latest_trajectory_data.get('latest_position')
                        if latest_pos:
                            camera_position = np.array([latest_pos['x'], latest_pos['y'], latest_pos['z']])
                    
                    # 更新2D可视化
                    self._update_2d_pointcloud(current_points, current_colors, camera_position)
                except Exception as e:
                    # 2D可视化错误不应该影响主SLAM流程
                    print(f"Warning: 2D visualization update failed: {e}")
                    pass

            # 更新Occupancy Grid可视化（完全仿照2D点云的调用方式）
            if self._enable_occupancy_grid:
                try:
                    # 直接使用accumulate_point_cloud中处理好的降采样点云数据
                    current_points = getattr(self, 'current_frame_2d_points', None)
                    
                    # 获取当前机器人位置（从最后一个相机位置）
                    robot_position = None
                    if cam_pts.shape[0] > 0 and DroidVisualizer._latest_trajectory_data is not None:
                        latest_pos = DroidVisualizer._latest_trajectory_data.get('latest_position')
                        if latest_pos:
                            robot_position = np.array([latest_pos['x'], latest_pos['y'], latest_pos['z']])
                    
                    # 更新occupancy grid可视化
                    self._update_occupancy_grid_plot(current_points, robot_position)
                except Exception as e:
                    # Occupancy grid可视化错误不应该影响主SLAM流程
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
        # 清理UDP socket
        if DroidVisualizer._udp_socket:
            try:
                DroidVisualizer._udp_socket.close()
                print("UDP socket closed")
            except Exception as e:
                print(f"Error closing UDP socket: {e}")
        
        # 清理角度UDP socket
        if DroidVisualizer._angle_udp_socket:
            try:
                DroidVisualizer._angle_udp_socket.close()
                print("Angle UDP socket closed")
            except Exception as e:
                print(f"Error closing angle UDP socket: {e}")

def create_udp_receiver(host="127.0.0.1", port=12345, timeout=1.0):
    """创建UDP接收器用于接收轨迹数据"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        sock.settimeout(timeout)
        print(f"UDP receiver created and listening on {host}:{port}")
        return sock
    except Exception as e:
        print(f"Failed to create UDP receiver: {e}")
        return None

def receive_trajectory_data_udp(sock):
    """从UDP接收轨迹数据"""
    try:
        data, addr = sock.recvfrom(65536)  # 64KB buffer
        json_data = data.decode('utf-8')
        trajectory_data = json.loads(json_data)
        return trajectory_data
    except socket.timeout:
        return None
    except Exception as e:
        print(f"Failed to receive trajectory data via UDP: {e}")
        return None
