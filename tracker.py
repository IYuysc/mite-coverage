"""
蓝色物体追踪器 - 追踪除螨仪上的蓝色标记
"""

import cv2
import numpy as np
import os
from config import config_manager
from typing import List, Tuple, Optional


class TrajectoryPoint:
    """轨迹点"""
    
    def __init__(self, x: int, y: int, timestamp: float, frame_idx: int):
        self.x = x
        self.y = y
        self.timestamp = timestamp  # 时间戳（秒）
        self.frame_idx = frame_idx  # 帧索引


class BlueTracker:
    """蓝色物体追踪器"""
    
    def __init__(self, video_path: str):
        """
        初始化追踪器
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 轨迹记录
        self.trajectory: List[TrajectoryPoint] = []
        
        # 配置
        self.config = config_manager.get_tracker_config()
        
        # 形态学核
        self.kernel = np.ones((self.config.morph_kernel_size, 
                              self.config.morph_kernel_size), 
                             np.uint8)
        
        print(f"视频信息:")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  帧率: {self.fps:.2f} FPS")
        print(f"  总帧数: {self.total_frames}")
        print(f"  时长: {self.total_frames/self.fps:.2f}秒")
        
    def _get_hsv_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        获取蓝色区域的掩码
        
        Args:
            frame: BGR格式图像
            
        Returns:
            二值掩码
        """
        # 转换到HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义蓝色范围
        lower_blue = np.array([
            self.config.blue_lower_h,
            self.config.blue_lower_s,
            self.config.blue_lower_v
        ])
        upper_blue = np.array([
            self.config.blue_upper_h,
            self.config.blue_upper_s,
            self.config.blue_upper_v
        ])
        
        # 提取蓝色区域
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 形态学操作去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        return mask
    
    def _find_blue_center(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        查找蓝色区域中心
        
        Args:
            mask: 二值掩码
            
        Returns:
            中心点坐标 (x, y)，未找到返回None
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # 找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        # 过滤小区域
        if area < self.config.min_area:
            return None
        
        # 计算中心
        M = cv2.moments(max_contour)
        if M["m00"] == 0:
            return None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def track(self, output_video: str = None, 
              show_progress: bool = True) -> List[TrajectoryPoint]:
        """
        执行追踪
        
        Args:
            output_video: 输出视频路径（可选）
            show_progress: 是否显示进度
            
        Returns:
            轨迹点列表
        """
        self.trajectory = []
        
        # 视频写入器
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_video, fourcc, self.fps,
                (self.width, self.height)
            )
        
        frame_idx = 0
        no_detection_count = 0
        
        print("\n开始追踪...")
        print("按 'Q' 键可提前退出\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 获取掩码
            mask = self._get_hsv_mask(frame)
            
            # 查找中心
            center = self._find_blue_center(mask)
            
            # 计算时间戳
            timestamp = frame_idx / self.fps
            
            if center:
                # 记录轨迹
                point = TrajectoryPoint(
                    x=center[0], y=center[1],
                    timestamp=timestamp, frame_idx=frame_idx
                )
                self.trajectory.append(point)
                no_detection_count = 0
                
                # 在帧上绘制
                cv2.circle(frame, center, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"({center[0]},{center[1]})",
                           (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                no_detection_count += 1
            
            # 显示信息
            cv2.putText(frame, f"Frame: {frame_idx}/{self.total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Points: {len(self.trajectory)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp:.2f}s",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示掩码
            cv2.imshow("Mask", mask)
            
            # 显示帧
            cv2.imshow("Tracking", frame)
            
            # 写入视频
            if writer:
                writer.write(frame)
            
            # 进度显示
            if show_progress and frame_idx % 30 == 0:
                progress = (frame_idx / self.total_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{self.total_frames})")
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n用户中断追踪")
                break
            
            frame_idx += 1
        
        # 清理
        self.cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n追踪完成!")
        print(f"  总帧数: {frame_idx}")
        print(f"  检测到轨迹点: {len(self.trajectory)}")
        print(f"  未检测帧数: {no_detection_count}")
        
        return self.trajectory
    
    def save_trajectory(self, output_file: str = "trajectory.csv"):
        """
        保存轨迹到CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        if not self.trajectory:
            print("没有轨迹数据可保存")
            return
        
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'timestamp', 'x', 'y'])
            
            for point in self.trajectory:
                writer.writerow([
                    point.frame_idx,
                    f"{point.timestamp:.3f}",
                    point.x,
                    point.y
                ])
        
        print(f"轨迹已保存到: {output_file}")
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        获取轨迹点数组
        
        Returns:
            numpy数组，形状为 (N, 2)，每行为 (x, y)
        """
        if not self.trajectory:
            return np.array([])
        
        points = np.array([[p.x, p.y] for p in self.trajectory])
        return points
    
    def get_trajectory_with_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取轨迹点数组和时间戳
        
        Returns:
            (points, timestamps) 其中:
            - points: (N, 2) 数组
            - timestamps: (N,) 数组
        """
        if not self.trajectory:
            return np.array([]), np.array([])
        
        points = np.array([[p.x, p.y] for p in self.trajectory])
        timestamps = np.array([p.timestamp for p in self.trajectory])
        
        return points, timestamps


def main():
    """测试函数"""
    # 查找视频文件
    video_dir = "videos"
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) 
                 if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        if videos:
            video_path = os.path.join(video_dir, videos[0])
            print(f"使用视频: {video_path}")
        else:
            print(f"在 {video_dir} 目录中未找到视频文件")
            return
    else:
        print(f"视频目录不存在: {video_dir}")
        return
    
    try:
        tracker = BlueTracker(video_path)
        trajectory = tracker.track(output_video="outputs/tracking_result.mp4")
        tracker.save_trajectory("outputs/trajectory.csv")
        
        print(f"\n追踪到 {len(trajectory)} 个点")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
