"""
蓝色物体追踪器 - 追踪除螨仪上的蓝色标记
"""

import cv2
import numpy as np
import os
import time
from config import config_manager
from typing import List, Tuple, Optional
from utils import imshow_adaptive


class SimpleKalmanFilter:
    """简单的二维卡尔曼滤波器，用于平滑轨迹预测并处理短暂遮挡"""
    def __init__(self, max_lost_frames=15):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.lost_frames = 0
        self.max_lost_frames = max_lost_frames

    def update(self, center):
        if center is None:
            self.lost_frames += 1
            if not self.initialized or self.lost_frames > self.max_lost_frames:
                self.initialized = False
                return None
            # 仅预测
            prediction = self.kf.predict()
            return (int(prediction[0]), int(prediction[1]))
        else:
            self.lost_frames = 0
            if not self.initialized:
                self.kf.statePre = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
                self.kf.statePost = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
                self.initialized = True
                return center
            # 预测并校准
            prediction = self.kf.predict()
            measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
            estimated = self.kf.correct(measurement)
            return (int(estimated[0]), int(estimated[1]))


class TrajectoryPoint:
    """轨迹点"""
    
    def __init__(self, x: int, y: int, timestamp: float, frame_idx: int, bed_x: int = None, bed_y: int = None, bed_angle: float = None, tail_x: int = None, tail_y: int = None, width: float = None):
        self.x = x
        self.y = y
        self.timestamp = timestamp  # 时间戳（秒）
        self.frame_idx = frame_idx  # 帧索引
        self.bed_x = bed_x
        self.bed_y = bed_y
        self.bed_angle = bed_angle
        self.tail_x = tail_x if tail_x is not None else x
        self.tail_y = tail_y if tail_y is not None else y
        self.width = width


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
        
        # 加载床铺与覆盖计算配置，用于实时轨迹和机身绘制
        self.bed_config = config_manager.get_bed_area_config()
        self.coverage_config = config_manager.get_coverage_config()
        
        # 卡尔曼滤波器
        self.blue_kf = SimpleKalmanFilter(max_lost_frames=15)
        self.green_kf = SimpleKalmanFilter(max_lost_frames=15)
        

        # 预先计算透视变换矩阵与逆矩阵
        self.has_bed_config = False
        if self.bed_config.points and len(self.bed_config.points) == 4:
            pts1 = np.float32(self.bed_config.points)
            pts2 = np.float32([
                [0, 0],
                [self.bed_config.width - 1, 0],
                [self.bed_config.width - 1, self.bed_config.height - 1],
                [0, self.bed_config.height - 1]
            ])
            self.matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.inv_matrix = np.linalg.inv(self.matrix)
            self.has_bed_config = True
            
            # 计算像素级比例与大小
            real_w = self.bed_config.real_width_cm or 200
            real_h = self.bed_config.real_height_cm or 150
            self.pixel_to_cm_x = real_w / self.bed_config.width
            self.pixel_to_cm_y = real_h / self.bed_config.height
            
            brush_w_cm = self.coverage_config.real_brush_width_cm or 13.0
            brush_h_cm = self.coverage_config.real_brush_height_cm or 2.5
            # 强制左右跨度为较大值，前后跨度为较小值，防止配置写反
            actual_w = max(brush_w_cm, brush_h_cm)
            actual_h = min(brush_w_cm, brush_h_cm)
            self.brush_width_px = max(1, int(round(actual_w / self.pixel_to_cm_x)))
            self.brush_height_px = max(1, int(round(actual_h / self.pixel_to_cm_y)))
            
            remover_w_cm = self.coverage_config.real_remover_width_cm or 25.0
            remover_h_cm = self.coverage_config.real_remover_height_cm or 22.0
            self.remover_width_px = max(1, int(round(remover_w_cm / self.pixel_to_cm_x)))
            self.remover_height_px = max(1, int(round(remover_h_cm / self.pixel_to_cm_y)))
            
            # 初始化床铺大小的扫掠掩码
            self.bed_mask = np.zeros((self.bed_config.height, self.bed_config.width), dtype=np.uint8)
            self.last_box_brush = None  # 用于帧间插值，防止倍速断点
        
        print(f"视频信息:")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  帧率: {self.fps:.2f} FPS")
        print(f"  总帧数: {self.total_frames}")
        print(f"  时长: {self.total_frames/self.fps:.2f}秒")
        
    def _get_color_mask(self, frame: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
        """根据指定HSV范围提取掩码"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask
    
    def _find_marker_center(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """查找指定颜色区域的质心"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        if area < self.config.min_area:
            return None
        M = cv2.moments(max_contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def track(self, output_video: str = None, 
              show_progress: bool = True,
              show_preview: bool = True) -> Tuple[List[TrajectoryPoint], np.ndarray, Optional[np.ndarray]]:
        """
        执行双色贴纸追踪并进行可视化绘制
        
        Args:
            output_video: 输出视频路径（可选）
            show_progress: 是否显示进度
            show_preview: 是否显示预览窗口（关闭可大幅提升速度）
            
        Returns:
            (轨迹点列表, 最终床面扫掠二值掩码)
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
        
        # 双色追踪相关的状态变量
        last_heading_angle = 0.0
        last_bg_vector = None  # 蓝点到绿点的相对向量 (dx, dy)
        
        # 定义蓝色与绿色的 HSV 边界
        lower_blue = np.array([self.config.blue_lower_h, self.config.blue_lower_s, self.config.blue_lower_v])
        upper_blue = np.array([self.config.blue_upper_h, self.config.blue_upper_s, self.config.blue_upper_v])
        lower_green = np.array([self.config.green_lower_h, self.config.green_lower_s, self.config.green_lower_v])
        upper_green = np.array([self.config.green_upper_h, self.config.green_upper_s, self.config.green_upper_v])
        
        # 引入并创建实时覆盖率计算器
        from coverage import CoverageCalculator
        realtime_calc = None
        if self.has_bed_config:
            realtime_calc = CoverageCalculator(
                self.bed_config.width,
                self.bed_config.height,
                self.bed_config.real_width_cm,
                self.bed_config.real_height_cm
            )
        
        print("\n开始追踪...")
        if show_preview:
            print("按 'Q' 键可提前退出\n")
        

        first_frame = None
        
        play_speed = 1
        
        while True:
            # 记录当前帧开始处理的时间点，用于精确控制倍速下的等待延迟
            frame_start_time = time.time()
            
            # 根据实时倍速选择，抓取并丢弃中间的帧以实现跳帧倍速播放
            if frame_idx > 0 and play_speed > 1:
                for _ in range(play_speed - 1):
                    ret_grab = self.cap.grab()
                    if not ret_grab:
                        break
                    frame_idx += 1
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if first_frame is None:
                first_frame = frame.copy()
            
            # 计算时间戳
            timestamp = frame_idx / self.fps
            
            # 提取蓝色和绿色掩码
            blue_mask = self._get_color_mask(frame, lower_blue, upper_blue)
            green_mask = self._get_color_mask(frame, lower_green, upper_green)
            
            # 查找两个标记的中心
            blue_center = self._find_marker_center(blue_mask)
            green_center = self._find_marker_center(green_mask)
            
            # 距离约束：蓝色和绿色标定纸的距离不能超过机身的大小 (严格限制)
            if blue_center and green_center and self.has_bed_config:
                pt_v_b = np.array([[[blue_center[0], blue_center[1]]]], dtype=np.float32)
                pt_v_g = np.array([[[green_center[0], green_center[1]]]], dtype=np.float32)
                pt_b_b = cv2.perspectiveTransform(pt_v_b, self.matrix)[0][0]
                pt_b_g = cv2.perspectiveTransform(pt_v_g, self.matrix)[0][0]
                dist_x_cm = (pt_b_g[0] - pt_b_b[0]) * self.pixel_to_cm_x
                dist_y_cm = (pt_b_g[1] - pt_b_b[1]) * self.pixel_to_cm_y
                dist_cm = np.hypot(dist_x_cm, dist_y_cm)
                
                max_size_cm = max(self.coverage_config.real_remover_width_cm or 25.0, 
                                  self.coverage_config.real_remover_height_cm or 22.0)
                
                # 真实贴纸距离大约十几厘米，最大限制设为机身最长边的 1.1 倍，绝对不允许拉跨到其他地方
                if dist_cm > max_size_cm * 1.1:
                    blue_center = None
                    green_center = None
            
            # === EMA平滑处理与防瞬间跳变 ===
            base_alpha = 0.15
            alpha = 1.0 - (1.0 - base_alpha) ** play_speed
            max_jump_px = 60 * play_speed  # 每帧最大允许的物理位移像素

            if blue_center:
                if getattr(self, 'smoothed_blue', None) is None:
                    self.smoothed_blue = np.array(blue_center, dtype=float)
                    self.blue_reject_count = 0
                else:
                    dist = np.hypot(blue_center[0] - self.smoothed_blue[0], blue_center[1] - self.smoothed_blue[1])
                    if dist > max_jump_px and getattr(self, 'blue_reject_count', 0) < 15:
                        # 瞬间跳变，拒绝接受，保持上一帧的位置 (最多容忍连续15帧跳变，防止死锁)
                        self.blue_reject_count = getattr(self, 'blue_reject_count', 0) + 1
                        blue_center = (int(round(self.smoothed_blue[0])), int(round(self.smoothed_blue[1])))
                    else:
                        self.blue_reject_count = 0
                        self.smoothed_blue = self.smoothed_blue * (1 - alpha) + np.array(blue_center, dtype=float) * alpha
                        blue_center = (int(round(self.smoothed_blue[0])), int(round(self.smoothed_blue[1])))
            else:
                self.smoothed_blue = None
                self.blue_reject_count = 0
                
            if green_center:
                if getattr(self, 'smoothed_green', None) is None:
                    self.smoothed_green = np.array(green_center, dtype=float)
                    self.green_reject_count = 0
                else:
                    dist = np.hypot(green_center[0] - self.smoothed_green[0], green_center[1] - self.smoothed_green[1])
                    if dist > max_jump_px and getattr(self, 'green_reject_count', 0) < 15:
                        self.green_reject_count = getattr(self, 'green_reject_count', 0) + 1
                        green_center = (int(round(self.smoothed_green[0])), int(round(self.smoothed_green[1])))
                    else:
                        self.green_reject_count = 0
                        self.smoothed_green = self.smoothed_green * (1 - alpha) + np.array(green_center, dtype=float) * alpha
                        green_center = (int(round(self.smoothed_green[0])), int(round(self.smoothed_green[1])))
            else:
                self.smoothed_green = None
                self.green_reject_count = 0
            
            # 2. 获取坐标
            bx_v, by_v = None, None
            gx_v, gy_v = None, None

            if blue_center:
                bx_v, by_v = float(blue_center[0]), float(blue_center[1])
            if green_center:
                gx_v, gy_v = float(green_center[0]), float(green_center[1])

            heading_angle = last_heading_angle
            brush_center = None
            remover_center = None

            # 3. 融合定位与朝向计算逻辑
            if bx_v is not None and gx_v is not None:
                dx = gx_v - bx_v
                dy = gy_v - by_v
                
                # 计算向量的原始夹角
                vector_angle = np.degrees(np.arctan2(dy, dx))
                
                # 左右并排模式 (仅保留此排布)：朝向角等于向量夹角 + 90度 (已匹配用户实际的：左绿右蓝)
                heading_angle = vector_angle + 90
                # 机身几何中心位于蓝绿中点
                remover_center = (int(bx_v + 0.5 * dx), int(by_v + 0.5 * dy))
                
                # 根据用户要求：蓝色区域（刷头）在黄线正下方，不要往前延伸
                offset_cm = 0.0
                
                D = np.hypot(dx, dy)
                rem_h = self.coverage_config.real_remover_height_cm or 27.0
                
                brush_center = remover_center  # 由于没有前伸量，刷头中心直接等于机身中心
                # (不再在这里使用视频空间粗略计算尾部，而是放到后面的物理坐标系中精准计算)
                tail_center = remover_center  # 占位，后面会覆盖
                last_heading_angle = heading_angle
                
            elif gx_v is not None:
                # 只有绿点被初始化过（单标定纸兜底）：朝向依靠运动轨迹向量
                brush_center = (int(gx_v), int(gy_v))
                remover_center = brush_center
                tail_center = brush_center
                
                # 计算运动位移朝向
                if self.trajectory:
                    prev_pt = self.trajectory[-1]
                    dx_move = gx_v - prev_pt.x
                    dy_move = gy_v - prev_pt.y
                    if np.hypot(dx_move, dy_move) > 2.0:
                        heading_angle = np.degrees(np.arctan2(dy_move, dx_move))
                else:
                    heading_angle = 0.0
                last_heading_angle = heading_angle
                
            elif bx_v is not None:
                # 只有蓝点被初始化过（单标定纸兜底）：朝向依靠运动轨迹向量
                brush_center = (int(bx_v), int(by_v))
                remover_center = brush_center
                tail_center = brush_center
                
                # 计算运动位移朝向
                if self.trajectory:
                    prev_pt = self.trajectory[-1]
                    dx_move = bx_v - prev_pt.x
                    dy_move = by_v - prev_pt.y
                    if np.hypot(dx_move, dy_move) > 2.0:
                        heading_angle = np.degrees(np.arctan2(dy_move, dx_move))
                else:
                    heading_angle = 0.0
                last_heading_angle = heading_angle
                
            if brush_center:
                # 稍后投影计算完毕后再记录轨迹
                no_detection_count = 0
                # 投影到床面坐标系以用于主刷口绘制和实时覆盖率计算
                bx, by, box_angle = None, None, None
                
                if self.has_bed_config:
                    # 1. 初始化变量，优先使用床面物理坐标系下的计算以彻底消除透视畸变偏差
                    rx, ry = None, None
                    
                    parallax_h_cm = getattr(self.config, 'parallax_height_cm', 12.0)
                    
                    if blue_center and green_center:
                        # 将视频坐标系下的左右标记点投影到床面坐标系下
                        pt_video_blue = np.array([[[blue_center[0], blue_center[1]]]], dtype=np.float32)
                        pt_video_green = np.array([[[green_center[0], green_center[1]]]], dtype=np.float32)
                        
                        pt_bed_blue = cv2.perspectiveTransform(pt_video_blue, self.matrix)[0][0]
                        pt_bed_green = cv2.perspectiveTransform(pt_video_green, self.matrix)[0][0]
                        
                        bx_b, by_b = pt_bed_blue[0], pt_bed_blue[1]
                        gx_b, gy_b = pt_bed_green[0], pt_bed_green[1]
                        
                        parallax_h_cm = self.config.parallax_height_cm
                        if parallax_h_cm > 0:
                            # 径向视差高度补偿：根据用户测量的标定纸真实高度 (3cm)
                            # 将投影点向摄像机所在的绝对位置进行“径向收缩”
                            cam_z = 120.0  # 假定摄像机高度 120cm
                            cam_x_b = self.bed_config.width / 2.0
                            cam_y_b = self.bed_config.height + (50.0 / self.pixel_to_cm_y)
                            
                            shrink_factor = parallax_h_cm / cam_z
                            
                            bx_b = pt_bed_blue[0] + (cam_x_b - pt_bed_blue[0]) * shrink_factor
                            by_b = pt_bed_blue[1] + (cam_y_b - pt_bed_blue[1]) * shrink_factor
                            gx_b = pt_bed_green[0] + (cam_x_b - pt_bed_green[0]) * shrink_factor
                            gy_b = pt_bed_green[1] + (cam_y_b - pt_bed_green[1]) * shrink_factor
                        else:
                            bx_b, by_b = pt_bed_blue[0], pt_bed_blue[1]
                            gx_b, gy_b = pt_bed_green[0], pt_bed_green[1]
                        
                        dx_bed = gx_b - bx_b
                        dy_bed = gy_b - by_b
                        D_bed = np.hypot(dx_bed, dy_bed)
                        
                        if D_bed > 0:
                            # 机器前行朝向单位向量 (恢复为 -dy, dx，因为标定点确实是蓝右绿左)
                            ux = -dy_bed / D_bed
                            uy = dx_bed / D_bed
                            
                            # 用户要求蓝色主刷口比贴纸距离稍微宽一点点（乘以 1.15 倍）
                            self.last_dynamic_width_px = float(D_bed) * 1.15
                            
                            # 放弃思路2的动态补偿，仅保留一个固定的基础物理前伸量
                            # 根据机身真实大小，设定恒定的前伸量 4.0 cm
                            offset_cm = 4.0
                            
                            # 机身中心 (直接由两点中点确定)
                            rx = 0.5 * (bx_b + gx_b)
                            ry = 0.5 * (by_b + gy_b)
                            
                            # 刷头中心 (机身中心沿前行朝向推 offset_cm 对应的像素)
                            bx = rx + (offset_cm / self.pixel_to_cm_x) * ux
                            by = ry + (offset_cm / self.pixel_to_cm_y) * uy
                            
                            # 尾部红点 (沿相反方向推，也要往前一点，所以缩短距离到机身长度的30%)
                            tail_cm = (self.coverage_config.real_remover_height_cm or 27.0) * 0.3
                            tx = rx - (tail_cm / self.pixel_to_cm_x) * ux
                            ty = ry - (tail_cm / self.pixel_to_cm_y) * uy
                            
                            # 将精准计算的物理尾部坐标，反向投影回视频空间，用来画红点
                            pt_bed_tail = np.array([[[tx, ty]]], dtype=np.float32)
                            pt_video_tail = cv2.perspectiveTransform(pt_bed_tail, self.inv_matrix)[0][0]
                            tail_center = (int(pt_video_tail[0]), int(pt_video_tail[1]))
                            
                            # 床面上的连线角度与主刷口摆放角度
                            vector_angle_bed = np.degrees(np.arctan2(dy_bed, dx_bed))
                            box_angle = vector_angle_bed
                            
                    # 2. 兜底情况：如果不完全具备双色标记 (仅有单点轨迹)，则采用视频空间计算后投影的老方法
                    if bx is None or by is None:
                        # 投影视频空间算出的刷头中心
                        pt_video_brush = np.array([[[brush_center[0], brush_center[1]]]], dtype=np.float32)
                        pt_bed_brush = cv2.perspectiveTransform(pt_video_brush, self.matrix)
                        bx, by = pt_bed_brush[0][0][0], pt_bed_brush[0][0][1]
                        
                        if parallax_h_cm > 0:
                            y_cm = by * self.pixel_to_cm_y
                            shift_cm = parallax_h_cm * (1.0 - y_cm / (self.bed_config.real_height_cm or 150))
                            by = by + shift_cm / self.pixel_to_cm_y
                            
                        # 投影机身中心
                        pt_video_remover = np.array([[[remover_center[0], remover_center[1]]]], dtype=np.float32)
                        pt_bed_remover = cv2.perspectiveTransform(pt_video_remover, self.matrix)
                        rx, ry = pt_bed_remover[0][0][0], pt_bed_remover[0][0][1]
                        if parallax_h_cm > 0:
                            y_cm_rem = ry * self.pixel_to_cm_y
                            shift_cm_rem = parallax_h_cm * (1.0 - y_cm_rem / (self.bed_config.real_height_cm or 150))
                            ry = ry + shift_cm_rem / self.pixel_to_cm_y
                            
                        box_angle = heading_angle + 90
                        
                    # 3. 边界剪裁与取整
                    bx = int(round(max(0, min(bx, self.bed_config.width - 1))))
                    by = int(round(max(0, min(by, self.bed_config.height - 1))))
                    rx = int(round(max(0, min(rx, self.bed_config.width - 1))))
                    ry = int(round(max(0, min(ry, self.bed_config.height - 1))))
                    
                    # 蓝框的长度（左右跨度）根据两点之间的实时距离动态调整，若无则使用配置默认值
                    dynamic_w = getattr(self, 'last_dynamic_width_px', float(self.brush_width_px))
                    
                    # 实时更新覆盖率计算器
                    if realtime_calc is not None:
                        realtime_calc.add_point(bx, by, float(box_angle), dynamic_w)
                    
                    # 4. 在床面掩码上绘制主刷口清洁区域
                    rect_brush = ((float(bx), float(by)), (dynamic_w, float(self.brush_height_px)), float(box_angle))
                    box_brush = cv2.boxPoints(rect_brush)
                    
                    if self.last_box_brush is not None and self.trajectory:
                        prev_pt = self.trajectory[-1]
                        dist = np.hypot(bx - prev_pt.bed_x, by - prev_pt.bed_y) if prev_pt.bed_x else 999
                        if dist < 200:
                            hull = cv2.convexHull(np.vstack((self.last_box_brush, box_brush)))
                            draw_poly = np.squeeze(hull)
                        else:
                            draw_poly = box_brush
                    else:
                        draw_poly = box_brush
                        
                    cv2.fillPoly(self.bed_mask, [np.int32(draw_poly)], 255)
                    self.last_box_brush = box_brush

                # 记录轨迹（包含视频原始坐标，以及计算出的床面坐标和角度）
                point = TrajectoryPoint(
                    x=brush_center[0], y=brush_center[1],
                    timestamp=timestamp, frame_idx=frame_idx,
                    bed_x=bx, bed_y=by, bed_angle=box_angle,
                    tail_x=tail_center[0], tail_y=tail_center[1],
                    width=dynamic_w
                )
                self.trajectory.append(point)
            else:
                if self.has_bed_config:
                    self.last_box_brush = None
                no_detection_count += 1

            # 绘制整条半透明淡蓝色路线（已清扫区域）
            if self.has_bed_config and np.any(self.bed_mask > 0):
                warped_back = cv2.warpPerspective(self.bed_mask, self.inv_matrix, (self.width, self.height))
                overlay = frame.copy()
                # 区分颜色：轨迹为较浅的半透明淡蓝色
                overlay[warped_back > 0] = [235, 206, 135]  # 淡蓝色 BGR [235, 206, 135]
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
            
            # 绘制标定点 (使用 LINE_AA 开启抗锯齿，边缘更平滑)
            if blue_center:
                cv2.circle(frame, blue_center, 2, (255, 0, 0), -1, cv2.LINE_AA)
            if green_center:
                cv2.circle(frame, green_center, 2, (0, 255, 0), -1, cv2.LINE_AA)
            # 用户要求取消黄线
            # if blue_center and green_center:
            #     cv2.line(frame, blue_center, green_center, (0, 255, 255), 2, cv2.LINE_AA)

            # 绘制机身走过的轨迹（红线），使用tail_x和tail_y，颜色淡一点，并且把线宽调得更细致
            if len(self.trajectory) > 1:
                pts = np.array([[p.tail_x, p.tail_y] for p in self.trajectory], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(150, 150, 255), thickness=2, lineType=cv2.LINE_AA)

            # 显示常规信息
            cv2.putText(frame, f"Frame: {frame_idx}/{self.total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Points: {len(self.trajectory)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp:.2f}s",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示实时覆盖率 (以明显的黄色/青色文字标注)
            if realtime_calc is not None:
                realtime_coverage = realtime_calc.calculate_coverage_rate()
                cv2.putText(frame, f"Coverage: {realtime_coverage:.2f}%",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示实时播放倍速提示 (以黄色文字标注)
            cv2.putText(frame, f"Speed: {play_speed}x (Keys: 1, 2, 4, 8, 0=16x)",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 写入视频
            if writer:
                writer.write(frame)
            
            # 进度显示
            if show_progress and frame_idx % 30 == 0:
                progress = (frame_idx / self.total_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{self.total_frames})")
            
            # 按键处理与窗口显示
            if show_preview:
                # 自适应分辨率显示跟踪窗口（不显示 Mask 窗口）
                imshow_adaptive("Tracking", frame)
                
                # 计算合适的 waitKey 延迟，以保证画面播放速度与倍速匹配
                elapsed_ms = (time.time() - frame_start_time) * 1000.0
                delay = max(1, int(round(1000.0 / self.fps - elapsed_ms)))
                
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('1'):
                    play_speed = 1
                    print("\n切换至 1x 播放速度")
                elif key == ord('2'):
                    play_speed = 2
                    print("\n切换至 2x 播放速度")
                elif key == ord('4'):
                    play_speed = 4
                    print("\n切换至 4x 播放速度")
                elif key == ord('8'):
                    play_speed = 8
                    print("\n切换至 8x 播放速度")
                elif key == ord('0'):
                    play_speed = 16
                    print("\n切换至 16x 播放速度")
                
                # 键盘按 Q 或 ESC，提前结束追踪并生成报告
                if key in [27, ord('q'), ord('Q')]:
                    print("\n用户主动结束追踪，准备生成报告...")
                    break
                # 如果用户直接点击右上角 X 关闭窗口，则直接退出整个程序
                elif cv2.getWindowProperty("Tracking", cv2.WND_PROP_VISIBLE) < 1:
                    print("\n用户关闭了追踪窗口，直接退出程序。")
                    import sys
                    sys.exit(0)
            
            frame_idx += 1
        
        # 清理
        self.cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        print(f"\n追踪完成!")
        print(f"  总帧数: {frame_idx}")
        print(f"  检测到轨迹点: {len(self.trajectory)}")
        print(f"  未检测帧数: {no_detection_count}")
        
        final_bed_mask = self.bed_mask if self.has_bed_config else np.zeros((100, 100), dtype=np.uint8)
        return self.trajectory, final_bed_mask, first_frame
    
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
        
        print(f"\n追踪到 {len(trajectory)} 个点")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    import sys
    import os
    # 将当前目录加入 path 确保正常导入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import main as app_main
    app_main()
