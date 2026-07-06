"""
覆盖率计算模块 - 计算除螨仪在床铺区域的覆盖率
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from config import config_manager


class CoverageCalculator:
    """覆盖率计算器"""
    
    def __init__(self, bed_width: int, bed_height: int,
                 real_width_cm: int = 0, real_height_cm: int = 0):
        """
        初始化计算器
        
        Args:
            bed_width: 床铺区域宽度（像素）
            bed_height: 床铺区域高度（像素）
            real_width_cm: 实际宽度（厘米），0表示未设置
            real_height_cm: 实际高度（厘米），0表示未设置
        """
        self.bed_width = bed_width
        self.bed_height = bed_height
        self.real_width_cm = real_width_cm
        self.real_height_cm = real_height_cm
        self.config = config_manager.get_coverage_config()
        
        # 计算像素到厘米的比例
        if real_width_cm > 0 and real_height_cm > 0:
            self.pixel_to_cm_x = real_width_cm / bed_width  # cm per pixel
            self.pixel_to_cm_y = real_height_cm / bed_height # cm per pixel
            self.use_real_unit = True
        else:
            self.pixel_to_cm_x = 1.0
            self.pixel_to_cm_y = 1.0
            self.use_real_unit = False
        
        # 计算像素级主刷口大小
        if self.use_real_unit and getattr(self.config, 'real_brush_width_cm', 0) > 0 and getattr(self.config, 'real_brush_height_cm', 0) > 0:
            # 像素 = 厘米 / (厘米/像素)
            self.brush_width_px = int(round(self.config.real_brush_width_cm / self.pixel_to_cm_x))
            self.brush_height_px = int(round(self.config.real_brush_height_cm / self.pixel_to_cm_y))
        else:
            self.brush_width_px = self.config.brush_width
            self.brush_height_px = self.config.brush_height
            
        self.brush_width_px = max(1, self.brush_width_px)
        self.brush_height_px = max(1, self.brush_height_px)
        
        # 网格化参数
        self.grid_size = self.config.grid_size
        self.grid_cols = bed_width // self.grid_size
        self.grid_rows = bed_height // self.grid_size
        
        # 覆盖网格（记录每个网格被覆盖的次数）
        self.coverage_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
        
        # 覆盖掩码（二值，是否被覆盖过）
        self.coverage_mask = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        
        # 热力图
        self.heatmap = np.zeros((bed_height, bed_width), dtype=np.float32)
        
        # 轨迹点
        self.trajectory_points: List[Tuple[int, int]] = []
        
        # 用于运动方向估算的变量
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.current_angle = 0.0  # 默认水平朝向
        self.direction_alpha = 0.2  # 运动向量平滑指数
        self.min_move_threshold = 2.0  # 触发方向更新的最小位移（像素）
        
        self.last_box = None  # 用于帧间插值，防止倍速播放产生断点
        
    def _point_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """
        将像素坐标转换为网格坐标
        
        Args:
            x: 像素x坐标
            y: 像素y坐标
            
        Returns:
            (grid_col, grid_row)
        """
        grid_col = min(x // self.grid_size, self.grid_cols - 1)
        grid_row = min(y // self.grid_size, self.grid_rows - 1)
        return (grid_col, grid_row)
    
    def _get_brush_mask(self, center_x: int, center_y: int, angle: float = 0.0) -> np.ndarray:
        """
        获取主刷口掩码（旋转的矩形）
        
        Args:
            center_x: 中心x坐标
            center_y: 中心y坐标
            angle: 旋转角度（角度制）
            
        Returns:
            二值掩码，形状为 (bed_height, bed_width)
        """
        mask = np.zeros((self.bed_height, self.bed_width), dtype=np.uint8)
        
        # 旋转矩形参数: ((cx, cy), (width, height), angle)
        rect = ((float(center_x), float(center_y)), 
                (float(self.brush_width_px), float(self.brush_height_px)), 
                float(angle))
        
        # 获取矩形四个角点并转为整数
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 填充多边形
        cv2.fillPoly(mask, [box], 255)
        
        return mask
    
    def add_point(self, x: int, y: int, angle: Optional[float] = None):
        """
        添加轨迹点并更新覆盖
        
        Args:
            x: x坐标
            y: y坐标
            angle: 如果提供，则使用此角度而不是根据轨迹自动估算
        """
        # 边界检查
        x = max(0, min(x, self.bed_width - 1))
        y = max(0, min(y, self.bed_height - 1))
        
        # 估算旋转角度
        if angle is not None:
            final_angle = angle
            self.current_angle = angle
        else:
            final_angle = self.current_angle
            if self.trajectory_points:
                prev_x, prev_y = self.trajectory_points[-1]
                raw_dx = x - prev_x
                raw_dy = y - prev_y
                
                # 使用指数移动平均平滑运动向量
                self.smooth_dx = self.direction_alpha * raw_dx + (1.0 - self.direction_alpha) * self.smooth_dx
                self.smooth_dy = self.direction_alpha * raw_dy + (1.0 - self.direction_alpha) * self.smooth_dy
                
                # 只有当平滑位移大于阈值时，才更新方向，避免微小抖动引起角度乱转
                disp = np.sqrt(self.smooth_dx**2 + self.smooth_dy**2)
                if disp > self.min_move_threshold:
                    # 运动方向角 (弧度 -> 角度)
                    angle_motion = np.degrees(np.arctan2(self.smooth_dy, self.smooth_dx))
                    # 主刷口朝向与运动方向垂直，所以矩形需要旋转 angle_motion - 90
                    final_angle = angle_motion - 90
                    self.current_angle = final_angle
        
        # 旋转矩形参数: ((cx, cy), (width, height), angle)
        rect = ((float(x), float(y)), 
                (float(self.brush_width_px), float(self.brush_height_px)), 
                float(final_angle))
        box = cv2.boxPoints(rect)
        
        # 倍速播放帧间插值：计算当前 box 与上一个 box 的凸包来填充扫过的整片区域
        if self.last_box is not None and len(self.trajectory_points) > 0:
            prev_x, prev_y = self.trajectory_points[-1]
            dist = np.hypot(x - prev_x, y - prev_y)
            if dist < 200:  # 如果距离在合理范围内（不超过 ~30cm），则插值
                hull = cv2.convexHull(np.vstack((self.last_box, box)))
                draw_poly = np.squeeze(hull)
            else:
                draw_poly = box
        else:
            draw_poly = box
            
        self.trajectory_points.append((x, y))
        self.last_box = box
        
        # 1. 优化热力图更新（只在包围盒范围内更新，避免全图大数组拷贝和加法）
        xs = draw_poly[:, 0]
        ys = draw_poly[:, 1]
        min_x = max(0, int(np.floor(np.min(xs))))
        max_x = min(self.bed_width - 1, int(np.ceil(np.max(xs))))
        min_y = max(0, int(np.floor(np.min(ys))))
        max_y = min(self.bed_height - 1, int(np.ceil(np.max(ys))))
        
        if max_x >= min_x and max_y >= min_y:
            h_sub = max_y - min_y + 1
            w_sub = max_x - min_x + 1
            sub_mask = np.zeros((h_sub, w_sub), dtype=np.uint8)
            # 将角点平移到子区域的局部坐标系下
            shifted_poly = draw_poly - [min_x, min_y]
            cv2.fillPoly(sub_mask, [np.int32(shifted_poly)], 255)
            # 在原热力图的对应区域进行加法（0.1 * 255.0 = 25.5 对应原来数值）
            self.heatmap[min_y:max_y+1, min_x:max_x+1] += sub_mask.astype(np.float32) * 0.1
            
        # 2. 优化覆盖网格更新（使用 NumPy + cv2.fillPoly 向量化操作提升数百倍速度）
        grid_min_x = max(0, int(np.floor(np.min(xs))) // self.grid_size)
        grid_max_x = min(self.grid_cols - 1, int(np.ceil(np.max(xs))) // self.grid_size)
        grid_min_y = max(0, int(np.floor(np.min(ys))) // self.grid_size)
        grid_max_y = min(self.grid_rows - 1, int(np.ceil(np.max(ys))) // self.grid_size)
        
        if grid_max_x >= grid_min_x and grid_max_y >= grid_min_y:
            # 创建与局部网格对应的二值掩码
            grid_h = grid_max_y - grid_min_y + 1
            grid_w = grid_max_x - grid_min_x + 1
            sub_grid_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
            
            # 将多边形顶点映射到网格坐标系
            scaled_poly = draw_poly.copy().astype(np.float32)
            scaled_poly[:, 0] = (scaled_poly[:, 0] - grid_min_x * self.grid_size) / self.grid_size
            scaled_poly[:, 1] = (scaled_poly[:, 1] - grid_min_y * self.grid_size) / self.grid_size
            
            # 由于 OpenCV 的 fillPoly 要求整数点，网格坐标系下数值较小，我们可以用 sub-pixel 级别绘制来保持精度
            # 乘以位移因子 (shift=4 即 16 倍精度)
            shift = 4
            multiplier = 1 << shift
            scaled_poly_int = np.int32(np.round(scaled_poly * multiplier))
            
            cv2.fillPoly(sub_grid_mask, [scaled_poly_int], 1, shift=shift)
            
            # 用 numpy 切片直接进行批量覆盖更新
            self.coverage_grid[grid_min_y:grid_max_y+1, grid_min_x:grid_max_x+1] += sub_grid_mask
            self.coverage_mask[grid_min_y:grid_max_y+1, grid_min_x:grid_max_x+1] |= sub_grid_mask

    
    def add_trajectory(self, points: List[Tuple]):
        """
        批量添加轨迹点
        
        Args:
            points: 轨迹点列表 [(x1,y1), (x1,y1,angle), ...]
        """
        for pt in points:
            if len(pt) >= 3 and pt[2] is not None:
                self.add_point(pt[0], pt[1], float(pt[2]))
            else:
                self.add_point(pt[0], pt[1])
    
    def calculate_coverage_rate(self) -> float:
        """
        计算覆盖率
        
        Returns:
            覆盖率百分比 (0-100)
        """
        total_cells = self.grid_rows * self.grid_cols
        covered_cells = np.count_nonzero(self.coverage_mask)
        
        if total_cells == 0:
            return 0.0
        
        return (covered_cells / total_cells) * 100
    
    def get_coverage_mask_image(self) -> np.ndarray:
        """
        获取覆盖掩码图像（可视化用）
        
        Returns:
            BGR图像，绿色=已覆盖，红色=未覆盖
        """
        # 放大到床铺尺寸
        mask_vis = cv2.resize(
            self.coverage_mask * 255,
            (self.bed_width, self.bed_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        # 创建BGR图像
        result = np.zeros((self.bed_height, self.bed_width, 3), dtype=np.uint8)
        
        # 已覆盖区域（绿色）
        result[mask_vis > 0] = [0, 255, 0]
        
        # 未覆盖区域（红色）
        result[mask_vis == 0] = [0, 0, 255]
        
        return result
    
    def get_heatmap_image(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        获取热力图图像
        
        Args:
            colormap: OpenCV颜色映射
            
        Returns:
            BGR热力图图像
        """
        # 归一化热力图
        if self.heatmap.max() > 0:
            heatmap_norm = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap_norm = self.heatmap.astype(np.uint8)
        
        # 应用颜色映射
        heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)
        
        return heatmap_color
    
    def get_coverage_image(self, bed_image: np.ndarray, 
                          alpha: float = 0.5) -> np.ndarray:
        """
        获取覆盖叠加图像
        
        Args:
            bed_image: 床铺背景图像
            alpha: 透明度 (0-1)
            
        Returns:
            叠加后的图像
        """
        # 获取覆盖掩码图像
        coverage_vis = self.get_coverage_mask_image()
        
        # 叠加
        result = cv2.addWeighted(bed_image, 1 - alpha, coverage_vis, alpha, 0)
        
        # 绘制轨迹
        for i, (x, y) in enumerate(self.trajectory_points):
            cv2.circle(result, (x, y), 2, (255, 255, 0), -1)
        
        return result
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        total_cells = self.grid_rows * self.grid_cols
        covered_cells = np.count_nonzero(self.coverage_mask)
        coverage_rate = self.calculate_coverage_rate()
        
        # 计算覆盖密度
        if covered_cells > 0:
            avg_coverage = self.coverage_grid[self.coverage_mask > 0].mean()
            max_coverage = self.coverage_grid.max()
        else:
            avg_coverage = 0
            max_coverage = 0
        
        # 计算实际覆盖面积
        if self.use_real_unit:
            total_area_cm2 = self.real_width_cm * self.real_height_cm
            covered_area_cm2 = total_area_cm2 * coverage_rate / 100
            area_info = f"{covered_area_cm2:.0f} cm2 ({covered_area_cm2/10000:.2f} m2)"
        else:
            area_info = "未设置实际尺寸"
        
        return {
            'bed_size': f"{self.bed_width}x{self.bed_height}",
            'real_bed_size': f"{self.real_width_cm}x{self.real_height_cm} cm" if self.use_real_unit else "未设置",
            'real_remover_size': f"{self.config.real_remover_width_cm}x{self.config.real_remover_height_cm} cm" if self.use_real_unit else "未设置",
            'grid_size': self.grid_size,
            'grid_count': f"{self.grid_rows}x{self.grid_cols} ({total_cells})",
            'covered_cells': int(covered_cells),
            'coverage_rate': f"{coverage_rate:.2f}%",
            'covered_area': area_info,
            'trajectory_points': len(self.trajectory_points),
            'avg_coverage_per_cell': f"{avg_coverage:.1f}",
            'max_coverage_per_cell': int(max_coverage),
            'brush_size': f"{self.brush_width_px}x{self.brush_height_px} px ({self.config.real_brush_width_cm}x{self.config.real_brush_height_cm} cm)" if self.use_real_unit else f"{self.brush_width_px}x{self.brush_height_px} px"
        }
    
    def get_trajectory_array(self) -> np.ndarray:
        """
        获取轨迹点数组
        
        Returns:
            numpy数组，形状为 (N, 2)，每行为 (x, y)
        """
        if not self.trajectory_points:
            return np.array([])
        
        points = np.array([[p[0], p[1]] for p in self.trajectory_points])
        return points
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n=== 覆盖率统计 ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("==================\n")


def main():
    """测试函数"""
    # 创建测试数据
    bed_width, bed_height = 800, 600
    
    calculator = CoverageCalculator(bed_width, bed_height)
    
    # 模拟轨迹（螺旋形）
    center_x, center_y = bed_width // 2, bed_height // 2
    radius = 200
    turns = 5
    points_per_turn = 100
    
    for i in range(turns * points_per_turn):
        angle = 2 * np.pi * i / points_per_turn
        r = radius * (i / (turns * points_per_turn))
        x = int(center_x + r * np.cos(angle))
        y = int(center_y + r * np.sin(angle))
        
        calculator.add_point(x, y)
    
    # 打印统计
    calculator.print_statistics()
    
    # 生成可视化
    bed_image = np.ones((bed_height, bed_width, 3), dtype=np.uint8) * 255
    
    # 覆盖图
    coverage_img = calculator.get_coverage_image(bed_image)
    cv2.imwrite("outputs/coverage_test.jpg", coverage_img)
    
    # 热力图
    heatmap_img = calculator.get_heatmap_image()
    cv2.imwrite("outputs/heatmap_test.jpg", heatmap_img)
    
    print("测试图像已保存到 outputs/ 目录")


if __name__ == "__main__":
    main()
