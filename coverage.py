"""
覆盖率计算模块 - 计算除螨仪在床铺区域的覆盖率
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from config import config_manager


class CoverageCalculator:
    """覆盖率计算器"""
    
    def __init__(self, bed_width: int, bed_height: int):
        """
        初始化计算器
        
        Args:
            bed_width: 床铺区域宽度（像素）
            bed_height: 床铺区域高度（像素）
        """
        self.bed_width = bed_width
        self.bed_height = bed_height
        self.config = config_manager.get_coverage_config()
        
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
    
    def _get_brush_mask(self, center_x: int, center_y: int) -> np.ndarray:
        """
        获取主刷口掩码（矩形）
        
        Args:
            center_x: 中心x坐标
            center_y: 中心y坐标
            
        Returns:
            二值掩码，形状为 (bed_height, bed_width)
        """
        mask = np.zeros((self.bed_height, self.bed_width), dtype=np.uint8)
        
        # 绘制矩形主刷口
        half_w = self.config.brush_width // 2
        half_h = self.config.brush_height // 2
        
        top_left = (center_x - half_w, center_y - half_h)
        bottom_right = (center_x + half_w, center_y + half_h)
        
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        
        return mask
    
    def add_point(self, x: int, y: int):
        """
        添加轨迹点并更新覆盖
        
        Args:
            x: x坐标
            y: y坐标
        """
        # 边界检查
        x = max(0, min(x, self.bed_width - 1))
        y = max(0, min(y, self.bed_height - 1))
        
        self.trajectory_points.append((x, y))
        
        # 获取主刷口掩码
        brush_mask = self._get_brush_mask(x, y)
        
        # 更新热力图
        self.heatmap += brush_mask.astype(np.float32) * 0.1
        
        # 更新网格覆盖
        # 找到主刷口覆盖的网格范围（矩形）
        half_w = self.config.brush_width // 2
        half_h = self.config.brush_height // 2
        
        min_x = max(0, (x - half_w) // self.grid_size)
        max_x = min(self.grid_cols - 1, (x + half_w) // self.grid_size)
        min_y = max(0, (y - half_h) // self.grid_size)
        max_y = min(self.grid_rows - 1, (y + half_h) // self.grid_size)
        
        # 标记覆盖的网格
        for row in range(min_y, max_y + 1):
            for col in range(min_x, max_x + 1):
                # 计算网格中心
                grid_center_x = col * self.grid_size + self.grid_size // 2
                grid_center_y = row * self.grid_size + self.grid_size // 2
                
                # 检查是否在主刷口矩形范围内
                if (abs(grid_center_x - x) <= half_w and
                    abs(grid_center_y - y) <= half_h):
                    self.coverage_grid[row, col] += 1
                    self.coverage_mask[row, col] = 1
    
    def add_trajectory(self, points: List[Tuple[int, int]]):
        """
        批量添加轨迹点
        
        Args:
            points: 轨迹点列表 [(x1,y1), (x2,y2), ...]
        """
        for x, y in points:
            self.add_point(x, y)
    
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
        
        return {
            'bed_size': f"{self.bed_width}x{self.bed_height}",
            'grid_size': self.grid_size,
            'grid_count': f"{self.grid_rows}x{self.grid_cols} ({total_cells})",
            'covered_cells': int(covered_cells),
            'coverage_rate': f"{coverage_rate:.2f}%",
            'trajectory_points': len(self.trajectory_points),
            'avg_coverage_per_cell': f"{avg_coverage:.1f}",
            'max_coverage_per_cell': int(max_coverage),
            'brush_size': f"{self.config.brush_width}x{self.config.brush_height}"
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
