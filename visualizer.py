"""
可视化模块 - 生成覆盖率报告和可视化结果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List
from config import config_manager

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class CoverageVisualizer:
    """覆盖率可视化器"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.config = config_manager.get_coverage_config()
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def save_coverage_image(self, coverage_image: np.ndarray, 
                           filename: str = "coverage_result.jpg"):
        """
        保存覆盖图像
        
        Args:
            coverage_image: 覆盖图像
            filename: 文件名
        """
        filepath = f"{self.output_dir}/{filename}"
        cv2.imwrite(filepath, coverage_image)
        print(f"覆盖图像已保存: {filepath}")
    
    def save_heatmap(self, heatmap: np.ndarray, 
                    filename: str = "heatmap.jpg"):
        """
        保存热力图
        
        Args:
            heatmap: 热力图
            filename: 文件名
        """
        filepath = f"{self.output_dir}/{filename}"
        cv2.imwrite(filepath, heatmap)
        print(f"热力图已保存: {filepath}")
    
    def create_coverage_report(self, bed_image: np.ndarray,
                              coverage_image: np.ndarray,
                              heatmap: np.ndarray,
                              statistics: dict,
                              trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建综合报告图像
        
        Args:
            bed_image: 床铺原图
            coverage_image: 覆盖图像
            heatmap: 热力图
            statistics: 统计信息字典
            trajectory: 轨迹点数组 (N, 2)
            
        Returns:
            报告图像
        """
        # 调整所有图像到相同高度
        h, w = bed_image.shape[:2]
        
        # 调整大小
        coverage_resized = cv2.resize(coverage_image, (w, h))
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # 创建轨迹图
        trajectory_img = bed_image.copy()
        if trajectory is not None and len(trajectory) > 0:
            # 绘制轨迹线
            for i in range(len(trajectory) - 1):
                pt1 = tuple(trajectory[i].astype(int))
                pt2 = tuple(trajectory[i + 1].astype(int))
                cv2.line(trajectory_img, pt1, pt2, (0, 255, 255), 2)
            
            # 绘制起点和终点
            if len(trajectory) > 0:
                start = tuple(trajectory[0].astype(int))
                end = tuple(trajectory[-1].astype(int))
                cv2.circle(trajectory_img, start, 8, (0, 255, 0), -1)  # 绿色起点
                cv2.circle(trajectory_img, end, 8, (0, 0, 255), -1)    # 红色终点
        
        # 组合图像（2x2网格）
        top_row = np.hstack([bed_image, coverage_resized])
        bottom_row = np.hstack([trajectory_img, heatmap_resized])
        report = np.vstack([top_row, bottom_row])
        
        # 添加统计信息文字
        self._add_statistics_text(report, statistics)
        
        return report
    
    def _add_statistics_text(self, image: np.ndarray, statistics: dict):
        """
        在图像上添加统计信息
        
        Args:
            image: 图像
            statistics: 统计信息字典
        """
        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 文字
        y_offset = 35
        cv2.putText(image, "=== 覆盖率统计 ===", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # 关键指标
        key_items = [
            ('coverage_rate', '覆盖率'),
            ('covered_cells', '覆盖网格数'),
            ('trajectory_points', '轨迹点数'),
            ('brush_size', '主刷口大小')
        ]
        
        for key, label in key_items:
            if key in statistics:
                value = statistics[key]
                text = f"{label}: {value}"
                cv2.putText(image, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
    
    def create_coverage_curve(self, coverage_history: List[float],
                             timestamps: List[float],
                             filename: str = "coverage_curve.png"):
        """
        创建覆盖率曲线图
        
        Args:
            coverage_history: 覆盖率历史列表
            timestamps: 时间戳列表
            filename: 文件名
        """
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, coverage_history, 'b-', linewidth=2)
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('覆盖率 (%)', fontsize=12)
        plt.title('除螨仪覆盖率随时间变化', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(timestamps) if timestamps else 10)
        plt.ylim(0, 100)
        
        # 添加平均线
        if coverage_history:
            avg_coverage = np.mean(coverage_history)
            plt.axhline(y=avg_coverage, color='r', linestyle='--', 
                       label=f'平均覆盖率: {avg_coverage:.2f}%')
            plt.legend()
        
        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        print(f"覆盖率曲线已保存: {filepath}")
    
    def create_comparison_image(self, images: List[np.ndarray],
                               titles: List[str],
                               filename: str = "comparison.jpg") -> np.ndarray:
        """
        创建对比图像
        
        Args:
            images: 图像列表
            titles: 标题列表
            filename: 文件名
            
        Returns:
            对比图像
        """
        n = len(images)
        if n == 0:
            return np.array([])
        
        # 计算网格布局
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        # 调整所有图像到相同大小
        h, w = images[0].shape[:2]
        resized_images = [cv2.resize(img, (w, h)) for img in images]
        
        # 创建网格
        rows_images = []
        for i in range(rows):
            row_imgs = resized_images[i * cols:(i + 1) * cols]
            # 补齐
            while len(row_imgs) < cols:
                row_imgs.append(np.zeros_like(resized_images[0]))
            rows_images.append(np.hstack(row_imgs))
        
        comparison = np.vstack(rows_images)
        
        # 添加标题
        y_offset = 30
        for i, title in enumerate(titles):
            row = i // cols
            col = i % cols
            x = col * w + 10
            y = row * h + y_offset
            
            cv2.putText(comparison, title, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存
        filepath = f"{self.output_dir}/{filename}"
        cv2.imwrite(filepath, comparison)
        print(f"对比图像已保存: {filepath}")
        
        return comparison
    
    def create_summary_image(self, bed_image: np.ndarray,
                            coverage_image: np.ndarray,
                            heatmap: np.ndarray,
                            statistics: dict) -> np.ndarray:
        """
        创建摘要图像（单张大图）
        
        Args:
            bed_image: 床铺原图
            coverage_image: 覆盖图像
            heatmap: 热力图
            statistics: 统计信息
            
        Returns:
            摘要图像
        """
        h, w = bed_image.shape[:2]
        
        # 创建画布
        canvas_height = h + 200  # 额外空间放文字
        canvas = np.ones((canvas_height, w * 3, 3), dtype=np.uint8) * 255
        
        # 放置图像
        canvas[0:h, 0:w] = bed_image
        canvas[0:h, w:w*2] = cv2.resize(coverage_image, (w, h))
        canvas[0:h, w*2:w*3] = cv2.resize(heatmap, (w, h))
        
        # 添加标题
        cv2.putText(canvas, "原图", (w//2 - 30, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "覆盖图", (w + w//2 - 30, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "热力图", (2*w + w//2 - 30, h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 添加统计信息
        y_offset = h + 70
        cv2.putText(canvas, "=== 统计信息 ===", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30
        
        # 显示关键指标
        key_stats = [
            f"覆盖率: {statistics.get('coverage_rate', 'N/A')}",
            f"覆盖网格: {statistics.get('covered_cells', 'N/A')}",
            f"轨迹点数: {statistics.get('trajectory_points', 'N/A')}",
            f"主刷口大小: {statistics.get('brush_size', 'N/A')}"
        ]
        
        for stat in key_stats:
            cv2.putText(canvas, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 25
        
        return canvas


def main():
    """测试函数"""
    from coverage import CoverageCalculator
    
    # 创建测试数据
    bed_width, bed_height = 800, 600
    bed_image = np.ones((bed_height, bed_width, 3), dtype=np.uint8) * 255
    
    calculator = CoverageCalculator(bed_width, bed_height)
    
    # 模拟轨迹
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
    
    # 获取可视化
    coverage_img = calculator.get_coverage_image(bed_image)
    heatmap_img = calculator.get_heatmap_image()
    stats = calculator.get_statistics()
    trajectory = calculator.get_trajectory_array()
    
    # 创建可视化器
    visualizer = CoverageVisualizer()
    
    # 保存结果
    visualizer.save_coverage_image(coverage_img)
    visualizer.save_heatmap(heatmap_img)
    
    # 创建报告
    report = visualizer.create_coverage_report(
        bed_image, coverage_img, heatmap_img, stats, trajectory
    )
    visualizer.save_coverage_image(report, "report.jpg")
    
    # 创建摘要
    summary = visualizer.create_summary_image(
        bed_image, coverage_img, heatmap_img, stats
    )
    visualizer.save_coverage_image(summary, "summary.jpg")
    
    # 创建覆盖率曲线
    coverage_history = []
    timestamps = []
    for i in range(0, len(calculator.trajectory_points), 10):
        temp_calc = CoverageCalculator(bed_width, bed_height)
        temp_calc.add_trajectory(calculator.trajectory_points[:i+1])
        coverage_history.append(temp_calc.calculate_coverage_rate())
        timestamps.append(i / 30.0)  # 假设30fps
    
    visualizer.create_coverage_curve(coverage_history, timestamps)
    
    print("\n可视化完成!")


if __name__ == "__main__":
    main()
