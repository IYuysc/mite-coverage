"""
除螨仪覆盖率测试软件 - 配置管理
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional


@dataclass
class TrackerConfig:
    """追踪器配置"""
    blue_lower_h: int = 105
    blue_lower_s: int = 130
    blue_lower_v: int = 30
    blue_upper_h: int = 130
    blue_upper_s: int = 255
    blue_upper_v: int = 255
    
    green_lower_h: int = 55
    green_lower_s: int = 120
    green_lower_v: int = 80
    green_upper_h: int = 85
    green_upper_s: int = 255
    green_upper_v: int = 255
    
    min_area: int = 50
    morph_kernel_size: int = 5
    sticker_layout: str = "side_by_side"    # 视差高度补偿 (cm) - 用户指定标定纸悬空高度为 3cm
    parallax_height_cm: float = 3.0     # 贴纸物理高度（厘米），用于消除斜视视差投影误差
    



@dataclass
class CoverageConfig:
    """覆盖率计算配置"""
    brush_width: int = 30   # 主刷口宽度（像素）
    brush_height: int = 15  # 主刷口高度（像素）
    grid_size: int = 10     # 网格大小（像素）
    output_dir: str = "outputs"
    
    # 实际物理尺寸（厘米）
    real_brush_width_cm: float =2.5    # 主刷口实际宽度
    real_brush_height_cm: float = 13.0    # 主刷口实际长度
    real_remover_width_cm: float = 20.0  # 除螨仪实际宽度
    real_remover_height_cm: float = 27.0 # 除螨仪实际长度


@dataclass
class BedAreaConfig:
    """床铺区域配置"""
    points: list = None  # 4个角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    width: int = 0           # 像素宽度
    height: int = 0          # 像素高度
    real_width_cm: int = 0   # 实际宽度（厘米）
    real_height_cm: int = 0  # 实际高度（厘米）
    calibration_video_width: int = 0   # 标定时的原视频宽度
    calibration_video_height: int = 0  # 标定时的原视频高度
    
    def __post_init__(self):
        if self.points is None:
            self.points = []

    def get_scaled_points(self, current_width: int, current_height: int) -> list:
        """根据当前视频尺寸自适应缩放标定的床面四角点"""
        if not self.points or len(self.points) != 4:
            return []
        # 如果没有记录标定分辨率，则回退到原始坐标
        if self.calibration_video_width <= 0 or self.calibration_video_height <= 0:
            return self.points
            
        scale_x = current_width / self.calibration_video_width
        scale_y = current_height / self.calibration_video_height
        
        return [[int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))] for pt in self.points]


@dataclass
class Config:
    """全局配置"""
    tracker: TrackerConfig = None
    coverage: CoverageConfig = None
    bed_area: BedAreaConfig = None
    
    def __post_init__(self):
        if self.tracker is None:
            self.tracker = TrackerConfig()
        if self.coverage is None:
            self.coverage = CoverageConfig()
        if self.bed_area is None:
            self.bed_area = BedAreaConfig()


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = Config()
        self.load()
    
    def load(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新配置
                if 'tracker' in data:
                    self.config.tracker = TrackerConfig(**data['tracker'])
                if 'coverage' in data:
                    self.config.coverage = CoverageConfig(**data['coverage'])
                if 'bed_area' in data:
                    self.config.bed_area = BedAreaConfig(**data['bed_area'])
                    
                print(f"配置已加载: {self.config_file}")
            except Exception as e:
                print(f"加载配置失败: {e}")
                self.config = Config()
    
    def save(self):
        """保存配置到文件"""
        try:
            data = {
                'tracker': asdict(self.config.tracker),
                'coverage': asdict(self.config.coverage),
                'bed_area': asdict(self.config.bed_area)
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"配置已保存: {self.config_file}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def get_tracker_config(self) -> TrackerConfig:
        return self.config.tracker
    
    def get_coverage_config(self) -> CoverageConfig:
        return self.config.coverage
    
    def get_bed_area_config(self) -> BedAreaConfig:
        return self.config.bed_area
    
    def set_bed_area(self, points: list, width: int, height: int, 
                     real_width_cm: int = 0, real_height_cm: int = 0,
                     calibration_video_width: int = 0, calibration_video_height: int = 0):
        """设置床铺区域"""
        self.config.bed_area = BedAreaConfig(
            points=points,
            width=width,
            height=height,
            real_width_cm=real_width_cm,
            real_height_cm=real_height_cm,
            calibration_video_width=calibration_video_width,
            calibration_video_height=calibration_video_height
        )
        self.save()


# 全局配置实例
config_manager = ConfigManager()
