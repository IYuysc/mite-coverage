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
    blue_lower_h: int = 90
    blue_lower_s: int = 50
    blue_lower_v: int = 50
    blue_upper_h: int = 140
    blue_upper_s: int = 255
    blue_upper_v: int = 255
    min_area: int = 50
    morph_kernel_size: int = 5


@dataclass
class CoverageConfig:
    """覆盖率计算配置"""
    brush_width: int = 30   # 主刷口宽度（像素）
    brush_height: int = 15  # 主刷口高度（像素）
    grid_size: int = 10     # 网格大小（像素）
    output_dir: str = "outputs"


@dataclass
class BedAreaConfig:
    """床铺区域配置"""
    image_path: str = ""
    points: list = None  # 4个角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        if self.points is None:
            self.points = []


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
    
    def set_bed_area(self, image_path: str, points: list, width: int, height: int):
        """设置床铺区域"""
        self.config.bed_area = BedAreaConfig(
            image_path=image_path,
            points=points,
            width=width,
            height=height
        )
        self.save()


# 全局配置实例
config_manager = ConfigManager()
