"""
床铺区域选择器 - 通过鼠标点击选择4个角点，进行透视矫正
支持从图片或视频第一帧加载
"""

import cv2
import numpy as np
import os
from config import config_manager
from utils import imshow_adaptive


class BedSelector:
    """床铺区域选择器"""
    
    def __init__(self, video_path: str):
        """
        初始化选择器
        
        Args:
            video_path: 视频文件路径（用于提取第一帧）
        """
        self.video_path = video_path
        self.image = None
        self.height = 0
        self.width = 0
        self.display = None
        self.points = []  # 存储4个角点
        self.window_name = "Bed Area Selection"
        self.warped_image = None  # 透视变换后的图像
        
        # 加载图像
        self._load_image()
        
    def _load_image(self):
        """加载图像（从视频第一帧）"""
        if self.video_path and os.path.exists(self.video_path):
            # 从视频提取第一帧
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.image = frame
                    print(f"已从视频提取第一帧: {self.video_path}")
                else:
                    raise ValueError(f"无法读取视频帧: {self.video_path}")
            else:
                raise ValueError(f"无法打开视频: {self.video_path}")
        else:
            raise ValueError("请提供有效的视频路径")
        
        self.height, self.width = self.image.shape[:2]
        self.display = self.image.copy()
        
    def create_window(self):
        """创建窗口并设置鼠标回调"""
        # 获取屏幕分辨率并自适应缩放窗口，保持比例且不超出屏幕
        from utils import get_screen_resolution
        screen_w, screen_h = get_screen_resolution()
        
        max_ratio = 0.82
        max_w = int(screen_w * max_ratio)
        max_h = int(screen_h * max_ratio)
        
        scale = min(max_w / self.width, max_h / self.height)
        
        win_w = int(self.width * scale) if scale < 1.0 else self.width
        win_h = int(self.height * scale) if scale < 1.0 else self.height
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, win_w, win_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        # 左键添加点
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 4:
                print("已选择4个点，按 'R' 重置，按 'S' 保存")
                return
            self.points.append((x, y))
            print(f"第{len(self.points)}个点: ({x}, {y})")
            self.redraw()
            
        # 右键回退一步
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                removed_point = self.points.pop()
                print(f"回退一步，取消了点: {removed_point}")
                self.redraw()
        
    def redraw(self):
        """重绘显示图像"""
        self.display = self.image.copy()
        
        # 绘制已选择的点
        for i, point in enumerate(self.points):
            # 画点
            cv2.circle(self.display, point, 8, (0, 0, 255), -1)
            # 画编号
            cv2.putText(self.display, str(i + 1), 
                       (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 绘制连线
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, self.points[i], 
                        self.points[i + 1], (255, 0, 0), 2)
        
        # 闭合四边形
        if len(self.points) == 4:
            cv2.line(self.display, self.points[3], 
                    self.points[0], (255, 0, 0), 2)
            
        # 显示提示文字
        self._draw_help_text()
        
    def _draw_help_text(self):
        """绘制帮助文字"""
        help_text = [
            f"Selected: {len(self.points)}/4 points",
            "Controls:",
            "Left-Click - Add point",
            "Right-Click - Undo last point",
            "R - Reset all",
            "S - Save & Exit",
            "Q - Quit"
        ]
        
        y_offset = 30
        for text in help_text:
            cv2.putText(self.display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
    def perspective_transform(self):
        """
        执行透视变换
        
        Returns:
            透视变换后的图像
        """
        if len(self.points) != 4:
            raise ValueError("需要4个点才能进行透视变换")
        
        # 获取4个点
        pts1 = np.float32(self.points)
        
        # 计算宽度和高度
        width_top = np.linalg.norm(pts1[1] - pts1[0])
        width_bottom = np.linalg.norm(pts1[2] - pts1[3])
        height_left = np.linalg.norm(pts1[3] - pts1[0])
        height_right = np.linalg.norm(pts1[2] - pts1[1])
        
        # 使用最大宽度和高度
        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))
        
        # 目标点（矩形）
        pts2 = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 执行变换
        self.warped_image = cv2.warpPerspective(
            self.image, matrix, (width, height)
        )
        
        return self.warped_image
    
    def save_config(self, output_path: str = "bed_area_config.json"):
        """
        保存床铺区域配置
        
        Args:
            output_path: 配置文件保存路径
        """
        if len(self.points) != 4:
            raise ValueError("需要4个点才能保存配置")
        
        # 执行透视变换获取尺寸
        self.perspective_transform()
        
        # 立即销毁 OpenCV 窗口，避免在等待控制台输入时发生“窗口未响应”挂起/崩溃
        cv2.destroyAllWindows()
        
        # 询问实际床铺尺寸，默认沿用之前配置
        prev_bed_config = config_manager.get_bed_area_config()
        default_width = prev_bed_config.real_width_cm if prev_bed_config.real_width_cm > 0 else 150
        default_height = prev_bed_config.real_height_cm if prev_bed_config.real_height_cm > 0 else 200
        
        print(f"\n床铺区域像素尺寸: {self.warped_image.shape[1]}x{self.warped_image.shape[0]}")
        print(f"使用默认床铺实际尺寸: {default_width}x{default_height} cm (如需修改请前往配置菜单)")
        
        real_width = default_width
        real_height = default_height
        
        # 保存配置
        config_manager.set_bed_area(
            points=self.points,
            width=self.warped_image.shape[1],
            height=self.warped_image.shape[0],
            real_width_cm=real_width,
            real_height_cm=real_height,
            calibration_video_width=self.width,
            calibration_video_height=self.height
        )
        
        # 保存透视变换后的图像
        warped_path = "bed_area_warped.jpg"
        cv2.imwrite(warped_path, self.warped_image)
        
        print(f"\n配置已保存！")
        print(f"  像素尺寸: {self.warped_image.shape[1]}x{self.warped_image.shape[0]}")
        if real_width > 0 and real_height > 0:
            print(f"  实际尺寸: {real_width}x{real_height} 厘米")
            print(f"  比例: {real_width/self.warped_image.shape[1]:.3f} 厘米/像素")
        else:
            print("  实际尺寸: 未设置（将使用像素计算）")
        
    def run(self):
        """
        运行选择器
        
        Returns:
            bool: 是否成功保存配置
        """
        self.create_window()
        
        print("\n=== 床铺区域选择 ===")
        print("1. 在图片上点击4个角点（顺时针或逆时针）")
        print("2. 按 'S' 保存并退出")
        print("3. 按 'R' 重置选择")
        print("4. 按 'Q' 退出不保存\n")
        
        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF
            
            # 支持 ESC (27), Q/q 键以及右上角 X 按钮关闭窗口退出
            if key in [27, ord('q'), ord('Q')] or \
               (cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1):
                print("退出选择")
                cv2.destroyAllWindows()
                return False
                
            elif key == ord('r') or key == ord('R'):
                self.points = []
                self.warped_image = None
                self.redraw()
                print("已重置选择")
                
            elif key == ord('s') or key == ord('S'):
                if len(self.points) == 4:
                    try:
                        self.save_config()
                        cv2.destroyAllWindows()
                        return True
                    except Exception as e:
                        print(f"保存失败: {e}")
                else:
                    print(f"请先选择4个点（当前已选{len(self.points)}个）")
                    
    def show_preview(self):
        """显示透视变换预览"""
        if self.warped_image is None and len(self.points) == 4:
            self.perspective_transform()
            
        if self.warped_image is not None:
            imshow_adaptive("Perspective Warp Preview", self.warped_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Perspective Warp Preview")


def main():
    """测试函数"""
    # 测试从视频加载
    video_dir = "videos"
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) 
                 if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        if videos:
            video_path = os.path.join(video_dir, videos[0])
            print(f"使用视频: {video_path}")
            
            selector = BedSelector(video_path=video_path)
            success = selector.run()
            
            if success:
                print("\n床铺区域配置完成！")
                selector.show_preview()
            else:
                print("\n未保存配置")
            return
    
    print("请将测试录像放入 videos/ 目录")

if __name__ == "__main__":
    main()
