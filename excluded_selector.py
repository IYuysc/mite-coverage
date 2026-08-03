"""
床铺排除区域（枕头/障碍物）选择器 - 通过鼠标点击多边形选择不参与覆盖率计算的区域
"""

import cv2
import numpy as np
import os
from config import config_manager
from utils import imshow_adaptive


class ExcludedAreaSelector:
    """床铺排除区域（枕头/障碍物）选择器"""
    
    def __init__(self, video_path: str = None):
        """
        初始化排除区域选择器
        
        Args:
            video_path: 视频文件路径（可选，当无 bed_area_warped.jpg 时提取第一帧生成）
        """
        self.video_path = video_path
        self.bed_config = config_manager.get_bed_area_config()
        self.warped_image = None
        self.display = None
        self.window_name = "Excluded Area Selection (Pillows / Obstacles)"
        
        self.polygons = []         # 已保存的多边形列表: [[(x1,y1), (x2,y2)...], ...]
        self.current_polygon = []  # 当前正在绘制的多边形顶点列表: [(x1,y1), ...]
        
        # 加载配置中已保存的排除多边形
        if hasattr(self.bed_config, 'excluded_polygons') and self.bed_config.excluded_polygons:
            for poly in self.bed_config.excluded_polygons:
                if poly and len(poly) >= 3:
                    self.polygons.append([tuple(pt) for pt in poly])
                    
        self._load_warped_image()
        
    def _load_warped_image(self):
        """加载或生成透视矫正后的床铺图像"""
        warped_path = "bed_area_warped.jpg"
        if os.path.exists(warped_path):
            self.warped_image = cv2.imread(warped_path)
            print(f"已加载床铺矫正图像: {warped_path}")
        else:
            # 如果没有现成的矫正图像，尝试通过视频和已有的床铺四角点矩阵生成
            if not self.bed_config.points or len(self.bed_config.points) != 4:
                raise ValueError("未找到床铺标定信息！请先执行“1. 标定床铺区域”。")
                
            # 寻找有效视频
            v_path = self.video_path
            if not v_path or not os.path.exists(v_path):
                if os.path.exists("videos"):
                    videos = [os.path.join("videos", f) for f in os.listdir("videos")
                              if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
                    if videos:
                        v_path = videos[0]
                        
            if not v_path or not os.path.exists(v_path):
                raise ValueError("未能获取用于透视矫正的视频画面，请先进行床铺区域标定。")
                
            cap = cv2.VideoCapture(v_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("无法读取视频画面进行透视矫正。")
                
            pts1 = np.float32(self.bed_config.points)
            pts2 = np.float32([
                [0, 0],
                [self.bed_config.width - 1, 0],
                [self.bed_config.width - 1, self.bed_config.height - 1],
                [0, self.bed_config.height - 1]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            self.warped_image = cv2.warpPerspective(frame, M, (self.bed_config.width, self.bed_config.height))
            cv2.imwrite(warped_path, self.warped_image)
            print(f"已基于视频帧生成并保存床铺矫正图像: {warped_path}")
            
        self.height, self.width = self.warped_image.shape[:2]
        self.display = self.warped_image.copy()
        
    def create_window(self):
        """创建窗口并设置鼠标回调"""
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
        """鼠标点击回调"""
        # 左键添加顶点
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
            print(f"添加排除区域顶点: ({x}, {y})")
            self.redraw()
            
        # 右键撤销
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) > 0:
                removed = self.current_polygon.pop()
                print(f"撤销顶点: {removed}")
            elif len(self.polygons) > 0:
                removed_poly = self.polygons.pop()
                print("撤销上一个完整的排除区域多边形")
            self.redraw()
            
    def redraw(self):
        """重绘显示图像"""
        base = self.warped_image.copy()
        overlay = base.copy()
        
        # 1. 绘制已完成的多边形排除区域
        for i, poly in enumerate(self.polygons):
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (40, 40, 40))  # 柔和深灰色半透明
            cv2.polylines(base, [pts], isClosed=True, color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)
            
            # 计算多边形中心并标注文字
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = poly[0]
            cv2.putText(base, f"Area #{i+1}", (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                        
        # 混合半透明图层
        cv2.addWeighted(overlay, 0.35, base, 0.65, 0, dst=base)
        
        # 2. 绘制正在添加的当前多边形
        if self.current_polygon:
            pts = np.array(self.current_polygon, dtype=np.int32)
            # 画圆点
            for pt in self.current_polygon:
                cv2.circle(base, pt, 5, (0, 255, 255), -1)
            # 画连线
            if len(self.current_polygon) > 1:
                cv2.polylines(base, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            # 如果点数 >= 3，用细线连接首尾 preview 封闭状态
            if len(self.current_polygon) >= 3:
                cv2.line(base, self.current_polygon[-1], self.current_polygon[0], (0, 255, 255), 1, cv2.LINE_AA)
                
        self.display = base
        self._draw_help_text()
        
    def _draw_help_text(self):
        """绘制控制提示"""
        help_text = [
            f"Saved Regions: {len(self.polygons)} | Current Points: {len(self.current_polygon)}",
            "Left-Click: Add point to polygon",
            "Right-Click: Undo point / region",
            "Enter / Space / N: Finish current region",
            "R: Reset all regions",
            "S: Save & Exit",
            "Q: Quit without save"
        ]
        
        y_offset = 25
        for text in help_text:
            cv2.putText(self.display, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(self.display, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            y_offset += 22
            
    def save_config(self):
        """保存排除区域配置"""
        # 如果当前有未完成的多边形且顶点 >= 3，自动闭合加入
        if len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon = []
            
        polygons_data = [[list(pt) for pt in poly] for poly in self.polygons]
        config_manager.set_excluded_polygons(polygons_data)
        
        print(f"\n✓ 排除区域配置已保存！共 {len(polygons_data)} 个排除区域（枕头/障碍物）。")
        
    def run(self) -> bool:
        """
        运行选择器
        
        Returns:
            bool: 是否成功保存
        """
        try:
            self.create_window()
        except Exception as e:
            print(f"打开排除区域标注窗口失败: {e}")
            return False
            
        print("\n=== 标注床铺排除区域（枕头等） ===")
        print("1. 鼠标左键连续点击多边形各顶点")
        print("2. 按 Enter / Space 键完成当前多边形（可标注多个枕头）")
        print("3. 按 'S' 键保存并退出")
        print("4. 按 'R' 重置，按 'Q' 退出\n")
        
        self.redraw()
        
        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(20) & 0xFF
            
            # 关闭/退出
            if key in [27, ord('q'), ord('Q')] or \
               (cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1):
                print("取消排除区域标注")
                cv2.destroyAllWindows()
                return False
                
            # 重置
            elif key in [ord('r'), ord('R'), ord('c'), ord('C')]:
                self.polygons = []
                self.current_polygon = []
                self.redraw()
                print("已重置所有排除区域")
                
            # 完成当前多边形
            elif key in [13, 32, ord('n'), ord('N')]:  # Enter, Space, N
                if len(self.current_polygon) >= 3:
                    self.polygons.append(list(self.current_polygon))
                    self.current_polygon = []
                    self.redraw()
                    print(f"已完成第 {len(self.polygons)} 个排除区域标注")
                else:
                    print("多边形至少需要3个顶点！")
                    
            # 保存并退出
            elif key in [ord('s'), ord('S')]:
                self.save_config()
                cv2.destroyAllWindows()
                return True


def main():
    """测试入口"""
    selector = ExcludedAreaSelector()
    selector.run()

if __name__ == "__main__":
    main()
