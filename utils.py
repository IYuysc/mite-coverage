"""
除螨仪覆盖率测试软件 - 辅助工具函数
"""

import cv2
import numpy as np

def get_screen_resolution():
    """获取屏幕分辨率，带有简单缓存并安全回退"""
    if hasattr(get_screen_resolution, "_cache"):
        return get_screen_resolution._cache
        
    try:
        import tkinter as tk
        root = tk.Tk()
        # 隐藏窗口
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        get_screen_resolution._cache = (w, h)
        return w, h
    except Exception:
        # 默认安全回退分辨率
        return 1280, 720

def imshow_adaptive(window_name: str, img: np.ndarray, max_ratio: float = 0.82):
    """自适应屏幕分辨率并保持宽高比展示图像"""
    if img is None:
        return
        
    if not hasattr(imshow_adaptive, "_initialized_windows"):
        imshow_adaptive._initialized_windows = set()
        
    if window_name not in imshow_adaptive._initialized_windows:
        h, w = img.shape[:2]
        screen_w, screen_h = get_screen_resolution()
        
        # 预留任务栏和窗口边框空间
        max_w = int(screen_w * max_ratio)
        max_h = int(screen_h * max_ratio)
        
        scale = min(max_w / w, max_h / h)
        
        # 总是按比例进行自适应缩放（无论是放大还是缩小），使其在 2K/4K 高分屏下也能清晰大图显示
        win_w = int(round(w * scale))
        win_h = int(round(h * scale))
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, win_w, win_h)
        cv2.moveWindow(window_name, int(screen_w/2 - win_w/2), int(screen_h/2 - win_h/2))
        imshow_adaptive._initialized_windows.add(window_name)
        
    cv2.imshow(window_name, img)




def create_coverage_curve(history: list, timestamps: list, output_path: str):
    """绘制覆盖率随时间变化的曲线并保存"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(timestamps, history, color='#29B6F6', linewidth=2.5)
    plt.title("Coverage Rate Trend", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Coverage Rate (%)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def draw_hatched_excluded_region(img: np.ndarray, mask: np.ndarray, label: str = None) -> np.ndarray:
    """
    在图像上的 mask 区域绘制柔和自然的灰色半透明阴影与浅灰细斜纹，表示未计算入覆盖率
    """
    if img is None or mask is None or not np.any(mask > 0):
        return img
        
    h, w = img.shape[:2]
    mask_uint8 = mask.astype(np.uint8)
    mask_bool = mask_uint8 > 0
    
    # 1. 柔和自然的半透明灰色阴影
    overlay = img.copy()
    overlay[mask_bool] = [60, 60, 60]  # 柔和中灰色 BGR
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, dst=img)
    
    # 2. 绘制极淡浅灰细斜纹 (非死黑，柔和半透明融入)
    hatch_mask = np.zeros((h, w), dtype=np.uint8)
    stripe_step = 14  # 斜纹间距
    for d in range(-h, w + h, stripe_step):
        cv2.line(hatch_mask, (d, 0), (d + h, h), 255, 1, cv2.LINE_AA)
        
    hatch_in_mask = cv2.bitwise_and(hatch_mask, hatch_mask, mask=mask_uint8)
    
    # 柔和混合斜纹，避免过深的黑色斜线条
    hatch_overlay = img.copy()
    hatch_overlay[hatch_in_mask > 0] = [80, 80, 80]
    cv2.addWeighted(hatch_overlay, 0.35, img, 0.65, 0, dst=img)
    
    # 3. 柔和细灰轮廓边框
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (110, 110, 110), 1, cv2.LINE_AA)
    
    return img



