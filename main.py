"""
除螨仪覆盖率测试软件 - 主入口
"""

import os
import sys
import cv2
import numpy as np

# 导入模块
from config import config_manager
from bed_selector import BedSelector
from tracker import BlueTracker
from coverage import CoverageCalculator
from utils import imshow_adaptive, create_coverage_curve


class MiteCoverageApp:
    """除螨仪覆盖率测试应用"""
    
    def __init__(self):
        self.config = config_manager
        self.output_dir = "outputs"
        
    def show_menu(self):
        """显示主菜单"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 50)
        print("    除螨仪覆盖率测试软件 v1.0")
        print("=" * 50)
        print()
        print("请选择操作:")
        print("  1. 标定床铺区域（选择4个角点）")
        print("  2. 分析录像（计算覆盖率）")
        print("  3. 查看结果")
        print("  4. 配置参数")
        print("  5. 退出")
        print()
        
    def calibrate_bed_area(self):
        """标定床铺区域"""
        print("\n=== 床铺区域标定 ===")
        
        # 从视频标定
        video_dir = "videos"
        if not os.path.exists(video_dir):
            print(f"错误: 视频目录不存在 {video_dir}")
            input("按回车键继续...")
            return
        
        videos = [f for f in os.listdir(video_dir)
                 if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        
        if not videos:
            print(f"错误: 在 {video_dir} 目录中未找到视频文件")
            print("请将测试录像放入 videos/ 目录")
            input("按回车键继续...")
            return
        
        # 显示视频列表
        print("\n找到以下视频文件:")
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video}")
        
        if len(videos) == 1:
            video_path = os.path.join(video_dir, videos[0])
        else:
            try:
                video_choice = int(input("\n请选择视频编号: ")) - 1
                if 0 <= video_choice < len(videos):
                    video_path = os.path.join(video_dir, videos[video_choice])
                else:
                    print("无效选择")
                    return
            except ValueError:
                print("无效输入")
                return
        
        try:
            # 从视频第一帧标定
            selector = BedSelector(video_path=video_path)
            
            # 运行选择
            success = selector.run()
            
            if success:
                print("\n✓ 床铺区域标定成功！")
                print(f"  区域尺寸: {selector.warped_image.shape[1]}x{selector.warped_image.shape[0]}")
                print("  配置已保存到 config.json")
            else:
                print("\n✗ 未保存配置")
                
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n按回车键继续...")
    
    def analyze_video(self):
        """分析录像"""
        print("\n=== 录像分析 ===")
        
        # 检查床铺区域配置
        bed_config = self.config.get_bed_area_config()
        if not bed_config.points or len(bed_config.points) != 4:
            print("错误: 请先标定床铺区域（选择菜单1）")
            input("按回车键继续...")
            return
        
        # 查找视频
        video_dir = "videos"
        if not os.path.exists(video_dir):
            print(f"错误: 视频目录不存在 {video_dir}")
            input("按回车键继续...")
            return
        
        videos = [f for f in os.listdir(video_dir) 
                 if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        
        if not videos:
            print(f"错误: 在 {video_dir} 目录中未找到视频文件")
            print("支持的格式: .mp4, .avi, .mov")
            input("按回车键继续...")
            return
        
        # 显示视频列表
        print("\n找到以下视频文件:")
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video}")
        
        if len(videos) == 1:
            video_path = os.path.join(video_dir, videos[0])
        else:
            try:
                choice = int(input("\n请选择视频编号: ")) - 1
                if 0 <= choice < len(videos):
                    video_path = os.path.join(video_dir, videos[choice])
                else:
                    print("无效选择")
                    return
            except ValueError:
                print("无效输入")
                return
        
        try:
            # 1. 追踪蓝色标记
            print(f"\n开始分析视频: {video_path}")
            tracker = BlueTracker(video_path)
            trajectory, bed_mask, first_frame = tracker.track(
                output_video="outputs/tracking_result.mp4"
            )
            if len(trajectory) < 10:
                print("警告: 轨迹点太少，请检查视频和颜色设置")
                return
            
            # 2. 透视变换轨迹点到床铺坐标系
            # 直接使用 tracker 在追踪过程中通过双色贴纸和视差计算出的高精度床面坐标和角度
            bed_points = []
            bed_config = self.config.get_bed_area_config()
            for pt in trajectory:
                if pt.bed_x is not None and pt.bed_y is not None:
                    # 边界检查
                    x = max(0, min(pt.bed_x, bed_config.width - 1))
                    y = max(0, min(pt.bed_y, bed_config.height - 1))
                    bed_points.append((x, y, pt.bed_angle))
            
            if not bed_points:
                print("错误: 坐标转换失败")
                return
            
            # 3. 计算覆盖率
            print("计算覆盖率...")
            bed_config = self.config.get_bed_area_config()
            calculator = CoverageCalculator(
                bed_config.width,
                bed_config.height,
                bed_config.real_width_cm,
                bed_config.real_height_cm
            )
            calculator.add_trajectory(bed_points)
            
            # 4. 生成可视化
            print("生成可视化结果...")
            
            # 1. 生成热力图 (透视变换回原始视频第一帧视角并进行半透明叠加)
            if first_frame is not None and tracker.has_bed_config:
                if calculator.heatmap.max() > 0:
                    heatmap_norm = (calculator.heatmap / calculator.heatmap.max() * 255).astype(np.uint8)
                else:
                    heatmap_norm = calculator.heatmap.astype(np.uint8)
                
                # 透视变换回第一帧视角
                heatmap_warped_back = cv2.warpPerspective(heatmap_norm, tracker.inv_matrix, (tracker.width, tracker.height))
                
                # 应用色彩映射得到彩色热力图
                heatmap_color = cv2.applyColorMap(heatmap_warped_back, cv2.COLORMAP_JET)
                
                # 叠加到第一帧上 (只在扫掠过的区域进行混合)
                heatmap_img = first_frame.copy()
                mask_swept = heatmap_warped_back > 0
                alpha_heatmap = 0.6
                heatmap_img[mask_swept] = cv2.addWeighted(
                    first_frame, 1 - alpha_heatmap, heatmap_color, alpha_heatmap, 0
                )[mask_swept]
            else:
                # 兜底：直接输出原版拉平后的热力图
                heatmap_img = calculator.get_heatmap_image()

            stats = calculator.get_statistics()
            coverage_rate = stats.get('coverage_rate', '0.00%')

            # 保存热力图到 outputs 中
            os.makedirs(self.output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.output_dir, "heatmap.jpg"), heatmap_img)
            
            # 2. 计算覆盖率随时间变化曲线并保存为临时图片
            coverage_history = []
            timestamps = []
            step = max(1, len(bed_points) // 100)
            
            # 实例化计算器，使用相同的尺寸与实际物理参数，避免网格和刷头比例不一致
            temp_calc = CoverageCalculator(
                bed_config.width,
                bed_config.height,
                bed_config.real_width_cm,
                bed_config.real_height_cm
            )
            
            for i in range(0, len(bed_points), step):
                # 增量式添加点，避免 O(N^2) 重复计算
                current_points = bed_points[len(temp_calc.trajectory_points) : i + 1]
                temp_calc.add_trajectory(current_points)
                
                coverage_history.append(temp_calc.calculate_coverage_rate())
                # 使用 tracker 传回的真实时间戳
                timestamps.append(trajectory[i].timestamp)
            
            temp_curve_name = "temp_curve.png"
            if coverage_history:
                create_coverage_curve(
                    coverage_history, timestamps, os.path.join(self.output_dir, temp_curve_name)
                )
            
            # 3. 创建完整的淡蓝色扫掠路线图 (透视变换回原始视频第一帧视角并进行半透明叠加)
            if first_frame is not None and tracker.has_bed_config:
                coverage_img = first_frame.copy()
                if bed_mask is not None and np.any(bed_mask > 0):
                    bed_mask_warped_back = cv2.warpPerspective(bed_mask, tracker.inv_matrix, (tracker.width, tracker.height))
                    overlay = first_frame.copy()
                    overlay[bed_mask_warped_back > 0] = [235, 206, 135]  # 淡蓝色 BGR [235, 206, 135]
                    cv2.addWeighted(overlay, 0.35, coverage_img, 0.65, 0, dst=coverage_img)
            else:
                # 兜底：采用白底画布
                bed_h = bed_config.height or 500
                bed_w = bed_config.width or 800
                coverage_img = np.ones((bed_h, bed_w, 3), dtype=np.uint8) * 255
                if bed_mask is not None and np.any(bed_mask > 0):
                    overlay = coverage_img.copy()
                    overlay[bed_mask > 0] = [235, 206, 135]
                    cv2.addWeighted(overlay, 0.35, coverage_img, 0.65, 0, dst=coverage_img)
            
            # 在图上叠加清洁覆盖率文字，放大边框和字体
            overlay_txt = coverage_img.copy()
            cv2.rectangle(overlay_txt, (20, 20), (550, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay_txt, 0.6, coverage_img, 0.4, 0, dst=coverage_img)
            cv2.putText(coverage_img, f"Coverage: {coverage_rate}", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
            
            # 4. 拼接床铺扫掠图与时间变化曲线图
            curve_path = os.path.join(self.output_dir, temp_curve_name)
            final_combined = coverage_img
            if os.path.exists(curve_path):
                curve_img = cv2.imread(curve_path)
                if curve_img is not None:
                    h_bed, w_bed = coverage_img.shape[:2]
                    h_curve, w_curve = curve_img.shape[:2]
                    w_curve_new = int(w_curve * h_bed / h_curve)
                    curve_resized = cv2.resize(curve_img, (w_curve_new, h_bed))
                    final_combined = np.hstack([coverage_img, curve_resized])
                
                # 删除临时曲线图片以保持目录整洁
                try:
                    os.remove(curve_path)
                except Exception:
                    pass
            
            # 5. 保存拼合后的综合结果图为 report.jpg
            cv2.imwrite(os.path.join(self.output_dir, "report.jpg"), final_combined)
            
            print("\n✓ 分析完成！")
            print(f"  热力图已保存: {self.output_dir}/heatmap.jpg")
            print(f"  覆盖率报告已保存: {self.output_dir}/report.jpg")
            print(f"  主刷口大小: {calculator.brush_width_px}x{calculator.brush_height_px} 像素 ({calculator.config.real_brush_width_cm}x{calculator.config.real_brush_height_cm} 厘米)")
            
            # 显示实际覆盖面积
            if calculator.use_real_unit:
                print(f"  床铺实际尺寸: {stats['real_bed_size']}")
                print(f"  实际覆盖面积: {stats['covered_area']}")
            
            # 6. 分析结束后，仅弹出一张完整淡蓝色路线+覆盖率+曲线的综合结果图
            win_name = "Analysis Report (Coverage & Trend Curve)"
            imshow_adaptive(win_name, final_combined, max_ratio=0.85)
            
            print("\n按任意键、ESC、或点击右上角 X 关闭窗口以继续...")
            while True:
                key = cv2.waitKey(100) & 0xFF
                # 支持按任意键或 ESC 退出
                if key != 255:
                    break
                # 检测右上角 X 关闭按钮
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n按回车键继续...")
    
    
    def view_results(self):
        """查看结果"""
        print("\n=== 查看结果 ===")
        
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            print(f"输出目录不存在: {output_dir}")
            input("按回车键继续...")
            return
        
        files = os.listdir(output_dir)
        image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print("未找到结果图像")
            input("按回车键继续...")
            return
        
        print("\n找到以下结果文件:")
        for i, file in enumerate(image_files, 1):
            print(f"  {i}. {file}")
        
        # 显示最新结果
        print("\n显示最新结果...")
        
        # 尝试显示各种结果
        result_files = {
            'report.jpg': 'Analysis Report (Coverage & Trend)',
            'heatmap.jpg': 'Heatmap (Standalone)'
        }
        
        for filename, title in result_files.items():
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    imshow_adaptive(title, img)
        
        # 我们用已打开的窗口来监控关闭按钮
        active_windows = [title for filename, title in result_files.items() if os.path.exists(os.path.join(output_dir, filename))]
        
        print("\n按任意键、ESC、或点击任意窗口右上角 X 关闭窗口以继续...")
        while active_windows:
            key = cv2.waitKey(100) & 0xFF
            # 如果按下了任何键
            if key != 255:
                break
            # 如果任何一个窗口被关闭了，我们也退出
            any_closed = False
            for win in active_windows:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    any_closed = True
                    break
            if any_closed:
                break
                
        cv2.destroyAllWindows()
        
        input("\n按回车键继续...")
    
    def configure_parameters(self):
        """配置参数"""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 50)
            print("    除螨仪覆盖率测试软件 - 参数配置")
            print("=" * 50)
            
            tracker_config = self.config.get_tracker_config()
            coverage_config = self.config.get_coverage_config()
            bed_config = self.config.get_bed_area_config()
            
            print("当前参数值:")
            print(f"  1. 床铺实际尺寸 (宽x高): {bed_config.real_width_cm}x{bed_config.real_height_cm} cm")
            print(f"  2. 除螨仪实际尺寸 (宽x高): {coverage_config.real_remover_width_cm}x{coverage_config.real_remover_height_cm} cm")
            print(f"  3. 主刷口实际尺寸 (宽x高): {coverage_config.real_brush_width_cm}x{coverage_config.real_brush_height_cm} cm")
            print(f"  4. 蓝色追踪颜色 HSV 范围下限: [{tracker_config.blue_lower_h}, {tracker_config.blue_lower_s}, {tracker_config.blue_lower_v}]")
            print(f"  5. 蓝色追踪颜色 HSV 范围上限: [{tracker_config.blue_upper_h}, {tracker_config.blue_upper_s}, {tracker_config.blue_upper_v}]")
            print(f"  6. 绿色追踪颜色 HSV 范围下限: [{tracker_config.green_lower_h}, {tracker_config.green_lower_s}, {tracker_config.green_lower_v}]")
            print(f"  7. 绿色追踪颜色 HSV 范围上限: [{tracker_config.green_upper_h}, {tracker_config.green_upper_s}, {tracker_config.green_upper_v}]")
            print(f"  8. 最小检测区域 (像素): {tracker_config.min_area}")
            print(f"  9. 覆盖网格大小 (像素): {coverage_config.grid_size}")
            print(f"  10. 贴纸高度视差补偿 (cm): {getattr(tracker_config, 'parallax_height_cm', 0.0)} cm")
            print(f"  11. 输出视频播放倍速: {getattr(tracker_config, 'output_video_speed', 1.0)}")
            print(f"  12. 返回主菜单")
            print()
            
            choice = input("请选择要修改的配置项 (1-12): ").strip()
            
            if choice == '1':
                try:
                    w = int(input(f"床铺实际宽度 (当前 {bed_config.real_width_cm} cm): ").strip() or str(bed_config.real_width_cm))
                    h = int(input(f"床铺实际长度 (当前 {bed_config.real_height_cm} cm): ").strip() or str(bed_config.real_height_cm))
                    bed_config.real_width_cm = w
                    bed_config.real_height_cm = h
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '2':
                try:
                    w = float(input(f"除螨仪实际宽度 (当前 {coverage_config.real_remover_width_cm} cm): ").strip() or str(coverage_config.real_remover_width_cm))
                    h = float(input(f"除螨仪实际长度 (当前 {coverage_config.real_remover_height_cm} cm): ").strip() or str(coverage_config.real_remover_height_cm))
                    coverage_config.real_remover_width_cm = w
                    coverage_config.real_remover_height_cm = h
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '3':
                try:
                    w = float(input(f"主刷口实际宽度 (当前 {coverage_config.real_brush_width_cm} cm): ").strip() or str(coverage_config.real_brush_width_cm))
                    h = float(input(f"主刷口实际高度 (当前 {coverage_config.real_brush_height_cm} cm): ").strip() or str(coverage_config.real_brush_height_cm))
                    coverage_config.real_brush_width_cm = w
                    coverage_config.real_brush_height_cm = h
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '4':
                try:
                    h_val = int(input(f"H (当前 {tracker_config.blue_lower_h}): ").strip() or str(tracker_config.blue_lower_h))
                    s_val = int(input(f"S (当前 {tracker_config.blue_lower_s}): ").strip() or str(tracker_config.blue_lower_s))
                    v_val = int(input(f"V (当前 {tracker_config.blue_lower_v}): ").strip() or str(tracker_config.blue_lower_v))
                    tracker_config.blue_lower_h = h_val
                    tracker_config.blue_lower_s = s_val
                    tracker_config.blue_lower_v = v_val
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '5':
                try:
                    h_val = int(input(f"H (当前 {tracker_config.blue_upper_h}): ").strip() or str(tracker_config.blue_upper_h))
                    s_val = int(input(f"S (当前 {tracker_config.blue_upper_s}): ").strip() or str(tracker_config.blue_upper_s))
                    v_val = int(input(f"V (当前 {tracker_config.blue_upper_v}): ").strip() or str(tracker_config.blue_upper_v))
                    tracker_config.blue_upper_h = h_val
                    tracker_config.blue_upper_s = s_val
                    tracker_config.blue_upper_v = v_val
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '6':
                try:
                    h_val = int(input(f"H (当前 {tracker_config.green_lower_h}): ").strip() or str(tracker_config.green_lower_h))
                    s_val = int(input(f"S (当前 {tracker_config.green_lower_s}): ").strip() or str(tracker_config.green_lower_s))
                    v_val = int(input(f"V (当前 {tracker_config.green_lower_v}): ").strip() or str(tracker_config.green_lower_v))
                    tracker_config.green_lower_h = h_val
                    tracker_config.green_lower_s = s_val
                    tracker_config.green_lower_v = v_val
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '7':
                try:
                    h_val = int(input(f"H (当前 {tracker_config.green_upper_h}): ").strip() or str(tracker_config.green_upper_h))
                    s_val = int(input(f"S (当前 {tracker_config.green_upper_s}): ").strip() or str(tracker_config.green_upper_s))
                    v_val = int(input(f"V (当前 {tracker_config.green_upper_v}): ").strip() or str(tracker_config.green_upper_v))
                    tracker_config.green_upper_h = h_val
                    tracker_config.green_upper_s = s_val
                    tracker_config.green_upper_v = v_val
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '8':
                try:
                    area = int(input(f"最小检测面积 (当前 {tracker_config.min_area}): ").strip() or str(tracker_config.min_area))
                    tracker_config.min_area = area
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '9':
                try:
                    size = int(input(f"覆盖网格大小 (当前 {coverage_config.grid_size}): ").strip() or str(coverage_config.grid_size))
                    coverage_config.grid_size = size
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '10':
                try:
                    val = float(input(f"贴纸高度视差补偿 (当前 {getattr(tracker_config, 'parallax_height_cm', 0.0)} cm): ").strip() or str(getattr(tracker_config, 'parallax_height_cm', 0.0)))
                    tracker_config.parallax_height_cm = val
                    self.config.save()
                    print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '11':
                try:
                    val = float(input(f"输出视频播放倍速 (当前 {getattr(tracker_config, 'output_video_speed', 1.0)}): ").strip() or str(getattr(tracker_config, 'output_video_speed', 1.0)))
                    if val <= 0:
                        print("✗ 倍速必须大于 0！")
                    else:
                        tracker_config.output_video_speed = val
                        self.config.save()
                        print("✓ 修改成功！已保存。")
                except ValueError:
                    print("✗ 输入无效！")
                input("按回车键继续...")
            elif choice == '12':
                break
            else:
                print("✗ 无效选择！")
                input("按回车键继续...")
    
    def run(self):
        """运行应用"""
        while True:
            self.show_menu()
            
            try:
                choice = input("请输入选项 (1-5): ").strip()
                
                if choice == '1':
                    self.calibrate_bed_area()
                elif choice == '2':
                    self.analyze_video()
                elif choice == '3':
                    self.view_results()
                elif choice == '4':
                    self.configure_parameters()
                elif choice == '5':
                    print("\n感谢使用！")
                    sys.exit(0)
                else:
                    print("\n无效选项，请重新选择")
                    input("按回车键继续...")
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                sys.exit(0)
            except Exception as e:
                print(f"\n错误: {e}")
                import traceback
                traceback.print_exc()
                input("按回车键继续...")


def main():
    """主函数"""
    # 检查依赖
    try:
        import cv2
        import numpy as np
        print("依赖检查通过")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 创建必要目录
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # 运行应用
    app = MiteCoverageApp()
    app.run()


if __name__ == "__main__":
    main()
