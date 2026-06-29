"""
除螨仪覆盖率测试软件 - 主入口
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional

# 导入模块
from config import config_manager
from bed_selector import BedSelector
from tracker import BlueTracker
from coverage import CoverageCalculator
from visualizer import CoverageVisualizer


class MiteCoverageApp:
    """除螨仪覆盖率测试应用"""
    
    def __init__(self):
        self.config = config_manager
        self.visualizer = CoverageVisualizer()
        
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
        print("请选择标定方式:")
        print("  1. 从视频第一帧标定（推荐）")
        print("  2. 从图片标定")
        
        choice = input("请输入选项 (1/2): ").strip()
        
        try:
            if choice == '1':
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
                
                # 从视频第一帧标定
                selector = BedSelector(video_path=video_path)
                
            else:
                # 从图片标定
                image_path = "images/test.jpg"
                
                if not os.path.exists(image_path):
                    print(f"错误: 找不到测试图片 {image_path}")
                    print("请将床铺照片放入 images/ 目录，命名为 test.jpg")
                    input("按回车键继续...")
                    return
                
                selector = BedSelector(image_path=image_path)
            
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
            trajectory = tracker.track(
                output_video="outputs/tracking_result.mp4"
            )
            tracker.save_trajectory("outputs/trajectory.csv")
            
            if len(trajectory) < 10:
                print("警告: 轨迹点太少，请检查视频和颜色设置")
                return
            
            # 2. 透视变换轨迹点到床铺坐标系
            print("\n转换轨迹坐标...")
            bed_points = self._transform_trajectory(trajectory)
            
            if not bed_points:
                print("错误: 坐标转换失败")
                return
            
            # 3. 计算覆盖率
            print("计算覆盖率...")
            bed_config = self.config.get_bed_area_config()
            calculator = CoverageCalculator(
                bed_config.width, bed_config.height
            )
            calculator.add_trajectory(bed_points)
            
            # 打印统计
            calculator.print_statistics()
            
            # 4. 生成可视化
            print("生成可视化结果...")
            
            # 加载床铺图像
            bed_image = cv2.imread(bed_config.image_path)
            if bed_image is None:
                bed_image = np.ones(
                    (bed_config.height, bed_config.width, 3), 
                    dtype=np.uint8
                ) * 255
            
            # 透视变换床铺图像
            selector = BedSelector(bed_config.image_path)
            selector.points = bed_config.points
            bed_warped = selector.perspective_transform()
            
            # 生成各种图像
            coverage_img = calculator.get_coverage_image(bed_warped)
            heatmap_img = calculator.get_heatmap_image()
            trajectory_array = calculator.get_trajectory_array()
            stats = calculator.get_statistics()
            
            # 保存结果
            self.visualizer.save_coverage_image(coverage_img, "coverage_result.jpg")
            self.visualizer.save_heatmap(heatmap_img, "heatmap.jpg")
            
            # 创建报告
            report = self.visualizer.create_coverage_report(
                bed_warped, coverage_img, heatmap_img, stats, trajectory_array
            )
            self.visualizer.save_coverage_image(report, "report.jpg")
            
            # 创建摘要
            summary = self.visualizer.create_summary_image(
                bed_warped, coverage_img, heatmap_img, stats
            )
            self.visualizer.save_coverage_image(summary, "summary.jpg")
            
            # 创建覆盖率曲线
            coverage_history = []
            timestamps = []
            step = max(1, len(bed_points) // 100)
            
            for i in range(0, len(bed_points), step):
                temp_calc = CoverageCalculator(
                    bed_config.width, bed_config.height
                )
                temp_calc.add_trajectory(bed_points[:i+1])
                coverage_history.append(temp_calc.calculate_coverage_rate())
                timestamps.append(bed_points[i][0] / 30.0)  # 假设30fps
            
            if coverage_history:
                self.visualizer.create_coverage_curve(
                    coverage_history, timestamps
                )
            
            print("\n✓ 分析完成！")
            print(f"  结果已保存到: {self.visualizer.output_dir}/")
            print(f"  主刷口大小: {calculator.config.brush_width}x{calculator.config.brush_height} 像素")
            
            # 显示结果
            cv2.imshow("Coverage Result", coverage_img)
            cv2.imshow("Heatmap", heatmap_img)
            cv2.imshow("Report", report)
            
            print("\n按任意键关闭结果窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n按回车键继续...")
    
    def _transform_trajectory(self, trajectory) -> list:
        """
        将视频坐标转换为床铺坐标
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            转换后的坐标列表
        """
        bed_config = self.config.get_bed_area_config()
        
        if not bed_config.points or len(bed_config.points) != 4:
            return []
        
        # 源点（视频中的4个角点）
        pts1 = np.float32(bed_config.points)
        
        # 目标点（床铺矩形）
        pts2 = np.float32([
            [0, 0],
            [bed_config.width - 1, 0],
            [bed_config.width - 1, bed_config.height - 1],
            [0, bed_config.height - 1]
        ])
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 转换所有轨迹点
        bed_points = []
        for point in trajectory:
            # 转换为数组
            pt = np.array([[point.x, point.y]], dtype=np.float32)
            # 透视变换
            transformed = cv2.perspectiveTransform(pt, matrix)
            # 转换为整数坐标
            x, y = int(transformed[0][0][0]), int(transformed[0][0][1])
            
            # 边界检查
            x = max(0, min(x, bed_config.width - 1))
            y = max(0, min(y, bed_config.height - 1))
            
            bed_points.append((x, y))
        
        return bed_points
    
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
            'summary.jpg': '摘要',
            'report.jpg': '报告',
            'coverage_result.jpg': '覆盖结果',
            'heatmap.jpg': '热力图'
        }
        
        for filename, title in result_files.items():
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    cv2.imshow(title, img)
        
        if cv2.waitKey(0) & 0xFF:
            pass
        cv2.destroyAllWindows()
        
        input("\n按回车键继续...")
    
    def configure_parameters(self):
        """配置参数"""
        print("\n=== 参数配置 ===")
        
        tracker_config = self.config.get_tracker_config()
        coverage_config = self.config.get_coverage_config()
        
        print("\n当前参数:")
        print(f"  蓝色HSV范围: [{tracker_config.blue_lower_h}, {tracker_config.blue_lower_s}, {tracker_config.blue_lower_v}] - "
              f"[{tracker_config.blue_upper_h}, {tracker_config.blue_upper_s}, {tracker_config.blue_upper_v}]")
        print(f"  最小区域: {tracker_config.min_area}")
        print(f"  主刷口大小: {coverage_config.brush_width}x{coverage_config.brush_height} 像素")
        print(f"  网格大小: {coverage_config.grid_size}")
        
        print("\n注意: 参数修改后请重新运行分析")
        print("（当前版本暂不支持运行时修改，请直接编辑 config.json）")
        
        # 显示当前配置
        config_path = "config.json"
        if os.path.exists(config_path):
            print(f"\n配置文件位置: {os.path.abspath(config_path)}")
        
        input("\n按回车键继续...")
    
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
        import matplotlib
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
