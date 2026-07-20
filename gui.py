import sys
import os
import io
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QFormLayout, QLineEdit, 
                             QSpinBox, QDoubleSpinBox, QMessageBox, QTabWidget, QScrollArea)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QThread

from config import config_manager
from bed_selector import BedSelector
from main import MiteCoverageApp

class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

class AnalysisThread(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        try:
            app = MiteCoverageApp()
            app.analyze_video(self.video_path)
            self.finished_signal.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("除螨仪覆盖率测试软件 v1.0 (GUI)")
        self.resize(900, 700)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Tabs
        self.tab_run = QWidget()
        self.tab_config = QWidget()
        self.tabs.addTab(self.tab_run, "主控制台")
        self.tabs.addTab(self.tab_config, "参数配置")
        
        self.setup_run_tab()
        self.setup_config_tab()
        self.setup_log_panel()
        
        # Redirect stdout
        self.redirector = StreamRedirector()
        self.redirector.text_written.connect(self.append_log)
        sys.stdout = self.redirector
        sys.stderr = self.redirector
        
        print("MiteCoverage GUI 已启动。")

    def setup_run_tab(self):
        layout = QVBoxLayout(self.tab_run)
        
        group = QGroupBox("操作控制")
        btn_layout = QVBoxLayout()
        
        self.btn_calibrate = QPushButton("1. 标定床铺区域 (选择视频)")
        self.btn_calibrate.setMinimumHeight(50)
        self.btn_calibrate.clicked.connect(self.on_calibrate)
        
        self.btn_analyze = QPushButton("2. 分析测试视频 (计算覆盖率)")
        self.btn_analyze.setMinimumHeight(50)
        self.btn_analyze.clicked.connect(self.on_analyze)
        
        self.btn_view = QPushButton("3. 查看结果目录")
        self.btn_view.setMinimumHeight(50)
        self.btn_view.clicked.connect(self.on_view_results)
        
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_analyze)
        btn_layout.addWidget(self.btn_view)
        group.setLayout(btn_layout)
        layout.addWidget(group)
        layout.addStretch()

    def setup_config_tab(self):
        layout = QVBoxLayout(self.tab_config)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form_layout = QFormLayout(content)
        
        self.fields = {}
        
        def add_spinbox(name, label, val, min_v, max_v, step=1, double=False):
            if double:
                w = QDoubleSpinBox()
                w.setDecimals(1)
            else:
                w = QSpinBox()
            w.setRange(min_v, max_v)
            w.setSingleStep(step)
            w.setValue(val)
            form_layout.addRow(label, w)
            self.fields[name] = w
            
        bed = config_manager.get_bed_area_config()
        cov = config_manager.get_coverage_config()
        trk = config_manager.get_tracker_config()
        
        add_spinbox("bed_w", "床铺实际宽度 (cm)", bed.real_width_cm, 10, 500)
        add_spinbox("bed_h", "床铺实际长度 (cm)", bed.real_height_cm, 10, 500)
        add_spinbox("rem_w", "除螨仪实际宽度 (cm)", cov.real_remover_width_cm, 1.0, 100.0, 0.5, True)
        add_spinbox("rem_h", "除螨仪实际长度 (cm)", cov.real_remover_height_cm, 1.0, 100.0, 0.5, True)
        add_spinbox("brush_w", "主刷口实际宽度 (cm)", cov.real_brush_width_cm, 1.0, 100.0, 0.5, True)
        add_spinbox("brush_h", "主刷口实际高度 (cm)", cov.real_brush_height_cm, 1.0, 100.0, 0.5, True)
        
        add_spinbox("b_l_h", "蓝追踪 H下限", trk.blue_lower_h, 0, 255)
        add_spinbox("b_l_s", "蓝追踪 S下限", trk.blue_lower_s, 0, 255)
        add_spinbox("b_l_v", "蓝追踪 V下限", trk.blue_lower_v, 0, 255)
        add_spinbox("b_u_h", "蓝追踪 H上限", trk.blue_upper_h, 0, 255)
        add_spinbox("b_u_s", "蓝追踪 S上限", trk.blue_upper_s, 0, 255)
        add_spinbox("b_u_v", "蓝追踪 V上限", trk.blue_upper_v, 0, 255)
        
        add_spinbox("g_l_h", "绿追踪 H下限", trk.green_lower_h, 0, 255)
        add_spinbox("g_l_s", "绿追踪 S下限", trk.green_lower_s, 0, 255)
        add_spinbox("g_l_v", "绿追踪 V下限", trk.green_lower_v, 0, 255)
        add_spinbox("g_u_h", "绿追踪 H上限", trk.green_upper_h, 0, 255)
        add_spinbox("g_u_s", "绿追踪 S上限", trk.green_upper_s, 0, 255)
        add_spinbox("g_u_v", "绿追踪 V上限", trk.green_upper_v, 0, 255)
        
        add_spinbox("min_area", "最小检测区域(像素)", trk.min_area, 1, 10000, 10)
        add_spinbox("grid_size", "覆盖网格大小(像素)", cov.grid_size, 1, 100)
        add_spinbox("parallax", "贴纸高度视差补偿(cm)", getattr(trk, 'parallax_height_cm', 0.0), 0.0, 50.0, 0.5, True)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        btn_save = QPushButton("保存配置")
        btn_save.setMinimumHeight(40)
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

    def setup_log_panel(self):
        group = QGroupBox("运行日志")
        layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        group.setLayout(layout)
        self.main_layout.addWidget(group)

    def append_log(self, text):
        self.log_text.moveCursor(self.log_text.textCursor().End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(self.log_text.textCursor().End)

    def on_calibrate(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频用于标定", "videos", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            try:
                print(f"正在从 {video_path} 标定...")
                selector = BedSelector(video_path=video_path)
                success = selector.run()
                if success:
                    print("✓ 床铺区域标定成功，配置已保存。")
                else:
                    print("✗ 标定取消或失败。")
            except Exception as e:
                print(f"错误: {e}")

    def on_analyze(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "选择要分析的视频", "videos", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.btn_analyze.setEnabled(False)
            print(f"准备分析视频: {video_path}")
            self.thread = AnalysisThread(video_path)
            self.thread.finished_signal.connect(self.on_analyze_finished)
            self.thread.error_signal.connect(self.on_analyze_error)
            self.thread.start()

    def on_analyze_finished(self):
        self.btn_analyze.setEnabled(True)
        print("分析线程完成。")

    def on_analyze_error(self, err):
        self.btn_analyze.setEnabled(True)
        print(f"分析线程报错: {err}")

    def on_view_results(self):
        output_dir = os.path.abspath("outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        import platform
        if platform.system() == "Windows":
            os.startfile(output_dir)
        elif platform.system() == "Darwin":
            import subprocess
            subprocess.Popen(["open", output_dir])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", output_dir])
        print(f"已打开目录: {output_dir}")

    def save_config(self):
        bed = config_manager.get_bed_area_config()
        cov = config_manager.get_coverage_config()
        trk = config_manager.get_tracker_config()
        
        bed.real_width_cm = self.fields["bed_w"].value()
        bed.real_height_cm = self.fields["bed_h"].value()
        
        cov.real_remover_width_cm = self.fields["rem_w"].value()
        cov.real_remover_height_cm = self.fields["rem_h"].value()
        cov.real_brush_width_cm = self.fields["brush_w"].value()
        cov.real_brush_height_cm = self.fields["brush_h"].value()
        cov.grid_size = self.fields["grid_size"].value()
        
        trk.blue_lower_h = self.fields["b_l_h"].value()
        trk.blue_lower_s = self.fields["b_l_s"].value()
        trk.blue_lower_v = self.fields["b_l_v"].value()
        trk.blue_upper_h = self.fields["b_u_h"].value()
        trk.blue_upper_s = self.fields["b_u_s"].value()
        trk.blue_upper_v = self.fields["b_u_v"].value()
        
        trk.green_lower_h = self.fields["g_l_h"].value()
        trk.green_lower_s = self.fields["g_l_s"].value()
        trk.green_lower_v = self.fields["g_l_v"].value()
        trk.green_upper_h = self.fields["g_u_h"].value()
        trk.green_upper_s = self.fields["g_u_s"].value()
        trk.green_upper_v = self.fields["g_u_v"].value()
        
        trk.min_area = self.fields["min_area"].value()
        trk.parallax_height_cm = self.fields["parallax"].value()
        
        config_manager.save()
        print("✓ 参数已成功保存到 config.json！")
        QMessageBox.information(self, "成功", "参数已保存")

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()
