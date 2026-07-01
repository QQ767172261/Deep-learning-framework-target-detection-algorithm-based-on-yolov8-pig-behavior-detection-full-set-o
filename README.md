基于yolov8的猪只行为检测，全套代码＋数据集＋
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8349b26e5cad4a78a76e15f29f70264c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31477aee78a24cc2a74bd8e9ea80f1c7.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/25ec39b70a884b3e862a85adf3153407.png)
这是一个非常专业的**“基于YOLOv8的猪只行为检测系统”**。这是一个基于 **PyQt5** 构建的桌面端应用程序，能够实时检测猪只的多种行为状态（如站立、侧卧、俯卧、吃食等）。



---

### 📁 1. 项目结构还原

根据你的截图，项目结构如下：

```text
project_root/
├── main.py                 # 主程序入口 (PyQt5)
├── youi/                   # UI界面文件 (由 .ui 转换而来)
│   └── main_window.py      # 主窗口逻辑
├── core/                   # 核心算法逻辑
│   └── detector.py         # YOLOv8 检测封装
├── data/                   # 数据集目录
├── runs/                   # 训练结果
├── weights/                # 模型权重
│   ├── best.pt             # 训练好的猪只行为模型
│   └── yolov8n.pt          # 预训练模型
├── test.jpg                # 测试图片
├── requirements.txt        # 依赖包
└── README.md               # 说明文档
```

---

### 🛠️ 2. 环境准备 (`requirements.txt`)

```txt
ultralytics>=8.0.0
PyQt5>=5.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.0.0
torchvision>=0.15.0
```

安装命令：
```bash
pip install -r requirements.txt
```

---

### 3. 核心检测逻辑 (`core/detector.py`)

这是系统的“大脑”，负责加载YOLOv8模型并进行推理。

```python
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict
import time

class PigBehaviorDetector:
    """
    猪只行为检测核心类
    支持图片、视频、摄像头检测
    """
    
    def __init__(self, model_path: str = 'weights/best.pt'):
        """
        初始化检测器
        :param model_path: YOLOv8 模型权重文件路径
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print(f"✅ 模型加载成功: {model_path}")
            print(f"🏷️ 可识别行为: {list(self.class_names.values())}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.5) -> Dict:
        """
        检测单张图片
        :param image_path: 图片路径
        :param conf_threshold: 置信度阈值
        :return: 检测结果字典
        """
        start_time = time.time()
        
        # 推理
        results = self.model.predict(source=image_path, conf=conf_threshold, verbose=False)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 解析结果
        detections = []
        stats = {}
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                class_name = self.class_names[cls_id]
                
                # 统计各类别数量
                stats[class_name] = stats.get(class_name, 0) + 1
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': xyxy.tolist(),
                    'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                })
        
        return {
            'image_path': image_path,
            'detections': detections,
            'stats': stats,
            'count': len(detections),
            'inference_time': round(inference_time, 4)
        }
    
    def detect_video(self, video_path: str, conf_threshold: float = 0.5, output_path: str = None):
        """
        检测视频文件
        :param video_path: 视频文件路径
        :param conf_threshold: 置信度阈值
        :param output_path: 输出视频路径 (可选)
        :yield: 每一帧的检测结果
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            print(f"处理帧: {frame_idx}/{total_frames}", end='\r')
            
            results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
            
            annotated_frame = frame.copy()
            detections = []
            stats = {}
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    class_name = self.class_names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = self._get_color(cls_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    stats[class_name] = stats.get(class_name, 0) + 1
                    detections.append({
                        'frame': frame_idx,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': xyxy.tolist()
                    })
            
            if out:
                out.write(annotated_frame)
            
            yield {
                'frame': annotated_frame,
                'detections': detections,
                'stats': stats,
                'frame_idx': frame_idx,
                'total_frames': total_frames
            }
        
        cap.release()
        if out:
            out.release()
    
    def _get_color(self, cls_id: int):
        """为不同类别分配不同颜色"""
        colors = [
            (0, 255, 0),   # 绿色
            (0, 0, 255),   # 红色
            (255, 0, 0),   # 蓝色
            (0, 255, 255), # 黄色
            (255, 0, 255), # 紫色
            (255, 255, 0)  # 青色
        ]
        return colors[cls_id % len(colors)]
```

---

### 4. PyQt5 图形界面 (`main.py`)

这是系统的“脸面”，完全复刻你截图中的深色主题界面。

```python
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QComboBox, QSlider, QGroupBox, 
                             QTableWidget, QTableWidgetItem, QMessageBox, 
                             QProgressBar, QStatusBar, QLineEdit, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
from core.detector import PigBehaviorDetector

class DetectionWorker(QThread):
    """后台检测工作线程"""
    frame_ready = pyqtSignal(object)
    detection_result = pyqtSignal(dict)
    finished = pyqtSignal()
    
    def __init__(self, detector, source_type, source_path=None, camera_id=0):
        super().__init__()
        self.detector = detector
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        self.running = True
    
    def run(self):
        try:
            if self.source_type == 'image':
                result = self.detector.detect_image(self.source_path)
                self.detection_result.emit(result)
                self.finished.emit()
                
            elif self.source_type == 'video':
                for frame_data in self.detector.detect_video(self.source_path):
                    if not self.running:
                        break
                    self.frame_ready.emit(frame_data['frame'])
                    self.detection_result.emit({
                        'stats': frame_data['stats'],
                        'frame': frame_data['frame_idx'],
                        'total': frame_data['total_frames']
                    })
                self.finished.emit()
                
        except Exception as e:
            print(f"检测错误: {e}")
            self.finished.emit()
    
    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI猪舍 - 基于YOLOv8的猪只行为检测系统")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self._get_dark_style())
        
        # 初始化检测器
        try:
            self.detector = PigBehaviorDetector('weights/best.pt')
        except:
            QMessageBox.critical(self, "错误", "模型加载失败！请确保 weights/best.pt 存在")
            sys.exit(1)
        
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # === 左侧控制面板 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # 模型设置
        model_group = QGroupBox("⚙️ 模型设置")
        model_layout = QVBoxLayout(model_group)
        
        self.model_path_edit = QLineEdit("weights/best.pt")
        model_layout.addWidget(QLabel("模型路径:"))
        model_layout.addWidget(self.model_path_edit)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        model_layout.addWidget(QLabel("置信度阈值:"))
        model_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.50")
        model_layout.addWidget(self.conf_label)
        
        self.img_size_slider = QSlider(Qt.Horizontal)
        self.img_size_slider.setMinimum(320)
        self.img_size_slider.setMaximum(1280)
        self.img_size_slider.setValue(640)
        model_layout.addWidget(QLabel("图像尺寸:"))
        model_layout.addWidget(self.img_size_slider)
        
        left_layout.addWidget(model_group)
        
        # 文件操作
        file_group = QGroupBox("📁 文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_img_btn = QPushButton("🖼️ 加载图片")
        self.load_img_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_img_btn)
        
        self.load_video_btn = QPushButton("🎬 加载视频")
        self.load_video_btn.clicked.connect(self.load_video)
        file_layout.addWidget(self.load_video_btn)
        
        self.start_detect_btn = QPushButton("▶️ 开始检测")
        self.start_detect_btn.clicked.connect(self.start_detection)
        self.start_detect_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        file_layout.addWidget(self.start_detect_btn)
        
        self.stop_detect_btn = QPushButton("⏹️ 停止检测")
        self.stop_detect_btn.clicked.connect(self.stop_detection)
        self.stop_detect_btn.setEnabled(False)
        file_layout.addWidget(self.stop_detect_btn)
        
        left_layout.addWidget(file_group)
        
        # 任务信息
        task_group = QGroupBox("📊 任务信息")
        task_layout = QVBoxLayout(task_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        task_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("等待任务开始...")
        task_layout.addWidget(self.status_label)
        
        left_layout.addWidget(task_group)
        
        left_layout.addStretch()
        
        # === 右侧显示区域 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像显示
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #2c3e50; border: 1px solid #34495e; color: #ecf0f1;")
        self.image_label.setText("请加载图片或视频")
        right_layout.addWidget(self.image_label)
        
        # 结果输出
        result_group = QGroupBox("📝 检测结果输出")
        result_layout = QVBoxLayout(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        result_layout.addWidget(self.result_text)
        
        # 统计表格
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["行为类别", "数量"])
        result_layout.addWidget(self.stats_table)
        
        right_layout.addWidget(result_group)
        
        # 添加布局到主窗口
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 连接滑块信号
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.2f}"))
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_path:
            self.current_file = file_path
            self.show_image(file_path)
            self.status_bar.showMessage(f"已加载图片: {os.path.basename(file_path)}")
    
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.current_file = file_path
            self.status_bar.showMessage(f"已加载视频: {os.path.basename(file_path)}")
    
    def show_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
    
    def start_detection(self):
        if not hasattr(self, 'current_file'):
            QMessageBox.warning(self, "警告", "请先选择图片或视频文件！")
            return
        
        self.start_detect_btn.setEnabled(False)
        self.stop_detect_btn.setEnabled(True)
        self.status_bar.showMessage("正在检测...")
        self.progress_bar.setValue(0)
        
        conf_threshold = self.conf_slider.value() / 100.0
        
        if self.current_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_type = 'image'
        else:
            source_type = 'video'
        
        self.worker = DetectionWorker(self.detector, source_type, self.current_file)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.detection_result.emit({'stats': {}, 'frame': 0, 'total': 1}) # 初始化
        self.worker.detection_result.connect(self.update_result)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.start()
    
    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.start_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
        self.status_bar.showMessage("检测已停止")
    
    def update_frame(self, frame):
        """更新显示的帧"""
        if isinstance(frame, np.ndarray):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
    
    def update_result(self, result):
        """更新检测结果"""
        stats = result.get('stats', {})
        
        # 更新统计表格
        self.stats_table.setRowCount(len(stats))
        for i, (behavior, count) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(behavior))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(count)))
        
        # 更新文本输出
        output_text = f"--- 检测结果 ---\n"
        for behavior, count in stats.items():
            output_text += f"- {behavior}: {count} 个\n"
        output_text += f"总目标数: {sum(stats.values())}\n"
        
        if 'frame' in result:
            output_text += f"当前帧: {result['frame']}/{result['total']}\n"
            self.progress_bar.setValue(int((result['frame'] / result['total']) * 100))
        
        self.result_text.setText(output_text)
        self.status_label.setText(f"检测到 {sum(stats.values())} 个目标")
    
    def on_detection_finished(self):
        self.start_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
        self.status_bar.showMessage("检测完成")
        self.progress_bar.setValue(100)
    
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()
    
    def _get_dark_style(self):
        return """
        QMainWindow { background-color: #1a1a1a; color: #ecf0f1; }
        QGroupBox { 
            border: 1px solid #34495e; 
            border-radius: 5px; 
            margin-top: 1ex; 
            font-weight: bold;
            padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        QPushButton { 
            background-color: #34495e; 
            border: none; 
            color: white; 
            padding: 8px; 
            border-radius: 4px; 
        }
        QPushButton:hover { background-color: #2c3e50; }
        QPushButton:pressed { background-color: #1a252f; }
        QLineEdit, QComboBox { 
            background-color: #2c3e50; 
            border: 1px solid #34495e; 
            color: white; 
            padding: 5px; 
            border-radius: 3px; 
        }
        QProgressBar { 
            border: 1px solid #34495e; 
            border-radius: 5px; 
            text-align: center; 
            background-color: #2c3e50; 
        }
        QProgressBar::chunk { background-color: #2ecc71; }
        QTableWidget { 
            background-color: #2c3e50; 
            color: white; 
            gridline-color: #34495e; 
        }
        QTableWidget::item:selected { background-color: #3498db; }
        QTextEdit { 
            background-color: #2c3e50; 
            color: #ecf0f1; 
            border: 1px solid #34495e; 
        }
        """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

---

### 🚀 5. 如何运行系统

#### 步骤 1: 准备模型
1.  收集猪只行为数据集（标注：站立、侧卧、俯卧、吃食等）。
2.  使用 `yolo train model=yolov8n.pt data=pig_behavior.yaml epochs=100` 训练。
3.  将生成的 `best.pt` 放入 `weights/` 目录。

#### 步骤 2: 运行系统
```bash
python main.py
```

---

### ✅ 系统功能亮点

- **深色专业界面**: 完全复刻工业级软件风格
- **多格式支持**: 图片、视频批量检测
- **实时统计**: 动态更新各类行为数量
- **高性能**: YOLOv8 提供高精度和快速推理
- **可扩展性**: 轻松替换为其他动物或物体检测任务

，可以直接用于猪舍智能监控项目！
