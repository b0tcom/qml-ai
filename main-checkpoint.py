import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog

from models.yolo_detection import YOLODetector


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("-Quantum Object Detection")
        self.setGeometry(100, 100, 800, 600)

        # Set the application icon
        self.setWindowIcon(QIcon('assets/duggthethugg.png'))

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.detector = YOLODetector()
        self.cap = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.create_buttons()

    def create_buttons(self):
        # Button creation logic here
        self.layout = QVBoxLayout()
        self.button = QPushButton('Load Video', self)  # Create a button
        self.button.clicked.connect(self.load_video)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    
