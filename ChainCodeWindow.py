import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt


class ChainCodeWindow(QDialog):
    def __init__(self, chain_code, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chain Code Representation")
        self.setGeometry(200, 200, 500, 550)
        self.setStyleSheet("background-color: #f8f9fa; border-radius: 10px;")

        self.image_path ="C:/Computer_Vision/task2_canny&activecountour/Edge-and-boundary-detection-Hough-transform-and-SNAKE-/Images/chaincode.jpg"  # Store the image path
        layout = QVBoxLayout()

        # Title
        self.title_label = QLabel("Chain Code Representation")
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Chain Code Display
        self.chain_code_label = QLabel("Computed Chain Code:")
        self.chain_code_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.chain_code_label)

        self.chain_code_text = QTextEdit()
        self.chain_code_text.setReadOnly(True)
        self.chain_code_text.setText(str(chain_code))
        self.chain_code_text.setFont(QFont("Arial", 8)) 
        self.chain_code_text.setStyleSheet("background-color: #fff; border: 1px solid #ccc; padding: 5px;")
        layout.addWidget(self.chain_code_text)

        # Explanation of directions
        direction_info = """
        <b>Chain Code Directions:</b><br>
        0 → Right &emsp; 1 → Bottom-Right &emsp; 2 → Down &emsp; 3 → Bottom-Left <br>
        4 → Left &emsp; 5 → Top-Left &emsp; 6 → Up &emsp; 7 → Top-Right
        """
        self.direction_label = QLabel(direction_info)
        self.direction_label.setWordWrap(True)
        layout.addWidget(self.direction_label)

        # Load and display the image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)
        self.load_and_display_image()  # Call function to load image

        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.close_button.setStyleSheet(
            "background-color: #007bff; color: white; border-radius: 5px; padding: 8px;"
        )
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)

        self.setLayout(layout)

    def load_and_display_image(self):
        """
        Loads the image using OpenCV from the given path and displays it.
        """
        if not os.path.exists(self.image_path):
            self.image_label.setText(f"⚠️ Image not found: {self.image_path}")
            self.image_label.setStyleSheet("color: red; font-weight: bold;")
            return

        
        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if img is None:
            self.image_label.setText("⚠️ Error loading image.")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignCenter)

    def open_window(self):
        """Opens the Chain Code Window when button is clicked."""
        self.show()



