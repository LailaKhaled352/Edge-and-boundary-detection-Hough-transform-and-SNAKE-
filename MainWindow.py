from PyQt5.QtWidgets import QMainWindow,QComboBox,QTabWidget, QSpinBox, QWidget, QApplication, QPushButton, QLabel, QSlider,QProgressBar,QGraphicsView,QGraphicsScene
from PyQt5.QtGui import QIcon
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from ImageViewer import ImageViewer
from CannyDetector import CannyDetector
from HoughTransformLine import HoughTransformLine
from HoughTransformCircle import HoughTransformCircle
from HoughTransformEllipse import HoughTransformEllipse

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)
        #images widgets
        self.input_image = self.findChild(QWidget, "inputImage")
        self.canny_image = self.findChild(QWidget, "cannyImage")
        self.hough_image = self.findChild(QWidget, "houghImage")
        self.accumulator_image = self.findChild(QWidget, "accumulatorImage")


        # Create ImageViewers for each widget
        self.input_viewer = ImageViewer(input_view=self.input_image)
        self.canny_viewer = ImageViewer(input_view=self.canny_image)
        self.hough_viewer = ImageViewer(input_view=self.hough_image)
        self.accumulator_viewer = ImageViewer(input_view=self.accumulator_image)

        #canny parameters
        self.highthreshold_slider = self.findChild(QSlider, "cannyHighThreshold")
        self.lowthreshold_slider = self.findChild(QSlider, "cannyLowThreshold")
        self.highthreshold_label = self.findChild(QLabel, "highThresholdLabel")
        self.lowthreshold_label = self.findChild(QLabel, "lowThresholdLabel")

        self.highthreshold_slider.setMinimum(1)
        self.highthreshold_slider.setMaximum(300)

        self.highthreshold_slider.valueChanged.connect(lambda: self.highthreshold_label.setText(str(self.highthreshold_slider.value())))
        self.lowthreshold_slider.valueChanged.connect(lambda: self.lowthreshold_label.setText(str(self.lowthreshold_slider.value())))
      
        self.applycanny = self.findChild(QPushButton, "cannyButton")
        self.applycanny.clicked.connect(self.apply_canny)

        #hough transform parameters
        self.hough_combobox = self.findChild(QComboBox, "houghCombo")
        self.hough_sliderlabel = self.findChild(QLabel, "houghLabel")
        self.hough_slider = self.findChild(QSlider, "houghThreshold")

        self.hough_slider.setMinimum(1)
        self.hough_slider.setMaximum(300)

        self.hough_slider.valueChanged.connect(lambda: self.hough_sliderlabel.setText(str(self.hough_slider.value())))

        self.applyhough = self.findChild(QPushButton, "houghButton")
        self.applyhough.clicked.connect(self.apply_hough_transform)

        self.edges=None



    def apply_canny(self):
        """ Apply Canny Edge Detection and display the result. """
        low_threshold = self.lowthreshold_slider.value()
        high_threshold = self.highthreshold_slider.value()

        img = self.input_viewer.get_loaded_image()
        if img is None:
          print("No image loaded for Canny Edge Detection.")
          return
        # Get Canny thresholds from UI sliders
        low_threshold = self.lowthreshold_slider.value()
        high_threshold = self.highthreshold_slider.value()
        # Convert to grayscale if necessary
        if len(self.input_viewer.img_data.shape) == 3:  
            grayscale_image = cv2.cvtColor(self.input_viewer.img_data, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = self.input_viewer.img_data

        # Apply custom Canny detection
        detector = CannyDetector(low_threshold, high_threshold)
        self.edges = detector.apply_canny_detector(grayscale_image)

        # Display processed image in the output widget
        self.input_viewer.display_output_image(self.edges, self.canny_image)        

    def apply_hough_transform(self):
        """Apply Hough Transform after ensuring edges are detected."""
        # Ensure input image is loaded
        if self.input_viewer.img_data is None:
            print("No image loaded.")
            return   
        threshold = self.hough_slider.value()
        mode = self.hough_combobox.currentText()
        original_image = self.input_viewer.img_data.copy()
        #canny_detector = CannyDetector(75, 150)
        # Convert to grayscale if necessary
        if len(self.input_viewer.img_data.shape) == 3:
            grayscale_image = cv2.cvtColor(self.input_viewer.img_data, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = self.input_viewer.img_data

        # Apply Canny
        #edges = canny_detector.apply_canny_detector(grayscale_image)     

        # Get the Hough threshold value from the slider
        hough_threshold = self.hough_slider.value() 

        if mode == "Lines":
            hough = HoughTransformLine(original_image, self.hough_image, self.accumulator_image, self.edges)
            detected_hough_image = hough.draw_lines(original_image, threshold=hough_threshold)

        elif mode == "Circles":
            hough = HoughTransformCircle(original_image, self.hough_image, self.accumulator_image, self.edges)
            detected_hough_image = hough.draw_circles(original_image, threshold=hough_threshold)

        elif mode == "Ellipses":
            hough = HoughTransformEllipse(original_image, self.hough_image, self.accumulator_image)
            detected_hough_image = hough.draw_ellipses(original_image, threshold=hough_threshold)

        else:
            print("Invalid Hough Transform mode selected.")
            return
          # Display results
        self.hough_viewer.display_output_image(detected_hough_image, self.hough_image)
        hough.plot_accumulator()    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        