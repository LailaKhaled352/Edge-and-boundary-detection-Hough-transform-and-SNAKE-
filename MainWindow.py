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
from ActiveContour import ActiveContour
from SignalManager import global_signal_manager  
from ChainCodeWindow import ChainCodeWindow

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



        #tab two connections
        self.input_image_snake = self.findChild(QWidget, "inputImage2")
        self.output_image_snake = self.findChild(QWidget, "outputImage2")
        self.alpha_slider = self.findChild(QSlider, "alphaSlider")
        self.gamma_slider = self.findChild(QSlider, "gammaSlider")
        self.iter_slider = self.findChild(QSlider, "iterationaSlider")

        self.alpha_label = self.findChild(QLabel, "alphaLabel")
        self.gamma_label = self.findChild(QLabel, "gammaLabel")
        self.iter_label = self.findChild(QLabel, "iterationsLabel")
        self.perimeterLabel=self.findChild(QLabel, "perimeterLabel")
        self.areaLabel=self.findChild(QLabel, "areaLabel")
        self.chainCodeButton = self.findChild(QPushButton, "chainCode")

        self.alpha_slider.setRange(0, 9)  # 0.1 to 0.9 (scaled by 10)
        self.gamma_slider.setRange(0, 9)  # 0 to 0.9 (scaled by 10)
        self.iter_slider.setRange(1, 100)  # 1000 to 10000 (scaled by 1000)

        self.alpha_slider.setSingleStep(1)
        self.gamma_slider.setSingleStep(1)
        self.iter_slider.setSingleStep(1)
       
         # Set default values (scaled accordingly)
        self.alpha_slider.setValue(int(0.1 * 10))  # 0.1 -> 1
        self.gamma_slider.setValue(int(0.7 * 10))  # 0.7 -> 7
        self.iter_slider.setValue(int(9000 / 1000))  # 9000 -> 9

        self.update_labels()
        # Create ImageViewers for each widget
        self.input_viewer = ImageViewer(input_view=self.input_image)
        self.canny_viewer = ImageViewer(input_view=self.canny_image)
        self.hough_viewer = ImageViewer(input_view=self.hough_image)
        self.accumulator_viewer = ImageViewer(input_view=self.accumulator_image)
        self.input_viewer_snake = ImageViewer(input_view=self.input_image_snake, output_view=self.output_image_snake ,index=0,mode=True, widget=2)
        #Active contour instance
        self.active_contour = ActiveContour(self.input_image_snake, self.output_image_snake, self.input_viewer_snake,self.perimeterLabel,self.areaLabel)

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

        self.chainCodeButton.clicked.connect(self.open_chainCodeWindow)
        global_signal_manager.image_loaded.connect(self.apply_active_contour)
        # Connect sliders to update function
        self.alpha_slider.valueChanged.connect(self.update_active_contour)
        self.gamma_slider.valueChanged.connect(self.update_active_contour)
        self.iter_slider.valueChanged.connect(self.update_active_contour)


    def apply_canny(self):
        """ Apply Canny Edge Detection and display the result. """
        # Get Canny thresholds from UI sliders  
        low_threshold = self.lowthreshold_slider.value()
        high_threshold = self.highthreshold_slider.value()

        img = self.input_viewer.get_loaded_image()
        if img is None:
          print("No image loaded for Canny Edge Detection.")
          return

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
        mode = self.hough_combobox.currentText()
        original_image = self.input_viewer.img_data.copy()
        #canny_detector = CannyDetector(75, 150)
        # Convert to grayscale if necessary
        if len(self.input_viewer.img_data.shape) == 3:
            grayscale_image = cv2.cvtColor(self.input_viewer.img_data, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = self.input_viewer.img_data     

        # Get the Hough threshold value from the slider
        hough_threshold = self.hough_slider.value() 

        if mode == "Lines":
            hough = HoughTransformLine(original_image, self.hough_image, self.accumulator_image, self.edges)
            detected_hough_image = hough.draw_lines(original_image, threshold=hough_threshold)
            hough.plot_accumulator() 

        elif mode == "Circles":
            hough = HoughTransformCircle(original_image, self.hough_image, self.accumulator_image, self.edges)
            detected_hough_image = hough.draw_circles(original_image, threshold=hough_threshold)
            hough.plot_accumulator() 

        elif mode == "Ellipses":
            hough = HoughTransformEllipse(grayscale_image, self.hough_image, self.edges)
            detected_ellipses = hough.detect_ellipses(threshold=hough_threshold)  
            if len(detected_ellipses) == 3:
                detected_ellipses = [detected_ellipses[1]]  
            fitted_ellipses = hough.fit_ellipses(detected_ellipses)         
            detected_hough_image = hough.draw_ellipses(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), fitted_ellipses)

            # Ensure the image is in RGB
            detected_hough_image = cv2.cvtColor(detected_hough_image, cv2.COLOR_BGR2RGB)

            if detected_hough_image is not None:
                self.hough_viewer.display_output_image(detected_hough_image, self.hough_image)
            else:
                print("Failed to display detected ellipses.")

        else:
            print("Invalid Hough Transform mode selected.")
            return
          # Display results
        self.hough_viewer.display_output_image(detected_hough_image, self.hough_image)   



    def apply_active_contour(self):
        # Load Image with color
         self.active_contour.load_image()




    def update_active_contour(self):
        """Update Active Contour parameters and reapply the function."""
        alpha = self.alpha_slider.value() / 10.0  # Scale back to 0-0.9
        gamma = self.gamma_slider.value() / 10.0  # Scale back to 0-0.9
        iterations = self.iter_slider.value() * 1000  # Scale to 1000-10000
        self.update_labels()
        # Update active contour parameters
        self.active_contour.alpha = alpha
        self.active_contour.gamma = gamma
        self.active_contour.iterations = iterations

        print(f"Updated Parameters -> Alpha: {alpha}, Gamma: {gamma}, Iterations: {iterations}")

        # Apply active contour with new parameters
        self.active_contour.apply_active_contour()


    def open_chainCodeWindow(self):
       chaincode= self.active_contour.get_chain_code()
       if chaincode:
        self.chainCodeWindow=ChainCodeWindow(chaincode)
        self.chainCodeWindow.exec_()
        
    def update_labels(self):
        """Update labels based on slider values."""
        alpha = self.alpha_slider.value() / 10.0  # Scale back to 0.0 - 0.9
        gamma = self.gamma_slider.value() / 10.0  # Scale back to 0.0 - 0.9
        iterations = self.iter_slider.value() * 10000  # Scale to 1000 - 10000

        self.alpha_label.setText(f"Alpha: {alpha:.1f}")
        self.gamma_label.setText(f"Gamma: {gamma:.1f}")
        self.iter_label.setText(f"Iterations: {iterations}")





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    