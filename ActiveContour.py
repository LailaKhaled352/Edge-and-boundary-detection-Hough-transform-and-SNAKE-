import numpy as np
from CannyDetector import CannyDetector
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QWidget
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from SignalManager import global_signal_manager  
from math import sqrt
from ChainCodeWindow import ChainCodeWindow
from scipy.signal import convolve2d
class ActiveContour:
    def __init__(self,input_widget: QWidget, output_widget: QWidget , image_viewer ,perimeterLabel=None,areaLabel=None,alpha=0.1, beta=20, gamma=0.7, iterations=9000):
        self.input_widget = input_widget
        self.output_widget = output_widget
        self.alpha = alpha  # Elasticity (smoothness)
        self.beta = beta    # Rigidity (curvature)
        self.gamma = gamma  # Edge attraction
        self.iterations = iterations  
        self.canny_detector = CannyDetector()
        self.rect_selector = None
        self.start_x, self.start_y, self.end_x, self.end_y = None, None, None, None  # Store selected region
        self.image_viewer = image_viewer  # ImageViewer instance
        self.perimeter = 0
        self.area=0
        self.pixel_to_cm = 0.1
        self.perimeterLabel = perimeterLabel
        self.areaLabel = areaLabel
        global_signal_manager.contour_requested.connect(self.apply_active_contour)
        self.chain_code=None
    def load_image(self):
        """Loads image from ImageViewer and converts it to grayscale."""
        img_data = self.image_viewer.get_loaded_image()
        if img_data is None:
            print("No image loaded.")
            return
        self.original_color = img_data.copy()
        self.image = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)



    def compute_gradients(self, edge_image):
        """Compute gradient maps using Sobel filtering (without OpenCV)."""
        edge_image = edge_image.astype(np.float32)  # Ensure computations are in float

        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1], 
                            [0,  0,  0], 
                            [1,  2,  1]], dtype=np.float32)

        grad_x = convolve2d(edge_image, sobel_x, mode='same', boundary='symm')
        grad_y = convolve2d(edge_image, sobel_y, mode='same', boundary='symm')


        # Normalize gradients to prevent overflow
        eps = 1e-6 
        magnitude = np.sqrt(grad_x**2 + grad_y**2)+eps 
        magnitude[magnitude == 0] = 1  # Avoid division by zero
        grad_x = np.clip(grad_x / magnitude, -1, 1)
        grad_y = np.clip(grad_y / magnitude, -1, 1)
        
       
        return grad_x, grad_y


    
    def apply_active_contour(self):
        """Applies Active Contour (Snake) algorithm."""
        
        initial_contour = self.image_viewer.get_initial_contour()
        if initial_contour is None:
         print("No region selected! Please use select_region() before running the snake algorithm.")
         return None
        x1, y1, x2, y2 = initial_contour
        # Step 1: Detect edges using Canny
        edge_image = self.canny_detector.apply_gaussian_blur(self.image)
        
        # Step 2: Compute image gradients
        grad_x, grad_y = self.compute_gradients(edge_image)

       # Step 3: Create an initial contour as a rectangle boundary
        t = np.linspace(0, 2 * np.pi, 300)  # 300 points on the contour
        x = np.concatenate([np.linspace(x1, x2, 25), np.full(25, x2),
                            np.linspace(x2, x1, 25), np.full(25, x1)])
        y = np.concatenate([np.full(25, y1), np.linspace(y1, y2, 25),
                            np.full(25, y2), np.linspace(y2, y1, 25)])
        contour = np.vstack((x, y)).T
         # Vectorized contour update
        for _ in range(self.iterations):
            prev_contour = np.roll(contour, shift=1, axis=0)
            next_contour = np.roll(contour, shift=-1, axis=0)
            internal_force = self.alpha * (prev_contour + next_contour - 2 * contour)

            x_idx = np.clip(contour[:, 0].astype(int), 0, self.image.shape[1] - 1)
            y_idx = np.clip(contour[:, 1].astype(int), 0, self.image.shape[0] - 1)

            external_force = self.gamma * np.column_stack((-grad_x[y_idx, x_idx], -grad_y[y_idx, x_idx]))
            contour += internal_force + external_force


        if contour is not None:
         self.compute_chain_code(contour)
         self.compute_area(contour)
         self.compute_perimeter(contour)
         #self.verify_contour_metrics(contour,self.pixel_to_cm)
         self.image_viewer.display_contour(contour)
        return contour


    def compute_perimeter(self,contour):
     """Computes the perimeter of a given contour without OpenCV."""


     if len(contour) < 2:
        self.perimeter = 0  # Reset perimeter if the contour is invalid
        self.perimeterLabel.setText("Perimeter: 0.00 cm")
        return 0

     self.perimeter = 0  # Reset perimeter before calculation
     for i in range(len(contour) - 1):
        pt1, pt2 = contour[i], contour[i + 1]
        self.perimeter += sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    # Closing the contour (connect last and first point)
     if len(contour) > 1:
       self. perimeter += sqrt((contour[-1][0] - contour[0][0]) ** 2 + (contour[-1][1] - contour[0][1]) ** 2)
     self.perimeter_cm = self.perimeter * self.pixel_to_cm
     print(f"Perimeter: {self.perimeter_cm:.2f}")
     self.perimeterLabel.setText(f"Perimeter: {self.perimeter_cm:.2f} cm")  # Update label

     return self.perimeter_cm

    def compute_area(self,contour):
     """Computes the area inside a contour using the Shoelace theorem."""
     if len(contour) < 3:
        self.area = 0  # Reset area if the contour is invalid
        self.areaLabel.setText("Area: 0.00 cm²")  # Update label

        return 0  # A valid area requires at least 3 points

     self.area = 0  # Reset area if the contour is invalid
     n = len(contour)
     for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]  # Wrap around for the last point
        self.area  += (x1 * y2 - x2 * y1)
     self.area_cm2 = abs(self.area / 2) * (self.pixel_to_cm ** 2) 
     print(f"Area: {self.area_cm2:.2f}")
     self.areaLabel.setText(f"Area: {self.area_cm2:.2f} cm²")  # Update label

     return abs( self.area_cm2 ) 




   # def verify_contour_metrics(self,contour, pixel_to_cm):
    # """Verify the computed perimeter and area using OpenCV functions."""
    
    # Convert contour to proper format
    # contour = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))

    # Compute perimeter using OpenCV
    # opencv_perimeter = cv2.arcLength(contour, True) * pixel_to_cm
    # print(f"OpenCV Perimeter (cm): {opencv_perimeter:.2f}")

    # Compute area using OpenCV
    # opencv_area = cv2.contourArea(contour) * (pixel_to_cm ** 2)
    # print(f"OpenCV Area (cm²): {opencv_area:.2f}")

    # return opencv_perimeter, opencv_area


    def compute_chain_code(self, contour):
     """Computes the chain code representation of a given contour."""
     if len(contour) < 2:
        return []  # No chain code for a single point

     directions = {
        (0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
        (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7
    }

     self.chain_code = []
    
     for i in range(len(contour) - 1):
        x1, y1 = contour[i]
        x2, y2 = contour[i + 1]
        dx, dy = np.sign(x2 - x1), np.sign(y2 - y1)
        
        if (dx, dy) in directions:
            self.chain_code.append(directions[(dx, dy)])

     print(f"Chain Code: {self.chain_code}")
     
     return self.chain_code



    def get_chain_code(self):
      return self.chain_code
   