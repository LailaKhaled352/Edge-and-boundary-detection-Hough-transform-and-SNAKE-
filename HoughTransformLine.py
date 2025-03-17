import cv2
import numpy as np
import matplotlib.pyplot as plt
from CannyDetector import CannyDetector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout 
class HoughTransformLine:
    def __init__(self, image, detected_obj_widget, hough_heatmap_widget, theta_res=1, rho_res=1):
        """Initialize with an image and resolution settings."""
        self.image = image #image data as numpy array
        self.canny = CannyDetector( 50, 150)  # Apply Canny edge detection
        self.edges= self.canny.apply_canny_detector(self.image)
        self.theta_res = np.deg2rad(theta_res)  # Convert degrees to radians
        self.rho_res = rho_res
        self.accumulator, self.thetas, self.rhos = self._hough_transform()
        self.detected_obj_widget, self.hough_heatmap_widget =detected_obj_widget, hough_heatmap_widget,

    def _hough_transform(self):
        """Compute Hough Transform (from cartesian to paramter space) and return accumulator, theta, and rho values."""
        height, width = self.edges.shape
        diag_len = int(np.sqrt(height**2 + width**2))  # Maximum possible rho
        rhos = np.arange(-diag_len, diag_len, self.rho_res)
        thetas = np.arange(-np.pi / 2, np.pi / 2, self.theta_res)

        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

        # Get edge points
        edge_points = np.argwhere(self.edges)  #Returns the indices (coordinates) of all nonzero (white/edge) pixels
                                               #The output is a list of (row, column) pairs, which correspond to (y, x) format.

        #here we do the hough transform
        for y, x in edge_points:
            for theta_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                rho_idx = np.argmin(np.abs(rhos - rho))  # Find closest rho index
                accumulator[rho_idx, theta_idx] += 1  # Vote in accumulator

        return accumulator, thetas, rhos

    def detect_lines(self, threshold=100):
        """Detect prominent lines from Hough accumulator."""
        line_indices = np.argwhere(self.accumulator > threshold)
        lines = []
        for rho_idx, theta_idx in line_indices:
            rho = self.rhos[rho_idx]
            theta = self.thetas[theta_idx]
            lines.append((rho, theta))
        return lines

    def draw_lines(self, image, threshold=100):
        """Draw detected lines on the original image."""
        lines = self.detect_lines(threshold)
        output_img = image.copy()

        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            #x0,y0 is a point on the detected line and it's the closest point to the origin
            x0 = a * rho
            y0 = b * rho
            #Computing Two Endpoints of the Line 
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #send the output image and the widget to the output viewer to show it
        return output_img 

    def plot_accumulator(self):
        """Display the Hough accumulator heatmap on the QWidget (self.hough_heatmap_widget)."""
        
        # Clear previous content
        for child in self.hough_heatmap_widget.children():
            child.deleteLater()

        # Create a new Matplotlib Figure and Canvas
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot the accumulator heatmap
        im = ax.imshow(
            self.accumulator, cmap='hot', aspect='auto',
            extent=[np.rad2deg(self.thetas[0]), np.rad2deg(self.thetas[-1]), self.rhos[-1], self.rhos[0]]
        )

        # Add colorbar
        fig.colorbar(im, ax=ax, label="Votes")

        # Labels and title
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Rho (pixels)")
        ax.set_title("Hough Transform Accumulator")

        # Embed the Matplotlib Figure inside the QWidget
        layout = QVBoxLayout(self.hough_heatmap_widget)
        layout.addWidget(canvas)


# if __name__ =='__main__':
#     # Load an image and apply Hough Transform
#     image = cv2.imread("Images/roberts.jpg")  # Change to your image path
#     hough = HoughTransformLine(image)

#     # Plot the Hough accumulator
#     hough.plot_accumulator()

#     # Draw detected lines on the image
#     output_image = hough.draw_lines(image, threshold=150)
#     # Show result
#     cv2.imshow("Detected Lines", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
