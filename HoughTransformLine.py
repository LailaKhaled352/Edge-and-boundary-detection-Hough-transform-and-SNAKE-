import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout 
class HoughTransformLine:
    def __init__(self, image, detected_obj_widget, hough_heatmap_widget,edges, theta_res=1, rho_res=1):
        """Initialize with an image and resolution settings."""
        self.image = image #image data as numpy array
        self.edges= edges
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
        lines = self.detect_lines(threshold)
        output_img = image.copy()
        origin = np.array([0, image.shape[1]])
    
        for hough_line in lines:
            rho, theta= hough_line
            if np.sin(theta) != 0:
                y0, y1 = (rho - origin * np.cos(theta)) / np.sin(theta)
            else:
                y0, y1 = 0, image.shape[0]  # If horizontal, extend from top to bottom                       
            cv2.line(output_img, (int(origin[0]), int(y0)), (int(origin[1]), int(y1)), (0,0, 255),1)
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

        if self.hough_heatmap_widget.layout() is None:
            layout = QVBoxLayout(self.hough_heatmap_widget)
            self.hough_heatmap_widget.setLayout(layout)
        else:
            layout = self.hough_heatmap_widget.layout()

        layout.addWidget(canvas)

