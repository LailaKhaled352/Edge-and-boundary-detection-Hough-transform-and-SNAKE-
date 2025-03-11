import cv2
import numpy as np
from CannyDetector import CannyDetector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout

class HoughTransformCircle:
    def __init__(self, image, detected_obj_widget, hough_heatmap_widget, radius_res=5, center_res=3):
        """Initialize with an image and resolution settings."""
        self.image = image  # Image data as a numpy array
        self.canny = CannyDetector(75, 150)  
        self.edges = self.canny.apply_canny_detector(self.image)
        
        self.radius_res = radius_res
        self.center_res = center_res

        self.detected_obj_widget = detected_obj_widget
        self.hough_heatmap_widget = hough_heatmap_widget

        self.accumulator, self.radii, self.center_x_vals, self.center_y_vals = self._hough_transform()

    def _hough_transform(self):
        """Compute Hough Transform (from Cartesian to parameter space) and return accumulator."""
        height, width = self.edges.shape
        edge_points = np.column_stack(np.where(self.edges))  # Efficiently get edge coordinates (y, x)

        # Define search space
        center_x_vals = np.arange(0, width, self.center_res)
        center_y_vals = np.arange(0, height, self.center_res)
        radii = np.arange(20, 200, self.radius_res)
        
        # Initialize accumulator
        accumulator = np.zeros((len(center_x_vals), len(center_y_vals), len(radii)), dtype=np.int32)

        # Vectorized approach: precompute cos & sin values for different angles
        thetas = np.linspace(0, 2 * np.pi, 100)
        cos_thetas, sin_thetas = np.cos(thetas), np.sin(thetas)

        # Iterate over edge points
        for y, x in edge_points:
            for radius_idx, radius in enumerate(radii):
                center_x_candidates = (x - radius * cos_thetas).astype(int)
                center_y_candidates = (y - radius * sin_thetas).astype(int)

                # Filter valid candidates inside image boundaries
                valid_idx = (0 <= center_x_candidates) & (center_x_candidates < width) & \
                            (0 <= center_y_candidates) & (center_y_candidates < height)

                center_x_candidates, center_y_candidates = center_x_candidates[valid_idx], center_y_candidates[valid_idx]

                # Map to the discretized Hough space
                idx_x = np.searchsorted(center_x_vals, center_x_candidates)
                idx_y = np.searchsorted(center_y_vals, center_y_candidates)

                # Filter valid indices
                valid_idx = (idx_x < len(center_x_vals)) & (idx_y < len(center_y_vals))
                np.add.at(accumulator, (idx_x[valid_idx], idx_y[valid_idx], radius_idx), 1)

        return accumulator, radii, center_x_vals, center_y_vals

    def detect_circles(self, threshold=150):
        """Find circles in the accumulator above a threshold."""
        circle_indices = np.argwhere(self.accumulator > threshold)
        circles = [
            (self.center_x_vals[ix], self.center_y_vals[iy], self.radii[ir])
            for ix, iy, ir in circle_indices
        ]
        return circles

    def draw_circles(self, image, threshold=100):
        """Overlay detected circles on the image."""
        circles = self.detect_circles(threshold)
        output_img = image.copy()

        for cx, cy, r in circles:
            cv2.circle(output_img, (cx, cy), r, (0, 255, 0), 2)  # Green circles

        return output_img

    def plot_accumulator(self, selected_radius_idx=50):
        """Display a heatmap of the Hough accumulator for a selected radius."""
        if selected_radius_idx >= len(self.radii):
            print("Invalid radius index for accumulator visualization.")
            return

        accumulator_2d = self.accumulator[:, :, selected_radius_idx]

        # Remove previous content only once
        for child in self.hough_heatmap_widget.children():
            child.deleteLater()

        # Create a Matplotlib Figure and Canvas
        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot heatmap
        im = ax.imshow(accumulator_2d, cmap='hot', aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax, label="Votes")

        ax.set_xlabel("Center X")
        ax.set_ylabel("Center Y")
        ax.set_title(f"Hough Transform Accumulator (Radius = {self.radii[selected_radius_idx]})")

        # Embed the plot into the PyQt Widget
        layout = QVBoxLayout(self.hough_heatmap_widget)
        layout.addWidget(canvas)


if __name__ =='__main__':
    # Load an image and apply Hough Transform
    image = cv2.imread("Images/bicycle.jpeg")  # Change to your image path
    hough = HoughTransformCircle(image)

    # Plot the Hough accumulator
    #hough.plot_accumulator()

    # Draw detected lines on the image
    output_image = hough.draw_lines(image, threshold=150)
    # Show result
    cv2.imshow("Detected circles", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
