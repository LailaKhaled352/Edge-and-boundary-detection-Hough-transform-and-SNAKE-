import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout 
from mpl_toolkits.mplot3d import Axes3D


class HoughTransformCircle:
    def __init__(self, image, detected_obj_widget, hough_heatmap_widget, edges, radius_res=5, center_res=3):
        """Initialize with an image and resolution settings."""
        self.image = image  # Image data as a numpy array
        self.edges = edges
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
        
        # Use uint16 to reduce memory usage
        accumulator = np.zeros((len(center_x_vals), len(center_y_vals), len(radii)), dtype=np.uint16)

        # Precompute cos & sin values for different angles
        thetas = np.linspace(0, 2 * np.pi, 100)
        cos_thetas, sin_thetas = np.cos(thetas), np.sin(thetas)

        # Iterate over edge points
        for y, x in edge_points:
            for radius_idx, radius in enumerate(radii):
                center_x_candidates = (x - radius * cos_thetas).astype(int)
                center_y_candidates = (y - radius * sin_thetas).astype(int)

                # Filter valid candidates inside image boundaries
                valid_mask = (0 <= center_x_candidates) & (center_x_candidates < width) & \
                             (0 <= center_y_candidates) & (center_y_candidates < height)

                center_x_candidates = center_x_candidates[valid_mask]
                center_y_candidates = center_y_candidates[valid_mask]

                # Map to the nearest discretized Hough space
                idx_x = np.abs(center_x_vals[:, None] - center_x_candidates).argmin(axis=0)
                idx_y = np.abs(center_y_vals[:, None] - center_y_candidates).argmin(axis=0)

                # Accumulate votes
                np.add.at(accumulator, (idx_x, idx_y, radius_idx), 1)

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
            cv2.circle(output_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)  # Ensure integers

        return output_img

    def plot_accumulator(self):
        """Display a 3D plot of the Hough accumulator (X, Y, Radius)"""
        X_vals, Y_vals, R_vals = self.center_x_vals, self.center_y_vals, self.radii
        X, Y, R = np.meshgrid(X_vals, Y_vals, R_vals, indexing='ij')  # Create a 3D grid

        # Flatten arrays for 3D scatter plot
        X_flat, Y_flat, R_flat = X.ravel(), Y.ravel(), R.ravel()
        votes_flat = self.accumulator.ravel()

        # Filter out low-vote points for clarity
        min_votes = np.percentile(votes_flat, 90)  # Show only top 10% votes
        valid_indices = votes_flat >= min_votes

        X_flat, Y_flat, R_flat, votes_flat = X_flat[valid_indices], Y_flat[valid_indices], R_flat[valid_indices], votes_flat[valid_indices]

        # Remove previous content only once
        for child in self.hough_heatmap_widget.children():
            child.deleteLater()

        # Create Matplotlib Figure and Canvas
        fig = Figure(figsize=(6, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')

        # Plot with intensity (color) based on votes
        img = ax.scatter(X_flat, Y_flat, R_flat, c=votes_flat, cmap='hot', alpha=0.7)

        # Labels and title
        ax.set_xlabel("Center X")
        ax.set_ylabel("Center Y")
        ax.set_zlabel("Radius")
        ax.set_title("3D Hough Accumulator (X, Y, R)")

        # Colorbar for reference
        fig.colorbar(img, ax=ax, label="Votes")

        # Embed the plot into the PyQt Widget
        layout = QVBoxLayout(self.hough_heatmap_widget)
        layout.addWidget(canvas)