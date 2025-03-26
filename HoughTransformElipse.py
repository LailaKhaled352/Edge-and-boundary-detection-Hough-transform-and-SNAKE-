import cv2
import numpy as np

class HoughTransformEllipse:
    def __init__(self, image, edges, center_res=3, a_range=(20, 100), b_range=(10, 80), angle_step=10):
        """Initialize with an image and resolution settings."""
        self.image = image
        self.edges = edges
        self.center_res = center_res
        self.a_range = a_range  # Semi-major axis range
        self.b_range = b_range  # Semi-minor axis range
        self.angle_step = angle_step  # Step size for rotation
        
        self.accumulator, self.center_x_vals, self.center_y_vals, self.a_vals, self.b_vals, self.angle_vals = self._hough_transform()

    def _hough_transform(self):
        """Compute Hough Transform for ellipses."""
        height, width = self.edges.shape
        edge_points = np.column_stack(np.where(self.edges))  # Get edge coordinates (y, x)

        # Define accumulator bins
        center_x_vals = np.arange(0, width, self.center_res)
        center_y_vals = np.arange(0, height, self.center_res)
        a_vals = np.arange(self.a_range[0], self.a_range[1], 5)
        b_vals = np.arange(self.b_range[0], self.b_range[1], 5)
        angle_vals = np.arange(0, 180, self.angle_step)

        # 5D accumulator (center_x, center_y, a, b, angle)
        accumulator = np.zeros((len(center_x_vals), len(center_y_vals), len(a_vals), len(b_vals), len(angle_vals)), dtype=np.uint16)

        # Iterate over edge points
        for y, x in edge_points:
            for ai, a in enumerate(a_vals):  # Semi-major axis
                for bi, b in enumerate(b_vals):  # Semi-minor axis (b <= a)
                    if b > a:
                        continue
                    for ti, theta in enumerate(angle_vals):  # Rotation angle
                        theta_rad = np.deg2rad(theta)
                        
                        # Compute possible ellipse center
                        x_c = x - a * np.cos(theta_rad)
                        y_c = y - b * np.sin(theta_rad)

                        # Find nearest center indices in accumulator
                        idx_x = np.abs(center_x_vals - x_c).argmin()
                        idx_y = np.abs(center_y_vals - y_c).argmin()

                        if 0 <= idx_x < len(center_x_vals) and 0 <= idx_y < len(center_y_vals):
                            accumulator[idx_x, idx_y, ai, bi, ti] += 1  # Vote

        return accumulator, center_x_vals, center_y_vals, a_vals, b_vals, angle_vals

    def detect_ellipses(self, threshold=10):
        """Find ellipses with accumulator votes above a threshold."""
        ellipse_candidates = np.argwhere(self.accumulator > threshold)
        ellipses = [
            (self.center_x_vals[x], self.center_y_vals[y], self.a_vals[a], self.b_vals[b], self.angle_vals[t])
            for x, y, a, b, t in ellipse_candidates
        ]
        return ellipses

    def draw_ellipses(self, image, threshold=100):
        """Overlay detected ellipses on the image."""
        ellipses = self.detect_ellipses(threshold)
        output_img = image.copy()

        for cx, cy, a, b, angle in ellipses:
            cv2.ellipse(output_img, (int(cx), int(cy)), (int(a), int(b)), int(angle), 0, 360, (0, 255, 0), 2)
        
        return output_img


