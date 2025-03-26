import cv2
import numpy as np
from skimage.measure import EllipseModel

class HoughTransformEllipse:
    def __init__(self, image, detected_obj_widget, edges ,center_res=3, major_axis_res=5):
        self.image = image
        self.edges = edges
        self.center_res = center_res
        self.major_axis_res = major_axis_res
        self.accumulator, self.center_x_vals, self.center_y_vals, self.major_axes = self._hough_transform()
        self.detected_obj_widget = detected_obj_widget

    def _hough_transform(self):
        height, width = self.edges.shape
        edge_points = np.column_stack(np.where(self.edges))  # (y, x) format

        # Define search space
        center_x_vals = np.arange(0, width, self.center_res)
        center_y_vals = np.arange(0, height, self.center_res)
        major_axes = np.arange(10, 200, self.major_axis_res)
        
        accumulator = np.zeros((len(center_x_vals), len(center_y_vals), len(major_axes)), dtype=np.uint16)
        
        # Compute edge gradients
        grad_x = cv2.Sobel(self.edges, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(self.edges, cv2.CV_64F, 0, 1)
        gradient_angles = np.arctan2(grad_y, grad_x)

   
        
        for y, x in edge_points:
            theta = gradient_angles[y, x]
            for a_idx, a in enumerate(major_axes):
                cx = int(x - a * np.cos(theta))
                cy = int(y - a * np.sin(theta))
                
                if 0 <= cx < width and 0 <= cy < height:
                    idx_x = np.abs(center_x_vals - cx).argmin()
                    idx_y = np.abs(center_y_vals - cy).argmin()
                    accumulator[idx_x, idx_y, a_idx] += 1
        
        return accumulator, center_x_vals, center_y_vals, major_axes

    def detect_ellipses(self, threshold=150, min_distance=15):
        ellipse_indices = np.argwhere(self.accumulator > threshold)
        ellipses = []
        for ix, iy, ia in ellipse_indices:
            cx, cy, a = self.center_x_vals[ix], self.center_y_vals[iy], self.major_axes[ia]
            if not any(np.linalg.norm(np.array([cx, cy]) - np.array([ex, ey])) < min_distance for ex, ey, _ in ellipses):
              ellipses.append((cx, cy, a))
        return ellipses

    def fit_ellipses(self, detected_ellipses):
        fitted_ellipses = []
        for cx, cy, a in detected_ellipses:
            nearby_edges = self.edges[max(0, cy-50):min(self.edges.shape[0], cy+50),
                                      max(0, cx-50):min(self.edges.shape[1], cx+50)]
            edge_points = np.column_stack(np.where(nearby_edges))
            
            if len(edge_points) >= 5:
                ellipse = EllipseModel()
                if ellipse.estimate(edge_points):
                    fitted_ellipses.append(ellipse.params)  # (xc, yc, a, b, theta)
        return fitted_ellipses
    
    def draw_ellipses(self, image, fitted_ellipses):
        output_img = image.copy()
        for xc, yc, a, b, theta in fitted_ellipses:
            cv2.ellipse(output_img, (int(xc), int(yc)), (int(a), int(b)), np.rad2deg(theta), 0, 360, (0, 0, 255), 2)
        return output_img
    
