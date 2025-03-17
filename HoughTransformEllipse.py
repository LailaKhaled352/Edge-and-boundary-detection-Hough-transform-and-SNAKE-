import cv2
import numpy as np
from CannyDetector import CannyDetector

class HoughTransformEllipse:
  def __init__(self, image, center_res=3, a_res=5, b_res=5, theta_res=5):
      """Initialize the Hough Transform for ellipse detection."""
      self.image = image
      self.canny = CannyDetector(75, 150)
      self.edges = self.canny.apply_canny_detector(self.image)

      # Resolution settings
      self.center_res = center_res
      self.a_res = a_res
      self.b_res = b_res
      self.theta_res = np.deg2rad(theta_res)  # Convert degrees to radians

      # Extract image dimensions
      self.height, self.width = self.edges.shape

      # Define search space
      self.center_x_vals = np.arange(0, self.width, self.center_res)
      self.center_y_vals = np.arange(0, self.height, self.center_res)
      self.a_vals = np.arange(10, min(self.width, self.height) // 2, self.a_res)
      self.b_vals = np.arange(10, min(self.width, self.height) // 2, self.b_res)
      self.theta_vals = np.arange(0, np.pi, self.theta_res)

      # Initialize 5D accumulator
      self.accumulator = np.zeros(
        (len(self.center_x_vals), len(self.center_y_vals), len(self.a_vals), len(self.b_vals), len(self.theta_vals)),
        dtype=np.int32
      )

  def _hough_transform(self):
      """Compute Hough Transform for ellipse detection."""
      edge_points = np.column_stack(np.where(self.edges))  # Get edge coordinates (y, x)

      for y, x in edge_points:
          for a_idx, a in enumerate(self.a_vals):
              for b_idx, b in enumerate(self.b_vals):
                  for theta_idx, theta in enumerate(self.theta_vals):
                      x_c = x - a * np.cos(theta)
                      y_c = y - b * np.sin(theta)

                      # Ensure center is within image bounds
                      if 0 <= x_c < self.width and 0 <= y_c < self.height:
                          # Find closest center indices
                          idx_x = np.searchsorted(self.center_x_vals, x_c)
                          idx_y = np.searchsorted(self.center_y_vals, y_c)

                          # Ensure indices are valid
                          if idx_x < len(self.center_x_vals) and idx_y < len(self.center_y_vals):
                              self.accumulator[idx_x, idx_y, a_idx, b_idx, theta_idx] += 1


  def detect_ellipses(self, threshold=100):
      """Extract detected ellipses from the accumulator."""
      ellipses = []
      idxs = np.argwhere(self.accumulator > threshold)

      for idx_x, idx_y, a_idx, b_idx, theta_idx in idxs:
          x_c = self.center_x_vals[idx_x]
          y_c = self.center_y_vals[idx_y]
          a = self.a_vals[a_idx]
          b = self.b_vals[b_idx]
          theta = self.theta_vals[theta_idx]
          ellipses.append((x_c, y_c, a, b, theta))

      return ellipses
  
  def draw_ellipses(self, image, threshold=100):
      """Draw detected ellipses on the image."""
      ellipses = self.detect_ellipses(threshold)
      output_img = image.copy()

      for x_c, y_c, a, b, theta in ellipses:
          cv2.ellipse(
              output_img,
              (int(x_c), int(y_c)), (int(a), int(b)),
              np.rad2deg(theta),
              0, 360, (0, 0, 255), 2
          )

      return output_img  

# #test
# if __name__ == '__main__':
#     image = cv2.imread('Images/Ellipse.png', cv2.IMREAD_GRAYSCALE)
#     hough = HoughTransformEllipse(image)

#     hough._hough_transform()
#     result_img = hough.draw_ellipses(image)

#     cv2.imshow("Detected Ellipses", result_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
