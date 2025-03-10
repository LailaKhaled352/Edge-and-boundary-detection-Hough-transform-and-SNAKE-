import cv2
import numpy as np
import matplotlib.pyplot as plt
from CannyDetector import CannyDetector

class HoughTransform:
    def __init__(self, image, theta_res=1, rho_res=1):
        """Initialize with an image and resolution settings."""
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.canny = CannyDetector( 50, 150)  # Apply Canny edge detection
        self.edges= self.canny.apply_canny_detector(self.image)
        self.theta_res = np.deg2rad(theta_res)  # Convert degrees to radians
        self.rho_res = rho_res
        self.accumulator, self.thetas, self.rhos = self._hough_transform()

    def _hough_transform(self):
        """Compute Hough Transform and return accumulator, theta, and rho values."""
        height, width = self.edges.shape
        diag_len = int(np.sqrt(height**2 + width**2))  # Maximum possible rho
        rhos = np.arange(-diag_len, diag_len, self.rho_res)
        thetas = np.arange(-np.pi / 2, np.pi / 2, self.theta_res)

        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

        # Get edge points
        edge_points = np.argwhere(self.edges)  # (y, x) format

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
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img

    def plot_accumulator(self):
        """Display the Hough accumulator heatmap."""
        plt.imshow(self.accumulator, cmap='hot', aspect='auto', extent=[np.rad2deg(self.thetas[0]), np.rad2deg(self.thetas[-1]), self.rhos[-1], self.rhos[0]])
        plt.colorbar(label="Votes")
        plt.xlabel("Theta (degrees)")
        plt.ylabel("Rho (pixels)")
        plt.title("Hough Transform Accumulator")
        plt.show()

if __name__ =='__main__':
    # Load an image and apply Hough Transform
    image = cv2.imread("Images/roberts.jpg")  # Change to your image path
    hough = HoughTransform(image)

    # Plot the Hough accumulator
    hough.plot_accumulator()

    # Draw detected lines on the image
    output_image = hough.draw_lines(image, threshold=150)
    # Show result
    cv2.imshow("Detected Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
