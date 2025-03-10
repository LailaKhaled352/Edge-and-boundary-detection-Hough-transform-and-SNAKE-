import cv2
import numpy as np

class CannyDetector:
    def __init__(self, low_threshold=50, high_threshold=150, kernel_size=5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def apply_gaussian_blur(self, image, sigma=1):
        """Applies Gaussian blur to reduce noise."""
        k = self.kernel_size // 2
        y, x = np.mgrid[-k:k+1, -k:k+1]
        gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        
        return cv2.filter2D(image, -1, gaussian_kernel)

    def sobel_gradients(self, image):
        """Computes gradient magnitude and direction using Sobel operator."""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        Gx = cv2.filter2D(image, -1, sobel_x)
        Gy = cv2.filter2D(image, -1, sobel_y)
        
        magnitude = np.sqrt(Gx**2 + Gy**2)
        magnitude = (magnitude / magnitude.max()) * 255  # Normalize to 0-255
        direction = np.arctan2(Gy, Gx)
        
        return magnitude, direction

    def non_maximum_suppression(self, magnitude, direction):
        """thining edges by comparing each pixel to its neigbors and supress it if it's not the highest"""
        M, N = magnitude.shape
        output = np.zeros((M, N), dtype=np.uint8)
        angle = direction * 180 / np.pi
        angle[angle < 0] += 180  # Normalize angles to 0-180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                # Angle quantization to 4 possible directions
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]  # Right neighbor
                    r = magnitude[i, j - 1]  # Left neighbor
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]  # Bottom-left neighbor
                    r = magnitude[i - 1, j + 1]  # Top-right neighbor
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]  # Bottom neighbor
                    r = magnitude[i - 1, j]  # Top neighbor
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]  # Top-left neighbor
                    r = magnitude[i + 1, j + 1]  # Bottom-right neighbor

                # Keep the maximum pixel, suppress others
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    output[i, j] = magnitude[i, j]
                else:
                    output[i, j] = 0
        return output

    def double_threshold(self, image):
        """Apply double threshold to classify pixels as strong, weak, or suppressed."""
        strong = 255
        weak = 75

        strong_i, strong_j = np.where(image >= self.high_threshold)
        weak_i, weak_j = np.where((image >= self.low_threshold) & (image < self.high_threshold))

        output = np.zeros_like(image, dtype=np.uint8)
        output[strong_i, strong_j] = strong
        output[weak_i, weak_j] = weak

        return output, weak, strong

    def edge_tracking_by_hysteresis(self, image, weak, strong):
        """Link weak edges to strong ones."""
        M, N = image.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if image[i, j] == weak:
                    if (
                        image[i + 1, j - 1] == strong or image[i + 1, j] == strong or image[i + 1, j + 1] == strong
                        or image[i, j - 1] == strong or image[i, j + 1] == strong
                        or image[i - 1, j - 1] == strong or image[i - 1, j] == strong or image[i - 1, j + 1] == strong
                    ):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image

    def apply_canny_detector(self, image):
        """Complete pipeline for Canny Edge Detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred = self.apply_gaussian_blur(gray)  # Step 1: Smooth image
        magnitude, direction = self.sobel_gradients(blurred)  # Step 2: Compute gradients
        suppressed = self.non_maximum_suppression(magnitude, direction)  # Step 3: Non-Maximum Suppression
        thresholded, weak, strong = self.double_threshold(suppressed)  # Step 4: Double Thresholding
        final_edges = self.edge_tracking_by_hysteresis(thresholded, weak, strong)  # Step 5: Hysteresis

        return final_edges

# Example usage
if __name__ == "__main__":
    image = cv2.imread("Images/roberts.jpg")  # Load an image
    # detector = CannyDetector(low_threshold=75, high_threshold=100)
    # edges = detector.apply_canny_detector(image)
    edges= cv2.Canny(image, 75, 150)

    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
