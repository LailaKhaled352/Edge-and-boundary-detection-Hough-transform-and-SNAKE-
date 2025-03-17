import numpy as np
from CannyDetector import CannyDetector

class ActiveContour:
    def __init__(self, image):
        self.image = image
        self.canny_detector = CannyDetector()

    def compute_gradients(self, edge_image):
        """Compute gradient maps using Sobel filtering (without OpenCV)."""
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1], 
                            [0,  0,  0], 
                            [1,  2,  1]], dtype=np.float32)

        grad_x = self.convolve(edge_image, sobel_x)
        grad_y = self.convolve(edge_image, sobel_y)
        
        return grad_x, grad_y

    def convolve(self, image, kernel):
        """Applies convolution manually."""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Pad image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        result = np.zeros_like(image)

        # Perform convolution
        for i in range(h):
            for j in range(w):
                region = padded_image[i:i+kh, j:j+kw]
                result[i, j] = np.sum(region * kernel)
        
        return result

    def apply_active_contour(self):
        """Applies Active Contour (Snake) algorithm."""
        alpha=0.1
        gamma=0.1 
        num_iterations=500

        # Step 1: Detect edges using Canny
        edge_image = self.canny_detector.apply_gaussian_blur(self.image)
        
        # Step 2: Compute image gradients
        grad_x, grad_y = self.compute_gradients(edge_image)

        # Step 3: Initialize contour as a circle
        t = np.linspace(0, 2 * np.pi, 100)  # 100 points on the contour
        x = 150 + 100 * np.cos(t)  # Center at (150,150), radius 100
        y = 150 + 100 * np.sin(t)
        contour = np.array([x, y]).T  # Shape: (100,2)

        # Step 4: Active Contour Optimization Loop
        for _ in range(num_iterations):
            new_contour = np.copy(contour)

            for i in range(len(contour)):
                # Compute internal forces (smoothness)
                prev_point = contour[i - 1]
                next_point = contour[(i + 1) % len(contour)]  # Wrap around (circular contour)
                #Alpha (α) - Smoothness Term (Elasticity)
                internal_force = alpha * (prev_point + next_point - 2 * contour[i])  # Smoothness term

                # Compute external forces (edge attraction)
                x, y = int(contour[i][0]), int(contour[i][1])
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:  # Check bounds
                    grad_fx = grad_x[y, x]  
                    grad_fy = grad_y[y, x]

                    #Gamma (γ) - External Force Weight
                    external_force = gamma * np.array([-grad_fx, -grad_fy])  # Move towards strong edges
                else:
                    external_force = np.array([0, 0])

                # Compute new position
                new_contour[i] = contour[i] + internal_force + external_force

            # Step 5: Update contour
            contour = new_contour

        return contour
