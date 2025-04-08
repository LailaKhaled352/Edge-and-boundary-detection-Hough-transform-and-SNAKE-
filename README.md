# Edge& Boundary Dectection: Hough Transform and SNAKE Contour Model

## Overview

This project implements edge and boundary detection techniques using the Canny edge detector, Hough transform (for lines, circles, and ellipses), and the Active Contour Model (SNAKE) with a greedy algorithm.

1. **Shape Detection:**
   - For each provided image (grayscale or color):
     - Detect edges using the Canny edge detector.
     - Detect lines, circles, and ellipses using the Hough transform.
     - Superimpose the detected shapes onto the original images.
     - Provide hough paramter space Image 

2. **Active Contour Model (SNAKE):**
   - For each provided image:
     - Initialize a contour around a given object.
     - Evolve the contour using the Active Contour Model (SNAKE) with a greedy algorithm.
     - Represent the output contour as a chain code.
