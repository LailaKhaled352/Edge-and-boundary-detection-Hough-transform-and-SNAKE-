import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from ActiveCountor import ActiveContour  # Import your class

# Load the color image
color_image = cv2.imread("C:/Computer_Vision/task2_canny&activecountour/Edge-and-boundary-detection-Hough-transform-and-SNAKE-/green.jpg")
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

# Convert to grayscale for processing
gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

# Initialize ActiveContour class
active_contour = ActiveContour(gray_image)

# Create a figure and axis
fig, ax = plt.subplots()
ax.imshow(color_image)  # Show the original color image
contour_line, = ax.plot([], [], '-r', linewidth=2) 
# Function to handle region selection
def on_select(eclick, erelease):
    """ Callback when user selects a region. """
    active_contour.on_select(eclick, erelease)  # Store coordinates in ActiveContour
    plt.close()  # Close selection window after selection

# Rectangle selector for region selection
rect_selector = RectangleSelector(ax, on_select, useblit=True, interactive=True)

plt.show()  # Show image for selection

# Now apply active contour if a region was selected
if None not in (active_contour.start_x, active_contour.start_y, active_contour.end_x, active_contour.end_y):
    print("Processing Active Contour...")

    # Apply active contour
    final_contour = active_contour.apply_active_contour()

    # Display the updated image with contour
    fig, ax = plt.subplots()
    ax.imshow(color_image)  # Show the original color image
    if final_contour is not None:
        ax.plot(final_contour[:, 0], final_contour[:, 1], '-r', linewidth=2)
    plt.show()
else:
    print("No region selected. Please run the script again.")

