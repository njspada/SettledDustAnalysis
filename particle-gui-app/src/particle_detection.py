import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.segmentation import clear_border

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

import cv2
import numpy as np
from skimage.measure import label, regionprops

def detect_particles(image_path, threshold_value):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding (ignores threshold_value slider, but you can blend with Otsu or use as C)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, threshold_value - 128
    )

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=2)

    # Label connected components
    labeled = label(morph)
    num_particles = np.max(labeled)

    # Draw circles around particles
    output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for region in regionprops(labeled):
        if region.area < 10:  # Filter out very small regions
            continue
        cy, cx = region.centroid
        radius = int(max(region.major_axis_length, region.minor_axis_length) / 2)
        cv2.circle(output_image, (int(cx), int(cy)), radius, (255, 0, 0), 2)

    # Convert to RGB for matplotlib/Tkinter
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    return num_particles, output_image

def count_particles(label_image):
    return np.max(label_image)

def get_particle_overlay(image, label_image):
    return label2rgb(label_image, image=image, bg_label=0)

def get_threshold_values(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return threshold_otsu(gray_image)