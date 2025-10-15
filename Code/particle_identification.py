# This code opens a picture file and identifies particles in the image using OpenCV.
import cv2
import skimage
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max

def identify_particles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    thresh = threshold_otsu(gray_image)
    binary = gray_image > thresh

    # Remove artifacts connected to image border
    cleared = clear_border(binary)

    # Label the image
    label_image = label(cleared)

    # Create an overlay of the labels on the original image
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    # Display the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # Draw rectangle around segmented particles
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Mark the center of the particle
        cy, cx = region.centroid
        ax.plot(cx, cy, 'o', markerfacecolor='blue', markeredgecolor='k', markersize=6)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# Example usage
identify_particles('C:/git/SettledDustAnalysis/Demo/LocationC.png')