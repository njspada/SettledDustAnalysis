def load_image(image_path):
    """Load an image from the specified file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    return image

def draw_circles(image, coordinates, radius=5, color=(255, 0, 0)):
    """Draw circles around detected particles on the image."""
    for (x, y) in coordinates:
        cv2.circle(image, (x, y), radius, color, thickness=-1)
    return image

def get_particle_coordinates(label_image):
    """Extract the coordinates of the centroids of detected particles."""
    coordinates = []
    for region in regionprops(label_image):
        cy, cx = region.centroid
        coordinates.append((int(cx), int(cy)))
    return coordinates