from PIL import Image, ImageFilter
import numpy as np
from PIL import Image
from math import cos, sin, pi, sqrt


def to_tiny_image(image_path, image_size):
    pillow_image = Image.open(image_path)
    smaller_image = pillow_image.resize(image_size, Image.Resampling.LANCZOS)
    # array_to_image = Image.fromarray(image_array)
    # array_to_image.show()
    return np.array(smaller_image).flatten()


def to_edges(image_path, image_size):
    pillow_image = Image.open(image_path)
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize(image_size, Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    flattened_edges = np.array(smaller_image).flatten()
    return flattened_edges


def to_edge_and_colour(image_path, tiny_image_size, edge_image_size):
    edge_and_colours = np.concatenate((to_tiny_image(image_path, tiny_image_size), to_edges(image_path, edge_image_size)))
    return edge_and_colours


# Function to extract RGB Histogram features using Pillow
def extract_rgb_histogram(image_path, bins=8):
    # Load image using Pillow
    img = Image.open(image_path)
    img = np.array(img)
    
    # Compute histograms for each channel (Red, Green, Blue)
    r_hist = np.histogram(img[:,:,0], bins=bins, range=(0, 256))[0]
    g_hist = np.histogram(img[:,:,1], bins=bins, range=(0, 256))[0]
    b_hist = np.histogram(img[:,:,2], bins=bins, range=(0, 256))[0]
    
    # Normalize the histograms
    r_hist = r_hist / r_hist.sum()
    g_hist = g_hist / g_hist.sum()
    b_hist = b_hist / b_hist.sum()
    
    # Combine into a single feature vector
    return np.concatenate([r_hist, g_hist, b_hist])


def local_binary_pattern(image_path, num_bins, radius):
    def get_circular_lbp(matrix, cx, cy, radius):
        rows, cols = len(matrix), len(matrix[0])
        center_value = matrix[cx][cy]
        binary_pattern = []

        # Get circular neighbors
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if 0 <= x < rows and 0 <= y < cols:  # Ensure within bounds
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:  # Circle equation
                        if not (x == cx and y == cy):  # Exclude center pixel
                            binary_pattern.append(1 if matrix[x][y] >= center_value else 0)
        # Convert binary list to integer (base 2)
        lbp_value = int("".join(map(str, binary_pattern)), 2)
        return lbp_value
    
    img = Image.open(image_path)
    greyscale = img.convert('L')
    smaller_image = greyscale.resize((128, 128), Image.Resampling.LANCZOS)
    img_lbp = np.zeros((128, 128)) 
    for i in range(0, 128): 
        for j in range(0, 128): 
            lpb_value = get_circular_lbp(np.array(smaller_image), i, j, radius)
            img_lbp[i, j] = lpb_value

    hist = np.histogram(img_lbp.flatten(), num_bins)[0]
    hist = hist / hist.sum()
    return hist


def colour_hist_and_lpb(image_path, lpb_bins, color_hist_bins):
    hist = extract_rgb_histogram(image_path, color_hist_bins)
    lbp = local_binary_pattern(image_path, lpb_bins, radius=1)
    result = np.concatenate([lbp, hist])
    return result
