from PIL import Image, ImageFilter
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from sklearn.preprocessing import normalize



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


def extract_hog_like_features(image_path, num_bins):
    """Extracts HOG-like gradient features from an image."""
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = np.array(image, dtype=np.float32)

    # Compute horizontal and vertical gradients
    gx = sobel(image, axis=1)  # Sobel filter in X direction
    gy = sobel(image, axis=0)  # Sobel filter in Y direction

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)

    # Histogram of oriented gradients (8 bins)
    hist, _ = np.histogram(orientation, bins=num_bins, range=(-np.pi, np.pi), weights=magnitude)

    # Normalize histogram
    hist = normalize(hist.reshape(1, -1))[0]

    return hist


def colour_hist_and_hog(image_path, color_hist_bins, hog_bins):
    hist = extract_rgb_histogram(image_path, color_hist_bins)
    hog = extract_hog_like_features(image_path, hog_bins)
    result = np.concatenate([hist, hog])
    return result
