from PIL import Image, ImageFilter
import numpy as np
from PIL import Image

# For LBP
from skimage.feature import local_binary_pattern



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


# 1. Grayscale Intensity Histogram
def extract_grayscale_histogram(image_path, bins=16):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    hist, _ = np.histogram(arr, bins=bins, range=(0, 256))
    hist = hist / hist.sum()
    return hist

# 2. Mean and Standard Deviation for each color channel
def extract_mean_std(image_path):
    img = Image.open(image_path)
    arr = np.array(img)
    if arr.ndim == 2:  # grayscale
        arr = arr[:, :, np.newaxis]
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([means, stds])

# 3. Local Binary Pattern (LBP) Texture Histogram
def extract_lbp_histogram(image_path, P=8, R=1, bins=10):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    lbp = local_binary_pattern(arr, P, R, method='uniform')
    hist, _ = np.histogram(lbp, bins=bins, range=(0, P + 2))
    hist = hist / hist.sum()
    return hist
