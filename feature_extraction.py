from PIL import Image, ImageFilter
import numpy as np
from PIL import Image
from math import cos, sin, pi, sqrt


def best_feature_extraction(pillow_image):
    smaller_image = pillow_image.resize([4,4], Image.Resampling.LANCZOS)
    tiny_image = np.array(smaller_image).flatten()
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize([8,8], Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    flattened_edges = np.array(smaller_image).flatten()
    edge_and_colours = np.concatenate((tiny_image, flattened_edges))
    return edge_and_colours


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


# Function to compute GLCM (Gray-Level Co-occurrence Matrix) manually
def compute_glcm(img_gray, distance=1, angle=0):
    # Convert image to 0-255 range (if it's not)
    img_gray = (img_gray * 255).astype(int)
    rows, cols = img_gray.shape
    
    # Get the displacement for the given angle and distance
    dx = int(round(distance * cos(angle)))
    dy = int(round(distance * sin(angle)))
    
    glcm = np.zeros((256, 256), dtype=int)
    
    # Iterate through image to calculate co-occurrences, making sure indices are valid
    for i in range(rows - dy):  # Make sure i + dy is within bounds
        for j in range(cols - dx):  # Make sure j + dx is within bounds
             current_pixel = img_gray[i, j]
            neighbor_pixel = img_gray[i + dy, j + dx]
            glcm[current_pixel, neighbor_pixel] += 1
    
    # Normalize the GLCM
    glcm = glcm / glcm.sum()
    return glcm

# Function to extract GLCM properties (contrast, correlation, energy, homogeneity)
def extract_glcm_features(image_path, distances=[1], angles=[0]):
    img = Image.open(image_path)
    img_gray = np.array(img.convert('L')) / 255.0  # Convert to grayscale and normalize
    
    contrast_list = []
    correlation_list = []
    energy_list = []
    homogeneity_list = []
    
    for distance in distances:
        for angle in angles:
            glcm = compute_glcm(img_gray, distance, angle)
            
            # Contrast
            contrast = np.sum((np.indices(glcm.shape)[0] - np.indices(glcm.shape)[1])**2 * glcm)
            contrast_list.append(contrast)
            
            # Correlation
            row_indices, col_indices = np.indices(glcm.shape)
            mean_i = np.sum(row_indices * glcm.sum(axis=1))
            mean_j = np.sum(col_indices * glcm.sum(axis=0))
            correlation = np.sum(((row_indices - mean_i) * (col_indices - mean_j)) * glcm) / np.sqrt(np.sum((row_indices - mean_i)**2) * np.sum((col_indices - mean_j)**2))
            correlation_list.append(correlation)
            
            # Energy
            energy = np.sum(glcm**2)
            energy_list.append(energy)
            
            # Homogeneity
            homogeneity = np.sum(glcm / (1 + np.abs(np.indices(glcm.shape)[0] - np.indices(glcm.shape)[1])))
            homogeneity_list.append(homogeneity)
    
    return np.concatenate([contrast_list, correlation_list, energy_list, homogeneity_list])

# Function to extract combined features (RGB + GLCM)
def extract_glcm_and_colour_hist(image_path, bins=8, distances=[1], angles=[0]):
    rgb_hist = extract_rgb_histogram(image_path, bins)
    glcm_features = extract_glcm_features(image_path, distances, angles)
    return np.concatenate([rgb_hist, glcm_features])
