from PIL import Image, ImageFilter
import numpy as np


def to_tiny_image(pillow_image, image_size):
    smaller_image = pillow_image.resize(image_size, Image.Resampling.LANCZOS)
    # array_to_image = Image.fromarray(image_array)
    # array_to_image.show()
    return np.array(smaller_image).flatten()


def to_tiny_image_with_edges(pillow_image, image_size):
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize(image_size, Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    flattened_edges = np.array(smaller_image).flatten()
    return flattened_edges


def to_edge_and_colour(pillow_image, tiny_image_size, edge_image_size):
    edge_and_colours = np.concatenate((to_tiny_image(pillow_image, tiny_image_size), to_tiny_image_with_edges(pillow_image, edge_image_size)))
    return edge_and_colours


def to_colour_histogram(pillow_image, num_bins):
    red, green, blue = pillow_image.split()
    red_hist, _ = np.histogram(red, bins=num_bins)
    green_hist, _ = np.histogram(green, bins=num_bins)
    blue_hist, _ = np.histogram(blue, bins=num_bins)
    return np.concatenate((red_hist, green_hist, blue_hist))


def to_tiny_image_then_colour_histogram(pillow_image, num_bins):
    pillow_image = pillow_image.resize((16,16), Image.Resampling.LANCZOS)
    red, green, blue = pillow_image.split()
    red_hist, _ = np.histogram(red, bins=num_bins)
    green_hist, _ = np.histogram(green, bins=num_bins)
    blue_hist, _ = np.histogram(blue, bins=num_bins)
    return np.concatenate((red_hist, green_hist, blue_hist))
