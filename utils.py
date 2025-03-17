import os
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageFilter
import numpy as np


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


def get_image_paths(folder_name: str):
    folder = os.path.join("data", folder_name)
    class_names_and_paths = {}
    for directory, _, _ in os.walk(folder):
        _, class_name = os.path.split(directory)
        class_names_and_paths[class_name] = []
        for image_path in listdir(directory):
            if isfile(join(directory, image_path)):
                class_names_and_paths[class_name].append(join(directory, image_path))
    del class_names_and_paths[folder_name]
    return class_names_and_paths
