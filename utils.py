import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

from create_data_points import NUM_FEATURES, NUM_CLUSTERS


def best_feature_extraction(pillow_image, kmeans):
    """
    Extract BoVW histogram for a single image using SIFT and a pre-trained KMeans vocabulary.
    Pass a preloaded kmeans model for efficiency, or leave as None to load from disk.
    """
    # Convert PIL image to grayscale numpy array
    img = np.array(pillow_image.convert('L'))
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)
    _, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        hist = np.zeros(NUM_CLUSTERS)
    else:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(NUM_CLUSTERS+1))
    return hist


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
    sorted_classes = {k: class_names_and_paths[k] for k in sorted(class_names_and_paths.keys())}
    return sorted_classes
