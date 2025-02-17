from PIL import Image
import os
import numpy as np
from os import listdir
from os.path import isfile, join

from nearest_neighbour import NearestNeighbourClassifier


def get_image_paths(use_train: bool):
    folder = os.path.join("data", "train") if use_train else os.path.join("data", "test")
    class_names_and_paths = {}
    for directory, _, _ in os.walk(folder):
        _, class_name = os.path.split(directory)
        class_names_and_paths[class_name] = []
        for image_path in listdir(directory):
            if isfile(join(directory, image_path)):
                class_names_and_paths[class_name].append(join(directory, image_path))
    if use_train:
        del class_names_and_paths['train']
    else:
        del class_names_and_paths['test']
    return class_names_and_paths


def to_tiny_image(pillow_image):
    size = 16, 16
    smaller_image = pillow_image.resize(size, Image.Resampling.LANCZOS)
    # array_to_image = Image.fromarray(image_array)
    # array_to_image.show()
    return np.array(smaller_image).flatten()


def main():
    # Create feature set
    class_name_and_paths = get_image_paths(use_train=True)
    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            image = Image.open(path)
            image_array = to_tiny_image(image)
            training_data[class_name].append(image_array)

    print('Created image feature set')

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data, k=20)

    print("Test the classifier")
    class_name_and_paths = get_image_paths(use_train=False)
    class_names = [x.lower() for x in list(class_name_and_paths.keys())]
    # test it
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    testing_data = {}
    for class_name, paths in class_name_and_paths.items():
        testing_data[class_name] = []
        for path in paths:
            image = Image.open(path)
            image_array = to_tiny_image(image)
            testing_data[class_name].append(image_array)

    print("Extracted features for the testing set")
    for class_name, list_of_image_features in testing_data.items():
        for image_features in list_of_image_features:
            predicted, _ = knn.classify(image_features)
            confusion_matrix[class_names.index(predicted.lower())][class_names.index(class_name.lower())] += 1

    print("Calculated accuracy and confusion matrix")
    print(confusion_matrix)

if __name__ == '__main__':
    main()
