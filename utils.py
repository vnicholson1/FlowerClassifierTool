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


def pretty_confusion_matrix(confusion_matrix, class_names):

    accuracy = calculate_accuracy(confusion_matrix) * 100
    balanced_accuracy = calculate_balanced_accuracy(confusion_matrix) * 100
    num_correct = 0
    for i in range(len(class_names)):
        num_correct += confusion_matrix[i][i]

    confusion_matrix.insert(0, [''] + class_names)
    for i, row in enumerate(confusion_matrix[1:]):
        row.insert(0, class_names[i])
    s = [[str(e) for e in row] for row in confusion_matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    text = '\n'.join(table)

    text += '\n\n' + f'Accuracy = {accuracy}%'
    text += '\n\n' + f'Balanced Accuracy = {balanced_accuracy}%'
    return text


def calculate_accuracy(confusion_matrix):
    total_data = 0
    num_correct = 0
    for i in range(len(confusion_matrix)):
        total_data += sum([x[i] for x in confusion_matrix])
        num_correct += confusion_matrix[i][i]
    return num_correct / total_data


def calculate_balanced_accuracy(confusion_matrix):
    accuracy_sum = 0
    for i in range(len(confusion_matrix)):
        num_in_class = sum([x[i] for x in confusion_matrix])
        class_accuracy = confusion_matrix[i][i] / num_in_class
        accuracy_sum += class_accuracy
    return accuracy_sum / len(confusion_matrix)


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
