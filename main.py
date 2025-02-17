from PIL import Image, ImageFilter
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


def to_tiny_image(pillow_image, image_size):
    smaller_image = pillow_image.resize(image_size, Image.Resampling.LANCZOS)
    # array_to_image = Image.fromarray(image_array)
    # array_to_image.show()
    return np.array(smaller_image).flatten()


def to_tiny_image_with_edges(pillow_image, image_size):
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize(image_size, Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    return np.array(smaller_image).flatten()


def combo(pillow_image, args):
    tiny_image_size, edge_size = args

    tiny_image = to_tiny_image(pillow_image, tiny_image_size)
    edge_image = to_tiny_image_with_edges(pillow_image, edge_size)
    return np.concatenate((tiny_image, edge_image))

def to_colour_histogram(pillow_image, num_bins):
    red, green, blue = pillow_image.split()
    red_hist, _ = np.histogram(red, bins=num_bins)
    green_hist, _ = np.histogram(green, bins=num_bins)
    blue_hist, _ = np.histogram(blue, bins=num_bins)
    return np.concatenate((red_hist, green_hist, blue_hist))


def pretty_confusion_matrix(confusion_matrix, class_names, total):

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

    text += '\n\n' + f'Accuracy = {(num_correct/total) * 100}%'
    return text


def evaluate(function, args, ks):
    for k in ks:
        for size in args:
            # Create feature set
            class_name_and_paths = get_image_paths(use_train=True)
            training_data = {}
            for class_name, paths in class_name_and_paths.items():
                training_data[class_name] = []
                for path in paths:
                    image = Image.open(path)
                    image_array = function(image, size)
                    training_data[class_name].append(image_array)

            print('Created image feature set')

            # Create the classifier
            knn = NearestNeighbourClassifier(training_data, k=k)

            print("Test the classifier")
            class_name_and_paths = get_image_paths(use_train=False)
            class_names = [x.lower() for x in list(class_name_and_paths.keys())]
            # test it
            confusion_matrix = np.zeros((len(class_names), len(class_names))).tolist()
            testing_data = {}
            total_data = 0
            for class_name, paths in class_name_and_paths.items():
                testing_data[class_name] = []
                for path in paths:
                    image = Image.open(path)
                    image_array = function(image, size)
                    testing_data[class_name].append(image_array)
                    total_data += 1

            print("Extracted features for the testing set")
            for class_name, list_of_image_features in testing_data.items():
                print(f"Predicting class name {class_name}")
                for image_features in list_of_image_features:
                    predicted, _ = knn.classify(image_features)
                    confusion_matrix[class_names.index(predicted.lower())][class_names.index(class_name.lower())] += 1

            results = pretty_confusion_matrix(confusion_matrix, class_names, total_data)
            print(results)

            if type(size) == tuple:
                with open(f'results_{function.__name__}_{size[0]}_{size[1]}_k_{knn.k}.txt', 'w') as f:
                    f.write(results)
            else:
                with open(f'results_{function.__name__}_{size}_k_{knn.k}.txt', 'w') as f:
                    f.write(results)


def run_evaluation(tiny_image_sizes, image_type):

    if image_type == "tiny_images":
        function = to_tiny_image
    elif image_type == 'tiny_images_with_edges':
        function = to_tiny_image_with_edges
    else:
        function = to_colour_histogram

    evaluate(function, tiny_image_sizes)
    

if __name__ == '__main__':
    evaluate(to_tiny_image, [(4,4), (8,8)], [1,2,3,4,5])
