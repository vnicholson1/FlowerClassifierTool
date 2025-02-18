from PIL import Image

from feature_extraction import to_colour_histogram, to_edge_and_colour, to_tiny_image, to_tiny_image_then_colour_histogram, to_tiny_image_with_edges
from nearest_neighbour import NearestNeighbourClassifier
from utils import get_image_paths, pretty_confusion_matrix


def evaluate(function, *params):
    # Create feature set
    class_name_and_paths = get_image_paths(use_train=True)
    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            image = Image.open(path)
            image_array = function(image, *params)
            training_data[class_name].append(image_array)

    print('Created image feature set')

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data)
    results = pretty_confusion_matrix(knn.training_confusion_matrix, list(class_name_and_paths.keys()))
    results += f'\n\n Best K = {knn.k}'

    # print("Test the classifier")
    # class_name_and_paths = get_image_paths(use_train=False)
    # class_names = [x.lower() for x in list(class_name_and_paths.keys())]
    # # test it
    # confusion_matrix = np.zeros((len(class_names), len(class_names))).tolist()
    # testing_data = {}
    # total_data = 0
    # for class_name, paths in class_name_and_paths.items():
    #     testing_data[class_name] = []
    #     for path in paths:
    #         image = Image.open(path)
    #         image_array = function(image, *params)
    #         testing_data[class_name].append(image_array)
    #         total_data += 1

    # print("Extracted features for the testing set")
    # for class_name, list_of_image_features in testing_data.items():
    #     print(f"Predicting class name {class_name}")
    #     for image_features in list_of_image_features:
    #         predicted, _ = knn.classify(image_features)
    #         confusion_matrix[class_names.index(predicted.lower())][class_names.index(class_name.lower())] += 1

    # results = pretty_confusion_matrix(confusion_matrix, class_names, total_data)
    # print(results)

    return results
    

if __name__ == '__main__':
    # for image_size in [(4,4), (8,8), (12,12), (16,16)]:
    #     results = evaluate(to_tiny_image, image_size)
    #     print(results)
    #     with open(f'results_tiny_images_{image_size[0]}_{image_size[1]}.txt', 'w') as f:
    #         f.write(results)


    # for image_size in [(4,4), (8,8), (12,12), (16,16)]:
    #     results = evaluate(to_tiny_image_with_edges, image_size)
    #     print(results)
    #     with open(f'results_tiny_images_with_edges_{image_size[0]}_{image_size[1]}.txt', 'w') as f:
    #         f.write(results)

    for tiny_image_size in [(8,8)]:
        for edge_image_size in [(16,16)]:
            results = evaluate(to_edge_and_colour, tiny_image_size, edge_image_size)
            print(results)
            with open(f'results_tiny_images_and_edges_tiny_{tiny_image_size[0]}_{tiny_image_size[1]}_edge_{edge_image_size[0]}_{edge_image_size[1]}.txt', 'w') as f:
                f.write(results)
