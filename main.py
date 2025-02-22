from bow import create_training_data, read_and_clusterize
from feature_extraction import get_sift_features, to_colour_histogram, to_edge_and_colour, to_tiny_image, to_tiny_image_then_colour_histogram, to_tiny_image_with_edges
from nearest_neighbour import NearestNeighbourClassifier
from utils import get_image_paths, pretty_confusion_matrix


def evaluate(function, *params):
    # Create feature set
    class_name_and_paths = get_image_paths(use_reduced_train=True)
    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            image_array = function(path, *params)
            training_data[class_name].append(image_array)

    print('Created image feature set')

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data)
    results = pretty_confusion_matrix(knn.training_confusion_matrix, list(class_name_and_paths.keys()))
    results += f'\n\n Best K = {knn.k}'
    return results


def evaluate_bow(num_clusters):
    # Create feature set
    class_name_and_paths = get_image_paths(use_reduced_train=True)
    kmeans = read_and_clusterize(class_name_and_paths, num_clusters)
    training_data = create_training_data(class_name_and_paths, kmeans, num_clusters)

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data)
    results = pretty_confusion_matrix(knn.training_confusion_matrix, list(class_name_and_paths.keys()))
    results += f'\n\n Best K = {knn.k}'
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

    # for num_clusters in [500, 1000, 1500, 2000, 2500]:
    #     results = evaluate_bow(num_clusters=num_clusters)
    #     print(results)
    #     with open(f'results_bow_{num_clusters}.txt', 'w') as f:
    #         f.write(results)
