from feature_extraction import to_edge_and_colour, to_tiny_image, to_tiny_image_with_edges
from nearest_neighbour import NearestNeighbourClassifier
from utils import get_image_paths, pretty_confusion_matrix
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def test_classifier(classifier, function, *params):
    # Create the test set
    class_name_and_paths = get_image_paths(folder_name='reduced_test')
    class_names = list(class_name_and_paths.keys())
    test_data = {}
    for class_name, paths in class_name_and_paths.items():
        test_data[class_name] = []
        for path in paths:
            image_array = function(path, *params)
            test_data[class_name].append(image_array)

    confusion_matrix = np.zeros((len(class_names), len(class_names))).tolist()
    for class_name, list_of_image_features in test_data.items():
        for image_features in list_of_image_features:
            predicted = classifier.predict(image_features.reshape(1,-1))[0]
            confusion_matrix[class_names.index(predicted.lower())][class_names.index(class_name.lower())] += 1
    results = pretty_confusion_matrix(confusion_matrix, list(class_name_and_paths.keys()))
    return results


def create_training_data(function, *params):
    # Create feature set
    class_name_and_paths = get_image_paths(folder_name='reduced_train')
    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            image_array = function(path, *params)
            training_data[class_name].append(image_array)
    return training_data
    

def evaluate_knn(function, *params):
    training_data = create_training_data(function, *params)

    print('Created image feature set')

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data, k=5)
    print('Created classifier')
    return test_classifier(knn, function, *params)


def evaluate_svm(function, *params):
    training_data = create_training_data(function, *params)
    print('Created image feature set')
    x_list = []
    y_list = []
    for class_name, list_of_feats in training_data.items():
        for feats in list_of_feats:
            x_list.append(feats)
            y_list.append(class_name)
    X = np.vstack(x_list)
    Y = np.asarray(y_list)
    print('Convert to numpy arrays')
    # tuning code
    #best -  {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0) 
    grid.fit(X, Y)
    # tuning code end
    svm_classifier = svm.SVC(**grid.best_params_)
    svm_classifier.fit(X, Y)
    print('Tuned classifier')
    results = test_classifier(svm_classifier, function, *params)
    results += f"\n\nC={grid.best_params_['C']}"
    results += f"\ngamma={grid.best_params_['gamma']}"
    results += f"\nkernel={grid.best_params_['kernel']}"
    return results

    

if __name__ == '__main__':
    # for image_size in [(4,4), (8,8), (12,12), (16,16)]:
    #     results = evaluate_knn(to_tiny_image, image_size)
    #     print(results)
    #     with open(f'results_tiny_images_{image_size[0]}_{image_size[1]}.txt', 'w') as f:
    #         f.write(results)


    # for image_size in [(4,4), (8,8), (12,12), (16,16)]:
    #     results = evaluate_knn(to_tiny_image_with_edges, image_size)
    #     print(results)
    #     with open(f'results_tiny_images_with_edges_{image_size[0]}_{image_size[1]}.txt', 'w') as f:
    #         f.write(results)

    # for tiny_image_size in [(8,8)]:
    #     for edge_image_size in [(16,16)]:
    #         results = evaluate_knn(to_edge_and_colour, tiny_image_size, edge_image_size)
    #         print(results)
    #         with open(f'results_knn.txt', 'w') as f:
    #             f.write(results)

    for tiny_image_size in [(4,4), (8,8), (12,12), (16,16)]:
        for edge_image_size in [(4,4), (8,8), (12,12), (16,16)]:
            results = evaluate_svm(to_edge_and_colour, tiny_image_size, edge_image_size)
            print(results)
            with open(f'results_svm_tiny_{tiny_image_size[0]}_{tiny_image_size[1]}_edge_{edge_image_size[0]}_{edge_image_size[1]}.txt', 'w') as f:
                f.write(results)

    for edge_image_size in [(4,4), (8,8), (12,12), (16,16)]:
        results = evaluate_svm(to_tiny_image_with_edges, edge_image_size)
        print(results)
        with open(f'results_svm_edge_{edge_image_size[0]}_{edge_image_size[1]}.txt', 'w') as f:
            f.write(results)
