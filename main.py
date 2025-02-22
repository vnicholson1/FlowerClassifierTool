from bow import create_training_data, read_and_clusterize
from feature_extraction import get_sift_features, to_colour_histogram, to_edge_and_colour, to_tiny_image, to_tiny_image_then_colour_histogram, to_tiny_image_with_edges
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


def evaluate_svm(c, gamma, kernel):
    training_data = create_training_data(to_edge_and_colour, [8,8], [16,16])
    x_list = []
    y_list = []
    for class_name, list_of_feats in training_data.items():
        for feats in list_of_feats:
            x_list.append(feats)
            y_list.append(class_name)
    X = np.vstack(x_list)
    Y = np.asarray(y_list)
    # tuning code
    #best -  {'C': 0.1, 'gamma': 1, 'kernel': 'rbf'}
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],  
    #           'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #           'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    # grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0, scoring=) 
    # grid.fit(X, Y)
    # print(grid.best_params_) 
    # print(grid.best_estimator_)
    # tuning code end
    svm_classifier = svm.SVC(C=c, gamma=gamma, kernel=kernel)
    svm_classifier.fit(X, Y)
    return test_classifier(svm_classifier, to_edge_and_colour, [8,8], [16,16])

    

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
    #         with open(f'results_svm.txt', 'w') as f:
    #             f.write(results)

    for c in [0.1, 1, 10, 100, 1000]:
        for gamma in [1, 0.1, 0.01, 0.001, 0.0001]:
            for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
                results = evaluate_svm(c, gamma, kernel)
                print(results)
                with open(f'results_svm_c_{c}_gamma_{gamma}_kernel_{kernel}.txt', 'w') as f:
                    f.write(results)
