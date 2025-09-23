from feature_extraction import colour_hist_and_hog, extract_rgb_histogram, to_edge_and_colour, to_tiny_image, to_edges
from nearest_neighbour import NearestNeighbourClassifier
from utils import pretty_confusion_matrix, get_image_paths
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV 
from math import pi


def test_classifier(classifier, function, *params):
    # Create the test set
    class_name_and_paths = get_image_paths(folder_name='test')
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
    class_name_and_paths = get_image_paths(folder_name='train')
    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            image_array = function(path, *params)
            training_data[class_name].append(image_array)
        print(f'{class_name} features created')
    return training_data
    

def evaluate_both(function, *params):
    training_data = create_training_data(function, *params)
    knn_results = evaluate_knn(function, training_data, *params)
    print(knn_results)
    svm_results = evaluate_svm(function, training_data, *params)
    print(svm_results)
    return knn_results, svm_results


def evaluate_knn(function, training_data=None, *params):
    if not training_data:
        training_data = create_training_data(function, *params)

    print('Created image feature set')

    # Create the classifier
    knn = NearestNeighbourClassifier(training_data, use_weighted=True)
    print('Created classifier')
    results = test_classifier(knn, function, *params)
    results += f"\n\nBest K={knn.k} Weighted={knn.use_weighted_votes}"
    return results


def evaluate_svm(function, training_data=None, *params):
    if not training_data:
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
              'kernel': ['linear', 'poly', 'rbf']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
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
    for hog_bins in [8, 12, 16, 20, 24]:
        for color_hist_bins in [8, 12, 16, 20, 24]:
            knn_results, svm_results = evaluate_both(colour_hist_and_hog, hog_bins, color_hist_bins)
            with open(f'results_knn_hog_{hog_bins}_hist_{color_hist_bins}.txt', 'w') as f:
                f.write(knn_results)
            with open(f'results_svm_hog_{hog_bins}_hist_{color_hist_bins}.txt', 'w') as f:
                f.write(svm_results)
