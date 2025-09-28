"""
bovw.py
Bag of Visual Words (BoVW) implementation using ORB features and k-means clustering.
This script extracts features from images, builds a visual vocabulary, and represents images as histograms of visual words.
"""

import os
from typing import Literal
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import json

# Parameters
TRAIN_PATH = 'data/train'  # Path to training images
TEST_PATH = 'data/test'    # Path to test images


# Function to extract RGB Histogram features
def extract_rgb_histogram(image_path, bins):
    img = cv2.imread(image_path)
    img = np.array(img)
    
    # Compute histograms for each channel (Red, Green, Blue)
    r_hist = np.histogram(img[:,:,0], bins=bins, range=(0, 256))[0]
    g_hist = np.histogram(img[:,:,1], bins=bins, range=(0, 256))[0]
    b_hist = np.histogram(img[:,:,2], bins=bins, range=(0, 256))[0]
    
    # Normalize the histograms
    r_hist = r_hist / r_hist.sum()
    g_hist = g_hist / g_hist.sum()
    b_hist = b_hist / b_hist.sum()
    
    # Combine into a single feature vector
    return np.concatenate([r_hist, g_hist, b_hist])


def get_image_paths(dataset_path):
    image_paths = []
    labels = []
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    return image_paths, labels


def extract_features(image_paths, extractor_params=None):
    if extractor_params is None:
        extractor_params = {}
    # Only use SIFT
    feature_extractor = cv2.SIFT_create(**extractor_params)
    all_descriptors = []
    descriptors_list = []
    for path in tqdm(image_paths, desc=f'Extracting features (sift, params={extractor_params})'):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            descriptors_list.append(None)
            continue
        keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
        # Uncomment below to visualize keypoints
        # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
        # cv2.imshow("Image", img_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if descriptors is not None:
            all_descriptors.append(descriptors)
        descriptors_list.append(descriptors)
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    return all_descriptors, descriptors_list


def build_vocabulary(descriptors, num_clusters):
    print(f"Clustering {len(descriptors)} descriptors into {num_clusters} visual words...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=0)
    kmeans.fit(descriptors)
    return kmeans


def compute_bovw_histograms(descriptors_list, kmeans, num_clusters, image_paths=None, use_color_histogram=False, color_bins=8):
    histograms = []
    for idx, descriptors in enumerate(tqdm(descriptors_list, desc='Building histograms')):
        if descriptors is None:
            bovw_hist = np.zeros(num_clusters)
        else:
            words = kmeans.predict(descriptors)
            bovw_hist, _ = np.histogram(words, bins=np.arange(num_clusters+1))
        if use_color_histogram:
            # Use image_paths for color histogram
            if image_paths is not None:
                img_path = image_paths[idx]
                color_hist = extract_rgb_histogram(img_path, bins=color_bins)
            else:
                color_hist = np.zeros(24)  # fallback
            full_hist = np.concatenate([bovw_hist, color_hist])
        else:
            full_hist = bovw_hist
        histograms.append(full_hist)
    return np.array(histograms)


def classify(X_train, X_test, y_train, classifier: Literal['svm', 'knn', 'decision'], classifier_params=None, grid_search_params=None):

    if classifier == 'svm':
        param_grid = {'C': [0.1, 1, 10],  
                        'gamma': [1, 0.1, 0.01, 0.001], 
                        'kernel': ['linear', 'poly', 'rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_} Best score: {grid.best_score_}")
        svm_classifier = SVC(**grid.best_params_)
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
    elif classifier == 'knn':
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=1)
        grid.fit(X_train, y_train)
        knn = KNeighborsClassifier(**grid.best_params_)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return y_pred


def run_bovw(classifier, use_color_histogram=True):
    cluster_options = [100]
    nfeatures_options = [1000]
    results = []
    train_image_paths, train_labels = get_image_paths(TRAIN_PATH)
    test_image_paths, test_labels = get_image_paths(TEST_PATH)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    for num_clusters in cluster_options:
        for nfeatures in nfeatures_options:
            print(f"\nTesting NUM_CLUSTERS={num_clusters}, nfeatures={nfeatures}")
            # Extract features
            all_descriptors, train_descriptors_list = extract_features(train_image_paths, nfeatures=nfeatures)
            if all_descriptors.size == 0:
                print("No descriptors found in training set. Skipping.")
                continue
            kmeans = build_vocabulary(all_descriptors, num_clusters)
            X_train = compute_bovw_histograms(train_descriptors_list, kmeans, num_clusters, image_paths=train_image_paths, use_color_histogram=use_color_histogram)
            # Test features
            _, test_descriptors_list = extract_features(test_image_paths, nfeatures=nfeatures)
            X_test = compute_bovw_histograms(test_descriptors_list, kmeans, num_clusters, image_paths=test_image_paths, use_color_histogram=use_color_histogram)
            y_pred = classify(X_train, X_test, y_train, classifier=classifier)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"Accuracy: {acc:.4f}")
            results.append({
                'num_clusters': num_clusters,
                'nfeatures': nfeatures,
                'accuracy': acc,
                'classification_report': report
            })
    # Save results to file
    file_name = f'bovw_results_{classifier}_sift.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {file_name}")


def run_experiments(config):
    results = []
    train_image_paths, train_labels = get_image_paths(TRAIN_PATH)
    test_image_paths, test_labels = get_image_paths(TEST_PATH)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    for extractor_params in config['extractor_params_list']:
        for num_clusters in config['num_clusters_list']:
            for color_bins in config['color_bins_list']:
                for classifier_type in config['classifier_types']:
                    for classifier_params in config['classifier_params_list']:
                        print(f"\nTesting SIFT, params={extractor_params}, clusters={num_clusters}, color_bins={color_bins}, classifier={classifier_type}, clf_params={classifier_params}")
                        all_descriptors, train_descriptors_list = extract_features(train_image_paths, extractor_params=extractor_params)
                        if all_descriptors.size == 0:
                            print("No descriptors found in training set. Skipping.")
                            continue
                        kmeans = build_vocabulary(all_descriptors, num_clusters)
                        X_train = compute_bovw_histograms(train_descriptors_list, kmeans, num_clusters, image_paths=train_image_paths, use_color_histogram=config['use_color_histogram'], color_bins=color_bins)
                        _, test_descriptors_list = extract_features(test_image_paths, extractor_params=extractor_params)
                        X_test = compute_bovw_histograms(test_descriptors_list, kmeans, num_clusters, image_paths=test_image_paths, use_color_histogram=config['use_color_histogram'], color_bins=color_bins)
                        y_pred = classify(X_train, X_test, y_train, classifier_type=classifier_type, classifier_params=classifier_params, grid_search_params=config.get('grid_search_params', None))
                        acc = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        print(f"Accuracy: {acc:.4f}")
                        results.append({
                            'extractor_params': extractor_params,
                            'num_clusters': num_clusters,
                            'color_bins': color_bins,
                            'classifier_type': classifier_type,
                            'classifier_params': classifier_params,
                            'accuracy': acc,
                            'classification_report': report
                        })
    file_name = config.get('results_file', 'experiment_results.json')
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {file_name}")


if __name__ == '__main__':
    config = {
        'extractor_types': ['sift'],
        'extractor_params_list': [
            # nfeatures - number of best features to retain
            # nOctaveLayers - number of layers in each octave. More octaves and layers mean SIFT can find features over a wider range of scales, but it also increases computation time and may produce more keypoints. Fewer octaves/layers make it faster but may miss features at some scales. This trade-off lets you balance accuracy and speed for your application.
            # contrastThreshold - higher values mean less features
            # edgeThreshold - higher values mean more features
            # sigma - amount of Gaussian blur applied to the image before extracting features. Higher values mean more blur and potentially fewer features, while lower values preserve more detail but may introduce noise.
            {'nfeatures': 1000, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6},
        ],
        'num_clusters_list': [32],
        'color_bins_list': [32],
        'classifier_types': ['svm'],
        'classifier_params_list': [
            {},  # default
        ],
        'use_color_histogram': True,
        'grid_search_params': {
            'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'poly', 'rbf']
        },
        'results_file': 'experiment_results.json'
    }
    run_experiments(config)
