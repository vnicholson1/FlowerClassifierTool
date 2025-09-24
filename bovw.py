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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import json

# Parameters
NUM_CLUSTERS = 50  # Number of visual words
TRAIN_PATH = 'data/train'  # Path to training images
TEST_PATH = 'data/test'    # Path to test images


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


def extract_features(image_paths, nfeatures=500, features: Literal['orb', 'sift', 'brisk', 'akaze', 'kaze'] = 'orb'):
    # Select feature extractor based on 'features' argument
    if features == 'orb':
        feature_extractor = cv2.ORB_create(nfeatures=nfeatures)
    elif features == 'sift':
        feature_extractor = cv2.SIFT_create(nfeatures=nfeatures)
    elif features == 'brisk':
        feature_extractor = cv2.BRISK_create()
    elif features == 'akaze':
        feature_extractor = cv2.AKAZE_create()
    elif features == 'kaze':
        feature_extractor = cv2.KAZE_create()
    else:
        raise ValueError(f"Unknown feature type: {features}")

    all_descriptors = []
    descriptors_list = []
    for path in tqdm(image_paths, desc=f'Extracting features ({features}, nfeatures={nfeatures})'):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            descriptors_list.append(None)
            continue
        _, descriptors = feature_extractor.detectAndCompute(img, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
        descriptors_list.append(descriptors)
    if all_descriptors:
        all_descriptors = np.vstack(all_descriptors)
    else:
        all_descriptors = np.array([])
    return all_descriptors, descriptors_list


def build_vocabulary(descriptors, num_clusters=NUM_CLUSTERS):
    print(f"Clustering {len(descriptors)} descriptors into {num_clusters} visual words...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=0)
    kmeans.fit(descriptors)
    return kmeans


def compute_bovw_histograms(descriptors_list, kmeans, num_clusters=NUM_CLUSTERS):
    histograms = []
    for descriptors in tqdm(descriptors_list, desc='Building histograms'):
        if descriptors is None:
            hist = np.zeros(num_clusters)
        else:
            words = kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(num_clusters+1))
        histograms.append(hist)
    return np.array(histograms)


def classify(X_train, X_test, y_train, classifier: Literal['svm', 'knn', 'decision']):

    if classifier == 'svm':
        param_grid = {'C': [0.1, 1, 10],  
                        'gamma': [1, 0.1, 0.01, 0.001], 
                        'kernel': ['linear', 'poly', 'rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
        grid.fit(X_train, y_train)
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
    elif classifier == 'decision':
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, refit=True, verbose=1)
        grid.fit(X_train, y_train)
        dt = DecisionTreeClassifier(**grid.best_params_, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return y_pred


def run_bovw(classifier, features):
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
            all_descriptors, train_descriptors_list = extract_features(train_image_paths, nfeatures=nfeatures, features=features)
            if all_descriptors.size == 0:
                print("No descriptors found in training set. Skipping.")
                continue
            kmeans = build_vocabulary(all_descriptors, num_clusters)
            X_train = compute_bovw_histograms(train_descriptors_list, kmeans, num_clusters)
            # Test features
            _, test_descriptors_list = extract_features(test_image_paths, nfeatures=nfeatures, features=features)
            X_test = compute_bovw_histograms(test_descriptors_list, kmeans, num_clusters)
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
    file_name = f'bovw_results_{classifier}_{features}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"All results saved to {file_name}")


if __name__ == '__main__':
    for feature in ['sift','kaze']:
        for classifier in ['knn', 'svm', 'decision']:
            run_bovw(classifier=classifier, features=feature)
