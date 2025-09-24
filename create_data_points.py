import json
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import joblib

NUM_CLUSTERS = 100
NUM_FEATURES = 1000
TRAIN_PATH = 'data/train'

def get_image_paths(dataset_path):
    image_paths = []
    labels = []
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    return image_paths, labels

def extract_sift_features(image_paths):
    feature_extractor = cv2.SIFT_create(nfeatures=NUM_FEATURES)
    all_descriptors = []
    descriptors_list = []
    for path in tqdm(image_paths, desc='Extracting SIFT features'):
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

def run_train():
    print('Extracting SIFT features for BoVW...')
    image_paths, labels = get_image_paths(TRAIN_PATH)
    all_descriptors, descriptors_list = extract_sift_features(image_paths)
    if all_descriptors.size == 0:
        print("No descriptors found in training set. Exiting.")
        return
    kmeans = build_vocabulary(all_descriptors, NUM_CLUSTERS)
    # Save the fitted KMeans model for later use
    joblib.dump(kmeans, 'bovw_kmeans.pkl')
    print('Saved KMeans model to bovw_kmeans.pkl')
    X = compute_bovw_histograms(descriptors_list, kmeans, NUM_CLUSTERS)
    Y = np.asarray(labels)
    print('Saving BoVW histograms and labels...')
    with open('training_features.json', 'w') as f:
        json.dump({'training_data': [x.tolist() for x in X], 'labels': Y.tolist()}, f)

if __name__ == '__main__':
    run_train()
