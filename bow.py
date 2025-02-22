import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import math


def get_sift_features(image_path):
    num_features = 50
    img = cv2.imread(image_path)
    img = cv2.resize(img,(150,150))
    # Applying SIFT detector
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(num_features)  # control number of features here
    _, des = sift.detectAndCompute(gray,None)
    if len(des) > num_features:
        des = des[:num_features]
    elif len(des) < num_features:
        des = np.concatenate((des, [des[0]] * (num_features - len(des))))
    return des

#this function will get SIFT descriptors from training images and 
#train a k-means classifier    
def read_and_clusterize(class_name_and_paths, num_clusters):

    sift_keypoints = []

    for _, paths in class_name_and_paths.items():
        for path in paths:
            des = get_sift_features(path)
            #append the descriptors to a list of descriptors
            sift_keypoints.append(des)

    #with the descriptors detected, lets clusterize them
    print("Training kmeans") 
    sift_keypoints = np.vstack(sift_keypoints)
    kmeans = MiniBatchKMeans(num_clusters).fit(sift_keypoints)
    #return the learned model
    return kmeans


def create_training_data(class_name_and_paths, kmeans, num_clusters):

    training_data = {}
    for class_name, paths in class_name_and_paths.items():
        training_data[class_name] = []
        for path in paths:
            des = get_sift_features(path)
            predict_kmeans=kmeans.predict(des)
            hist, _ = np.histogram(predict_kmeans, math.floor(math.sqrt(num_clusters)))
            training_data[class_name].append(hist)

    return training_data

