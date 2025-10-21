# To run this script do:

python experiments/experiments_feature_extraction_knn.py --k 3

## k-NN Results:

Feature extraction with no noise reduction:

k=3, Accuracy = 41.7%, distance method = manhattan, scoring = weighted

Feature extraction with simple 20% center cropping:

k=3, Accuracy = 44.5%, distance method = manhattan, scoring = weighted

## CNN Results:

flower_classifier_transfer.keras - 84.13% accuracy
flower_classifier_transfer_tf.keras - 86.37% accuracy