import json
import numpy as np
from feature_extraction import to_edge_and_colour
from utils import get_image_paths


print('Creating image feature set')
class_name_and_paths = get_image_paths(folder_name='train')
training_data = {}
for class_name, paths in class_name_and_paths.items():
    training_data[class_name] = []
    for path in paths:
        image_array = to_edge_and_colour(path, (4,4), (8,8))  # currently the best feature extraction with an svm
        normalised_input = (image_array-np.min(image_array))/(np.max(image_array)-np.min(image_array))
        training_data[class_name].append(normalised_input)

x_list = []
y_list = []
for class_name, list_of_feats in training_data.items():
    for feats in list_of_feats:
        x_list.append(feats)
        y_list.append(class_name)
X = np.vstack(x_list)
Y = np.asarray(y_list)
print('Convert to numpy arrays')

with open('training_features.json', 'w') as f:
    json.dump({'training_data': [x.tolist() for x in X], 'labels': Y.tolist()}, f)
