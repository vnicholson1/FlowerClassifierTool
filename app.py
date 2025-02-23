from flask import Flask, render_template, request

from feature_extraction import best_feature_extraction
from PIL import Image
import base64
import json
import numpy as np

from utils import get_image_paths
from sklearn import svm

app = Flask(__name__)


# Initialising the classifier
print('Loading training data')
with open('training_features.json') as f:
    training_data = json.load(f)
class_name_and_paths = get_image_paths(folder_name='train')
class_names = list(class_name_and_paths.keys())
X = np.array([np.array(xi) for xi in training_data['training_data']])
Y = np.array(training_data['labels'])

print('Creating the classifier')
svm_classifier = svm.SVC(C=0.1, gamma=1, kernel='linear', probability=True)
svm_classifier.fit(X, Y)

print('Initialisation Complete')


def classify_top_x(image_features, x: int):
    predict_probablilities = svm_classifier.predict_proba(image_features.reshape(1,-1))
    predict_probablilities = predict_probablilities.tolist()[0]
    props_dict = {
        i: prob
        for i, prob in enumerate(predict_probablilities)
    }
    sorted_probs = dict(sorted(props_dict.items(), key=lambda item: item[1], reverse=True))
    result = []
    i = 0
    for class_index, prob in sorted_probs.items():
        if i >= x:
            break
        result.append((class_names[class_index], prob))
        i += 1
    return result


@app.route('/', methods=['GET'])
def main():
    counts = dict()
    for i in training_data['labels']:
        counts[i] = counts.get(i, 0) + 1
    return render_template('index.html', class_counts=counts)


@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['upload']
    img = Image.open(file)
    file.seek(0)
    b64_encoded_upload = 'data:image/png;base64, ' + base64.b64encode(file.read()).decode('utf-8')
    feature_array = best_feature_extraction(img)
    normalised_input = (feature_array-np.min(feature_array))/(np.max(feature_array)-np.min(feature_array))
    top_x = classify_top_x(normalised_input, 5)

    top_x_with_images = []
    for top in top_x:
        image_path = class_name_and_paths[top[0]][0]
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        top_x_with_images.append(top + ('data:image/png;base64, ' + encoded_string.decode('utf-8'),))
    sorted_by_second = sorted(top_x_with_images, key=lambda tup: tup[1], reverse=True)
    return render_template('classify.html', uploaded_file=b64_encoded_upload, predictions=sorted_by_second)


if __name__ == '__main__':
    app.run(port=4000, host='0.0.0.0')
