from flask import Flask, render_template, request

from feature_extraction import to_edge_and_colour
from nearest_neighbour import NearestNeighbourClassifier
from utils import get_image_paths
from PIL import Image
import base64


app = Flask(__name__)


class_name_and_paths = get_image_paths(use_train=True)
training_data = {}
for class_name, paths in class_name_and_paths.items():
    training_data[class_name] = []
    for path in paths:
        image = Image.open(path)
        image_array = to_edge_and_colour(image, (8,8), (16,16))
        training_data[class_name].append(image_array)

k = 5
knn = NearestNeighbourClassifier(training_data, k=k)


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['upload']
    img = Image.open(file)
    feature_array = to_edge_and_colour(img, (8,8), (16,16))
    top_three = knn.classify_top_x(feature_array, k)

    top_x_with_images = []
    for top in top_three:
        image_path = class_name_and_paths[top[0]][0]
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        top_x_with_images.append(top + ('data:image/png;base64, ' + encoded_string.decode('utf-8'),))
    sorted_by_second = sorted(top_x_with_images, key=lambda tup: tup[1], reverse=True)
    return render_template('classify.html', predictions=sorted_by_second)


if __name__ == '__main__':
    app.run(port=4000, host='0.0.0.0')
