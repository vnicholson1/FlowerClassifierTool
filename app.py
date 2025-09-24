import os
import random
import string
from flask import Flask, render_template, request
import joblib
from werkzeug.utils import secure_filename
from pathlib import Path


from PIL import Image
import base64
import json
import numpy as np

from create_data_points import run_train
from utils import best_feature_extraction, get_image_paths
from sklearn import svm

app = Flask(__name__)

kmeans = joblib.load('bovw_kmeans.pkl')

def initialise():
    # Initialising the classifier
    print('Loading training data')
    with open('training_features.json') as f:
        training_data = json.load(f)

    print(f"Number of training data {len(training_data['training_data'])}")
    class_name_and_paths = get_image_paths(folder_name='train')
    class_names = list(class_name_and_paths.keys())
    class_names = sorted(class_names)
    X = np.array([np.array(xi) for xi in training_data['training_data']])
    Y = np.array(training_data['labels'])

    # from sklearn.model_selection import GridSearchCV
    # print('Tuning the classifier')
    # param_grid = {'C': [0.1, 1, 10],  
    #                     'gamma': [1, 0.1, 0.01, 0.001], 
    #                     'kernel': ['linear', 'poly', 'rbf']}
    # grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) 
    # grid.fit(X, Y)
    # print(grid.best_params_)
    # svm_classifier = svm.SVC(**grid.best_params_, probability=True)


    print('Creating the classifier')
    # print(f'Best params: {grid.best_params_} Best score: {grid.best_score_}')
    svm_classifier = svm.SVC(C=1, gamma=0.1, kernel='linear', probability=True)
    svm_classifier.fit(X, Y)
    counts = dict()
    for i in training_data['labels']:
        counts[i] = counts.get(i, 0) + 1
    counts = dict(sorted(counts.items()))
    return class_names, class_name_and_paths, counts, svm_classifier

class_names, class_name_and_paths, counts, svm_classifier = initialise()

print('Initialisation Complete')
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'tiff', 'jfif'])


def generate_random_string():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))


def allowed_file(filename):     
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        result.append((class_names[class_index], round(prob*100, 4)))
        i += 1
    return result


# navigation pane
@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['upload']
        if file and allowed_file(file.filename):
            img = Image.open(file)
            file.seek(0)
            b64_encoded_upload = 'data:image/png;base64, ' + base64.b64encode(file.read()).decode('utf-8')
            feature_array = best_feature_extraction(img, kmeans)
            normalised_input = (feature_array-np.min(feature_array))/(np.max(feature_array)-np.min(feature_array))
            top_x = classify_top_x(normalised_input, 5)

            top_x_with_images = []
            for top in top_x:
                image_path = class_name_and_paths[top[0]][0]
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                top_x_with_images.append(top + ('data:image/png;base64, ' + encoded_string.decode('utf-8'),))
            sorted_by_second = sorted(top_x_with_images, key=lambda tup: tup[1], reverse=True)
            return render_template('index.html', uploaded_file=b64_encoded_upload, predictions=sorted_by_second)
        else:
            return render_template('index.html', status='Error uploading file, try again')
    except Exception as e:
        return render_template('index.html', status=str(e))


@app.route('/upload', methods=['GET'])
def upload_training():
    return render_template('upload.html', class_counts=counts)


@app.route('/new_class', methods=['POST'])
def create_new_class():
    new_class = request.form['new_class'].lower()
    global counts
    if new_class not in counts:
        counts[new_class] = 0
        counts = dict(sorted(counts.items()))
        return render_template('upload.html', class_counts=counts, status=f'{new_class} created successfully!')
    else:
        return render_template('upload.html', class_counts=counts, status=f'{new_class} already exists!')


@app.route('/training', methods=['POST'])
def upload_for_training():
    files = request.files.getlist('upload')
    try:
        for file in files:
            if file.filename == '':
                return render_template('upload.html', class_counts=counts, status='No file selected, try again')
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                path = os.path.join('data', 'user_input', request.form['classes'])
                new_filename = generate_random_string() + '.' + filename.split('.')[-1]
                Path(path).mkdir(parents=True, exist_ok=True)
                file.save(os.path.join(path, new_filename))
            else:
                return render_template('upload.html', status='Error uploading file, try again', class_counts=counts)
        return render_template('upload.html', status=f"Training upload successful for flower(s) {request.form['classes']}", class_counts=counts)

    except Exception as e:
        return render_template('upload.html', status=str(e), class_counts=counts)
    

@app.route('/class/<class_name>', methods=['GET'])
def class_photos(class_name: str):
    try:
        file_paths = class_name_and_paths[class_name]
        base64_images = []
        for path in file_paths:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            base64_images.append('data:image/png;base64, ' + encoded_string.decode('utf-8'))
        return render_template('photos_of_classes.html', images=base64_images, class_name=class_name)
    except Exception as e:
        return render_template('upload.html', status=str(e), class_counts=counts)
    


def get_existing_images():
    folder = os.path.join("data", "user_input")
    class_names_and_base64_filename = {}
    for directory, _, _ in os.walk(folder):
        _, class_name = os.path.split(directory)
        for image_path in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, image_path)):
                file_path = os.path.join(directory, image_path)
                with open(file_path, "rb") as image_file:
                    b64_encoded_upload = 'data:image/png;base64, ' + base64.b64encode(image_file.read()).decode('utf-8')
                if class_name not in class_names_and_base64_filename:
                    class_names_and_base64_filename[class_name] = []
                class_names_and_base64_filename[class_name].append((b64_encoded_upload, file_path))
    return class_names_and_base64_filename


@app.route('/validate', methods=['GET'])
def validate_training():
    return render_template('validate.html', existing_images=get_existing_images())


@app.route('/submit_training', methods=['POST'])
def validate_training_submittion():
    approved = request.form.get('approve_button') is not None
    file_path = request.form['filepath']
    if approved:
        # bit of a crud but to stop people tampering with the filepath
        if file_path.startswith('data'):
            new_path = file_path.replace('user_input', 'train')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(file_path, new_path)
    else:
        # bit of a crud but to stop people tampering with the filepath
        if file_path.startswith('data'):
            os.remove(file_path)
    class_name = request.form['class_name']
    return_text = "Approved" if approved else "Rejected"
    return render_template('validate.html', existing_images=get_existing_images(), status=f"Flower of class '{class_name}' successfully '{return_text}'.")


@app.route('/classes', methods=['GET'])
def view_classes():
    return render_template('classes.html', class_counts=counts)


@app.route('/run_training', methods=['POST'])
def run_training():
    try:
        run_train()
        global class_names, class_name_and_paths, counts, svm_classifier
        class_names, class_name_and_paths, counts, svm_classifier = initialise()
        return render_template('classes.html', class_counts=counts, status="Training run successfully")
    except Exception as e:
        return render_template('classes.html', class_counts=counts, status=str(e))


if __name__ == '__main__':
    app.run(port=4000, host='0.0.0.0')
