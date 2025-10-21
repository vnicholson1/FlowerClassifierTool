import os
import random
import string
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# -------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------
MODEL_PATH = "flower_classifier_transfer_ft.tflite"
LABELS_PATH = "flower_labels.txt"
IMG_SIZE = (224, 224)  # must match training input

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
class_name_and_paths = {}
counts = {}

print(f"✅ Model loaded with {len(class_names)} classes.")


# -------------------------------------------------------------
# UTILITY HELPERS
# -------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "jfif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_random_string():
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + normalize image"""
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_top_x(image: Image.Image, x: int = 5):
    """Run inference on image and return top x predictions."""
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    sorted_indices = np.argsort(output)[::-1][:x]
    return [(class_names[i], float(output[i])) for i in sorted_indices]

def get_image_paths():
    """Return dict: {class_name: [list of file paths]}"""
    class_name_and_paths = {}
    for root, dirs, files in os.walk(os.path.join("data", "train")):
        if root == "train":
            continue
        class_name = os.path.basename(root)
        class_name_and_paths[class_name] = [
            os.path.join(root, f) for f in files if allowed_file(f)
        ]
    return class_name_and_paths


def initialise():
    """Initialise paths + counts (for upload pages)"""
    global counts, class_name_and_paths
    print("Loading class paths and counts...")
    class_name_and_paths = get_image_paths()
    counts = {cls: len(paths) for cls, paths in class_name_and_paths.items()}
    counts = dict(sorted(counts.items()))
    return class_names, class_name_and_paths, counts


# Initialise on start
class_names, class_name_and_paths, counts = initialise()
print("Initialisation Complete")

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    try:
        file = request.files["upload"]
        if file and allowed_file(file.filename):
            img = Image.open(file.stream)
            top_x = predict_top_x(img, 5)

            file.stream.seek(0)
            b64_encoded_upload = (
                "data:image/png;base64," + base64.b64encode(file.read()).decode("utf-8")
            )

            top_x_with_images = []
            for label, prob in top_x:
                # Show example image if available
                image_path = None
                if label in class_name_and_paths and class_name_and_paths[label]:
                    image_path = class_name_and_paths[label][0]
                encoded_img = ""
                if image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        encoded_img = "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
                top_x_with_images.append((label, round(prob * 100, 2), encoded_img))

            sorted_by_second = sorted(top_x_with_images, key=lambda tup: tup[1], reverse=True)
            return render_template("index.html", uploaded_file=b64_encoded_upload, predictions=sorted_by_second)
        else:
            return render_template("index.html", status="Error: invalid file type.")
    except Exception as e:
        return render_template("index.html", status=str(e))


@app.route("/upload", methods=["GET"])
def upload_training():
    return render_template("upload.html", class_counts=counts)


@app.route("/new_class", methods=["POST"])
def create_new_class():
    new_class = request.form["new_class"].lower()
    global counts
    if new_class not in counts:
        counts[new_class] = 0
        counts = dict(sorted(counts.items()))
        return render_template("upload.html", class_counts=counts, status=f"{new_class} created successfully!")
    else:
        return render_template("upload.html", class_counts=counts, status=f"{new_class} already exists!")


@app.route("/training", methods=["POST"])
def upload_for_training():
    files = request.files.getlist("upload")
    try:
        for file in files:
            if file.filename == "":
                return render_template("upload.html", class_counts=counts, status="No file selected, try again")

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join("data", "user_input", request.form["classes"])
                new_filename = generate_random_string() + "." + filename.split(".")[-1]
                Path(path).mkdir(parents=True, exist_ok=True)
                file.save(os.path.join(path, new_filename))
            else:
                return render_template("upload.html", status="Error uploading file, try again", class_counts=counts)
        return render_template("upload.html", status=f"Training upload successful for flower(s) {request.form['classes']}", class_counts=counts)
    except Exception as e:
        return render_template("upload.html", status=str(e), class_counts=counts)


@app.route("/class/<class_name>", methods=["GET"])
def class_photos(class_name: str):
    try:
        file_paths = class_name_and_paths.get(class_name, [])
        base64_images = []
        for path in file_paths:
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            base64_images.append("data:image/png;base64," + encoded_string.decode("utf-8"))
        return render_template("photos_of_classes.html", images=base64_images, class_name=class_name)
    except Exception as e:
        return render_template("upload.html", status=str(e), class_counts=counts)


def get_existing_images():
    folder = os.path.join("data", "user_input")
    class_names_and_base64_filename = {}
    for directory, _, _ in os.walk(folder):
        _, class_name = os.path.split(directory)
        for image_path in os.listdir(directory):
            file_path = os.path.join(directory, image_path)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as image_file:
                    b64_encoded_upload = "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
                if class_name not in class_names_and_base64_filename:
                    class_names_and_base64_filename[class_name] = []
                class_names_and_base64_filename[class_name].append((b64_encoded_upload, file_path))
    return class_names_and_base64_filename


@app.route("/validate", methods=["GET"])
def validate_training():
    return render_template("validate.html", existing_images=get_existing_images())


@app.route("/submit_training", methods=["POST"])
def validate_training_submittion():
    approved = request.form.get("approve_button") is not None
    file_path = request.form["filepath"]
    if approved:
        if file_path.startswith("data"):
            new_path = file_path.replace("user_input", "train")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(file_path, new_path)
    else:
        if file_path.startswith("data"):
            os.remove(file_path)
    class_name = request.form["class_name"]
    return_text = "Approved" if approved else "Rejected"
    return render_template("validate.html", existing_images=get_existing_images(), status=f"Flower of class '{class_name}' successfully '{return_text}'.")


@app.route("/classes", methods=["GET"])
def view_classes():
    return render_template("classes.html", class_counts=counts)


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(port=4000, host="0.0.0.0")
