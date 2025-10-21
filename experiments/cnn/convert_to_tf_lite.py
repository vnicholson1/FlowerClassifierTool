import os
import tensorflow as tf
import keras

# Load your trained model
model = keras.models.load_model("experiments/cnn/flower_classifier_transfer_ft.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Convert
tflite_model = converter.convert()

# Save
with open("flower_classifier_transfer_ft.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as flower_classifier_transfer_ft.tflite")

data_dir = os.path.join("data", "train")
img_size = (224, 224)
batch_size = 32
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

with open("flower_labels.txt", "w") as f:
    for name in train_ds.class_names:  # or use train_ds.class_names
        f.write(name + "\n")
print("✅ Class labels saved as flower_labels.txt")
