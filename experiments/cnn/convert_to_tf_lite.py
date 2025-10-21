import tensorflow as tf
import keras

# Load your trained model
model = keras.models.load_model("experiments/cnn/flower_classifier_transfer_ft.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: enable optimization for smaller size (slightly slower but great for mobile)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open("experiments/cnn/flower_classifier_transfer_ft.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as flower_classifier_transfer_ft.tflite")


with open("experiments/cnn/flower_labels.txt", "w") as f:
    for name in model.class_names:  # or use train_ds.class_names
        f.write(name + "\n")
print("✅ Class labels saved as flower_labels.txt")
