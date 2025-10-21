import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import keras

# ⚙️ CONFIG
MODEL_PATH = "flower_classifier_transfer.keras"  # or "flower_classifier_108.keras"
DATA_DIR = os.path.join("data", "train")  # used to recover class names
TEST_DIR = os.path.join("data", "test")
IMG_SIZE = (224, 224)  # 224 for MobileNetV2, 180 for your custom CNN
BATCH_SIZE = 32

# 1️⃣ Load trained model
print(f"🔍 Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

# 2️⃣ Load test dataset (structured like train/)
print(f"📂 Loading test data from: {TEST_DIR}")
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  # important: don't shuffle, keeps labels in order
)

class_names = test_ds.class_names
num_classes = len(class_names)
print(f"✅ Found {num_classes} test classes: {class_names[:5]}...")

# Normalize and prefetch
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.map(lambda x, y: (x / 255.0, y)).prefetch(AUTOTUNE)

# 3️⃣ Evaluate the model
loss, acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# 4️⃣ Generate predictions
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

# 5️⃣ Print classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))
