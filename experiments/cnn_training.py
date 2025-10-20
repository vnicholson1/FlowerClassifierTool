import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import os

# 1️⃣ Set dataset paths
# Make sure 'train/' contains one subfolder per flower class
data_dir = os.path.join("data", "train")

# 2️⃣ Create training and validation datasets
img_size = (180, 180)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,        # 20% for validation
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

num_classes = len(train_ds.class_names)
print(f"✅ Found {num_classes} flower classes.")

# 3️⃣ Prefetch and normalize data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(buffer_size=AUTOTUNE)

# 4️⃣ Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=img_size + (3,)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# 5️⃣ Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6️⃣ Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# 7️⃣ Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# 8️⃣ Save the model
model.save("flower_classifier_108.keras")
print("✅ Model saved as flower_classifier_108.keras")
