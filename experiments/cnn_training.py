import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import os

# ⚡ Toggle: Use transfer learning or custom CNN
USE_TRANSFER_LEARNING = True  # Set to False to use your custom CNN

# 1️⃣ Set dataset paths
data_dir = os.path.join("data", "train")

# 2️⃣ Create training and validation datasets
img_size = (224, 224) if USE_TRANSFER_LEARNING else (180, 180)
batch_size = 32

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
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

# 4️⃣ Define model
if USE_TRANSFER_LEARNING:
    print("⚡ Using MobileNetV2 Transfer Learning")

    # Load pre-trained MobileNetV2 without top layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze backbone

    # Add classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
else:
    print("⚡ Using Custom CNN")
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
epochs = 10 if USE_TRANSFER_LEARNING else 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
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

# 8️⃣ Save model
filename = "flower_classifier_transfer.keras" if USE_TRANSFER_LEARNING else "flower_classifier_108.keras"
model.save(filename)
print(f"✅ Model saved as {filename}")
