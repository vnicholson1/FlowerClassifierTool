import tensorflow as tf
import keras
from keras import layers, models
import os

# ⚙️ CONFIG
FINE_TUNE = True          # ← enable this to fine-tune MobileNetV2
FINE_TUNE_AT = 100        # ← unfreeze layers from this index upward

# 1️⃣ Dataset paths
data_dir = os.path.join("data", "train")

# 2️⃣ Dataset loading
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
val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
num_classes = len(train_ds.class_names)
print(f"✅ Found {num_classes} classes.")

# 3️⃣ Prefetch + normalize
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (x / 255.0, y)).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(AUTOTUNE)

# 4️⃣ Model
print("⚡ Using MobileNetV2 backbone")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# 5️⃣ Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6️⃣ Train (phase 1 — frozen)
epochs = 10
print(f"🚀 Starting training for {epochs} epochs (frozen)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 7️⃣ Optional fine-tuning
if FINE_TUNE:
    print("\n🎯 Fine-tuning MobileNetV2 from layer", FINE_TUNE_AT)

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_epochs = 5
    total_epochs = epochs + fine_tune_epochs
    print(f"🚀 Continuing training for {fine_tune_epochs} fine-tuning epochs...")
    history_fine = model.fit(train_ds, validation_data=val_ds,
                             epochs=total_epochs, initial_epoch=history.epoch[-1])

# 9️⃣ Save
filename = "flower_classifier_transfer_ft.keras" if FINE_TUNE else \
           "flower_classifier_transfer.keras"

model.save(filename)
print(f"✅ Model saved as {filename}")
