import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Load Dataset
# -------------------------

(train_ds, test_ds), info = tfds.load(
    'oxford_iiit_pet',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = info.features['label'].num_classes
CLASS_NAMES = info.features['label'].names

print("Classes:", NUM_CLASSES)

# -------------------------
# Resize Images
# -------------------------

IMG_SIZE = 128

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# -------------------------
# Show Dataset
# -------------------------

plt.figure(figsize=(10,10))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        plt.title(CLASS_NAMES[labels[i]])
        plt.axis("off")

plt.show()

# -------------------------
# Build CNN
# -------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')

])

model.summary()

# -------------------------
# Compile
# -------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Train
# -------------------------

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# -------------------------
# Evaluate
# -------------------------

loss, accuracy = model.evaluate(test_ds)

print("Test Accuracy:", accuracy)

# -------------------------
# Save Model
# -------------------------

model.save("OxfordPetClassifier.keras")

print("Model Saved!")

# -------------------------
# Accuracy Graph
# -------------------------

plt.figure(figsize=(10,5))

plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.show()

# -------------------------
# Loss Graph
# -------------------------

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.show()