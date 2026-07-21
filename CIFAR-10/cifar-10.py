import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==========================================================
# Load CIFAR-10 Dataset
# ==========================================================

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

# ==========================================================
# Normalize Images
# ==========================================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# ==========================================================
# Class Names
# ==========================================================

classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# ==========================================================
# Data Augmentation
# ==========================================================

augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.10),
    RandomZoom(0.10)
])

# ==========================================================
# CNN Model
# ==========================================================

model = Sequential()

model.add(augmentation)

# Block 1
model.add(Conv2D(32, (3,3), padding='same',
                 activation='relu',
                 input_shape=(32,32,3)))

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.25))

# Block 2

model.add(Conv2D(64,(3,3),padding='same',
                 activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.30))

# Block 3

model.add(Conv2D(128,(3,3),padding='same',
                 activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.40))

# Dense Layers

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

# ==========================================================
# Compile
# ==========================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)

# ==========================================================
# Train
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=40,
    batch_size=64,
    callbacks=[early_stop,reduce_lr]
)

# ==========================================================
# Evaluate
# ==========================================================

loss, accuracy = model.evaluate(X_test,y_test)

print("\nTest Accuracy : {:.2f}%".format(accuracy*100))

# ==========================================================
# Accuracy Graph
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")

plt.legend()

plt.grid()

plt.show()

# ==========================================================
# Loss Graph
# ==========================================================

plt.figure(figsize=(10,5))

plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Graph")

plt.legend()

plt.grid()

plt.show()

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions,axis=1)

# ==========================================================
# Classification Report
# ==========================================================

print(classification_report(
    y_test.flatten(),
    predicted_classes,
    target_names=classes
))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test.flatten(),
    predicted_classes
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# ==========================================================
# Display Sample Predictions
# ==========================================================

plt.figure(figsize=(14,8))

for i in range(12):

    plt.subplot(3,4,i+1)

    plt.imshow(X_test[i])

    plt.title(
        "Actual : {}\nPred : {}".format(
            classes[y_test[i][0]],
            classes[predicted_classes[i]]
        ),
        fontsize=9
    )

    plt.axis("off")

plt.tight_layout()

plt.show()

# ==========================================================
# Save Model
# ==========================================================

model.save("CIFAR10_CNN_Model.keras")

print("\nModel Saved Successfully.")