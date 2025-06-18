import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ✅ Paths to your dataset
train_path = "C:/Users/Admin/OneDrive/Desktop/face_detection/images/train"
test_path = "C:/Users/Admin/OneDrive/Desktop/face_detection/images/test"

# ✅ Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    label_mode='categorical'
)

print("✅ Dataset loaded successfully!")

# ✅ Save class names before normalization
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# ✅ Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ Build CNN model
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model
model.fit(train_ds, validation_data=test_ds, epochs=10)

# ✅ Save the model
model.save("emotion_model.h5")
print("🎉 Model training complete. File saved as emotion_model.h5 ✅")
