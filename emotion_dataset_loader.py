import tensorflow as tf

# Dataset paths
train_path = r"C:\Users\Admin\OneDrive\Desktop\face_detection\images\train"
test_path = r"C:\Users\Admin\OneDrive\Desktop\face_detection\images\test"

img_size = (48, 48)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=img_size,
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=32
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=32
)

class_names = train_ds.class_names
print("✅ Dataset loaded successfully!")
print("Classes:", class_names)
